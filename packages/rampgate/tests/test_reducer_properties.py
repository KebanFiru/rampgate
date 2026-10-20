"""Hypothesis state machine over the reducer.

L1 checks structural consistency (I6, I12, I13 and the OOM part of I7) under
random but well-formed message sequences. The full invariant suite (I1-I14)
runs against decide and the simulator in L2.
"""

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from rampgate import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    CancelRequested,
    Config,
    DeviceObserved,
    ExitKind,
    InvalidTransition,
    JobExited,
    JobState,
    PreemptRequested,
    Resources,
    ResumeRequested,
    Stop,
    StopReason,
)
from rampgate_testkit import GPU, GPU1, Driver, gpus

JOBS = ["a", "b", "c"]
PLACEMENTS = [(GPU,), (GPU1,), (GPU, GPU1)]
CFG = Config(max_oom_retries=2, max_lost_retries=2)


class ReducerMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.d = Driver(gpus(GPU, GPU1), CFG)
        self.t = 0.0
        self.max_attempt: dict[str, int] = {}
        self.reservations: dict[tuple[str, int, str], Resources] = {}

    def tick(self, dt: float) -> float:
        self.t += dt
        return self.t

    def jobs_in(self, *states: JobState) -> list[str]:
        return sorted(j for j, rt in self.d.state.jobs.items() if rt.state in states)

    @rule(
        job=st.sampled_from(JOBS),
        placement=st.sampled_from(PLACEMENTS),
        hint=st.one_of(st.none(), st.integers(1, 120)),
    )
    def submit(self, job: str, placement: tuple[str, ...], hint: int | None) -> None:
        if job in self.d.state.jobs:
            return
        hints = {dev: None if hint is None else Resources(vram=hint) for dev in placement}
        self.d.submit(job, hints)

    @rule(data=st.data(), free=st.integers(0, 100))
    def start(self, data: st.DataObject, free: int) -> None:
        pending = self.jobs_in(JobState.PENDING)
        if pending:
            job = data.draw(st.sampled_from(pending))
            devices = self.d.rt(job).job.devices
            self.d.start(job, {dev: Resources(vram=free) for dev in devices})

    @rule(data=st.data(), dt=st.floats(0.1, 50))
    def started(self, data: st.DataObject, dt: float) -> None:
        starting = self.jobs_in(JobState.STARTING)
        if starting:
            self.d.started(data.draw(st.sampled_from(starting)), t=self.tick(dt))

    @rule(
        data=st.data(),
        dt=st.floats(0.1, 50),
        vram=st.integers(0, 120),
        steps=st.one_of(st.none(), st.integers(0, 1000)),
    )
    def observe(self, data: st.DataObject, dt: float, vram: int, steps: int | None) -> None:
        active = self.jobs_in(*ACTIVE_STATES)
        if active:
            job = data.draw(st.sampled_from(active))
            dev = data.draw(st.sampled_from(self.d.rt(job).job.devices))
            self.d.observe(self.tick(dt), job, dev, steps, vram=vram)

    @rule(dev=st.sampled_from([GPU, GPU1]), dt=st.floats(0.1, 50), used=st.integers(0, 150))
    def device_observed(self, dev: str, dt: float, used: int) -> None:
        self.d(DeviceObserved(dev, self.tick(dt), Resources(vram=used)))

    @rule(data=st.data())
    def mark_stable(self, data: st.DataObject) -> None:
        warming = self.jobs_in(JobState.WARMING)
        if warming:
            self.d.stable(data.draw(st.sampled_from(warming)))

    @rule(
        data=st.data(),
        kind=st.sampled_from(list(ExitKind)),
        dt=st.floats(0.1, 50),
    )
    def exit(self, data: st.DataObject, kind: ExitKind, dt: float) -> None:
        active = self.jobs_in(*ACTIVE_STATES)
        if not active:
            return
        job = data.draw(st.sampled_from(active))
        before = self.d.rt(job)
        self.d(JobExited(job, before.attempt, kind, self.tick(dt)))
        after = self.d.rt(job)
        if before.stop_intent is not None:
            assert after.state in (JobState.SUCCEEDED, JobState.PREEMPTED, JobState.CANCELLED)
            assert after.attempt == before.attempt
        if kind is ExitKind.LOST:
            assert after.oom_count == before.oom_count
            assert after.learned == before.learned
        if kind is ExitKind.OOM and before.stop_intent is not StopReason.CANCEL:
            assert after.oom_count == before.oom_count + 1
            for dev in before.job.devices:
                prev = before.prev_estimate(dev) or Resources()
                new = after.learned[dev]
                cap = self.d.state.devices[dev].capacity
                assert prev.fits_in(new)
                assert (
                    new["vram"] > prev["vram"] or new["vram"] >= cap["vram"] or (new["vram"] == 0)
                )

    @rule(data=st.data(), reason=st.sampled_from(list(StopReason)))
    def request_stop(self, data: st.DataObject, reason: StopReason) -> None:
        if not self.d.state.jobs:
            return
        job = data.draw(st.sampled_from(sorted(self.d.state.jobs)))
        ev = PreemptRequested(job) if reason is StopReason.PREEMPT else CancelRequested(job)
        try:
            self.d(ev)
        except InvalidTransition:
            assert self.d.rt(job).state in TERMINAL_STATES

    @rule(data=st.data())
    def send_stop(self, data: st.DataObject) -> None:
        cands = [
            j
            for j in self.jobs_in(JobState.STARTING, JobState.WARMING, JobState.STABLE)
            if self.d.rt(j).stop_intent is not None
        ]
        if cands:
            rt = self.d.rt(data.draw(st.sampled_from(cands)))
            assert rt.stop_intent is not None
            self.d(Stop(rt.id, rt.attempt, rt.stop_intent, 1.0))

    @rule(data=st.data())
    def resume(self, data: st.DataObject) -> None:
        preempted = self.jobs_in(JobState.PREEMPTED)
        if preempted:
            self.d(ResumeRequested(data.draw(st.sampled_from(preempted))))

    @rule()
    def drain_outbox(self) -> None:
        for action in self.d.state.outbox:
            self.d(action)
        assert self.d.state.outbox == ()

    @invariant()
    def consistent(self) -> None:
        for job, rt in self.d.state.jobs.items():
            assert rt.attempt >= self.max_attempt.get(job, 1)
            self.max_attempt[job] = rt.attempt
            assert rt.oom_count <= CFG.max_oom_retries + 1
            assert rt.lost_count <= CFG.max_lost_retries + 1
            if rt.state is JobState.STOPPING:
                assert rt.stop_intent is not None
            if rt.state not in ACTIVE_STATES:
                assert rt.usage == {}
                assert rt.stop_intent is None
            if rt.reservation:
                assert rt.state in (JobState.STABLE, JobState.STOPPING)
            for dev, res in rt.reservation.items():
                key = (job, rt.attempt, dev)
                if key in self.reservations:
                    assert self.reservations[key].fits_in(res)
                self.reservations[key] = res


ReducerMachine.TestCase.settings = settings(max_examples=150, stateful_step_count=60, deadline=None)
TestReducerMachine = ReducerMachine.TestCase
