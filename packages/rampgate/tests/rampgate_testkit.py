"""Small driver that feeds events/actions through `reduce` for tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from hypothesis import settings

from rampgate import (
    CancelRequested,
    Config,
    Decision,
    Device,
    DeviceObserved,
    ExitKind,
    FifoPolicy,
    IgnoreReason,
    InMemoryHistoryStore,
    Job,
    JobExited,
    JobObserved,
    JobRuntime,
    JobStarted,
    JobState,
    JobSubmitted,
    MarkStable,
    Message,
    OrderingPolicy,
    PreemptRequested,
    Resources,
    Snapshot,
    Start,
    State,
    Stop,
    StopReason,
    decide,
    reduce,
)

GPU = "gpu:0"
GPU1 = "gpu:1"
CAP = Resources(vram=100)


def gpus(*ids: str, cap: Resources = CAP) -> list[Device]:
    return [Device(i, cap) for i in (ids or (GPU,))]


class Driver:
    def __init__(self, devices: Iterable[Device] | None = None, config: Config | None = None):
        self.config = config or Config()
        self.state = State.initial(gpus() if devices is None else devices)

    def __call__(self, *msgs: Message) -> Driver:
        for m in msgs:
            self.state = reduce(self.state, m, self.config)
        return self

    def rt(self, job_id: str = "j") -> JobRuntime:
        return self.state.jobs[job_id]

    @property
    def ignored(self) -> list[IgnoreReason]:
        return [i.reason for i in self.state.ignored_events]

    def submit(
        self,
        job_id: str = "j",
        placement: Mapping[str, Resources | None] | None = None,
        *,
        t: float | None = None,
        **kw: object,
    ) -> Driver:
        job = Job(job_id, placement if placement is not None else {GPU: None}, **kw)  # type: ignore[arg-type]
        return self(JobSubmitted(job, t))

    def start_action(self, job_id: str = "j", free: Mapping[str, Resources] | None = None) -> Start:
        rt = self.rt(job_id)
        devices = rt.job.devices
        admitted = free if free is not None else {d: Resources() for d in devices}
        return Start(job_id, rt.attempt, devices, rt.has_run, admitted, issued_at=0.0)

    def start(self, job_id: str = "j", free: Mapping[str, Resources] | None = None) -> Driver:
        return self(self.start_action(job_id, free))

    def started(self, job_id: str = "j", t: float = 0.0) -> Driver:
        return self(JobStarted(job_id, self.rt(job_id).attempt, t))

    def observe(
        self,
        t: float,
        job_id: str = "j",
        device: str = GPU,
        *,
        steps: int | None = None,
        attempt: int | None = None,
        epoch: int | None = None,
        **usage: float,
    ) -> Driver:
        return self(
            JobObserved(
                job_id, device, t, Resources(usage), steps=steps, epoch=epoch, attempt=attempt
            )
        )

    def foreign(self, t: float = 99.0, device: str = GPU, **usage: float) -> Driver:
        """A ``DeviceObserved`` whose total is the managed usage plus ``usage`` as foreign load."""
        managed = Resources()
        for rt in self.state.jobs_on(device):
            managed = managed + rt.usage.get(device, Resources())
        return self(DeviceObserved(device, t, managed + Resources(usage)))

    def room(self, t: float = 99.0, device: str = GPU, **room: float) -> Driver:
        """Leave exactly ``room`` of the device to its jobs, via foreign load observed at ``t``."""
        cap = self.state.devices[device].capacity
        return self.foreign(t, device, **{k: cap[k] - v for k, v in room.items()})

    def exited(
        self,
        kind: ExitKind,
        job_id: str = "j",
        t: float = 100.0,
        oom_dims: tuple[str, ...] | None = None,
    ) -> Driver:
        return self(JobExited(job_id, self.rt(job_id).attempt, kind, t, oom_dims))

    def mark_stable_action(self, job_id: str = "j") -> MarkStable:
        rt = self.rt(job_id)
        reservation = {
            d: rt.peak.get(d, Resources()).scale(self.config.safety_margin) for d in rt.job.devices
        }
        return MarkStable(job_id, rt.attempt, reservation)

    def stable(self, job_id: str = "j") -> Driver:
        return self(self.mark_stable_action(job_id))

    def stop(self, reason: StopReason, job_id: str = "j") -> Driver:
        event = (
            PreemptRequested(job_id) if reason is StopReason.PREEMPT else CancelRequested(job_id)
        )
        self(event)
        return self(Stop(job_id, self.rt(job_id).attempt, reason, self.config.preempt_grace_s))

    def to(self, target: JobState, job_id: str = "j", **submit_kw: object) -> Driver:
        """Drive a fresh job to ``target`` along the main path."""
        self.submit(job_id, **submit_kw)  # type: ignore[arg-type]
        if target is JobState.PENDING:
            return self
        if target is JobState.CANCELLED:
            return self(CancelRequested(job_id))
        self.start(job_id)
        if target is JobState.STARTING:
            return self
        self.started(job_id, t=0.0)
        if target is JobState.WARMING:
            return self
        if target is JobState.STOPPING:
            return self.stop(StopReason.PREEMPT, job_id)
        if target is JobState.PREEMPTED:
            return self.stop(StopReason.PREEMPT, job_id).exited(ExitKind.PREEMPTED, job_id)
        self.observe(1.0, job_id, vram=10).stable(job_id)
        if target is JobState.STABLE:
            return self
        if target is JobState.SUCCEEDED:
            return self.exited(ExitKind.SUCCESS, job_id)
        if target is JobState.FAILED:
            return self.exited(ExitKind.FAILURE, job_id)
        raise AssertionError(target)


def decide_on(
    d: Driver,
    now: float,
    policy: OrderingPolicy | None = None,
    estimates: Mapping[str, Mapping[str, Resources | None]] | None = None,
) -> Decision:
    """Run ``decide`` on the driver's state (estimates given per job and device)."""
    snapshot = Snapshot(d.state, estimates or {})
    return decide(snapshot, now, d.config, policy or FifoPolicy(strict=False))


def apply(d: Driver, decision: Decision) -> Driver:
    return d(*decision.actions)


def examples(fraction: float) -> int:
    """Example count for an expensive hypothesis test, scaled from the active profile."""
    return max(5, int(settings().max_examples * fraction))


def stored(store: InMemoryHistoryStore, key: str, kind: str) -> Resources:
    """The value a store holds for (key, kind): ``max(complete, newer lower bound)``."""
    entry = store.get(key, kind)
    assert entry is not None, (key, kind)
    return entry.value
