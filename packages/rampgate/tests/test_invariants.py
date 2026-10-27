"""SPEC §10 invariants under hypothesis-generated scenarios run through the simulator.

Checked here: I1-I7, I9, I10, I13, I15, I17 (per tick), I11 (fact-event replay) and
I11b (command idempotence). I12 is checked by the reducer state machine
(the simulator never produces LOST); I14 by ``test_sim``; I8 belongs to L3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from rampgate import (
    TERMINAL_STATES,
    CancelRequested,
    Config,
    Device,
    DeviceAdded,
    DeviceRemoved,
    Event,
    Fail,
    FailReason,
    FailurePolicy,
    FifoPolicy,
    HistoryEstimator,
    InMemoryHistoryStore,
    InvalidTransition,
    Job,
    JobExited,
    JobObserved,
    JobStarted,
    JobState,
    JobSubmitted,
    JobUpdated,
    OrderingPolicy,
    Paused,
    PreemptRequested,
    PriorityPolicy,
    Resources,
    ResumeRequested,
    Snapshot,
    StagePolicy,
    Start,
    State,
    Stop,
    Unpaused,
    Warn,
    WarnCleared,
    decide,
    device_loads,
    is_edge_triggered,
    key_id,
    reduce,
    warning_key,
)
from rampgate import DeviceObserved as DeviceObs
from rampgate.reducer import _room_at_oom
from rampgate.sim import (
    Command,
    EpochEval,
    RampPlateau,
    Scenario,
    SimJob,
    Spiky,
    Tick,
    Trace,
    simulate,
)
from rampgate_testkit import examples

CFG = Config(
    min_warmup_s=5,
    stable_window_s=10,
    stable_min_steps=5,
    max_warmup_s=60,
    start_timeout_s=30,
    preempt_grace_s=5,
)
GROUPS = ["g0", "g1", "g2"]
# Shared fingerprints give later jobs trusted (level 0) or fallback (level 1) estimates.
FINGERPRINTS: list[tuple[str, ...]] = [(), ("f0",), ("f1",), ("f1", "f0")]
SETTINGS = settings(max_examples=examples(0.8))

vram = st.integers(1, 95).map(lambda v: Resources(vram=v))
common = {
    "duration_s": st.integers(10, 150).map(float),
    "checkpoint_every_s": st.integers(5, 60).map(float),
}
workloads = st.one_of(
    st.builds(
        RampPlateau,
        load_s=st.integers(1, 20).map(float),
        ramp_s=st.integers(1, 30).map(float),
        plateau=vram,
        noise=st.sampled_from([0.0, 0.05]),
        **common,
    ),
    st.builds(Spiky, base=vram, spike=vram, every_s=st.integers(10, 60).map(float), **common),
    st.builds(
        EpochEval,
        base=vram,
        eval_spike=vram,
        epoch_s=st.integers(10, 40).map(float),
        eval_s=st.integers(2, 10).map(float),
        load_s=st.integers(0, 10).map(float),
        **common,
    ),
)


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[Scenario, OrderingPolicy]:
    n_dev = draw(st.integers(1, 3))
    devices = tuple(Device(f"gpu:{i}", Resources(vram=100)) for i in range(n_dev))
    ids = [d.id for d in devices]
    jobs = []
    for i in range(draw(st.integers(1, 7))):
        placement_ids = draw(st.lists(st.sampled_from(ids), min_size=1, max_size=2, unique=True))
        hint = draw(st.one_of(st.none(), vram))
        job = Job(
            f"j{i}",
            dict.fromkeys(placement_ids, hint),
            group=draw(st.one_of(st.none(), st.sampled_from(GROUPS))),
            position=draw(st.integers(0, 3)),
            priority=draw(st.integers(0, 5)),
            fingerprint=draw(st.sampled_from(FINGERPRINTS)),
            force=draw(st.booleans().map(lambda b: b and draw(st.integers(0, 9)) == 0)),
        )
        jobs.append(SimJob(job, draw(workloads), submit_at=float(draw(st.integers(0, 60)))))
    job_ids = [sj.job.id for sj in jobs]
    makers: list[Callable[[str], Event]] = [PreemptRequested, ResumeRequested, CancelRequested]
    commands = tuple(
        Command(float(draw(st.integers(0, 200))), make(draw(st.sampled_from(job_ids))))
        for make in draw(st.lists(st.sampled_from(makers), max_size=4))
    )
    foreign = {
        d: draw(st.one_of(st.just(Resources()), vram.map(lambda r: r.scale(0.2)))) for d in ids
    }
    policy: OrderingPolicy = draw(
        st.one_of(
            st.builds(
                StagePolicy,
                st.permutations(GROUPS),
                backfill=st.booleans(),
                failure_policy=st.dictionaries(
                    st.sampled_from(GROUPS), st.sampled_from(list(FailurePolicy))
                ),
            ),
            st.builds(FifoPolicy, strict=st.booleans()),
            st.builds(
                PriorityPolicy,
                aging_per_s=st.sampled_from([0.0, 0.05]),
                head_of_line_blocking=st.booleans(),
            ),
        )
    )
    seed = draw(st.integers(0, 3))
    return Scenario(devices, tuple(jobs), commands, foreign, seed), policy


def config_for(scenario: Scenario) -> Config:
    """Both Rule P modes are exercised: odd seeds use ``probation="all"``."""
    return replace(CFG, probation="all" if scenario.seed % 2 else "untrusted_only")


def run(scenario: Scenario, policy: OrderingPolicy, **kw: Any) -> Trace:
    estimator = HistoryEstimator(InMemoryHistoryStore())
    cfg = config_for(scenario)
    return simulate(scenario, policy, config=cfg, estimator=estimator, max_time=400.0, **kw)


# -- per-tick checks ----------------------------------------------------------------------------


def check_i1(state: State, cfg: Config = CFG) -> None:
    """I1: at most one untrusted STARTING/WARMING job per device (every job with "all")."""
    for d in state.devices:
        warming = [
            rt.id
            for rt in state.jobs_on(d)
            if rt.state in (JobState.STARTING, JobState.WARMING) and d not in rt.trusted
        ]
        assert len(warming) <= 1, (d, warming)
        if cfg.probation == "all":
            assert not any(rt.trusted for rt in state.jobs_on(d))


def check_starts(
    tick: Tick, policy: OrderingPolicy, cfg: Config = CFG, *, history: bool = False
) -> None:
    """I2, I3, I9, I10 (resume flag), I15 and I17, replaying the decision on its snapshot.

    ``history``: the run used a `HistoryEstimator`, so a sibling of a STABLE job
    (shared fingerprint key, same device) must start with an estimate (Rule H + P).
    """
    work = tick.snapshot.state
    order: list[str] | None = None
    started: list[str] = []
    for action in tick.decision.actions:
        if isinstance(action, Start):
            rt = work.jobs[action.job_id]
            if order is None:
                order = list(policy.candidates(work, tick.now))
            assert action.placement == rt.job.devices  # I9
            assert action.resume == rt.has_run  # I10
            loads = device_loads(work, tick.now, cfg)
            for d in action.trusted:  # trusted only from the most specific level, no OOM yet
                assert tick.snapshot.is_trusted(rt, d), (rt.id, d)
            for d in action.placement:  # Rule H + P: a sibling never starts blind next to it
                if history and tick.snapshot.estimate(rt, d) is None and rt.job.fingerprint:
                    for o in work.jobs_on(d):
                        shared = set(o.job.fingerprint) & set(rt.job.fingerprint)
                        recorded = (
                            o.stable_at is not None
                            and o.stable_at < tick.now
                            and not o.peak.get(d, Resources()).is_zero
                        )
                        assert not (o.state is JobState.STABLE and shared and recorded), (
                            rt.id,
                            o.id,
                        )
            for d in action.placement:  # I17: an untrusted warming job holds d against any start
                for o in work.jobs_on(d):
                    holds = o.state in (JobState.STARTING, JobState.WARMING) or (
                        o.state is JobState.STABLE and o.stable_at == tick.now
                    )
                    assert not (holds and d not in o.trusted), (rt.id, o.id, d)
            for d in action.placement:  # I2 (unknown estimates are checked with unknown_min_free)
                estimate = tick.snapshot.estimate(rt, d)
                if rt.job.force:
                    continue
                load = loads[d]
                if estimate is None:
                    estimate = cfg.unknown_min_free_for(load.capacity)
                used = load.reserved + load.warming_usage + load.foreign + estimate
                for dim, value in used.items():
                    assert value <= load.capacity[dim] * (1 + 1e-9) + 1e-9, (d, dim, used)
            if isinstance(policy, StagePolicy) and not policy.backfill and rt.job.group:  # I3
                groups = policy.groups(work)
                for g in groups[: groups.index(rt.job.group)]:
                    members = [o for o in work.jobs.values() if o.job.group == g]
                    assert all(o.state in TERMINAL_STATES for o in members), (g, rt.id)
            if policy.head_of_line_blocking:  # I15
                for earlier in order[: order.index(rt.id)]:
                    shared = set(work.jobs[earlier].job.placement) & set(rt.job.placement)
                    assert not shared or earlier in started, (earlier, rt.id, shared)
            started.append(rt.id)
        work = reduce(work, action, cfg)


def check_purity(tick: Tick, policy: OrderingPolicy, cfg: Config = CFG) -> None:
    """I4 (determinism) and I5 (no duplicate Start/Stop right after applying a decision)."""
    assert decide(tick.snapshot, tick.now, cfg, policy) == replace(tick.decision, ignored_events=())
    snapshot = Snapshot(tick.after, tick.snapshot.estimates, tick.snapshot.trusted)
    again = decide(snapshot, tick.now, cfg, policy)
    assert not [a for a in again.actions if isinstance(a, Start | Stop)]


def check_exits(prev: State, tick: Tick) -> None:
    """I7 and I13, replaying the tick's events one by one."""
    work = prev
    for event in tick.events:
        before = work.jobs.get(getattr(event, "job_id", ""))
        state_before = work
        try:
            work = reduce(work, event, CFG)
        except InvalidTransition:
            continue
        if not isinstance(event, JobExited) or before is None or before.attempt != event.attempt:
            continue
        after = work.jobs[event.job_id]
        if before.stop_intent is not None:  # I13
            assert after.state in (JobState.SUCCEEDED, JobState.PREEMPTED, JobState.CANCELLED)
            assert after.attempt == before.attempt
        elif event.kind.value == "OOM":  # I7
            assert after.oom_count == before.oom_count + 1
            assert after.oom_count <= CFG.max_oom_retries + 1
            whole_device = False
            for d in before.job.devices:
                prev_est = before.prev_estimate(d) or Resources()
                new_est = after.learned[d]
                cap = work.devices[d].capacity
                room, room_known = _room_at_oom(state_before, before, d, event.t, CFG)
                assert prev_est.fits_in(new_est)
                for dim in event.oom_dims or new_est.keys():
                    lower = max(before.peak.get(d, Resources())[dim], prev_est[dim], room[dim])
                    whole = room[dim] >= cap[dim] * (1 - CFG.foreign_tolerance)
                    if cap[dim] > 0 and whole and room_known:
                        whole_device = True
                    assert new_est[dim] > prev_est[dim] or new_est[dim] >= cap[dim] or lower == 0, (
                        d,
                        dim,
                        prev_est,
                        new_est,
                    )
            capacity_fail = Fail(before.id, before.attempt, FailReason.EXCEEDS_DEVICE_CAPACITY)
            assert (capacity_fail in work.outbox) == whole_device


def check_explained(tick: Tick) -> None:
    """SPEC §3.7: every waiting job gets an explanation (built-in policies)."""
    for rt in tick.after.jobs.values():
        if rt.state in (JobState.PENDING, JobState.PREEMPTED):
            assert rt.id in tick.decision.explanations, rt.id


def check_i16(tick: Tick, raised: dict[str, bool]) -> None:
    """I16: per key, Warn and WarnCleared strictly alternate, starting with Warn."""
    for action in tick.decision.actions:
        if isinstance(action, Warn) and is_edge_triggered(action.warning):
            k = key_id(*warning_key(action.warning))
            assert not raised.get(k, False), ("raised twice", k)
            raised[k] = True
        elif isinstance(action, WarnCleared):
            k = key_id(action.kind, action.key)
            assert raised.get(k, False), ("cleared before raised", k)
            raised[k] = False


def check_groups(state: State) -> None:
    """I10: a PREEMPTED job keeps its group open."""
    for rt in state.jobs.values():
        if rt.state is JobState.PREEMPTED and rt.job.group is not None:
            assert rt.job.group not in state.completed_groups


@SETTINGS
@given(scenarios())
def test_invariants_hold_in_simulation(case: tuple[Scenario, OrderingPolicy]) -> None:
    scenario, policy = case
    cfg = config_for(scenario)
    trace = run(scenario, policy)
    prev = State.initial(scenario.devices)
    reservations: dict[tuple[str, int, str], Resources] = {}
    raised: dict[str, bool] = {}
    for tick in trace.ticks:
        check_exits(prev, tick)
        check_starts(tick, policy, cfg, history=True)
        check_purity(tick, policy, cfg)
        check_explained(tick)
        check_i16(tick, raised)
        for state in (tick.snapshot.state, tick.after):
            check_i1(state, cfg)
            check_groups(state)
            for rt in state.jobs.values():  # I6
                for d, r in rt.reservation.items():
                    key = (rt.id, rt.attempt, d)
                    if key in reservations:
                        assert reservations[key].fits_in(r)
                    reservations[key] = r
        prev = tick.after

    if any(rt.attempt > 1 for rt in trace.final.jobs.values()):  # I14
        reset = run(scenario, policy, reset_counters_on_resume=True)
        assert [t.decision.actions for t in reset.ticks] == [
            t.decision.actions for t in trace.ticks
        ]


# -- I11 / I11b -----------------------------------------------------------------------------------

FACTS = (JobSubmitted, JobStarted, JobExited, JobObserved, DeviceObs)


@settings(max_examples=examples(0.3))
@given(scenarios(), st.data())
def test_i11_replaying_fact_events_changes_nothing(
    case: tuple[Scenario, OrderingPolicy], data: st.DataObject
) -> None:
    scenario, policy = case
    trace = run(scenario, policy)
    facts = [e for t in trace.ticks for e in t.events if isinstance(e, FACTS)]
    n = data.draw(st.integers(0, len(facts)))
    final = trace.final
    replayed = final
    for event in facts[:n]:
        replayed = reduce(replayed, event, CFG)
    assert replace(replayed, ignored_events=()) == replace(final, ignored_events=())
    assert len(replayed.ignored_events) == len(final.ignored_events) + n


def commands_for(state: State) -> st.SearchStrategy[Event]:
    job_ids = sorted(state.jobs) or ["none"]
    dev_ids = sorted(state.devices)
    job = st.sampled_from(job_ids)
    simple = st.one_of(
        st.just(Paused()),
        st.just(Unpaused()),
        job.map(PreemptRequested),
        job.map(ResumeRequested),
        job.map(CancelRequested),
        st.builds(JobUpdated, job, position=st.integers(0, 3), priority=st.integers(0, 3)),
        st.builds(JobUpdated, job, group=st.sampled_from([None, *GROUPS])),
        st.just(DeviceAdded(Device("extra", Resources(vram=10)))),
    )
    return st.one_of(simple, st.sampled_from(dev_ids).map(DeviceRemoved)) if dev_ids else simple


@settings(max_examples=examples(0.3))
@given(scenarios(), st.data())
def test_i11b_commands_are_idempotent(
    case: tuple[Scenario, OrderingPolicy], data: st.DataObject
) -> None:
    scenario, policy = case
    trace = run(scenario, policy)
    for _ in range(5):
        tick = data.draw(st.sampled_from(trace.ticks))
        state = tick.after
        command = data.draw(commands_for(state))
        try:
            once = reduce(state, command, CFG)
        except InvalidTransition:
            continue
        twice = reduce(once, command, CFG)
        assert replace(twice, ignored_events=()) == replace(once, ignored_events=())


def test_scenarios_strategy_produces_work() -> None:
    """Sanity: a plain generated-style scenario actually starts and finishes jobs."""
    scenario = Scenario(
        (Device("gpu:0", Resources(vram=100)),),
        (
            SimJob(
                Job("j0", {"gpu:0": None}),
                RampPlateau(load_s=2, ramp_s=2, plateau=Resources(vram=40), duration_s=20),
            ),
        ),
    )
    trace = run(scenario, FifoPolicy())
    assert trace.final.jobs["j0"].state is JobState.SUCCEEDED
