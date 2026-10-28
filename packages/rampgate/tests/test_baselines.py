"""Pluggable admission, baselines and ablations (SPEC §8)."""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from rampgate import (
    PROBATION,
    Admission,
    Config,
    Decision,
    Device,
    ExitKind,
    FifoPolicy,
    Job,
    JobState,
    OrderingPolicy,
    ProbationAdmission,
    RejectedByAdmission,
    Requeue,
    RequeueReason,
    Resources,
    Snapshot,
    Start,
    State,
    WaitingForWarmup,
    decide,
    dump_state,
    load_state,
    measured_free,
)
from rampgate.sim import (
    ABLATIONS,
    MAIN,
    STRATEGIES,
    DeclaredReservation,
    GreedyFree,
    GreedyFreeRuleO,
    Scenario,
    Serial,
    SimJob,
    StepProfile,
    Strategy,
    simulate,
)
from rampgate_testkit import GPU, Driver, examples
from test_invariants import check_groups, check_i1, check_i16, check_i20, scenarios

DEV = Device(GPU, Resources(vram=100))
_ACTIVE = (JobState.STARTING, JobState.WARMING, JobState.STABLE, JobState.STOPPING)


def flat(vram: float, duration: float = 200.0) -> StepProfile:
    return StepProfile(steps=((0, Resources(vram=vram)),), duration_s=duration)


def decide_on_admission(d: Driver, admission: Admission) -> Decision:
    return decide(Snapshot(d.state, {}), 3.0, d.config, FifoPolicy(), admission)


# -- admissions on hand-built states ---------------------------------------------------------


def test_measured_free_counts_observed_usage_and_foreign() -> None:
    d = Driver().submit("a").start("a").started("a").observe(1.0, "a", vram=30)
    d.foreign(2.0, vram=20)
    assert measured_free(d.state, GPU, 2.0, d.config) == Resources(vram=50)


def test_greedy_free_starts_while_anything_is_free_and_ignores_probation() -> None:
    d = Driver().submit("a").start("a").started("a").observe(1.0, "a", vram=99)
    d.submit("b")
    dec = decide_on_admission(d, GreedyFree())
    assert [a.job_id for a in dec.actions if isinstance(a, Start)] == ["b"]
    d2 = Driver().submit("a").start("a").started("a").observe(1.0, "a", vram=100)
    d2.submit("b")
    dec2 = decide_on_admission(d2, GreedyFree())
    assert not [a for a in dec2.actions if isinstance(a, Start)]
    assert dec2.explanations["b"] == RejectedByAdmission(GPU, "GreedyFree")


def test_declared_reservation_sums_hints() -> None:
    d = Driver().submit("a", {GPU: Resources(vram=60)}).start("a")
    d.submit("b", {GPU: Resources(vram=40)}).submit("c", {GPU: Resources(vram=41)})
    started = [
        a.job_id
        for a in decide_on_admission(d, DeclaredReservation()).actions
        if isinstance(a, Start)
    ]
    assert "b" in started
    assert "c" not in started


def test_serial_allows_one_job_per_device() -> None:
    d = Driver().submit("a").submit("b")
    dec = decide_on_admission(d, Serial())
    assert [a.job_id for a in dec.actions if isinstance(a, Start)] == ["a"]


def test_greedy_rule_o_uses_learned_estimates() -> None:
    d = Driver().submit("a").start("a").started("a").observe(1.0, "a", vram=50)
    d.submit("b").start("b").exited(ExitKind.OOM, "b")  # room 50: learned 62.5
    assert d.rt("b").state is JobState.PENDING
    dec = decide_on_admission(d, GreedyFreeRuleO())
    assert not [a for a in dec.actions if isinstance(a, Start) and a.job_id == "b"]
    dec_greedy = decide_on_admission(d, GreedyFree())
    assert [a.job_id for a in dec_greedy.actions if isinstance(a, Start)] == ["b"]


def test_probation_off_ablation_starts_several_warming_jobs() -> None:
    d = Driver().submit("a").submit("b")
    on = decide_on_admission(d, PROBATION)
    off = decide_on_admission(d, ProbationAdmission("off", probation=False))
    assert [a.job_id for a in on.actions if isinstance(a, Start)] == ["a"]
    assert isinstance(on.explanations["b"], WaitingForWarmup)
    assert [a.job_id for a in off.actions if isinstance(a, Start)] == ["a", "b"]


def test_registries() -> None:
    assert [s.name for s in MAIN] == [
        "Probation",
        "Probation (all)",
        "GreedyFree",
        "DeclaredReservation",
        "Serial",
    ]
    assert MAIN[0].config.probation == "untrusted_only"
    assert MAIN[1].config.probation == "all"
    assert len(ABLATIONS) == 3
    assert STRATEGIES["Probation (warm)"].warm
    assert STRATEGIES["Probation (all, warm)"].config.probation == "all"
    assert STRATEGIES["rampgate - history"].estimator == "null"
    for strategy in (*MAIN[2:], ABLATIONS[0]):
        assert strategy.config.max_oom_retries >= 1_000_000


# -- in the simulator --------------------------------------------------------------------------


def test_greedy_free_oom_requeues_with_resume_and_never_runs_out_of_retries() -> None:
    jobs = tuple(SimJob(Job(f"j{i}", {GPU: None}), flat(45 + i)) for i in range(3))
    trace = simulate(
        Scenario((DEV,), jobs),
        FifoPolicy(),
        config=STRATEGIES["GreedyFree"].config,
        admission=GreedyFree(),
        max_time=5_000,
    )
    requeues = [
        a
        for t in trace.ticks
        for a in t.decision.actions
        if isinstance(a, Requeue) and a.reason is RequeueReason.OOM
    ]
    assert len(requeues) > Config().max_oom_retries + 1
    resumed = [a for t in trace.ticks for a in t.decision.actions if isinstance(a, Start)]
    assert all(a.resume for a in resumed if a.attempt > 1)
    assert {rt.state for rt in trace.final.jobs.values()} == {JobState.SUCCEEDED}


def test_declared_reservation_ooms_when_under_declared() -> None:
    jobs = tuple(SimJob(Job(f"j{i}", {GPU: Resources(vram=30)}), flat(60)) for i in range(2))
    trace = simulate(
        Scenario((DEV,), jobs),
        FifoPolicy(),
        config=STRATEGIES["DeclaredReservation"].config,
        admission=DeclaredReservation(),
        max_time=5_000,
    )
    assert trace.world is not None
    assert trace.world.oom_count > 0
    assert trace.finished


def test_serial_runs_one_job_at_a_time() -> None:
    jobs = tuple(SimJob(Job(f"j{i}", {GPU: None}), flat(10, 50)) for i in range(3))
    trace = simulate(Scenario((DEV,), jobs), FifoPolicy(), admission=Serial())
    for tick in trace.ticks:
        active = [rt for rt in tick.after.jobs.values() if rt.state in _ACTIVE]
        assert len(active) <= 1


# -- reducer-level invariants hold for every strategy -------------------------------------------


@settings(max_examples=examples(0.3))
@given(scenarios(), st.sampled_from(sorted(STRATEGIES)))
def test_reducer_invariants_hold_for_every_strategy(
    case: tuple[Scenario, OrderingPolicy], name: str
) -> None:
    """Baselines break the admission invariants by design; the reducer's must still hold."""
    scenario, policy = case
    strategy: Strategy = STRATEGIES[name]
    trace = simulate(
        scenario, policy, config=strategy.config, admission=strategy.admission, max_time=300.0
    )
    raised: dict[str, bool] = {}
    for tick in trace.ticks:
        check_i16(tick, raised)
        check_i20(tick)
        for state in (tick.snapshot.state, tick.after):
            check_groups(state)
            if strategy.admission.probation:
                check_i1(state)
    final: State = trace.final
    assert load_state(json.loads(json.dumps(dump_state(final)))) == final  # I8
