"""Example tests for ``decide``: Rules P and A, explanations, warnings
(docs/concepts.md, docs/algorithm.md)."""

import pytest

from rampgate import (
    AwaitingResume,
    BasePolicy,
    BlockedByForeignUsage,
    CancelRequested,
    CapacityExceeded,
    Config,
    DeviceHeldFor,
    DeviceObserved,
    EnginePaused,
    ExitKind,
    FailurePolicy,
    FifoPolicy,
    ForeignUsageHigh,
    GangPartiallyBlocked,
    GroupBarrier,
    GroupCompleted,
    GroupOutcome,
    HintExceeded,
    InsufficientResources,
    InvalidTransition,
    JobState,
    MarkStable,
    Overcommitted,
    Paused,
    PreemptRequested,
    PriorityPolicy,
    QueueStopped,
    Requeue,
    RequeueReason,
    Resources,
    Snapshot,
    StagePolicy,
    Start,
    Stop,
    StopReason,
    UnknownEstimateBlocked,
    Unpaused,
    WaitingForWarmup,
    Warn,
    WarnCleared,
    decide,
    device_loads,
)
from rampgate_testkit import GPU, GPU1, Driver, apply, decide_on, gpus


def starts(decision: object) -> list[str]:
    return [a.job_id for a in decision.actions if isinstance(a, Start)]  # type: ignore[attr-defined]


def of_type(decision: object, cls: type) -> list[object]:
    return [a for a in decision.actions if isinstance(a, cls)]  # type: ignore[attr-defined]


# -- Rule P / Rule A -------------------------------------------------------------------


def test_one_warming_job_per_device() -> None:
    d = Driver().submit("a").submit("b")
    dec = decide_on(d, 0.0)
    assert starts(dec) == ["a"]
    assert dec.explanations["b"] == WaitingForWarmup(GPU, "a")
    start = of_type(dec, Start)[0]
    assert start == Start("a", 1, (GPU,), False, {GPU: Resources(vram=100)}, 0.0, {GPU: None})


def test_waiting_for_warmup_names_existing_warming_job() -> None:
    d = Driver().to(JobState.WARMING, "a").submit("b")
    assert decide_on(d, 1.0).explanations["b"] == WaitingForWarmup(GPU, "a")


def test_admission_against_free_after_stable_reservations() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d.submit("b", {GPU: Resources(vram=40)}).submit("c", {GPU: Resources(vram=50)})
    dec = decide_on(d, 2.0)
    assert starts(dec) == ["b"]  # free = 100 - 57.5
    d2 = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d2.submit("c", {GPU: Resources(vram=50)})
    why = decide_on(d2, 2.0).explanations["c"]
    assert isinstance(why, InsufficientResources)
    assert (why.device_id, why.dim, why.need) == (GPU, "vram", 50)
    assert why.available == pytest.approx(42.5)


def test_device_loads_accounting() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.WARMING, "s").observe(1.0, "s", vram=40).stable("s")
    d.submit("w").start("w").started("w").observe(2.0, "w", vram=10)
    d(DeviceObserved(GPU, 2.0, Resources(vram=60)))
    load = device_loads(d.state, 3.0, d.config)[GPU]
    assert load.reserved == Resources(vram=46.0)
    assert load.warming_usage == Resources(vram=10)
    assert load.warming_job == "w"
    assert load.foreign == Resources(vram=10)
    assert load.free == Resources(vram=34.0)
    assert load.committed == Resources(vram=66.0)
    assert load.admits(Resources(vram=34))
    assert not load.admits(Resources(vram=35))
    assert load.admits(Resources(vram=44), ignore_foreign=True)
    assert load.available == Resources(vram=90)
    assert device_loads(d.state, 3.0, d.config)[GPU1].free == Resources(vram=100)


def test_unknown_estimate_uses_unknown_min_free() -> None:
    cfg = Config(unknown_min_free=Resources(vram=60))
    d = Driver(config=cfg).to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d.submit("u")
    dec = decide_on(d, 2.0)
    assert dec.explanations["u"] == UnknownEstimateBlocked(GPU)
    d2 = Driver(config=cfg).submit("u")
    assert starts(decide_on(d2, 0.0)) == ["u"]


@pytest.mark.parametrize(("reserved_peak", "admitted"), [(78, True), (79, False)])
def test_unknown_estimate_needs_ten_percent_free_by_default(
    reserved_peak: float, admitted: bool
) -> None:
    # 78 * 1.15 = 89.7 leaves 10.3 free; 79 * 1.15 = 90.85 leaves 9.15.
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=reserved_peak).stable("a")
    d.submit("u")
    assert (starts(decide_on(d, 2.0)) == ["u"]) is admitted


def test_unknown_estimate_refused_on_overcommitted_device() -> None:
    d = Driver(config=Config(unknown_min_free=Resources())).to(JobState.WARMING, "a")
    d.observe(1.0, "a", vram=95).stable("a").submit("u")
    dec = decide_on(d, 2.0)
    assert starts(dec) == []
    assert dec.explanations["u"] == UnknownEstimateBlocked(GPU)


def test_estimate_is_max_of_estimator_hint_and_learned() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d.submit("b", {GPU: Resources(vram=10)})
    why = decide_on(d, 2.0, estimates={"b": {GPU: Resources(vram=45)}}).explanations["b"]
    assert isinstance(why, InsufficientResources)
    assert why.need == 45


def test_force_skips_capacity_but_not_probation() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d.submit("f", {GPU: Resources(vram=500)}, force=True)
    assert starts(decide_on(d, 2.0)) == ["f"]
    d2 = Driver().to(JobState.WARMING, "a").submit("f", {GPU: Resources(vram=1)}, force=True)
    assert decide_on(d2, 2.0).explanations["f"] == WaitingForWarmup(GPU, "a")


def test_gang_is_all_or_nothing() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.WARMING, "a").submit("g", {GPU: None, GPU1: None})
    dec = decide_on(d, 1.0)
    assert starts(dec) == []
    assert dec.explanations["g"] == GangPartiallyBlocked((GPU,))


def test_gang_counts_as_warming_on_every_device() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("g", {GPU: None, GPU1: None}).submit("b", {GPU1: None})
    dec = decide_on(d, 0.0)
    assert starts(dec) == ["g"]
    assert dec.explanations["b"] == WaitingForWarmup(GPU1, "g")
    assert of_type(dec, Start)[0].placement == (GPU, GPU1)  # type: ignore[attr-defined]


def test_stopping_job_that_never_stabilized_holds_the_slot() -> None:
    d = Driver().to(JobState.WARMING, "a").stop(StopReason.PREEMPT, "a").submit("b")
    assert decide_on(d, 1.0).explanations["b"] == WaitingForWarmup(GPU, "a")
    d2 = Driver().to(JobState.STABLE, "a").stop(StopReason.PREEMPT, "a").submit("b")
    assert starts(decide_on(d2, 1.0)) == ["b"]


def test_foreign_usage_window() -> None:
    cfg = Config(foreign_window_s=10)
    d = Driver(config=cfg)(DeviceObserved(GPU, 0.0, Resources(vram=80)))
    d(DeviceObserved(GPU, 5.0, Resources(vram=10))).submit("b", {GPU: Resources(vram=50)})
    dec = decide_on(d, 9.0)
    foreign = Resources(vram=80)
    assert dec.explanations["b"] == BlockedByForeignUsage(GPU, "vram", 50, 20, foreign)
    assert of_type(dec, Warn) == [Warn(ForeignUsageHigh(GPU, foreign, ("b",)))]
    apply(d, dec)
    later = decide_on(d, 10.5)  # the t=0 sample left the window
    assert starts(later) == ["b"]
    assert of_type(later, WarnCleared) == [WarnCleared("ForeignUsageHigh", (GPU,))]


# -- head-of-line blocking ---------------------------------------------------------------


def test_lenient_policy_skips_jobs_that_do_not_fit() -> None:
    d = (
        Driver(gpus(GPU, GPU1))
        .submit("big", {GPU: Resources(vram=500)})
        .submit("small", {GPU1: None})
    )
    d.submit("same", {GPU: Resources(vram=10)})
    dec = decide_on(d, 0.0, FifoPolicy(strict=False))
    assert starts(dec) == ["small", "same"]


def test_strict_fifo_holds_device_for_the_head() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("big", {GPU: Resources(vram=500)})
    d.submit("same", {GPU: Resources(vram=10)}).submit("other", {GPU1: None})
    dec = decide_on(d, 0.0, FifoPolicy(strict=True))
    assert starts(dec) == ["other"]
    assert dec.explanations["same"] == DeviceHeldFor(GPU, "big")


def test_held_candidate_holds_its_other_devices_too() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("big", {GPU: Resources(vram=500)})
    d.submit("gang", {GPU: None, GPU1: None}).submit("late", {GPU1: None})
    dec = decide_on(d, 0.0, FifoPolicy(strict=True))
    assert starts(dec) == []
    assert dec.explanations["gang"] == DeviceHeldFor(GPU, "big")
    assert dec.explanations["late"] == DeviceHeldFor(GPU1, "gang")


def test_rule_p_failure_also_holds_the_device() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.WARMING, "w").submit("a").submit("b", {GPU1: None})
    d.submit("c", {GPU: None, GPU1: None})
    dec = decide_on(d, 1.0, FifoPolicy(strict=True))
    assert starts(dec) == ["b"]
    assert dec.explanations["c"] == DeviceHeldFor(GPU, "a")


def test_priority_aging_with_blocking() -> None:
    def setup(new_submitted_at: float) -> Driver:
        d = Driver().to(JobState.WARMING, "w").observe(1.0, "w", vram=60).stable("w")
        d.submit("old", {GPU: Resources(vram=50)}, t=0.0, priority=0)
        return d.submit("new", {GPU: Resources(vram=10)}, t=new_submitted_at, priority=5)

    policy = PriorityPolicy(aging_per_s=0.1)
    early = decide_on(setup(5.0), 10.0, policy)  # old: 1.0 < new: 5.5
    assert starts(early) == ["new"]
    d = setup(190.0)
    late = decide_on(d, 200.0, policy)  # old: 20.0 > new: 6.0, and old does not fit
    assert starts(late) == []
    assert late.explanations["new"] == DeviceHeldFor(GPU, "old")
    lenient = decide_on(d, 200.0, PriorityPolicy(aging_per_s=0.1, head_of_line_blocking=False))
    assert starts(lenient) == ["new"]


# -- other actions -----------------------------------------------------------------------


def test_outbox_is_emitted_and_drained() -> None:
    d = Driver().to(JobState.WARMING).room(vram=50).exited(ExitKind.OOM)
    dec = decide_on(d, 1.0)
    assert isinstance(dec.actions[0], Requeue)
    apply(d, dec)
    assert d.state.outbox == ()


def test_stop_emitted_for_stop_requests() -> None:
    d = Driver().to(JobState.STABLE)(PreemptRequested("j"))
    dec = decide_on(d, 5.0)
    assert of_type(dec, Stop) == [Stop("j", 1, StopReason.PREEMPT, 120)]
    apply(d, dec)
    assert of_type(decide_on(d, 6.0), Stop) == []


def test_start_timeout() -> None:
    cfg = Config(start_timeout_s=300)
    d = Driver(config=cfg).submit()
    apply(d, decide_on(d, 10.0))
    assert d.rt().starting_since == 10.0
    assert of_type(decide_on(d, 309.0), Requeue) == []
    dec = decide_on(d, 310.0)
    assert of_type(dec, Requeue) == [Requeue("j", 2, {GPU: None}, RequeueReason.START_TIMEOUT)]
    assert starts(dec) == ["j"]  # re-admitted at once as attempt 2
    apply(d, dec)
    assert (d.rt().state, d.rt().attempt) == (JobState.STARTING, 2)


def test_mark_stable_reservation_and_fallbacks() -> None:
    cfg = Config(min_warmup_s=0, stable_window_s=0)
    d = Driver(gpus(GPU, GPU1), cfg).submit("g", {GPU: None, GPU1: Resources(vram=20)})
    d.start("g").started("g", t=0.0).observe(1.0, "g", GPU, vram=40)
    (mark,) = of_type(decide_on(d, 2.0), MarkStable)
    assert mark == MarkStable(
        "g", 1, {GPU: Resources(vram=40).scale(1.15), GPU1: Resources(vram=20).scale(1.15)}, 2.0
    )
    d2 = Driver(gpus(GPU), cfg).submit("u").start("u").started("u")
    (mark2,) = of_type(decide_on(d2, 2.0), MarkStable)
    assert mark2 == MarkStable("u", 1, {GPU: Resources()}, 2.0)


def test_newly_stable_untrusted_job_keeps_the_slot_until_the_next_tick() -> None:
    """Rule P/H: the estimator must see the running record before a sibling is admitted."""
    cfg = Config(min_warmup_s=0, stable_window_s=0)
    d = Driver(config=cfg).to(JobState.WARMING, "a").observe(1.0, "a", vram=10).submit("b")
    dec = decide_on(d, 2.0)
    assert starts(dec) == []
    assert dec.explanations["b"] == WaitingForWarmup(GPU, "a")
    apply(d, dec)
    assert d.rt("a").stable_at == 2.0
    assert starts(decide_on(d, 2.0)) == []  # I5: the same tick again
    assert starts(decide_on(d, 3.0)) == ["b"]


# -- warnings ------------------------------------------------------------------------------


def test_capacity_exceeded_raised_once_and_cleared_when_job_leaves_pending() -> None:
    d = Driver().submit("big", {GPU: Resources(vram=500)}, group="g")
    dec = decide_on(d, 0.0)
    warning = CapacityExceeded("g", GPU, Resources(vram=500), Resources(vram=100), "big")
    assert of_type(dec, Warn) == [Warn(warning)]
    apply(d, dec)
    assert of_type(decide_on(d, 1.0), Warn) == []
    d(CancelRequested("big"))
    cleared = decide_on(d, 2.0)
    assert of_type(cleared, WarnCleared) == [WarnCleared("CapacityExceeded", ("big", GPU))]
    apply(d, cleared)
    assert decide_on(d, 3.0).actions == ()


def test_capacity_exceeded_cleared_when_estimate_drops() -> None:
    d = Driver().submit("big", {GPU: None})
    raised = decide_on(d, 0.0, estimates={"big": {GPU: Resources(vram=500)}})
    assert len(of_type(raised, Warn)) == 1
    apply(d, raised)
    dropped = decide_on(d, 1.0, estimates={"big": {GPU: Resources(vram=50)}})
    assert of_type(dropped, WarnCleared) == [WarnCleared("CapacityExceeded", ("big", GPU))]
    assert starts(dropped) == ["big"]


def test_overcommitted_raised_once_and_cleared() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=95).stable("a")
    dec = decide_on(d, 2.0)
    reserved = Resources(vram=95).scale(1.15)
    assert of_type(dec, Warn) == [Warn(Overcommitted(GPU, reserved, Resources(vram=100)))]
    apply(d, dec)
    assert of_type(decide_on(d, 3.0), Warn) == []
    d.exited(ExitKind.SUCCESS, "a")
    assert of_type(decide_on(d, 4.0), WarnCleared) == [WarnCleared("Overcommitted", (GPU,))]


def test_foreign_usage_warning_lists_blocked_jobs_and_survives_pause() -> None:
    d = Driver(gpus(GPU, GPU1))(DeviceObserved(GPU, 0.0, Resources(vram=80)))
    d.submit("b", {GPU: Resources(vram=50)}).submit("c", {GPU: Resources(vram=30)})
    dec = decide_on(d, 1.0, FifoPolicy(strict=False))
    (warn,) = of_type(dec, Warn)
    assert warn == Warn(ForeignUsageHigh(GPU, Resources(vram=80), ("b", "c")))
    apply(d, dec)(Paused())
    assert decide_on(d, 2.0).actions == ()  # paused: neither raised again nor cleared


def test_no_foreign_warning_when_foreign_is_not_the_reason() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d(DeviceObserved(GPU, 1.0, Resources(vram=55))).submit("b", {GPU: Resources(vram=50)})
    dec = decide_on(d, 2.0)
    assert isinstance(dec.explanations["b"], InsufficientResources)
    assert of_type(dec, Warn) == []


def test_hint_exceeded_once_per_attempt_and_cleared_when_it_ends() -> None:
    hint = Resources(vram=40)
    d = Driver().to(JobState.WARMING, "a", placement={GPU: hint}).observe(1.0, "a", vram=30)
    assert of_type(decide_on(d, 1.0), Warn) == []
    d.observe(2.0, "a", vram=50)
    raised = decide_on(d, 2.0)
    assert of_type(raised, Warn) == [Warn(HintExceeded("a", 1, GPU, hint, Resources(vram=50)))]
    apply(d, raised).observe(3.0, "a", vram=70)
    assert of_type(decide_on(d, 3.0), Warn) == []  # the peak grew, but the key is already active
    d.exited(ExitKind.OOM, "a", t=4.0)
    cleared = decide_on(d, 4.0)
    assert of_type(cleared, WarnCleared) == [WarnCleared("HintExceeded", ("a", "1", GPU))]
    assert not any(
        isinstance(a, Warn) and isinstance(a.warning, HintExceeded) for a in cleared.actions
    )


def test_hint_exceeded_is_informational() -> None:
    hint = Resources(vram=10)
    d = Driver().to(JobState.WARMING, "a", placement={GPU: hint}).observe(1.0, "a", vram=20)
    d.stable("a").submit("b", {GPU: Resources(vram=30)})
    dec = decide_on(d, 2.0)
    assert len(of_type(dec, Warn)) == 1
    assert starts(dec) == ["b"]  # admission is unaffected


def test_unknown_estimate_never_warns() -> None:
    cfg = Config(unknown_min_free=Resources(vram=500))
    d = Driver(config=cfg).submit("u")
    assert of_type(decide_on(d, 0.0), Warn) == []


# -- explanations ------------------------------------------------------------------------------


def test_paused_explanation_and_no_admission() -> None:
    d = Driver().submit("a")(Paused())
    dec = decide_on(d, 0.0)
    assert dec.actions == ()
    assert dec.explanations["a"] == EnginePaused()


def test_awaiting_resume() -> None:
    d = Driver().submit("a")(PreemptRequested("a"))
    assert decide_on(d, 0.0).explanations["a"] == AwaitingResume()


def test_group_barrier_and_completion() -> None:
    policy = StagePolicy(["s1", "s2"])
    d = Driver().submit("a", group="s1").submit("b", group="s2")
    dec = decide_on(d, 0.0, policy)
    assert starts(dec) == ["a"]
    assert dec.explanations["b"] == GroupBarrier("s1")
    apply(d, dec).started("a").exited(ExitKind.FAILURE, "a")
    dec2 = decide_on(d, 5.0, policy)
    assert GroupCompleted("s1", GroupOutcome.HAD_FAILURES) in dec2.actions
    assert starts(dec2) == ["b"]
    apply(d, dec2)
    assert GroupCompleted("s1", GroupOutcome.HAD_FAILURES) not in decide_on(d, 6.0, policy).actions


def test_stop_queue_pauses_with_reason() -> None:
    policy = StagePolicy(["s1", "s2"], failure_policy={"s1": FailurePolicy.STOP_QUEUE})
    d = Driver().submit("a", group="s1").submit("b", group="s2")
    apply(d, decide_on(d, 0.0, policy)).started("a").exited(ExitKind.FAILURE, "a")
    dec = decide_on(d, 5.0, policy)
    assert Warn(QueueStopped("s1", "a")) in dec.actions
    assert starts(dec) == []
    assert dec.explanations["b"] == EnginePaused(QueueStopped("s1", "a"))
    apply(d, dec)
    assert d.state.paused
    d(Unpaused())
    assert starts(decide_on(d, 6.0, policy)) == ["b"]


# -- purity -------------------------------------------------------------------------------------


def test_decide_is_deterministic_and_idempotent() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("a").submit("b", {GPU1: None}).submit("c")
    snap = Snapshot(d.state, {})
    first = decide(snap, 0.0, d.config, FifoPolicy())
    assert decide(snap, 0.0, d.config, FifoPolicy()) == first
    apply(d, first)
    again = decide_on(d, 0.0, FifoPolicy())
    assert not of_type(again, Start)
    assert not of_type(again, Stop)


@pytest.mark.parametrize("now", [0.0, 50.0])
def test_decision_carries_ignored_events(now: float) -> None:
    d = Driver().submit("a").submit("a")
    assert len(decide_on(d, now).ignored_events) == 1


def test_overcommitted_device_refuses_even_zero_estimates() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=95).stable("a")
    d.submit("z", {GPU: Resources()})
    dec = decide_on(d, 2.0)
    assert starts(dec) == []
    assert dec.explanations["z"] == InsufficientResources(GPU, "vram", 0, 0)


class _OddPolicy(BasePolicy):
    """Yields unknown and non-pending ids, and leaves one pending job unexplained."""

    def candidates(self, view: object, now: float) -> list[str]:
        return ["ghost", "running", "a"]


def test_decide_tolerates_odd_custom_policies() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.WARMING, "running").submit("a", {GPU1: None})
    d.submit("left-out", {GPU1: None})
    dec = decide_on(d, 1.0, _OddPolicy())
    assert starts(dec) == ["a"]
    assert "left-out" not in dec.explanations


class _DuckPolicy:
    """Implements only the three methods of the original protocol."""

    def candidates(self, view: object, now: float) -> list[str]:
        return ["b", "a"]

    def explain_blocked(self, view: object, job_id: str) -> None:
        return None

    def completed_groups(self, view: object) -> list[tuple[str, GroupOutcome]]:
        return []


def test_duck_typed_policy_defaults_to_no_blocking_and_no_queue_stops() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("a", {GPU1: None}).submit("b", {GPU: Resources(vram=500)})
    dec = decide_on(d, 0.0, _DuckPolicy())  # type: ignore[arg-type]
    assert starts(dec) == ["a"]


# -- trusted estimates and Rule P modes ------------------------------------------------------------


def trusted_snapshot(d: Driver, job: str, vram: float, observations: int = 3) -> Snapshot:
    return Snapshot(
        d.state,
        {job: {GPU: Resources(vram=vram)}},
        frozenset({(job, GPU)}),
        {(job, GPU): observations},
    )


def test_trusted_job_skips_probation_and_counts_estimate_times_trusted_margin() -> None:
    d = Driver().submit("b")
    dec = decide(trusted_snapshot(d, "b", 20), 2.0, d.config, FifoPolicy(strict=False))
    (start,) = [a for a in dec.actions if isinstance(a, Start)]
    assert (start.job_id, start.trusted) == ("b", frozenset({GPU}))
    apply(d, dec)
    loads = device_loads(d.state, 2.0, d.config)
    assert loads[GPU].warming_job is None  # b does not take the probation slot
    assert loads[GPU].warming_usage == Resources(vram=20).scale(1.05)
    d.started("b", t=2.0).observe(3.0, "b", vram=30)  # usage above E * margin wins
    assert device_loads(d.state, 3.0, d.config)[GPU].warming_usage == Resources(vram=30)


def test_trusted_job_waits_while_an_untrusted_job_warms_on_its_device() -> None:
    # Rule P (after held-out evaluation 1), I17: a's peak is unknown, so it gets everything
    # currently free until it is stable; a trusted job does not start next to it.
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=10).submit("b")
    dec = decide(trusted_snapshot(d, "b", 20), 2.0, d.config, FifoPolicy(strict=False))
    assert starts(dec) == []
    assert dec.explanations["b"] == WaitingForWarmup(GPU, "a")


def test_trusted_job_waits_for_an_untrusted_job_started_in_the_same_tick() -> None:
    # The held-out (c) failures: an unknown job and trusted ones admitted in one tick.
    d = Driver().submit("a").submit("b")
    dec = decide(trusted_snapshot(d, "b", 20), 0.0, d.config, FifoPolicy(strict=False))
    assert starts(dec) == ["a"]
    assert dec.explanations["b"] == WaitingForWarmup(GPU, "a")


def test_trusted_margin_is_separate_from_safety_margin() -> None:
    # Same-tick bookkeeping: b (trusted, 60) counts 60 * 1.05 = 63, so c (trusted, 30) fits
    # in 100 - 63 = 37 but would not with 60 * 1.15 = 69 (31 left).
    cfg = Config(trusted_margin=1.05, safety_margin=1.15)
    d = Driver(config=cfg).submit("b").submit("c")
    snapshot = Snapshot(
        d.state,
        {"b": {GPU: Resources(vram=60)}, "c": {GPU: Resources(vram=35)}},
        frozenset({("b", GPU), ("c", GPU)}),
        {("b", GPU): 3, ("c", GPU): 3},
    )
    assert starts(decide(snapshot, 0.0, d.config, FifoPolicy(strict=False))) == ["b", "c"]
    strict = Driver(config=Config(trusted_margin=1.15)).submit("b").submit("c")
    snapshot = Snapshot(strict.state, snapshot.estimates, snapshot.trusted, snapshot.observations)
    assert starts(decide(snapshot, 0.0, strict.config, FifoPolicy(strict=False))) == ["b"]


def test_rule_a_few_observations_count_with_safety_margin() -> None:
    # One observation behind b's estimate: b stays trusted (no probation slot) but counts
    # with safety_margin, 60 * 1.15 = 69, and c (35) no longer fits next to it.
    d = Driver().submit("b").submit("c")
    estimates = {"b": {GPU: Resources(vram=60)}, "c": {GPU: Resources(vram=35)}}
    trusted = frozenset({("b", GPU), ("c", GPU)})
    few = Snapshot(d.state, estimates, trusted, {("b", GPU): 1, ("c", GPU): 1})
    dec = decide(few, 0.0, d.config, FifoPolicy(strict=False))
    assert starts(dec) == ["b"]
    (start,) = of_type(dec, Start)
    assert start.observations == {GPU: 1}  # type: ignore[attr-defined]
    apply(d, dec)
    assert device_loads(d.state, 0.0, d.config)[GPU].warming_usage == Resources(vram=69)
    assert device_loads(d.state, 0.0, d.config)[GPU].warming_job is None
    enough = Snapshot(d.state, estimates, trusted, {("c", GPU): 3})
    assert starts(decide(enough, 0.0, d.config, FifoPolicy(strict=False))) == []


def test_untrusted_job_waits_next_to_a_trusted_warming_job_only_for_capacity() -> None:
    d = Driver().submit("b").submit("c")
    first = decide(trusted_snapshot(d, "b", 50), 0.0, d.config, FifoPolicy(strict=False))
    # c (unknown) may start next to trusted b: b holds no slot, 100 - 57.5 >= 10 % free.
    assert starts(first) == ["b", "c"]


def test_probation_all_restores_one_warming_job_per_device() -> None:
    d = Driver(config=Config(probation="all")).to(JobState.WARMING, "a").submit("b")
    dec = decide(trusted_snapshot(d, "b", 20), 2.0, d.config, FifoPolicy(strict=False))
    assert of_type(dec, Start) == []
    assert dec.explanations["b"] == WaitingForWarmup(GPU, "a")


def test_fallback_level_and_oomed_jobs_are_untrusted() -> None:
    d = Driver().to(JobState.WARMING, "a").submit("b")
    level1 = Snapshot(d.state, {"b": {GPU: Resources(vram=20)}})  # not in `trusted`
    assert of_type(decide(level1, 2.0, d.config, FifoPolicy(strict=False)), Start) == []
    d.submit("c").start("c").room(vram=50).exited(ExitKind.OOM, "c")
    snap = trusted_snapshot(d, "c", 20)
    assert not snap.is_trusted(d.rt("c"), GPU)


def test_config_rejects_unknown_probation_mode() -> None:
    with pytest.raises(ValueError, match="probation"):
        Config(probation="some")  # type: ignore[arg-type]


def test_start_rejects_trusted_device_without_estimate() -> None:
    d = Driver().submit()
    with pytest.raises(InvalidTransition, match="trusted"):
        d(Start("j", 1, (GPU,), False, {}, 0.0, {GPU: None}, frozenset({GPU})))
