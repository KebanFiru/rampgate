"""Example tests for ``decide``: Rules P and A, explanations, warnings (SPEC §3.7, §5)."""

import pytest

from rampgate import (
    AwaitingResume,
    BasePolicy,
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
    InsufficientResources,
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
    assert start == Start("a", 1, (GPU,), False, {GPU: Resources(vram=100)}, 0.0)


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
    assert dec.explanations["b"] == InsufficientResources(GPU, "vram", 50, 20)
    assert of_type(dec, Warn) == [Warn(ForeignUsageHigh(GPU, Resources(vram=80), "b"))]
    assert dec.explanations["b"] == InsufficientResources(GPU, "vram", 50, 20)
    assert starts(decide_on(d, 10.5)) == ["b"]  # the t=0 sample left the window


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
    d = Driver().to(JobState.WARMING).exited(ExitKind.OOM)
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
        "g", 1, {GPU: Resources(vram=40).scale(1.15), GPU1: Resources(vram=20).scale(1.15)}
    )
    d2 = Driver(gpus(GPU), cfg).submit("u").start("u").started("u")
    (mark2,) = of_type(decide_on(d2, 2.0), MarkStable)
    assert mark2 == MarkStable("u", 1, {GPU: Resources()})


def test_newly_stable_job_frees_the_slot_in_the_same_tick() -> None:
    cfg = Config(min_warmup_s=0, stable_window_s=0)
    d = Driver(config=cfg).to(JobState.WARMING, "a").observe(1.0, "a", vram=10).submit("b")
    assert starts(decide_on(d, 2.0)) == ["b"]


# -- warnings ------------------------------------------------------------------------------


def test_capacity_exceeded_warns_once() -> None:
    d = Driver().submit("big", {GPU: Resources(vram=500)}, group="g")
    dec = decide_on(d, 0.0)
    warning = CapacityExceeded("g", GPU, Resources(vram=500), Resources(vram=100), "big")
    assert of_type(dec, Warn) == [Warn(warning)]
    apply(d, dec)
    assert of_type(decide_on(d, 1.0), Warn) == []


def test_overcommitted_warns_once() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=95).stable("a")
    dec = decide_on(d, 2.0)
    reserved = Resources(vram=95).scale(1.15)
    assert of_type(dec, Warn) == [Warn(Overcommitted(GPU, reserved, Resources(vram=100)))]
    apply(d, dec)
    assert of_type(decide_on(d, 3.0), Warn) == []


def test_no_foreign_warning_when_foreign_is_not_the_reason() -> None:
    d = Driver().to(JobState.WARMING, "a").observe(1.0, "a", vram=50).stable("a")
    d(DeviceObserved(GPU, 1.0, Resources(vram=55))).submit("b", {GPU: Resources(vram=50)})
    dec = decide_on(d, 2.0)
    assert isinstance(dec.explanations["b"], InsufficientResources)
    assert of_type(dec, Warn) == []


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
