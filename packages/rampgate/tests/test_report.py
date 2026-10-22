"""Simulator ground truth, `Report` and `run` (SPEC §8)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rampgate import (
    Config,
    Device,
    ExitKind,
    FifoPolicy,
    InMemoryHistoryStore,
    Job,
    JobExited,
    JobState,
    PreemptRequested,
    Resources,
    ResumeRequested,
    Start,
    StaticEstimator,
)
from rampgate.sim import (
    PROBATION_COLD,
    STRATEGIES,
    Command,
    Scenario,
    SimExecutor,
    SimJob,
    StepProfile,
    TimelineEntry,
    quantile,
    report,
    run,
    simulate,
    summarize,
)
from rampgate_testkit import examples, stored

GPU = Device("gpu:0", Resources(vram=100))
FAST = Config(min_warmup_s=5, stable_window_s=10, stable_min_steps=5, stable_requires_epoch=False)


def flat(vram: float, duration: float = 100.0, **kw: float) -> StepProfile:
    return StepProfile(steps=((0, Resources(vram=vram)),), duration_s=duration, **kw)


# -- executor ground truth ------------------------------------------------------------------


def test_preempt_saves_a_checkpoint_after_ckpt_save_s_and_keeps_progress() -> None:
    job = SimJob(Job("a", {GPU.id: None}), flat(10, duration=100))
    commands = (Command(20, PreemptRequested("a")), Command(40, ResumeRequested("a")))
    trace = simulate(Scenario((GPU,), (job,), commands), FifoPolicy(), ckpt_save_s=7)
    assert trace.world is not None
    exits = [e for t in trace.ticks for e in t.events if isinstance(e, JobExited)]
    assert exits[0].kind is ExitKind.PREEMPTED
    assert exits[0].t == 20 + 7
    assert trace.world.lost_work_step_s == 0
    began = [x.t for x in trace.world.timeline if x.kind == "start"]
    # 1 s start delay each time; the second attempt only runs what is left.
    assert exits[-1].t - began[-1] == pytest.approx(100 - (exits[0].t - began[0]))


def test_oom_lost_work_is_the_rollback_to_the_last_checkpoint() -> None:
    ex = SimExecutor(
        [GPU],
        {
            "a": StepProfile(
                steps=((0, Resources(vram=10)), (25, Resources(vram=200))),
                duration_s=1000,
                checkpoint_every_s=10,
                steps_per_s=2,
            )
        },
    )
    ex.apply([Start("a", 1, (GPU.id,), False, {}, 0.0)], 0.0, {"a": Job("a", {GPU.id: None})})
    for t in range(1, 30):
        ex.advance(float(t))
    stats = ex.stats()
    assert stats.oom_count == 1
    # began at 1, died at 26: 25 s elapsed, checkpoint at 20 s, 5 s * 2 steps/s lost.
    assert stats.lost_work_step_s == pytest.approx(10)
    assert stats.max_overcommit[GPU.id] == pytest.approx(1.0)  # 200 on 100
    assert stats.timeline == (
        TimelineEntry(1.0, "a", 1, "start"),
        TimelineEntry(26.0, "a", 1, "OOM", GPU.id),
    )


def test_start_timeout_kill_of_a_running_process_is_lost_work() -> None:
    ex = SimExecutor([GPU], {"a": flat(10, duration=1000)})
    job = Job("a", {GPU.id: None})
    ex.apply([Start("a", 1, (GPU.id,), False, {}, 0.0)], 0.0, {"a": job})
    for t in range(1, 11):
        ex.advance(float(t))
    ex.apply([Start("a", 2, (GPU.id,), True, {}, 0.0)], 10.0, {"a": job})  # replaced
    assert ex.stats().lost_work_step_s == pytest.approx(9)


def test_foreign_load_schedule() -> None:
    ex = SimExecutor([GPU], {}, foreign={GPU.id: ((0, Resources(vram=5)), (10, Resources()))})
    assert ex.foreign_at(GPU.id, 3) == Resources(vram=5)
    assert ex.foreign_at(GPU.id, 10) == Resources()
    assert ex.foreign_at("other", 3) == Resources()
    events = ex.advance(3.0)
    assert events[-1].used_total == Resources(vram=5)  # type: ignore[union-attr]


def test_usage_integral_excludes_foreign() -> None:
    ex = SimExecutor([GPU], {"a": flat(40, duration=1000)}, foreign={GPU.id: Resources(vram=5)})
    ex.apply([Start("a", 1, (GPU.id,), False, {}, 0.0)], 0.0, {"a": Job("a", {GPU.id: None})})
    for t in range(0, 11):
        ex.advance(float(t))
    assert ex.stats().usage_s[GPU.id] == Resources(vram=400)  # 40 for t = 1..10
    assert ex.stats().observed_until == 10.0


# -- reactive submission -----------------------------------------------------------------


def test_after_submits_on_success_and_drops_on_failure() -> None:
    parent = SimJob(Job("p", {GPU.id: None}), flat(10, duration=20))
    child = SimJob(Job("c", {GPU.id: None}), flat(10, duration=20), after="p")
    grandchild = SimJob(Job("g", {GPU.id: None}), flat(10, duration=20), after="c")
    trace = simulate(Scenario((GPU,), (child, parent, grandchild)), FifoPolicy())
    assert {j: rt.state for j, rt in trace.final.jobs.items()} == dict.fromkeys(
        "pcg", JobState.SUCCEEDED
    )
    submit_c = next(t.now for t in trace.ticks if any(_submits(t, "c")))
    p_exit = next(
        t.now for t in trace.ticks for e in t.events if isinstance(e, JobExited) and e.job_id == "p"
    )
    assert submit_c == p_exit

    too_big = SimJob(Job("p", {GPU.id: None}), flat(150, duration=20))
    trace = simulate(Scenario((GPU,), (too_big, child, grandchild)), FifoPolicy())
    assert trace.final.jobs["p"].state is JobState.FAILED
    assert set(trace.final.jobs) == {"p"}
    assert trace.dropped == ("c", "g")
    assert trace.finished


def _submits(tick: object, job_id: str) -> list[bool]:
    return [getattr(e, "job", None) is not None and e.job.id == job_id for e in tick.events]  # type: ignore[attr-defined]


# -- Report --------------------------------------------------------------------------------


def test_report_on_a_tiny_scenario() -> None:
    jobs = (
        SimJob(Job("a", {GPU.id: None}), flat(40, duration=50)),
        SimJob(Job("b", {GPU.id: None}), flat(40, duration=50), submit_at=2),
    )
    scenario = Scenario((GPU,), jobs)
    rep = report(simulate(scenario, FifoPolicy(), config=FAST), scenario)
    assert not rep.censored
    assert (rep.completed, rep.total, rep.oom_count, rep.lost_work_step_s) == (2, 2, 0, 0)
    assert rep.max_overcommit == 0
    assert rep.makespan is not None
    # a runs 50 s at 40, b runs 50 s at 40: 4000 vram-seconds over makespan * 100.
    assert rep.utilization[GPU.id]["vram"] == pytest.approx(4000 / (rep.makespan * 100))
    assert rep.mean_utilization == rep.utilization[GPU.id]["vram"]
    assert rep.wait_p50 >= 0
    assert rep.wait_p95 > 0  # b waits for a's probation
    metrics = rep.metrics()
    assert metrics["completed"] == 2
    assert metrics["makespan"] == rep.makespan


def test_report_censors_runs_cut_at_max_time() -> None:
    jobs = (
        SimJob(Job("a", {GPU.id: Resources(vram=60)}), flat(60, duration=500)),
        SimJob(Job("b", {GPU.id: Resources(vram=60)}), flat(60, duration=500)),
    )
    scenario = Scenario((GPU,), jobs)
    rep = report(simulate(scenario, FifoPolicy(), max_time=100), scenario)
    assert rep.censored
    assert rep.makespan is None
    assert rep.completed == 0
    assert rep.wait_p95 == math.inf  # b never started
    assert rep.metrics()["wait_p95"] is None
    assert rep.metrics()["makespan"] is None


def test_run_builds_the_estimator_and_uses_the_store() -> None:
    jobs = (SimJob(Job("a", {GPU.id: None}, fingerprint=("k",)), flat(40, duration=50)),)
    scenario = Scenario((GPU,), jobs)
    store = InMemoryHistoryStore()
    rep = run(scenario, PROBATION_COLD, policy=FifoPolicy())
    assert rep.completed == 1
    run(scenario, PROBATION_COLD, policy=FifoPolicy(), history_store=store)
    assert stored(store, "k", GPU.id) == Resources(vram=40)
    null = run(scenario, STRATEGIES["rampgate - history"], policy=FifoPolicy())
    assert null.completed == 1


@settings(max_examples=examples(0.2))
@given(st.integers(0, 80), st.integers(0, 3))
def test_report_is_the_same_after_a_restart(at: int, seed: int) -> None:
    jobs = tuple(
        SimJob(Job(f"j{i}", {GPU.id: None}), flat(20 + 10 * i, duration=60), submit_at=5 * i)
        for i in range(3)
    )
    scenario = Scenario((GPU,), jobs, seed=seed)
    plain = report(simulate(scenario, FifoPolicy(), config=FAST), scenario)
    restarted = report(simulate(scenario, FifoPolicy(), config=FAST, restart_at=at), scenario)
    assert restarted == plain


# -- quantiles ------------------------------------------------------------------------------


def test_quantile_interpolates_and_handles_inf() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile([4, 1, 3, 2], 0.25) == 1.75
    assert quantile([5], 0.95) == 5
    assert quantile([1, 2, math.inf], 0.5) == 2
    assert quantile([1, 2, math.inf], 0.75) == math.inf
    assert summarize([1, 2, 3, 4, 5]) == (2, 3, 4)
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_first_contact_ooms_are_counted_separately() -> None:
    # A known job runs at 60; an unknown job ramps next to it and an OOM follows.
    known = SimJob(Job("known", {GPU.id: Resources(vram=60)}), flat(60, duration=300))
    unknown = SimJob(Job("unknown", {GPU.id: None}), flat(50, duration=100), submit_at=60)
    scenario = Scenario((GPU,), (known, unknown))
    rep = report(simulate(scenario, FifoPolicy(), config=FAST), scenario)
    assert rep.oom_count >= 1
    assert rep.first_contact_ooms == 1  # later attempts are no longer first contact
    assert rep.metrics()["first_contact_ooms"] == 1
    assert (rep.first_contact_unknown, rep.first_contact_hint) == (1, 0)


def _beside_a_stable_job(hint: Resources | None) -> Scenario:
    known = SimJob(Job("known", {GPU.id: Resources(vram=60)}), flat(60, duration=300))
    under = SimJob(
        Job("under", {GPU.id: hint}, fingerprint=("tmpl:x", "tmpl")),
        flat(50, duration=100),
        submit_at=60,
    )
    return Scenario((GPU,), (known, under))


def test_an_under_declared_hint_is_first_contact_of_its_own_kind() -> None:
    scenario = _beside_a_stable_job(Resources(vram=10))
    rep = report(simulate(scenario, FifoPolicy(), config=FAST), scenario)
    assert rep.oom_count >= 1
    assert (rep.first_contact_unknown, rep.first_contact_hint) == (0, 1)
    assert rep.first_contact_ooms == 1


def test_an_estimate_from_a_fallback_history_level_is_not_first_contact() -> None:
    scenario = _beside_a_stable_job(None)
    estimator = StaticEstimator({"under": {GPU.id: Resources(vram=10)}}, level=1)
    rep = report(simulate(scenario, FifoPolicy(), config=FAST, estimator=estimator), scenario)
    assert rep.oom_count >= 1
    assert rep.first_contact_ooms == 0


def test_ooms_per_job_and_repeat_ooms_on_a_device() -> None:
    scenario = _beside_a_stable_job(None)
    greedy = STRATEGIES["GreedyFree"]
    rep = run(scenario, greedy, policy=FifoPolicy(), max_time=400)
    assert rep.max_job_ooms >= 2  # GreedyFree never learns: "under" OOMs again and again
    assert rep.repeat_device_ooms == 1
    assert rep.ooms_per_job[0] == 1  # "known" never OOMs
    assert rep.metrics()["ooms_per_job"]["0"] == 1
