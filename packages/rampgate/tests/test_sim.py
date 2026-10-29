"""The minimal simulator and the epoch gate (docs/algorithm.md, docs/simulator.md)."""

import pytest

from rampgate import (
    CancelRequested,
    Config,
    Device,
    EstimateResult,
    ExitKind,
    FifoPolicy,
    HistoryEstimator,
    InMemoryHistoryStore,
    Job,
    JobExited,
    JobState,
    MarkStable,
    PreemptRequested,
    Requeue,
    RequeueReason,
    Resources,
    ResumeRequested,
    Start,
    Stop,
    StopReason,
)
from rampgate.sim import (
    Command,
    EpochEval,
    RampPlateau,
    Scenario,
    SimExecutor,
    SimJob,
    Spiky,
    StepProfile,
    VirtualClock,
    simulate,
)

GPU = Device("gpu:0", Resources(vram=100))


def ramp(plateau: float, duration: float = 100.0) -> RampPlateau:
    return RampPlateau(load_s=5, ramp_s=10, plateau=Resources(vram=plateau), duration_s=duration)


def stable_times(trace: object, job_id: str) -> list[float]:
    return [
        t.now
        for t in trace.ticks  # type: ignore[attr-defined]
        for a in t.decision.actions
        if isinstance(a, MarkStable) and a.job_id == job_id
    ]


# -- workloads -----------------------------------------------------------------------------


def test_ramp_plateau() -> None:
    w = ramp(100)
    assert w.usage(0) == Resources()
    assert w.usage(5)["vram"] == pytest.approx(30)
    assert w.usage(10)["vram"] == pytest.approx(65)
    assert w.usage(20) == Resources(vram=100)
    assert w.epoch(20) is None


def test_step_profile_and_spiky() -> None:
    w = StepProfile(steps=((10, Resources(vram=5)), (0, Resources(vram=1))))
    assert (w.usage(5), w.usage(10), w.usage(99)) == (
        Resources(vram=1),
        Resources(vram=5),
        Resources(vram=5),
    )
    s = Spiky(base=Resources(vram=10), spike=Resources(vram=5), every_s=30, spike_s=2)
    assert [s.usage(t)["vram"] for t in (0, 1, 29, 30, 31, 32, 60)] == [10, 10, 10, 15, 15, 10, 15]


def test_epoch_eval_workload() -> None:
    w = EpochEval(
        base=Resources(vram=20), eval_spike=Resources(vram=30), epoch_s=50, eval_s=10, load_s=10
    )
    assert w.usage(5) == Resources(vram=10)
    assert w.usage(30) == Resources(vram=20)
    assert w.usage(65) == Resources(vram=50)
    assert [w.epoch(t) for t in (0, 30, 65, 69.9, 70, 130)] == [0, 0, 0, 0, 1, 2]
    silent = EpochEval(
        base=Resources(vram=20), eval_spike=Resources(), epoch_s=5, eval_s=5, reports_epoch=False
    )
    assert silent.epoch(100) is None


def test_virtual_clock() -> None:
    c = VirtualClock(0.1)
    for _ in range(10):
        c.advance()
    assert c.now == 1.0
    with pytest.raises(ValueError, match="dt"):
        VirtualClock(0)


# -- executor --------------------------------------------------------------------------------


def test_executor_oom_kills_most_recent_grower() -> None:
    job_a, job_b = Job("a", {GPU.id: None}), Job("b", {GPU.id: None})
    ex = SimExecutor(
        [GPU],
        {
            "a": StepProfile(steps=((0, Resources(vram=60)),), duration_s=1000),
            "b": StepProfile(
                steps=((0, Resources(vram=10)), (5, Resources(vram=50))), duration_s=1000
            ),
        },
    )
    jobs = {"a": job_a, "b": job_b}
    ex.apply(
        [Start("a", 1, (GPU.id,), False, {}, 0.0), Start("b", 1, (GPU.id,), False, {}, 0.0)],
        0.0,
        jobs,
    )
    for t in range(1, 6):
        events = ex.advance(float(t))
    assert not any(isinstance(e, JobExited) for e in events)
    events = ex.advance(7.0)
    exits = [e for e in events if isinstance(e, JobExited)]
    assert exits == [JobExited("b", 1, ExitKind.OOM, 7.0, ("vram",))]


# -- simulate ----------------------------------------------------------------------------------


def test_probation_serializes_ramp_ups_without_oom() -> None:
    jobs = tuple(SimJob(Job(f"j{i}", {GPU.id: None}), ramp(30)) for i in range(3))
    trace = simulate(Scenario((GPU,), jobs), FifoPolicy(strict=False))
    final = trace.final
    assert {rt.state for rt in final.jobs.values()} == {JobState.SUCCEEDED}
    assert all(rt.oom_count == 0 for rt in final.jobs.values())
    starts = [t.now for t in trace.ticks for a in t.decision.actions if isinstance(a, Start)]
    assert starts == sorted(starts)
    assert len(set(starts)) == 3


def test_simulate_is_deterministic() -> None:
    jobs = tuple(
        SimJob(
            Job(f"j{i}", {GPU.id: None}),
            RampPlateau(load_s=5, ramp_s=5, plateau=Resources(vram=40), noise=0.1, duration_s=80),
        )
        for i in range(3)
    )
    scenario = Scenario((GPU,), jobs, seed=7)
    a = simulate(scenario, FifoPolicy())
    b = simulate(scenario, FifoPolicy())
    assert [t.decision for t in a.ticks] == [t.decision for t in b.ticks]


def test_commands_and_rejections() -> None:
    jobs = (SimJob(Job("j", {GPU.id: None}), ramp(30, duration=500)),)
    cmds = (
        Command(50.0, PreemptRequested("j")),
        Command(51.0, ResumeRequested("j")),  # still stopping: rejected
        Command(60.0, ResumeRequested("j")),
        Command(400.0, CancelRequested("j")),
    )
    trace = simulate(Scenario((GPU,), jobs, cmds), FifoPolicy())
    assert [r for t in trace.ticks for r in t.rejected] == [ResumeRequested("j")]
    final = trace.final.jobs["j"]
    assert final.state is JobState.CANCELLED
    resumed = [
        a
        for t in trace.ticks
        for a in t.decision.actions
        if isinstance(a, Start) and a.attempt == 2
    ]
    assert resumed
    assert resumed[0].resume


def test_history_makes_the_second_run_reserve_up_front() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    first = (SimJob(Job("a", {GPU.id: None}, fingerprint=("m",)), ramp(70, duration=50)),)
    simulate(Scenario((GPU,), first), FifoPolicy(), estimator=est)
    assert est.estimate(first[0].job, GPU) == EstimateResult(Resources(vram=70), observations=1)
    big = SimJob(Job("b", {GPU.id: None}, fingerprint=("m",)), ramp(70, duration=50))
    trace = simulate(Scenario((GPU,), (big,)), FifoPolicy(), estimator=est)
    assert trace.ticks[0].snapshot.estimates["b"][GPU.id] == Resources(vram=70)


def test_start_timeout_in_simulation() -> None:
    jobs = (SimJob(Job("j", {GPU.id: None}), ramp(10, duration=20)),)
    trace = simulate(
        Scenario((GPU,), jobs),
        FifoPolicy(),
        config=Config(start_timeout_s=3),
        start_delay_s=5,
        max_time=20,
    )
    rt = trace.final.jobs["j"]
    assert rt.state is JobState.STARTING
    assert rt.attempt == 7  # started at t=0, then re-started at t=3, 6, 9, 12, 15, 18


# -- epoch gate ----------------------------------------------------------------------------------


def eval_job(job_id: str, reports_epoch: bool = True) -> SimJob:
    return SimJob(
        Job(job_id, {GPU.id: None}),
        EpochEval(
            base=Resources(vram=30),
            eval_spike=Resources(vram=40),
            epoch_s=120,
            eval_s=20,
            load_s=5,
            duration_s=400,
            reports_epoch=reports_epoch,
        ),
    )


def test_epoch_gate_prevents_premature_stability() -> None:
    gated = simulate(Scenario((GPU,), (eval_job("e"),)), FifoPolicy())
    (t_gated,) = stable_times(gated, "e")
    eval_end = 1 + 5 + 120 + 20  # started at t=1, load, first epoch, eval
    assert t_gated >= eval_end
    reservation = next(
        a.reservation for t in gated.ticks for a in t.decision.actions if isinstance(a, MarkStable)
    )
    assert reservation[GPU.id]["vram"] == pytest.approx(70 * 1.15)

    ungated = simulate(Scenario((GPU,), (eval_job("e", reports_epoch=False),)), FifoPolicy())
    (t_ungated,) = stable_times(ungated, "e")
    assert t_ungated < 1 + 5 + 120  # judged stable before the eval spike


def test_epoch_gate_prevents_neighbour_oom() -> None:
    """Without the gate, a neighbour admitted against the pre-eval peak OOMs at eval time."""
    neighbour = SimJob(Job("n", {GPU.id: Resources(vram=40)}), ramp(40, duration=400), submit_at=1)
    scenario = Scenario((GPU,), (eval_job("e"), neighbour))
    with_gate = simulate(scenario, FifoPolicy())
    assert all(rt.oom_count == 0 for rt in with_gate.final.jobs.values())
    no_gate = simulate(scenario, FifoPolicy(), config=Config(stable_requires_epoch=False))
    assert sum(rt.oom_count for rt in no_gate.final.jobs.values()) > 0


def test_epoch_gate_is_capped_by_max_warmup() -> None:
    cfg = Config(max_warmup_s=60)
    trace = simulate(Scenario((GPU,), (eval_job("e"),)), FifoPolicy(), config=cfg)
    (t,) = stable_times(trace, "e")
    assert t == pytest.approx(1 + 1 + 60, abs=1)


def test_counter_reset_and_cumulative_counters_stabilize_alike() -> None:
    """I14 in the simulator: the job resumes after a preemption with either counter style."""
    jobs = (
        SimJob(
            Job("e", {GPU.id: None}),
            EpochEval(
                base=Resources(vram=30),
                eval_spike=Resources(vram=10),
                epoch_s=40,
                eval_s=10,
                duration_s=1000,
                steps_per_s=2,
            ),
        ),
    )
    cmds = (Command(300.0, PreemptRequested("e")), Command(305.0, ResumeRequested("e")))
    scenario = Scenario((GPU,), jobs, cmds)
    cumulative = simulate(scenario, FifoPolicy())
    reset = simulate(scenario, FifoPolicy(), reset_counters_on_resume=True)
    assert stable_times(cumulative, "e") == stable_times(reset, "e")
    assert len(stable_times(cumulative, "e")) == 2


def test_executor_ignores_actions_for_unknown_or_stale_processes() -> None:
    ex = SimExecutor([GPU], {"a": ramp(10)})
    ex.apply(
        [
            Stop("a", 1, StopReason.PREEMPT, 1.0),
            Requeue("a", 2, {}, RequeueReason.START_TIMEOUT),
        ],
        0.0,
        {},
    )
    assert ex.procs == {}
    ex.apply([Start("a", 2, (GPU.id,), False, {}, 0.0)], 0.0, {"a": Job("a", {GPU.id: None})})
    ex.apply(
        [Requeue("a", 2, {}, RequeueReason.START_TIMEOUT), Stop("a", 1, StopReason.CANCEL, 1.0)],
        0.0,
        {},
    )
    assert ex.procs["a"].stop is None


def test_unknown_job_waits_for_real_free_capacity_on_a_full_gpu() -> None:
    known = SimJob(Job("known", {GPU.id: Resources(vram=85)}), ramp(85, duration=120))
    unknown = SimJob(Job("unknown", {GPU.id: None}), ramp(30, duration=60), submit_at=5)
    scenario = Scenario((GPU,), (known, unknown))
    trace = simulate(scenario, FifoPolicy())
    final = trace.final.jobs
    assert {rt.state for rt in final.values()} == {JobState.SUCCEEDED}
    assert sum(rt.oom_count for rt in final.values()) == 0
    start = next(
        t.now
        for t in trace.ticks
        for a in t.decision.actions
        if isinstance(a, Start) and a.job_id == "unknown"
    )
    known_exit = next(
        t.now
        for t in trace.ticks
        for e in t.events
        if isinstance(e, JobExited) and e.job_id == "known"
    )
    assert start >= known_exit
    old_rule = simulate(scenario, FifoPolicy(), config=Config(unknown_min_free=Resources()))
    assert old_rule.final.jobs["known"].oom_count + old_rule.final.jobs["unknown"].oom_count > 0


# -- Rule O: room at OOM (regressions for the admitted_free rule) ---------------------------

SPIKE_CFG = Config(
    min_warmup_s=5, stable_window_s=10, stable_min_steps=5, stable_requires_epoch=False
)


def _steps(*steps: tuple[float, float], duration: float = 600.0) -> StepProfile:
    return StepProfile(steps=tuple((t, Resources(vram=v)) for t, v in steps), duration_s=duration)


def test_stable_spike_next_to_a_neighbour_retries_instead_of_failing() -> None:
    a = SimJob(Job("a", {GPU.id: None}), _steps((0, 40), (300, 70)))
    b = SimJob(Job("b", {GPU.id: None}), _steps((0, 40)), submit_at=1)
    trace = simulate(Scenario((GPU,), (a, b)), FifoPolicy(), config=SPIKE_CFG)
    final = trace.final.jobs
    assert final["a"].state is JobState.SUCCEEDED
    assert final["a"].oom_count == 1
    # The spike is killed before it is observed: lower bound = max(peak 40, room 60).
    assert final["a"].learned[GPU.id] == Resources(vram=75.0)
    # 75 + 40 > 100: after the OOM, a waits for b instead of running next to it.
    restart = next(
        t.now
        for t in trace.ticks
        for x in t.decision.actions
        if isinstance(x, Start) and x.job_id == "a" and x.attempt == 2
    )
    b_exit = next(
        t.now for t in trace.ticks for e in t.events if isinstance(e, JobExited) and e.job_id == "b"
    )
    assert restart >= b_exit


def test_oom_while_alone_fails_even_if_the_device_was_shared_at_start() -> None:
    a = SimJob(Job("a", {GPU.id: None}), _steps((0, 40), (300, 120)))
    b = SimJob(Job("b", {GPU.id: None}), _steps((0, 40), duration=100), submit_at=1)
    trace = simulate(Scenario((GPU,), (a, b)), FifoPolicy(), config=SPIKE_CFG)
    final = trace.final.jobs
    assert final["b"].state is JobState.SUCCEEDED
    assert final["a"].state is JobState.FAILED
    assert final["a"].oom_count == 1


def test_sibling_of_a_template_is_admitted_with_the_running_estimate() -> None:
    """Rule H + P: the second job of a template never starts blind on the same device."""
    first = SimJob(Job("first", {GPU.id: None}, fingerprint=("tmpl",)), ramp(40, duration=300))
    sibling = SimJob(
        Job("sibling", {GPU.id: None}, fingerprint=("tmpl",)), ramp(40, duration=300), submit_at=1
    )
    est = HistoryEstimator(InMemoryHistoryStore())
    trace = simulate(Scenario((GPU,), (first, sibling)), FifoPolicy(strict=False), estimator=est)
    starts = {
        a.job_id: (t.now, a)
        for t in trace.ticks
        for a in t.decision.actions
        if isinstance(a, Start)
    }
    stable_at = stable_times(trace, "first")[0]
    sib_t, sib_start = starts["sibling"]
    assert sib_t > stable_at  # it waited for the probation slot, plus one tick
    assert sib_start.estimate[GPU.id] == Resources(vram=40)  # the running record
    assert sib_start.trusted == frozenset({GPU.id})
    assert {rt.state for rt in trace.final.jobs.values()} == {JobState.SUCCEEDED}
