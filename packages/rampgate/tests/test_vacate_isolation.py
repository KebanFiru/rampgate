"""Rule V (vacate) and the isolation retry (docs/algorithm.md, after held-out evaluation 1)."""

from typing import Any

from rampgate import (
    CancelRequested,
    Config,
    DeviceHeldFor,
    ExitKind,
    FifoPolicy,
    JobState,
    PreemptRequested,
    Requeue,
    RequeueReason,
    Resources,
    Snapshot,
    Start,
    Stop,
    StopReason,
    WaitingForIsolation,
    decide,
    dump_state,
    load_state,
)
from rampgate.sim import STRATEGIES
from rampgate_testkit import GPU, Driver

# Rule S at once: the warming job becomes STABLE at the next decide.
FAST = Config(min_warmup_s=0, stable_window_s=0, stable_min_steps=0)


def vacates(dec: object) -> list[str]:
    return [a.job_id for a in dec.actions if isinstance(a, Stop) and a.vacate]  # type: ignore[attr-defined]


def running(d: Driver, job: str, vram: float, *, stable: bool) -> Driver:
    d.submit(job).start(job).started(job).observe(1.0, job, vram=vram)
    return d.stable(job) if stable else d


def decide_fast(d: Driver, now: float = 5.0, admission: object = None) -> object:
    kwargs = {} if admission is None else {"admission": admission}
    return decide(Snapshot(d.state), now, d.config, FifoPolicy(strict=False), **kwargs)  # type: ignore[arg-type]


# -- Rule V -------------------------------------------------------------------------------------


def test_mark_stable_above_headroom_vacates_the_newest_stable_job() -> None:
    d = running(Driver(config=FAST), "a", 50, stable=True)
    running(d, "b", 49.5, stable=False)  # 50 + 49.5 > 100 * 0.98 once b is STABLE
    dec = decide_fast(d)
    assert vacates(dec) == ["b"]
    (stop,) = [a for a in dec.actions if isinstance(a, Stop)]  # type: ignore[attr-defined]
    assert stop.reason is StopReason.PREEMPT


def test_below_headroom_nothing_is_vacated() -> None:
    d = running(Driver(config=FAST), "a", 50, stable=True)
    running(d, "b", 47.0, stable=False)  # 97 <= 98
    assert vacates(decide_fast(d)) == []


def test_warming_growth_alone_does_not_trigger() -> None:
    slow = Config(min_warmup_s=1000, max_warmup_s=1000)
    d = running(Driver(config=slow), "a", 50, stable=True)
    running(d, "b", 49.5, stable=False).observe(2.0, "b", vram=49.9)
    assert vacates(decide_fast(d)) == []


def test_foreign_load_alone_does_not_trigger() -> None:
    d = running(Driver(config=FAST), "a", 40, stable=True)
    running(d, "b", 40, stable=True).foreign(3.0, vram=30)
    assert vacates(decide_fast(d)) == []


def test_the_last_job_on_a_device_is_never_vacated() -> None:
    d = running(Driver(config=FAST), "a", 99.5, stable=False)
    assert vacates(decide_fast(d)) == []


def test_a_warming_job_is_never_the_victim() -> None:
    slow = Driver(config=Config(min_warmup_s=1000, max_warmup_s=1000))
    running(slow, "a", 40, stable=True)
    running(slow, "b", 30, stable=True)
    running(slow, "c", 29.5, stable=False)  # warming, admitted last
    # An UpdateReservation of b triggers; the victim is b (newest STABLE), not warming c.
    slow.observe(2.0, "b", vram=40)
    dec = decide(Snapshot(slow.state), 5.0, slow.config, FifoPolicy(strict=False))
    assert vacates(dec) == ["b"]


def test_baselines_never_vacate() -> None:
    d = running(Driver(config=FAST), "a", 50, stable=True)
    running(d, "b", 49.5, stable=False)
    assert vacates(decide_fast(d, admission=STRATEGIES["GreedyFree"].admission)) == []


def test_a_vacated_job_returns_to_pending_and_waits_with_its_peak() -> None:
    d = running(Driver(config=FAST), "a", 50, stable=True)
    running(d, "b", 49.5, stable=False)
    d(*decide_fast(d).actions)  # type: ignore[attr-defined]
    assert d.rt("b").state is JobState.STOPPING
    assert d.rt("b").vacating
    d.exited(ExitKind.PREEMPTED, "b")
    b = d.rt("b")
    assert (b.state, b.attempt, b.has_run) == (JobState.PENDING, 2, True)
    assert b.learned[GPU] == Resources(vram=49.5)
    (requeue,) = [a for a in d.state.outbox if isinstance(a, Requeue)]
    assert requeue.reason is RequeueReason.VACATED
    # It does not start again next to a (49.5 + a's reservation 57.5 > 100).
    dec = decide(Snapshot(d.state), 9.0, d.config, FifoPolicy(strict=False))
    assert [a for a in dec.actions if isinstance(a, Start)] == []


def test_a_user_preempt_or_cancel_wins_over_a_vacate() -> None:
    for request, end in (
        (PreemptRequested("b"), JobState.PREEMPTED),
        (CancelRequested("b"), JobState.CANCELLED),
    ):
        d = running(Driver(config=FAST), "a", 50, stable=True)
        running(d, "b", 49.5, stable=False)
        d(*decide_fast(d).actions)  # type: ignore[attr-defined]
        d(request).exited(ExitKind.PREEMPTED, "b")
        assert d.rt("b").state is end


def test_a_vacated_job_that_succeeds_while_stopping_succeeds() -> None:
    d = running(Driver(config=FAST), "a", 50, stable=True)
    running(d, "b", 49.5, stable=False)
    d(*decide_fast(d).actions)  # type: ignore[attr-defined]
    assert d.exited(ExitKind.SUCCESS, "b").rt("b").state is JobState.SUCCEEDED


# -- isolation retry ----------------------------------------------------------------------------


def oom_while_warming(d: Driver, job: str) -> Driver:
    d.submit(job).start(job).started(job).observe(1.0, job, vram=20)
    return d.room(vram=20).exited(ExitKind.OOM, job)


def test_an_oom_before_stable_sets_the_isolation_flag_and_mark_stable_clears_it() -> None:
    d = oom_while_warming(Driver(), "j")
    assert d.rt().isolate_seq is not None
    assert d.rt().state is JobState.PENDING
    d.start().started().observe(2.0, vram=30).stable()
    assert d.rt().isolate_seq is None
    stable_oom = Driver().to(JobState.STABLE)
    stable_oom.room(vram=10).exited(ExitKind.OOM)
    assert stable_oom.rt().isolate_seq is None


def test_an_isolation_retry_waits_for_an_empty_device_and_holds_it() -> None:
    d = oom_while_warming(Driver(), "j")
    running(d, "other", 10, stable=True)
    d.submit("late")
    dec = decide(Snapshot(d.state), 5.0, d.config, FifoPolicy(strict=False))
    assert [a for a in dec.actions if isinstance(a, Start)] == []
    assert dec.explanations["j"] == WaitingForIsolation(GPU)
    assert dec.explanations["late"] == DeviceHeldFor(GPU, "j")
    d.exited(ExitKind.SUCCESS, "other").foreign(150.0, vram=0)  # the OOM's room was foreign
    dec = decide(Snapshot(d.state), 200.0, d.config, FifoPolicy(strict=False))
    assert [a.job_id for a in dec.actions if isinstance(a, Start)] == ["j"]


def test_one_isolation_job_holds_a_device_the_oldest_requeue() -> None:
    d = oom_while_warming(Driver(), "first")
    oom_while_warming(d, "second")
    running(d, "other", 10, stable=True)
    dec = decide(Snapshot(d.state), 5.0, d.config, FifoPolicy(strict=False))
    assert dec.explanations["first"] == WaitingForIsolation(GPU)
    assert dec.explanations["second"] == DeviceHeldFor(GPU, "first")


def test_baselines_ignore_the_isolation_flag() -> None:
    d = oom_while_warming(Driver(), "j")
    running(d, "other", 10, stable=True)
    dec = decide(
        Snapshot(d.state),
        5.0,
        d.config,
        FifoPolicy(strict=False),
        STRATEGIES["GreedyFree"].admission,
    )
    assert "j" in [a.job_id for a in dec.actions if isinstance(a, Start)]


def test_v4_fields_survive_dump_and_load_and_v3_dumps_load() -> None:
    d = oom_while_warming(Driver(), "j")
    assert load_state(dump_state(d.state)) == d.state
    data = dump_state(d.state)

    def strip(x: Any) -> Any:
        if isinstance(x, dict):
            out = {k: strip(v) for k, v in x.items()}
            if out.get("$t") == "JobRuntime":
                for f in ("admit_seq", "vacating", "isolate_seq"):
                    out["f"].pop(f, None)
            if out.get("$t") == "State":
                out["f"].pop("next_order", None)
            return out
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    v3 = {**data, "schema_version": 3, "state": strip(data["state"])}
    loaded = load_state(v3)
    assert loaded.jobs["j"].isolate_seq is None
    assert loaded.next_order == 0
