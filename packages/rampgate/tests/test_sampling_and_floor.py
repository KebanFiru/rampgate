"""L6 items from the real-hardware runs (SPEC §12 L6): atomic sampling, missing samples,
and the Rule S usage floor."""

from __future__ import annotations

from dataclasses import replace

from rampgate import (
    Config,
    Device,
    DeviceObserved,
    Engine,
    FifoPolicy,
    ForeignUsageHigh,
    JobNotObserved,
    JobObserved,
    JobSample,
    MarkStable,
    NoUsageObserved,
    Resources,
    Start,
    Warn,
    WarnCleared,
)
from rampgate.model import Job, JobState
from rampgate.serialization import SCHEMA_VERSION, dump_state, load_state, to_data
from rampgate.sim import Scenario, SimJob, StepProfile, simulate

GPU = "gpu0"
CFG = Config(min_warmup_s=5, stable_window_s=10, max_warmup_s=60)


def _engine(*, floor: Resources | None = None, config: Config = CFG) -> Engine:
    device = Device(GPU, Resources(vram=1000), stable_min_usage=floor or Resources())
    return Engine([device], FifoPolicy(), config=config)


def _run(engine: Engine, job_id: str = "j", t: float = 0.0) -> None:
    engine.submit(Job(job_id, {GPU: None}), t=t)
    decision = engine.tick(t)
    assert any(isinstance(a, Start) and a.job_id == job_id for a in decision.actions)
    engine.on_started(job_id, 1, t)


def _foreign(engine: Engine) -> Resources:
    return engine.state.devices[GPU].foreign


# -- atomic sampling ------------------------------------------------------------------------------


def test_observe_sample_equals_the_correct_order_not_the_old_misordering() -> None:
    """A job grows 0 → 300 → 600 while the device's own share is 100 (foreign)."""
    samples = [(1.0, 100.0, 0.0), (2.0, 400.0, 300.0), (3.0, 700.0, 600.0)]

    wrong, right, atomic = _engine(), _engine(), _engine()
    for e in (wrong, right, atomic):
        _run(e)
    for t, total, job in samples:
        usage = Resources(vram=job)
        # The L5 harness bug: the device total first, paired with the job's older usage.
        wrong.observe(DeviceObserved(GPU, t, Resources(vram=total)))
        wrong.observe(JobObserved("j", GPU, t, usage))
        right.observe(JobObserved("j", GPU, t, usage))
        right.observe(DeviceObserved(GPU, t, Resources(vram=total)))
        atomic.observe_sample(GPU, t, Resources(vram=total), {"j": JobSample(1, usage)})

    assert _foreign(wrong) == Resources(vram=400)  # 300 MiB of growth counted as foreign
    assert _foreign(right) == Resources(vram=100)
    assert _foreign(atomic) == _foreign(right)
    assert atomic.state == right.state


def test_observe_sample_accepts_plain_tuples_with_steps_and_epoch() -> None:
    engine = _engine()
    _run(engine)
    engine.observe_sample(GPU, 1.0, Resources(vram=150), {"j": (None, Resources(vram=50), 7, 0)})
    rt = engine.state.jobs["j"]
    assert rt.usage[GPU] == Resources(vram=50)
    assert rt.steps_latest == 7
    assert rt.epoch_latest == 0
    assert _foreign(engine) == Resources(vram=100)


# -- a job missing from a sample ------------------------------------------------------------------


def test_a_job_skipped_for_one_sample_changes_nothing_and_raises_no_warning() -> None:
    engine = _engine()
    _run(engine)
    engine.observe_sample(GPU, 1.0, Resources(vram=500), {"j": JobSample(1, Resources(vram=400))})
    engine.tick(1.0)
    before = engine.state.jobs["j"]
    # NVML skipped the process in this sample; the device total still contains its memory.
    engine.observe_sample(GPU, 2.0, Resources(vram=500), {})
    decision = engine.tick(2.0)
    after = engine.state.jobs["j"]
    assert _foreign(engine) == Resources(vram=100)  # its last known usage still counts
    assert replace(after, obs_sample=before.obs_sample) == before  # usage, peak, stability
    assert not [a for a in decision.actions if isinstance(a, Warn | WarnCleared)]


def test_job_not_observed_after_missing_samples_warn_then_cleared() -> None:
    engine = _engine(config=replace(CFG, missing_samples_warn=3))
    _run(engine)
    warnings: list[object] = []
    for t in range(1, 5):
        engine.observe_sample(GPU, float(t), Resources(vram=100), {})
        warnings += [a for a in engine.tick(float(t)).actions if isinstance(a, Warn | WarnCleared)]
    assert warnings == [Warn(JobNotObserved("j", 1, GPU))]  # at the 3rd sample, once
    engine.observe_sample(GPU, 5.0, Resources(vram=150), {"j": JobSample(1, Resources(vram=50))})
    cleared = [a for a in engine.tick(5.0).actions if isinstance(a, WarnCleared)]
    assert [c.kind for c in cleared] == ["JobNotObserved"]


def test_missing_samples_warn_zero_disables_the_warning() -> None:
    engine = _engine(config=replace(CFG, missing_samples_warn=0))
    _run(engine)
    for t in range(1, 10):
        engine.observe_sample(GPU, float(t), Resources(vram=100), {})
        assert not [a for a in engine.tick(float(t)).actions if isinstance(a, Warn)]


def test_foreign_usage_warning_not_raised_by_a_skipped_sample() -> None:
    """Counting a skipped job as 0 would make its memory look foreign."""
    engine = _engine()
    _run(engine)
    engine.observe_sample(GPU, 1.0, Resources(vram=900), {"j": JobSample(1, Resources(vram=850))})
    engine.tick(1.0)
    engine.submit(Job("k", {GPU: Resources(vram=100)}), t=1.0)
    engine.observe_sample(GPU, 2.0, Resources(vram=900), {})
    decision = engine.tick(2.0)
    assert not [
        a
        for a in decision.actions
        if isinstance(a, Warn) and isinstance(a.warning, ForeignUsageHigh)
    ]


# -- Rule S usage floor ---------------------------------------------------------------------------


def _stable_at(engine: Engine, usage_at: dict[float, float], until: float) -> float | None:
    for t in range(1, int(until) + 1):
        used = usage_at.get(float(t), 0.0)
        engine.observe_sample(
            GPU, float(t), Resources(vram=used + 10), {"j": JobSample(1, Resources(vram=used))}
        )
        if any(isinstance(a, MarkStable) for a in engine.tick(float(t)).actions):
            return float(t)
    return None


def test_no_stability_below_the_floor_even_after_max_warmup() -> None:
    engine = _engine(floor=Resources(vram=256))
    _run(engine)
    warns: list[object] = []
    for t in range(1, 80):
        engine.observe_sample(
            GPU, float(t), Resources(vram=92), {"j": JobSample(1, Resources(vram=82))}
        )
        decision = engine.tick(float(t))
        assert not any(isinstance(a, MarkStable) for a in decision.actions), t
        warns += [(t, a) for a in decision.actions if isinstance(a, Warn)]
    assert engine.state.jobs["j"].state is JobState.WARMING
    assert warns == [(60, Warn(NoUsageObserved("j", 1, GPU)))]  # at max_warmup_s, once


def test_floor_off_keeps_the_old_behaviour() -> None:
    engine = _engine()
    _run(engine)
    assert _stable_at(engine, {}, 80) == 10  # quiet for stable_window_s, no floor


def test_floor_is_skipped_for_jobs_that_report_steps() -> None:
    engine = _engine(floor=Resources(vram=256))
    _run(engine)
    stable = None
    for t in range(1, 40):
        engine.observe_sample(
            GPU, float(t), Resources(vram=60), {"j": JobSample(1, Resources(vram=50), 100 * t)}
        )
        if any(isinstance(a, MarkStable) for a in engine.tick(float(t)).actions):
            stable = t
            break
    assert stable is not None


def test_no_usage_warning_cleared_when_the_job_starts_using_the_device() -> None:
    engine = _engine(floor=Resources(vram=256))
    _run(engine)
    usage = {float(t): 400.0 for t in range(65, 200)}
    actions = []
    for t in range(1, 90):
        used = usage.get(float(t), 0.0)
        engine.observe_sample(
            GPU, float(t), Resources(vram=used + 10), {"j": JobSample(1, Resources(vram=used))}
        )
        actions += [(t, a) for a in engine.tick(float(t)).actions]
    warns = [(t, type(a).__name__) for t, a in actions if isinstance(a, Warn | WarnCleared)]
    assert warns == [(60, "Warn"), (65, "WarnCleared")]
    stable = [t for t, a in actions if isinstance(a, MarkStable)]
    # Warm-up starts at the crossing (65): quiet for stable_window_s after it, although
    # max_warmup_s since the start has long passed.
    assert stable == [75]
    assert engine.state.jobs["j"].floor_crossed_t == 65


def test_simulated_data_loading_phase_before_any_gpu_allocation() -> None:
    """A job without a step counter loads data for 90 s with no GPU memory, then ramps
    (SPEC §12 L6). Without a floor, Rule S marks it STABLE on its empty loading phase."""
    plateau = Resources(vram=40)
    workload = StepProfile(
        steps=((0.0, Resources()), (90.0, Resources(vram=12)), (93.0, plateau)),
        duration_s=300.0,
        checkpoint_every_s=60.0,
        reports_steps=False,
    )
    config = Config(min_warmup_s=30, stable_window_s=60, max_warmup_s=600)
    for floor, before_load_ends in ((Resources(), True), (Resources(vram=1), False)):
        device = Device(GPU, Resources(vram=100), stable_min_usage=floor)
        scenario = Scenario((device,), (SimJob(Job("j", {GPU: None}), workload, 0.0),))
        trace = simulate(scenario, FifoPolicy(), config=config, max_time=600.0)
        marks = [
            (t.now, a) for t in trace.ticks for a in t.decision.actions if isinstance(a, MarkStable)
        ]
        assert len(marks) == 1, floor
        t_stable, mark = marks[0]
        assert (t_stable < 90) is before_load_ends, (floor, t_stable)
        if not before_load_ends:
            assert mark.reservation[GPU]["vram"] >= plateau["vram"]  # what it really uses
        else:
            assert mark.reservation[GPU]["vram"] == 0  # the L5 risk: nothing reserved


def test_slow_growth_after_data_loading_is_capped_at_crossing_plus_max_warmup() -> None:
    """90 s of data loading, then growth that never goes quiet: STABLE at the floor
    crossing + max_warmup_s, so the probation slot is still bounded (SPEC §5 Rule S)."""
    growth = tuple((90.0 + 10 * k, Resources(vram=10 * 1.05**k)) for k in range(90))
    workload = StepProfile(
        steps=((0.0, Resources()), *growth),
        duration_s=1200.0,
        checkpoint_every_s=60.0,
        reports_steps=False,
    )
    config = Config(min_warmup_s=30, stable_window_s=60, max_warmup_s=600)
    stable = {}
    for floor in (Resources(), Resources(vram=1)):
        device = Device(GPU, Resources(vram=10_000), stable_min_usage=floor)
        scenario = Scenario((device,), (SimJob(Job("j", {GPU: None}), workload, 0.0),))
        trace = simulate(scenario, FifoPolicy(), config=config, max_time=900.0)
        stable[floor["vram"]] = [
            t.now for t in trace.ticks for a in t.decision.actions if isinstance(a, MarkStable)
        ]
    # The job starts at t = 1, so its loading ends (and it passes the floor) at t = 91.
    assert stable[0] == [61.0]  # no floor: "stable" while still loading data
    assert stable[1] == [691.0]  # never quiet: the crossing + max_warmup_s


# -- persistence ----------------------------------------------------------------------------------


def test_v4_dump_loads_with_floor_off_and_no_samples() -> None:
    engine = _engine(floor=Resources(vram=256))
    _run(engine)
    engine.observe_sample(GPU, 1.0, Resources(vram=50), {"j": JobSample(1, Resources(vram=40))})
    state = engine.state
    v5 = dump_state(state)
    assert v5["schema_version"] == SCHEMA_VERSION == 6

    def strip(node: object) -> object:
        if isinstance(node, dict):
            out = {k: strip(v) for k, v in node.items()}
            fields = out.get("f")
            if isinstance(fields, dict):
                for name in ("stable_min_usage", "samples", "obs_sample", "floor_crossed_t"):
                    if out.get("$t") in ("Device", "DeviceState", "JobRuntime"):
                        fields.pop(name, None)
            return out
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    v4 = {"format": "rampgate.state", "schema_version": 4, "state": strip(to_data(state))}
    loaded = load_state(v4)
    ds = loaded.devices[GPU]
    assert ds.device.stable_min_usage == Resources()
    assert ds.samples == 0
    assert loaded.jobs["j"].obs_sample == {}
    assert load_state(v5) == state


def test_v5_dump_loads_without_a_floor_crossing() -> None:
    engine = _engine(floor=Resources(vram=256))
    _run(engine)
    engine.observe_sample(GPU, 1.0, Resources(vram=400), {"j": JobSample(1, Resources(vram=300))})
    state = engine.state
    assert state.jobs["j"].floor_crossed_t == 1.0

    def strip(node: object) -> object:
        if isinstance(node, dict):
            out = {k: strip(v) for k, v in node.items()}
            fields = out.get("f")
            if out.get("$t") == "JobRuntime" and isinstance(fields, dict):
                fields.pop("floor_crossed_t", None)
            return out
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    v5 = {"format": "rampgate.state", "schema_version": 5, "state": strip(to_data(state))}
    loaded = load_state(v5)
    assert loaded.jobs["j"].floor_crossed_t is None
    assert loaded.jobs["j"] == replace(state.jobs["j"], floor_crossed_t=None)
