"""Rule S (stability) and steps semantics, SPEC §3.5 and §5."""

import pytest

from rampgate import Config, ExitKind, JobState, Resources, is_peak_growth, is_stable
from rampgate_testkit import Driver

CFG = Config(min_warmup_s=30, stable_window_s=60, stable_min_steps=20, max_warmup_s=600)


def warming(t0: float = 0.0) -> Driver:
    return Driver(config=CFG).to(JobState.STARTING).started(t=t0)


def test_not_stable_before_min_warmup() -> None:
    d = warming().observe(1.0, vram=10, steps=100)
    assert not is_stable(d.rt(), 29.9, CFG)


def test_stable_with_steps_and_quiet_window() -> None:
    d = warming().observe(5.0, vram=10, steps=0).observe(10.0, vram=10, steps=20)
    assert not is_stable(d.rt(), 64.9, CFG)
    assert is_stable(d.rt(), 65.0, CFG)


def test_not_stable_without_enough_steps() -> None:
    d = warming().observe(5.0, vram=10, steps=0).observe(10.0, vram=10, steps=19)
    assert not is_stable(d.rt(), 500.0, CFG)
    assert is_stable(d.rt(), 600.0, CFG)


def test_peak_growth_restarts_window() -> None:
    d = warming().observe(5.0, vram=10, steps=0).observe(50.0, vram=20, steps=50)
    assert not is_stable(d.rt(), 100.0, CFG)
    assert is_stable(d.rt(), 110.0, CFG)


def test_small_growth_within_tolerance_is_not_growth() -> None:
    d = warming().observe(5.0, vram=100, steps=0).observe(50.0, vram=101, steps=50)
    assert d.rt().last_peak_growth_t == 5.0
    assert d.rt().peak["gpu:0"] == Resources(vram=101)


def test_time_only_without_steps() -> None:
    d = warming().observe(5.0, vram=10)
    assert d.rt().steps_in_attempt is None
    assert not is_stable(d.rt(), 64.0, CFG)
    assert is_stable(d.rt(), 65.0, CFG)


def test_no_observations_quiet_since_warm_start() -> None:
    d = warming(t0=10.0)
    assert not is_stable(d.rt(), 69.0, CFG)
    assert is_stable(d.rt(), 70.0, CFG)


def test_last_peak_growth_at_zero_is_honoured() -> None:
    d = Driver(config=CFG).to(JobState.STARTING).observe(0.0, vram=10).started(t=-100.0)
    assert d.rt().last_peak_growth_t == 0.0
    assert not is_stable(d.rt(), 59.0, CFG)
    assert is_stable(d.rt(), 60.0, CFG)


@pytest.mark.parametrize("state", [JobState.STARTING, JobState.STABLE])
def test_only_warming_jobs_can_become_stable(state: JobState) -> None:
    d = Driver(config=CFG).to(state)
    assert not is_stable(d.rt(), 10_000.0, CFG)


def test_steps_baseline_cumulative_counter_after_resume() -> None:
    d = warming().observe(1.0, vram=10, steps=50_000).observe(2.0, vram=10, steps=50_019)
    assert d.rt().steps_in_attempt == 19
    d.observe(3.0, vram=10, steps=50_020)
    assert d.rt().steps_in_attempt == 20


def test_steps_counter_reset_rebaselines() -> None:
    d = warming().observe(1.0, vram=10, steps=500).observe(2.0, vram=10, steps=510)
    d.observe(3.0, vram=10, steps=3)
    rt = d.rt()
    assert (rt.steps_baseline, rt.steps_latest, rt.steps_in_attempt) == (3, 3, 0)


def test_cumulative_and_reset_counters_stabilize_at_same_time() -> None:
    """Early form of I14: only steps_in_attempt matters."""

    def run(offset: int) -> float:
        d = warming()
        for i in range(1, 40):
            d.observe(float(i), vram=10, steps=offset + i)
        rt = d.rt()
        return next(t for t in range(0, 700) if is_stable(rt, float(t), CFG))

    assert run(0) == run(123_456)


def test_steps_reset_by_start() -> None:
    d = warming().observe(1.0, vram=10, steps=5).observe(2.0, vram=10, steps=9)
    d.room(vram=50).exited(ExitKind.OOM).start()
    assert d.rt().steps_in_attempt == 0
    assert d.rt().peak == {}


def test_is_peak_growth() -> None:
    assert is_peak_growth(Resources(), Resources(vram=1), 0.02)
    assert not is_peak_growth(Resources(vram=100), Resources(vram=102), 0.02)
    assert is_peak_growth(Resources(vram=100), Resources(vram=102.1), 0.02)
    assert is_peak_growth(Resources(vram=100), Resources(vram=100, ram=1), 0.02)


def test_step_reporting_job_keeps_needing_steps_after_resume() -> None:
    d = warming().observe(1.0, vram=10, steps=100).exited(ExitKind.LOST, t=2.0)
    d.start().started(t=10.0).observe(11.0, vram=10)
    assert d.rt().steps_in_attempt == 0
    assert not is_stable(d.rt(), 500.0, CFG)
    assert is_stable(d.rt(), 610.0, CFG)
