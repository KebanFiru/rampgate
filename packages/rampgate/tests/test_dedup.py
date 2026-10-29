"""Stale, duplicate and out-of-order events (docs/concepts.md)."""

from dataclasses import replace

import pytest

from rampgate import (
    Config,
    DeviceObserved,
    Event,
    ExitKind,
    IgnoreReason,
    InvalidTransition,
    JobExited,
    JobObserved,
    JobStarted,
    JobState,
    Resources,
)
from rampgate_testkit import GPU, GPU1, Driver, gpus

S, D, OOO = IgnoreReason.STALE_ATTEMPT, IgnoreReason.DUPLICATE, IgnoreReason.OUT_OF_ORDER


def test_duplicate_started_and_exited() -> None:
    d = Driver().to(JobState.WARMING).started(t=0.0)
    assert d.ignored == [D]
    d(JobExited("j", 1, ExitKind.SUCCESS, 5.0), JobExited("j", 1, ExitKind.SUCCESS, 5.0))
    assert d.ignored == [D, D]


def test_stale_attempt_events_are_ignored() -> None:
    d = Driver().to(JobState.WARMING).room(vram=50).exited(ExitKind.OOM)
    before = d.state.jobs
    d(JobStarted("j", 1, 0.0), JobExited("j", 1, ExitKind.FAILURE, 9.0))
    d(JobObserved("j", GPU, 50.0, Resources(vram=1), attempt=1))
    assert d.ignored == [S, S, S]
    assert d.state.jobs == before


def test_exit_overtaking_start_makes_start_out_of_order() -> None:
    d = Driver().submit().start()(JobExited("j", 1, ExitKind.SUCCESS, 1.0))
    d(JobStarted("j", 1, 0.5))
    assert d.ignored == [OOO]
    assert d.rt().state is JobState.SUCCEEDED


def test_observation_duplicate_and_out_of_order() -> None:
    d = Driver().to(JobState.WARMING).observe(10.0, vram=5).observe(10.0, vram=5)
    d.observe(9.0, vram=50)
    assert d.ignored == [D, OOO]
    rt = d.rt()
    assert rt.peak[GPU] == Resources(vram=5)
    assert rt.last_peak_growth_t == 10.0


def test_out_of_order_observation_never_moves_growth_time_back() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("j", {GPU: None, GPU1: None}).start().started()
    d.observe(20.0, device=GPU, vram=10).observe(5.0, device=GPU1, vram=10)
    assert d.rt().last_peak_growth_t == 20.0


def test_observation_without_attempt_after_exit_is_stale() -> None:
    d = Driver().to(JobState.WARMING).room(29.0, vram=50).exited(ExitKind.OOM, t=30.0)
    d.observe(25.0, vram=7)
    assert d.ignored == [S]
    d.start().started(t=40.0).observe(29.0, vram=7)
    assert d.ignored == [S, S]
    assert d.rt().peak == {}


def test_late_observation_on_finished_job_is_stale() -> None:
    d = Driver().to(JobState.SUCCEEDED).observe(500.0, vram=1)
    assert d.ignored == [S]


def test_observation_of_never_started_job_raises() -> None:
    with pytest.raises(InvalidTransition, match="never started"):
        Driver().submit().observe(1.0, vram=1)


def test_observation_on_foreign_device_raises() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.WARMING)
    with pytest.raises(InvalidTransition, match="placement"):
        d.observe(1.0, device=GPU1, vram=1)


def test_observation_future_attempt_raises() -> None:
    d = Driver().to(JobState.WARMING)
    with pytest.raises(InvalidTransition, match="future"):
        d.observe(1.0, attempt=2, vram=1)


def test_device_observed_foreign_window() -> None:
    d = Driver(config=Config(foreign_window_s=10)).to(JobState.WARMING).observe(1.0, vram=30)
    d(DeviceObserved(GPU, 1.0, Resources(vram=50)))
    d(DeviceObserved(GPU, 5.0, Resources(vram=35)))
    assert d.state.devices[GPU].foreign == Resources(vram=20)
    d(DeviceObserved(GPU, 12.0, Resources(vram=31)))
    assert d.state.devices[GPU].foreign == Resources(vram=5)
    d(DeviceObserved(GPU, 12.0, Resources(vram=99)), DeviceObserved(GPU, 3.0, Resources(vram=99)))
    assert d.ignored == [D, OOO]
    assert d.state.devices[GPU].foreign == Resources(vram=5)


def test_device_observed_foreign_is_clamped_at_zero() -> None:
    d = Driver().to(JobState.WARMING).observe(1.0, vram=30)
    d(DeviceObserved(GPU, 1.0, Resources(vram=10)))
    assert d.state.devices[GPU].foreign == Resources()


def test_device_observed_unknown_device_raises() -> None:
    with pytest.raises(InvalidTransition, match="unknown device"):
        Driver()(DeviceObserved("nope", 0.0, Resources()))


def test_replaying_keyed_events_only_grows_the_ignored_log() -> None:
    """Early form of I11, for events that carry an attempt or a timestamp."""
    d = Driver().submit()
    events: list[Event] = [
        JobStarted("j", 1, 0.0),
        JobObserved("j", GPU, 1.0, Resources(vram=10), steps=1),
        DeviceObserved(GPU, 1.0, Resources(vram=60)),
        JobObserved("j", GPU, 2.0, Resources(vram=20), steps=5),
        JobExited("j", 1, ExitKind.OOM, 3.0),
        JobStarted("j", 2, 4.0),
        JobObserved("j", GPU, 5.0, Resources(vram=25), steps=9),
    ]
    d.start()
    for ev in events[:5]:
        d(ev)
    d.start()
    for ev in events[5:]:
        d(ev)
    final = replace(d.state, ignored_events=())
    for n in range(len(events) + 1):
        for ev in events[:n]:
            d(ev)
        assert replace(d.state, ignored_events=()) == final
