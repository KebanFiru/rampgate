"""Reservation updates after stabilizing (SPEC §5, I6)."""

import pytest

from rampgate import ExitKind, JobState, Resources, StopReason, UpdateReservation
from rampgate_testkit import GPU, Driver


def stable(peak: float = 40) -> Driver:
    return Driver().to(JobState.WARMING).observe(1.0, vram=peak).stable()


def test_peak_below_reservation_changes_nothing() -> None:
    d = stable().observe(2.0, vram=45)  # 45 < 40 * 1.15
    assert d.rt().reservation == {GPU: Resources(vram=40).scale(1.15)}
    assert d.state.outbox == ()


def test_peak_above_reservation_grows_it() -> None:
    d = stable().observe(2.0, vram=50)
    assert d.rt().reservation[GPU]["vram"] == pytest.approx(57.5)
    assert d.state.outbox == (UpdateReservation("j", 1, d.rt().reservation),)


def test_only_latest_update_kept_in_outbox() -> None:
    d = stable().observe(2.0, vram=50).observe(3.0, vram=70)
    (update,) = d.state.outbox
    assert isinstance(update, UpdateReservation)
    assert update.reservation[GPU]["vram"] == pytest.approx(70 * 1.15)
    d(update)
    assert d.state.outbox == ()


def test_reservation_never_decreases_across_dimensions() -> None:
    d = Driver().to(JobState.WARMING).observe(1.0, vram=40, ram=10).stable()
    before = d.rt().reservation[GPU]
    d.observe(2.0, vram=60, ram=1)
    after = d.rt().reservation[GPU]
    assert before.fits_in(after)
    assert after["ram"] == before["ram"]


def test_reservation_kept_while_stopping_and_released_on_exit() -> None:
    d = stable().stop(StopReason.PREEMPT).observe(2.0, vram=80)
    assert d.rt().reservation[GPU]["vram"] == pytest.approx(92.0)
    d.exited(ExitKind.PREEMPTED)
    assert d.rt().reservation == {}


def test_warming_job_has_no_reservation() -> None:
    d = Driver().to(JobState.WARMING).observe(1.0, vram=99)
    assert d.rt().reservation == {}
    assert d.state.outbox == ()
