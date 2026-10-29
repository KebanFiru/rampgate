"""Rule O (OOM feedback), docs/algorithm.md."""

import pytest

from rampgate import (
    Config,
    ExitKind,
    Fail,
    FailReason,
    HistoryRecord,
    JobState,
    Requeue,
    RequeueReason,
    Resources,
    Start,
)
from rampgate_testkit import GPU, GPU1, Driver, gpus


def test_oom_with_no_data_learns_from_room_at_oom() -> None:
    d = Driver().submit().start().room(vram=60).exited(ExitKind.OOM)
    rt = d.rt()
    assert rt.learned == {GPU: Resources(vram=75.0)}
    assert d.state.history_records == (HistoryRecord("j", GPU, Resources(vram=60), True, 1),)
    assert d.state.outbox == (Requeue("j", 2, {GPU: Resources(vram=75.0)}, RequeueReason.OOM),)


def test_oom_lower_bound_is_max_of_peak_estimate_and_room() -> None:
    d = Driver().submit("j", {GPU: Resources(vram=30)})
    d.start(free={GPU: Resources(vram=90)}).started().observe(1.0, vram=50)
    d.room(vram=20).exited(ExitKind.OOM)
    assert d.rt().learned[GPU]["vram"] == pytest.approx(62.5)
    assert d.state.history_records[0].peak == Resources(vram=50)


def test_oom_estimate_is_capped_at_capacity() -> None:
    d = Driver().submit().start().room(vram=90).exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=100)


def test_oom_estimate_strictly_increases_until_capacity() -> None:
    d = Driver(config=Config(max_oom_retries=20)).submit()
    last = 0.0
    t = 100.0
    while d.rt().state is JobState.PENDING:
        d.start().room(t - 1, vram=10).exited(ExitKind.OOM, t=t)
        est = d.rt().learned[GPU]["vram"]
        assert est > last or est == 100
        last = est
        t += 100
    assert d.rt().state is JobState.FAILED
    assert d.state.outbox[-1].reason is FailReason.OOM_RETRIES_EXHAUSTED  # type: ignore[union-attr]
    assert last == 100


def test_oom_with_the_whole_device_to_itself_fails_immediately() -> None:
    d = Driver().submit().start().exited(ExitKind.OOM)
    assert d.rt().state is JobState.FAILED
    assert d.state.outbox == (Fail("j", 1, FailReason.EXCEEDS_DEVICE_CAPACITY),)


@pytest.mark.parametrize(("room", "fails"), [(98.0, True), (97.9, False)])
def test_whole_device_threshold_uses_foreign_tolerance(room: float, fails: bool) -> None:
    d = Driver().submit().start().room(vram=room).exited(ExitKind.OOM)
    assert (d.rt().state is JobState.FAILED) is fails


def test_capacity_fail_ignores_where_the_estimate_came_from() -> None:
    # A hint of the full device, but only half of it left when it died: retry with growth.
    d = Driver().submit("j", {GPU: Resources(vram=100)}, force=True)
    d.start().room(vram=50).exited(ExitKind.OOM)
    assert d.rt().state is JobState.PENDING
    assert d.rt().learned[GPU] == Resources(vram=100)
    # No hint, no history, but it had the whole device: fail.
    d2 = Driver().submit().start().exited(ExitKind.OOM)
    assert d2.rt().state is JobState.FAILED


def test_capacity_fail_only_counts_the_oom_dimension() -> None:
    cap = Resources(vram=100, ram=100)
    d = Driver(gpus(cap=cap)).submit().start().room(vram=40, ram=100)
    d.exited(ExitKind.OOM, oom_dims=("vram",))
    assert d.rt().state is JobState.PENDING
    d2 = Driver(gpus(cap=cap)).submit().start().room(vram=40, ram=100)
    d2.exited(ExitKind.OOM, oom_dims=("ram",))
    assert d2.rt().state is JobState.FAILED


def test_oom_dims_limits_growth() -> None:
    cap = Resources(vram=100, ram=100)
    d = Driver(gpus(cap=cap)).submit().start()
    d.started().observe(1.0, vram=30, ram=20)
    d.room(vram=40, ram=40).exited(ExitKind.OOM, oom_dims=("vram",))
    assert d.rt().learned[GPU] == Resources(vram=50.0, ram=20)
    assert d.state.history_records[0].peak == Resources(vram=40, ram=20)


def test_oom_retries_exhausted() -> None:
    d = Driver(config=Config(max_oom_retries=2)).submit()
    for t in (100.0, 200.0):
        d.start().room(t - 1, vram=1).exited(ExitKind.OOM, t=t)
        assert d.rt().state is JobState.PENDING
    d.start().room(299.0, vram=1).exited(ExitKind.OOM, t=300.0)
    assert d.rt().state is JobState.FAILED
    assert d.rt().oom_count == 3
    assert d.state.outbox[-1] == Fail("j", 3, FailReason.OOM_RETRIES_EXHAUSTED)


def test_zero_oom_retries_fails_first_oom() -> None:
    d = Driver(config=Config(max_oom_retries=0)).submit().start().room(vram=1)
    d.exited(ExitKind.OOM)
    assert d.rt().state is JobState.FAILED


def test_oom_keeps_group_and_position() -> None:
    d = Driver().submit("j", group="g", position=4).start().room(vram=1).exited(ExitKind.OOM)
    assert (d.rt().job.group, d.rt().job.position) == ("g", 4)


def test_gang_oom_learns_per_device() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("j", {GPU: None, GPU1: Resources(vram=10)})
    d.start().room(vram=20).room(device=GPU1, vram=5).exited(ExitKind.OOM)
    assert d.rt().learned == {GPU: Resources(vram=25.0), GPU1: Resources(vram=12.5)}


def test_hint_above_capacity_never_decreases() -> None:
    d = Driver().submit("j", {GPU: Resources(vram=150)}).submit("x").start("x")
    d.start("j").exited(ExitKind.OOM, "j")
    assert d.rt("j").learned[GPU] == Resources(vram=150)


def test_oom_in_dimension_the_device_lacks_still_grows() -> None:
    d = (
        Driver()
        .submit("j", {GPU: Resources(ram=8)})
        .start()
        .exited(ExitKind.OOM, oom_dims=("ram",))
    )
    assert d.rt().learned[GPU] == Resources(ram=10.0)


def test_zero_lower_bound_is_not_recorded() -> None:
    d = Driver().submit().start().room(vram=0).exited(ExitKind.OOM)
    assert d.state.history_records == ()


def test_unknown_job_oom_learns_at_least_unknown_min_free() -> None:
    d = Driver().submit()
    d(Start("j", 1, (GPU,), False, {GPU: Resources(vram=90)}, 0.0, {GPU: None}))
    d.room(vram=4).exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=12.5)  # max(4, 10 % of 100) * 1.25
    assert d.state.history_records[0].peak == Resources(vram=10.0)


def test_oom_previous_estimate_includes_the_estimator_value() -> None:
    d = Driver().submit()
    d(Start("j", 1, (GPU,), False, {GPU: Resources(vram=90)}, 0.0, {GPU: Resources(vram=40)}))
    d.room(vram=4).exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=50.0)


def test_admitted_free_no_longer_drives_rule_o() -> None:
    # Admitted with the whole device free, but a neighbour took 40 before the OOM.
    d = Driver().submit().submit("x").start().start("x").started("x").observe(98.0, "x", vram=40)
    d.exited(ExitKind.OOM)
    assert d.rt().state is JobState.PENDING
    assert d.rt().learned[GPU] == Resources(vram=75.0)  # room 60 * 1.25, not 100


def test_room_uses_the_max_over_the_window() -> None:
    d = Driver().submit().submit("x").start().start("x").started("x")
    d.observe(92.0, "x", vram=60).observe(99.0, "x", vram=5)  # spike inside the 10 s window
    d.exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=50.0)  # room 40, not 95
    d2 = Driver().submit().submit("x").start().start("x").started("x")
    d2.observe(85.0, "x", vram=60).observe(99.0, "x", vram=5)  # spike before the window
    d2.exited(ExitKind.OOM)
    assert d2.rt().learned[GPU] == Resources(vram=100.0)  # room 95 * 1.25, capped


def test_no_capacity_fail_while_a_neighbours_usage_is_unknown() -> None:
    # The neighbour began in the same instant and has no sample yet: the room is a guess.
    d = Driver().submit().submit("x").start().start("x").started("x")
    d.exited(ExitKind.OOM)
    assert d.rt().state is JobState.PENDING
    assert d.rt().learned[GPU] == Resources(vram=100)  # room counted as 100, capped
    # Once the neighbour has reported (even 0), the room is known.
    d2 = Driver().submit().submit("x").start().start("x").started("x").observe(95.0, "x")
    d2.exited(ExitKind.OOM)
    assert d2.rt().state is JobState.FAILED
