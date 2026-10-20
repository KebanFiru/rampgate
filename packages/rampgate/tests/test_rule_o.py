"""Rule O (OOM feedback), SPEC §5."""

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


def test_oom_with_no_data_learns_from_admitted_free() -> None:
    d = Driver().submit().start(free={GPU: Resources(vram=60)}).exited(ExitKind.OOM)
    rt = d.rt()
    assert rt.learned == {GPU: Resources(vram=75.0)}
    assert d.state.history_records == (HistoryRecord("j", GPU, Resources(vram=60), True, 1),)
    assert d.state.outbox == (Requeue("j", 2, {GPU: Resources(vram=75.0)}, RequeueReason.OOM),)


def test_oom_lower_bound_is_max_of_peak_estimate_and_admitted_free() -> None:
    d = Driver().submit("j", {GPU: Resources(vram=30)})
    d.start(free={GPU: Resources(vram=20)}).started().observe(1.0, vram=50)
    d.exited(ExitKind.OOM)
    assert d.rt().learned[GPU]["vram"] == pytest.approx(62.5)
    assert d.state.history_records[0].peak == Resources(vram=50)


def test_oom_estimate_is_capped_at_capacity() -> None:
    d = Driver().submit().start(free={GPU: Resources(vram=90)}).exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=100)


def test_oom_estimate_strictly_increases_until_capacity() -> None:
    d = Driver(config=Config(max_oom_retries=20)).submit()
    last = 0.0
    free = Resources(vram=10)
    while d.rt().state is JobState.PENDING:
        d.start(free={GPU: free}).exited(ExitKind.OOM)
        est = d.rt().learned[GPU]["vram"]
        assert est > last or est == 100
        last = est
        free = Resources()
    assert d.rt().state is JobState.FAILED
    assert d.state.outbox[-1].reason is FailReason.OOM_RETRIES_EXHAUSTED  # type: ignore[union-attr]
    assert last == 100


def test_oom_with_the_whole_device_free_fails_immediately() -> None:
    d = Driver().submit().start(free={GPU: Resources(vram=100)}).exited(ExitKind.OOM)
    assert d.rt().state is JobState.FAILED
    assert d.state.outbox == (Fail("j", 1, FailReason.EXCEEDS_DEVICE_CAPACITY),)


@pytest.mark.parametrize(("free", "fails"), [(98.0, True), (97.9, False)])
def test_whole_device_threshold_uses_foreign_tolerance(free: float, fails: bool) -> None:
    d = Driver().submit().start(free={GPU: Resources(vram=free)}).exited(ExitKind.OOM)
    assert (d.rt().state is JobState.FAILED) is fails


def test_capacity_fail_ignores_where_the_estimate_came_from() -> None:
    # A hint of the full device, but admitted with half of it free: retry with growth.
    d = Driver().submit("j", {GPU: Resources(vram=100)}, force=True)
    d.start(free={GPU: Resources(vram=50)}).exited(ExitKind.OOM)
    assert d.rt().state is JobState.PENDING
    assert d.rt().learned[GPU] == Resources(vram=100)
    # No hint, no history, but the whole device was free: fail.
    d2 = Driver().submit().start(free={GPU: Resources(vram=100)}).exited(ExitKind.OOM)
    assert d2.rt().state is JobState.FAILED


def test_capacity_fail_only_counts_the_oom_dimension() -> None:
    cap = Resources(vram=100, ram=100)
    d = Driver(gpus(cap=cap)).submit().start(free={GPU: Resources(vram=40, ram=100)})
    d.exited(ExitKind.OOM, oom_dims=("vram",))
    assert d.rt().state is JobState.PENDING
    d2 = Driver(gpus(cap=cap)).submit().start(free={GPU: Resources(vram=40, ram=100)})
    d2.exited(ExitKind.OOM, oom_dims=("ram",))
    assert d2.rt().state is JobState.FAILED


def test_oom_dims_limits_growth() -> None:
    cap = Resources(vram=100, ram=100)
    d = Driver(gpus(cap=cap)).submit().start(free={GPU: Resources(vram=40, ram=40)})
    d.started().observe(1.0, vram=30, ram=20)
    d.exited(ExitKind.OOM, oom_dims=("vram",))
    assert d.rt().learned[GPU] == Resources(vram=50.0, ram=20)
    assert d.state.history_records[0].peak == Resources(vram=40, ram=20)


def test_oom_retries_exhausted() -> None:
    d = Driver(config=Config(max_oom_retries=2)).submit()
    for _ in range(2):
        d.start(free={GPU: Resources(vram=1)}).exited(ExitKind.OOM)
        assert d.rt().state is JobState.PENDING
    d.start(free={GPU: Resources(vram=1)}).exited(ExitKind.OOM)
    assert d.rt().state is JobState.FAILED
    assert d.rt().oom_count == 3
    assert d.state.outbox[-1] == Fail("j", 3, FailReason.OOM_RETRIES_EXHAUSTED)


def test_zero_oom_retries_fails_first_oom() -> None:
    d = Driver(config=Config(max_oom_retries=0)).submit().start(free={GPU: Resources(vram=1)})
    d.exited(ExitKind.OOM)
    assert d.rt().state is JobState.FAILED


def test_oom_keeps_group_and_position() -> None:
    d = Driver().submit("j", group="g", position=4).start().exited(ExitKind.OOM)
    assert (d.rt().job.group, d.rt().job.position) == ("g", 4)


def test_gang_oom_learns_per_device() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("j", {GPU: None, GPU1: Resources(vram=10)})
    d.start(free={GPU: Resources(vram=20), GPU1: Resources(vram=5)}).exited(ExitKind.OOM)
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
    d = Driver().submit().start().exited(ExitKind.OOM)
    assert d.state.history_records == ()


def test_unknown_job_oom_learns_at_least_unknown_min_free() -> None:
    d = Driver().submit()
    d(Start("j", 1, (GPU,), False, {GPU: Resources(vram=4)}, 0.0, {GPU: None}))
    d.exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=12.5)  # max(4, 10 % of 100) * 1.25
    assert d.state.history_records[0].peak == Resources(vram=10.0)


def test_oom_previous_estimate_includes_the_estimator_value() -> None:
    d = Driver().submit()
    d(Start("j", 1, (GPU,), False, {GPU: Resources(vram=4)}, 0.0, {GPU: Resources(vram=40)}))
    d.exited(ExitKind.OOM)
    assert d.rt().learned[GPU] == Resources(vram=50.0)
