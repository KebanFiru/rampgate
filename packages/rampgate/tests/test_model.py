"""Static model, events, actions and config validation."""

import json

import pytest

from rampgate import (
    CapacityExceeded,
    Config,
    Device,
    ExitKind,
    Fail,
    FailReason,
    ForeignUsageHigh,
    GroupCompleted,
    GroupOutcome,
    Job,
    JobExited,
    JobObserved,
    JobStarted,
    MarkStable,
    Overcommitted,
    Requeue,
    RequeueReason,
    Resources,
    Start,
    Stop,
    StopReason,
    UncleanStop,
    UpdateReservation,
    Warn,
)
from rampgate.actions import canonical

# -- Device / Job --------------------------------------------------------------------


def test_device_validation_and_frozen_labels() -> None:
    labels = {"kind": "a100"}
    dev = Device("node-a/gpu:0", Resources(vram=1), labels)
    labels["kind"] = "changed"
    assert dev.labels == {"kind": "a100"}
    with pytest.raises(ValueError, match="non-empty"):
        Device("", Resources())
    with pytest.raises(TypeError):
        Device("x", {"vram": 1})  # type: ignore[arg-type]


def test_job_validation() -> None:
    job = Job("j", {"b": None, "a": Resources(vram=1)}, fingerprint=("x", "y"))
    assert job.devices == ("a", "b")
    assert job.is_gang
    assert not Job("k", {"a": None}).is_gang
    with pytest.raises(ValueError, match="non-empty"):
        Job("", {"a": None})
    with pytest.raises(ValueError, match="at least one"):
        Job("j", {})
    with pytest.raises(ValueError, match="device ids"):
        Job("j", {"": None})
    with pytest.raises(TypeError, match="hint"):
        Job("j", {"a": 5})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="fingerprint"):
        Job("j", {"a": None}, fingerprint=(1,))  # type: ignore[arg-type]


def test_job_equality_ignores_mapping_identity() -> None:
    assert Job("j", {"a": None}, metadata={"k": "v"}) == Job("j", {"a": None}, metadata={"k": "v"})


# -- events ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [float("nan"), float("inf"), True, "1"])
def test_event_time_must_be_finite(t: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        JobStarted("j", 1, t)  # type: ignore[arg-type]


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5])
def test_event_attempt_must_be_positive_int(attempt: object) -> None:
    with pytest.raises(ValueError, match="attempt"):
        JobStarted("j", attempt, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="attempt"):
        JobObserved("j", "a", 0.0, Resources(), attempt=attempt)  # type: ignore[arg-type]


def test_job_observed_steps_validation() -> None:
    with pytest.raises(ValueError, match="steps"):
        JobObserved("j", "a", 0.0, Resources(), steps=-1)


def test_oom_dims_only_for_oom() -> None:
    assert JobExited("j", 1, ExitKind.OOM, 0.0, ("vram", "ram", "vram")).oom_dims == ("ram", "vram")
    with pytest.raises(ValueError, match="oom_dims"):
        JobExited("j", 1, ExitKind.FAILURE, 0.0, ("vram",))


# -- actions ----------------------------------------------------------------------------

R = Resources(vram=1)
ALL_ACTIONS = [
    Start("j", 1, ("a",), False, {"a": R}),
    Stop("j", 1, StopReason.PREEMPT, 120),
    MarkStable("j", 1, {"a": R}),
    UpdateReservation("j", 1, {"a": R}),
    Requeue("j", 2, {"a": None}, RequeueReason.OOM),
    Fail("j", 1, FailReason.OOM_RETRIES_EXHAUSTED),
    GroupCompleted("g", GroupOutcome.OK),
    Warn(CapacityExceeded("g", "a", R, Resources())),
    Warn(Overcommitted("a", R, R)),
    Warn(ForeignUsageHigh("a", R)),
    Warn(UncleanStop("j", 1)),
]


def test_action_ids_are_deterministic_and_distinct() -> None:
    ids = [a.action_id for a in ALL_ACTIONS]
    assert len(set(ids)) == len(ids)
    assert ids == [a.action_id for a in ALL_ACTIONS]
    assert all(len(i) == 32 for i in ids)


def test_action_id_depends_on_attempt_and_content() -> None:
    a = Start("j", 1, ("a",), False, {"a": R})
    assert a.action_id == Start("j", 1, ("a",), False, {"a": Resources(vram=1)}).action_id
    assert a.action_id != Start("j", 2, ("a",), False, {"a": R}).action_id
    assert a.action_id != Start("j", 1, ("a",), True, {"a": R}).action_id
    assert MarkStable("j", 1, {"a": R}).action_id != UpdateReservation("j", 1, {"a": R}).action_id


def test_canonical_is_json() -> None:
    for a in ALL_ACTIONS:
        json.dumps(canonical(a))
    with pytest.raises(TypeError):
        canonical(object())


# -- config -----------------------------------------------------------------------------


def test_config_defaults_match_spec() -> None:
    c = Config()
    assert (c.safety_margin, c.min_warmup_s, c.stable_window_s, c.stable_min_steps) == (
        1.15,
        30,
        60,
        20,
    )
    assert (c.growth_tol, c.max_warmup_s, c.oom_growth, c.max_oom_retries) == (0.02, 600, 1.25, 2)
    assert (c.max_lost_retries, c.preempt_grace_s, c.foreign_window_s) == (3, 120, 10)
    assert (c.foreign_tolerance, c.start_timeout_s, c.unknown_min_free) == (0.02, 300, Resources())


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"safety_margin": 0.9}, ValueError),
        ({"oom_growth": 1.0}, ValueError),
        ({"min_warmup_s": 700}, ValueError),
        ({"foreign_tolerance": 1.0}, ValueError),
        ({"growth_tol": -0.1}, ValueError),
        ({"stable_window_s": float("inf")}, ValueError),
        ({"max_oom_retries": 1.5}, TypeError),
        ({"min_warmup_s": True}, TypeError),
        ({"unknown_min_free": {"vram": 1}}, TypeError),
    ],
)
def test_config_validation(kwargs: dict[str, object], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        Config(**kwargs)  # type: ignore[arg-type]
