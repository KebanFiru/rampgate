"""Output actions (SPEC §3.6), each with a deterministic ``action_id``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, TypeAlias

from rampgate.model import DeviceId, GroupId, JobId, StopReason
from rampgate.resources import Resources

__all__ = [
    "Action",
    "CapacityExceeded",
    "Fail",
    "FailReason",
    "ForeignUsageHigh",
    "GroupCompleted",
    "GroupOutcome",
    "MarkStable",
    "Overcommitted",
    "QueueStopped",
    "Requeue",
    "RequeueReason",
    "Start",
    "Stop",
    "UncleanStop",
    "UpdateReservation",
    "Warn",
    "WarningKind",
    "canonical",
    "warning_key",
]


class RequeueReason(StrEnum):
    OOM = "OOM"
    START_TIMEOUT = "START_TIMEOUT"
    LOST = "LOST"


class FailReason(StrEnum):
    OOM_RETRIES_EXHAUSTED = "OOM_RETRIES_EXHAUSTED"
    LOST_RETRIES_EXHAUSTED = "LOST_RETRIES_EXHAUSTED"
    EXCEEDS_DEVICE_CAPACITY = "EXCEEDS_DEVICE_CAPACITY"
    EXITED_FAILURE = "EXITED_FAILURE"


class GroupOutcome(StrEnum):
    OK = "OK"
    HAD_FAILURES = "HAD_FAILURES"


def canonical(value: Any) -> Any:
    """Convert a value made of dataclasses, enums, mappings and Resources into plain JSON data."""
    if isinstance(value, Resources):
        return {"__resources__": value.to_json()}
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [canonical(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


class _ActionMixin:
    __slots__ = ()

    @property
    def action_id(self) -> str:
        """Hash of the action's type and content (which includes the attempt, where relevant)."""
        blob = json.dumps(canonical(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class Start(_ActionMixin):
    job_id: JobId
    attempt: int
    placement: tuple[DeviceId, ...]
    resume: bool
    admitted_free: Mapping[DeviceId, Resources]
    issued_at: float


@dataclass(frozen=True, slots=True)
class Stop(_ActionMixin):
    job_id: JobId
    attempt: int
    reason: StopReason
    grace_s: float


@dataclass(frozen=True, slots=True)
class MarkStable(_ActionMixin):
    job_id: JobId
    attempt: int
    reservation: Mapping[DeviceId, Resources]


@dataclass(frozen=True, slots=True)
class UpdateReservation(_ActionMixin):
    job_id: JobId
    attempt: int
    reservation: Mapping[DeviceId, Resources]


@dataclass(frozen=True, slots=True)
class Requeue(_ActionMixin):
    job_id: JobId
    next_attempt: int
    new_estimate: Mapping[DeviceId, Resources | None]
    reason: RequeueReason


@dataclass(frozen=True, slots=True)
class Fail(_ActionMixin):
    job_id: JobId
    attempt: int
    reason: FailReason


@dataclass(frozen=True, slots=True)
class GroupCompleted(_ActionMixin):
    group: GroupId
    outcome: GroupOutcome


@dataclass(frozen=True, slots=True)
class CapacityExceeded:
    """A candidate's estimate exceeds the device's capacity: it can never fit there."""

    group: GroupId | None
    device_id: DeviceId
    estimated: Resources
    available: Resources
    job_id: JobId


@dataclass(frozen=True, slots=True)
class Overcommitted:
    """``sum(R) > C - F`` on a device; ``capacity`` is ``C - F``."""

    device_id: DeviceId
    reserved: Resources
    capacity: Resources


@dataclass(frozen=True, slots=True)
class ForeignUsageHigh:
    """Foreign usage is what keeps a candidate from fitting on a device."""

    device_id: DeviceId
    foreign: Resources
    job_id: JobId


@dataclass(frozen=True, slots=True)
class UncleanStop:
    """A stopped job exited non-zero: work since its last checkpoint may be lost."""

    job_id: JobId
    attempt: int


@dataclass(frozen=True, slots=True)
class QueueStopped:
    """A job of a STOP_QUEUE group failed; reducing this warning pauses the engine."""

    group: GroupId
    job_id: JobId


WarningKind: TypeAlias = (
    CapacityExceeded | Overcommitted | ForeignUsageHigh | UncleanStop | QueueStopped
)


def warning_key(w: WarningKind) -> str:
    """Dedup key of a warning: the same key is never emitted twice while it is recorded."""
    match w:
        case CapacityExceeded() | ForeignUsageHigh():
            parts: list[Any] = [type(w).__name__, w.job_id, w.device_id]
        case Overcommitted():
            parts = [type(w).__name__, w.device_id, canonical(w.reserved)]
        case UncleanStop():
            parts = [type(w).__name__, w.job_id, w.attempt]
        case QueueStopped():
            parts = [type(w).__name__, w.group, w.job_id]
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class Warn(_ActionMixin):
    warning: WarningKind


Action: TypeAlias = (
    Start | Stop | MarkStable | UpdateReservation | Requeue | Fail | GroupCompleted | Warn
)
