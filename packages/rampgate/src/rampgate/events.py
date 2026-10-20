"""Input events (SPEC §3.5)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, TypeAlias

from rampgate.model import Device, DeviceId, ExitKind, GroupId, Job, JobId
from rampgate.resources import Resources

__all__ = [
    "UNSET",
    "CancelRequested",
    "DeviceAdded",
    "DeviceObserved",
    "DeviceRemoved",
    "Event",
    "JobExited",
    "JobObserved",
    "JobStarted",
    "JobSubmitted",
    "JobUpdated",
    "Paused",
    "PreemptRequested",
    "ResumeRequested",
    "Unpaused",
    "Unset",
]


class Unset(Enum):
    """Marker for "field not given" in `JobUpdated` (``None`` is a real value for ``group``)."""

    UNSET = "UNSET"


UNSET: Final = Unset.UNSET


def _check_t(t: float) -> None:
    if isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t):
        raise ValueError(f"t must be a finite number, got {t!r}")


def _check_attempt(attempt: int) -> None:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt must be an int >= 1, got {attempt!r}")


@dataclass(frozen=True, slots=True)
class JobSubmitted:
    job: Job


@dataclass(frozen=True, slots=True)
class JobUpdated:
    job_id: JobId
    group: GroupId | Literal[Unset.UNSET] | None = UNSET
    position: int | Literal[Unset.UNSET] = UNSET
    placement: Mapping[DeviceId, Resources | None] | Literal[Unset.UNSET] = UNSET
    priority: int | Literal[Unset.UNSET] = UNSET
    force: bool | Literal[Unset.UNSET] = UNSET


@dataclass(frozen=True, slots=True)
class JobStarted:
    job_id: JobId
    attempt: int
    t: float

    def __post_init__(self) -> None:
        _check_attempt(self.attempt)
        _check_t(self.t)


@dataclass(frozen=True, slots=True)
class JobExited:
    job_id: JobId
    attempt: int
    kind: ExitKind
    t: float
    # Dimensions that ran out (OOM only); None = all dimensions.
    oom_dims: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _check_attempt(self.attempt)
        _check_t(self.t)
        if self.oom_dims is not None:
            if self.kind is not ExitKind.OOM:
                raise ValueError("oom_dims is only meaningful for ExitKind.OOM")
            object.__setattr__(self, "oom_dims", tuple(sorted(set(self.oom_dims))))


@dataclass(frozen=True, slots=True)
class JobObserved:
    """Usage of one job on one device at time ``t``.

    ``steps`` is the job's own cumulative counter (SPEC §3.5). ``attempt`` is an
    optional extension: adapters that know the attempt (e.g. from a container
    label) should pass it so late samples of a past attempt are recognized
    exactly; ``None`` means "the current attempt".
    """

    job_id: JobId
    device_id: DeviceId
    t: float
    usage: Resources
    steps: int | None = None
    attempt: int | None = None

    def __post_init__(self) -> None:
        _check_t(self.t)
        if self.attempt is not None:
            _check_attempt(self.attempt)
        if self.steps is not None and (isinstance(self.steps, bool) or self.steps < 0):
            raise ValueError(f"steps must be a non-negative int, got {self.steps!r}")


@dataclass(frozen=True, slots=True)
class DeviceObserved:
    """Total usage of a device by everything, managed or not."""

    device_id: DeviceId
    t: float
    used_total: Resources

    def __post_init__(self) -> None:
        _check_t(self.t)


@dataclass(frozen=True, slots=True)
class PreemptRequested:
    job_id: JobId


@dataclass(frozen=True, slots=True)
class ResumeRequested:
    job_id: JobId


@dataclass(frozen=True, slots=True)
class CancelRequested:
    job_id: JobId


@dataclass(frozen=True, slots=True)
class Paused:
    pass


@dataclass(frozen=True, slots=True)
class Unpaused:
    pass


@dataclass(frozen=True, slots=True)
class DeviceAdded:
    device: Device


@dataclass(frozen=True, slots=True)
class DeviceRemoved:
    device_id: DeviceId


Event: TypeAlias = (
    JobSubmitted
    | JobUpdated
    | JobStarted
    | JobExited
    | JobObserved
    | DeviceObserved
    | PreemptRequested
    | ResumeRequested
    | CancelRequested
    | Paused
    | Unpaused
    | DeviceAdded
    | DeviceRemoved
)
