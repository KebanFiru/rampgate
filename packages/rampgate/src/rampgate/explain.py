"""Explanations: why a job is waiting (docs/concepts.md)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from rampgate.actions import QueueStopped
from rampgate.model import DeviceId, GroupId, JobId
from rampgate.resources import Number, Resources

__all__ = [
    "AwaitingResume",
    "BlockedByForeignUsage",
    "DeviceHeldFor",
    "EnginePaused",
    "Explanation",
    "GangPartiallyBlocked",
    "GroupBarrier",
    "InsufficientResources",
    "RejectedByAdmission",
    "UnknownEstimateBlocked",
    "WaitingForIsolation",
    "WaitingForWarmup",
]


@dataclass(frozen=True, slots=True)
class WaitingForWarmup:
    """Another job is warming up on the device (Rule P)."""

    device_id: DeviceId
    warming_job_id: JobId


@dataclass(frozen=True, slots=True)
class InsufficientResources:
    """The estimate does not fit in ``free_d`` (Rule A)."""

    device_id: DeviceId
    dim: str
    need: Number
    available: Number


@dataclass(frozen=True, slots=True)
class GroupBarrier:
    """The job's group is not the active one."""

    active_group: GroupId


@dataclass(frozen=True, slots=True)
class EnginePaused:
    """The engine is paused; ``reason`` is set when a STOP_QUEUE failure paused it."""

    reason: QueueStopped | None = None


@dataclass(frozen=True, slots=True)
class UnknownEstimateBlocked:
    """No estimate, and ``free_d`` is below ``unknown_min_free``."""

    device_id: DeviceId


@dataclass(frozen=True, slots=True)
class GangPartiallyBlocked:
    """Some devices of a gang job admit it and these do not."""

    blocked_devices: tuple[DeviceId, ...]


@dataclass(frozen=True, slots=True)
class DeviceHeldFor:
    """An earlier blocking candidate holds the device (head-of-line blocking)."""

    device_id: DeviceId
    job_id: JobId


@dataclass(frozen=True, slots=True)
class BlockedByForeignUsage:
    """The estimate would fit if the device had no foreign usage."""

    device_id: DeviceId
    dim: str
    need: Number
    available: Number
    foreign: Resources


@dataclass(frozen=True, slots=True)
class RejectedByAdmission:
    """A non-default admission (a benchmark baseline) refused the device by its own rule."""

    device_id: DeviceId
    admission: str


@dataclass(frozen=True, slots=True)
class WaitingForIsolation:
    """An isolation retry (it OOMed before it was stable) waits for ``device_id`` to be empty
    of other managed jobs (docs/algorithm.md)."""

    device_id: DeviceId


@dataclass(frozen=True, slots=True)
class AwaitingResume:
    """The job is PREEMPTED and waits for ``ResumeRequested``."""


Explanation: TypeAlias = (
    WaitingForWarmup
    | InsufficientResources
    | GroupBarrier
    | EnginePaused
    | UnknownEstimateBlocked
    | GangPartiallyBlocked
    | DeviceHeldFor
    | BlockedByForeignUsage
    | RejectedByAdmission
    | AwaitingResume
    | WaitingForIsolation
)
