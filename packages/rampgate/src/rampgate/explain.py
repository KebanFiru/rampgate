"""Explanations: why a job is waiting (SPEC §3.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from rampgate.actions import QueueStopped
from rampgate.model import DeviceId, GroupId, JobId
from rampgate.resources import Number

__all__ = [
    "AwaitingResume",
    "DeviceHeldFor",
    "EnginePaused",
    "Explanation",
    "GangPartiallyBlocked",
    "GroupBarrier",
    "InsufficientResources",
    "UnknownEstimateBlocked",
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
    | AwaitingResume
)
