"""Static model (`Device`, `Job`) and the reducer's immutable state (SPEC §3)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from rampgate.resources import Resources

if TYPE_CHECKING:
    from rampgate.actions import Action
    from rampgate.reducer import IgnoredEvent

__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "Device",
    "DeviceId",
    "DeviceState",
    "ExitKind",
    "GroupId",
    "HistoryRecord",
    "Job",
    "JobId",
    "JobRuntime",
    "JobState",
    "State",
    "StopReason",
    "frozen_map",
]

DeviceId: TypeAlias = str
JobId: TypeAlias = str
GroupId: TypeAlias = str

K = TypeVar("K")
V = TypeVar("V")


def frozen_map(m: Mapping[K, V] | None = None) -> Mapping[K, V]:
    """A read-only copy of ``m`` (compares equal to any mapping with the same items)."""
    return MappingProxyType(dict(m or {}))


class JobState(StrEnum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    WARMING = "WARMING"
    STABLE = "STABLE"
    STOPPING = "STOPPING"
    PREEMPTED = "PREEMPTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})
ACTIVE_STATES = frozenset({JobState.STARTING, JobState.WARMING, JobState.STABLE, JobState.STOPPING})


class ExitKind(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    OOM = "OOM"
    PREEMPTED = "PREEMPTED"
    CANCELLED = "CANCELLED"
    LOST = "LOST"


class StopReason(StrEnum):
    PREEMPT = "PREEMPT"
    CANCEL = "CANCEL"


@dataclass(frozen=True, slots=True)
class Device:
    id: DeviceId
    capacity: Resources
    labels: Mapping[str, str] = field(default_factory=frozen_map)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Device.id must be a non-empty string")
        if not isinstance(self.capacity, Resources):
            raise TypeError("Device.capacity must be a Resources")
        object.__setattr__(self, "labels", frozen_map(self.labels))


@dataclass(frozen=True, slots=True)
class Job:
    """A unit of work. The caller chooses ``placement``; rampgate only decides *when*.

    ``placement`` maps each device the job runs on to an optional declared hint
    (a floor for the estimate) or ``None`` (unknown). More than one device makes
    it a gang job.
    """

    id: JobId
    placement: Mapping[DeviceId, Resources | None]
    group: GroupId | None = None
    position: int = 0
    priority: int = 0
    fingerprint: tuple[str, ...] = ()
    force: bool = False
    metadata: Mapping[str, str] = field(default_factory=frozen_map)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Job.id must be a non-empty string")
        if not self.placement:
            raise ValueError(f"job {self.id!r}: placement must name at least one device")
        for dev, hint in self.placement.items():
            if not isinstance(dev, str) or not dev:
                raise ValueError(f"job {self.id!r}: device ids must be non-empty strings")
            if hint is not None and not isinstance(hint, Resources):
                raise TypeError(f"job {self.id!r}: hint for {dev!r} must be Resources or None")
        if not all(isinstance(k, str) for k in self.fingerprint):
            raise TypeError(f"job {self.id!r}: fingerprint must be a tuple of strings")
        object.__setattr__(self, "placement", frozen_map(self.placement))
        object.__setattr__(self, "fingerprint", tuple(self.fingerprint))
        object.__setattr__(self, "metadata", frozen_map(self.metadata))

    @property
    def devices(self) -> tuple[DeviceId, ...]:
        """Placement devices in sorted order."""
        return tuple(sorted(self.placement))

    @property
    def is_gang(self) -> bool:
        return len(self.placement) > 1


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    """A peak to hand to the estimator. Drained by the Engine (L3); the reducer does no I/O."""

    job_id: JobId
    device_id: DeviceId
    peak: Resources
    lower_bound: bool


@dataclass(frozen=True, slots=True)
class JobRuntime:
    """Everything the reducer tracks for one job.

    ``attempt`` is the current attempt, or the next one while PENDING. Fields
    marked *per attempt* are reset by ``Start``.
    """

    job: Job
    state: JobState
    attempt: int
    submit_seq: int
    has_run: bool = False
    oom_count: int = 0
    lost_count: int = 0
    learned: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    stop_intent: StopReason | None = None
    reports_steps: bool = False
    applied: frozenset[tuple[int, str]] = frozenset()
    last_exit_t: float | None = None
    admitted_free: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    admitted_alone: Mapping[DeviceId, bool] = field(default_factory=frozen_map)
    warm_start: float | None = None
    usage: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    peak: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    last_obs_t: Mapping[DeviceId, float] = field(default_factory=frozen_map)
    last_peak_growth_t: float | None = None
    steps_baseline: int | None = None
    steps_latest: int | None = None
    reservation: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)

    @property
    def id(self) -> JobId:
        return self.job.id

    @property
    def steps_in_attempt(self) -> int | None:
        if self.steps_latest is None or self.steps_baseline is None:
            return 0 if self.reports_steps else None
        return self.steps_latest - self.steps_baseline

    def prev_estimate(self, device_id: DeviceId) -> Resources | None:
        """``max(declared hint, learned)`` on a device, or None if both are unknown."""
        hint = self.job.placement.get(device_id)
        learned = self.learned.get(device_id)
        if hint is None:
            return learned
        return hint if learned is None else hint.max(learned)


@dataclass(frozen=True, slots=True)
class DeviceState:
    device: Device
    used_total: Resources | None = None
    last_obs_t: float | None = None
    foreign_samples: tuple[tuple[float, Resources], ...] = ()

    @property
    def capacity(self) -> Resources:
        return self.device.capacity

    @property
    def foreign(self) -> Resources:
        """``F_d``: max foreign usage over the samples kept in the window."""
        out = Resources()
        for _, sample in self.foreign_samples:
            out = out.max(sample)
        return out


@dataclass(frozen=True, slots=True)
class State:
    """The whole reducer state. Immutable; ``reduce`` returns a new one."""

    devices: Mapping[DeviceId, DeviceState] = field(default_factory=frozen_map)
    jobs: Mapping[JobId, JobRuntime] = field(default_factory=frozen_map)
    next_seq: int = 0
    paused: bool = False
    outbox: tuple[Action, ...] = ()
    history_records: tuple[HistoryRecord, ...] = ()
    ignored_events: tuple[IgnoredEvent, ...] = ()

    @classmethod
    def initial(cls, devices: Iterable[Device] = ()) -> State:
        devs = list(devices)
        ids = [d.id for d in devs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate device ids")
        return cls(devices=frozen_map({d.id: DeviceState(d) for d in devs}))

    def jobs_on(self, device_id: DeviceId) -> list[JobRuntime]:
        """Jobs whose placement includes ``device_id``, in submit order."""
        return sorted(
            (rt for rt in self.jobs.values() if device_id in rt.job.placement),
            key=lambda rt: rt.submit_seq,
        )
