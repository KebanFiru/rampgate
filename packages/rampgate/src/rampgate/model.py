"""Static model (`Device`, `Job`) and the reducer's immutable state (SPEC §3)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from rampgate.resources import Resources

if TYPE_CHECKING:
    from rampgate.actions import Action, QueueStopped, WarningKind
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
    "Snapshot",
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
    """A device rampgate admits jobs onto.

    ``stable_min_usage`` is the Rule S usage floor (SPEC §5): a job that reports no
    steps becomes STABLE on this device only after its usage here exceeded the floor
    in some dimension. Off (``Resources()``) by default, because rampgate does not know
    the device's units; ``rampgate_nvml.NvmlProbe`` sets 256 MiB of ``vram``. Without
    that adapter, set it yourself (a CUDA context alone was measured at ~82 MiB).
    """

    id: DeviceId
    capacity: Resources
    labels: Mapping[str, str] = field(default_factory=frozen_map)
    stable_min_usage: Resources = field(default_factory=Resources)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Device.id must be a non-empty string")
        if not isinstance(self.capacity, Resources):
            raise TypeError("Device.capacity must be a Resources")
        if not isinstance(self.stable_min_usage, Resources):
            raise TypeError("Device.stable_min_usage must be a Resources")
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
    attempt: int
    running: bool = False

    @property
    def kind(self) -> str:
        """``peak`` (SUCCESS), ``lower_bound`` (OOM or stop) or ``running`` (STABLE, Rule H).

        Only ``lower_bound`` records have ``lower_bound=True``.
        """
        if self.running:
            return "running"
        return "lower_bound" if self.lower_bound else "peak"

    @property
    def record_id(self) -> str:
        """``job:attempt:device:kind``; stores max-merge records with the same id (SPEC §7)."""
        return f"{self.job_id}:{self.attempt}:{self.device_id}:{self.kind}"


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
    submit_t: float | None = None
    has_run: bool = False
    oom_count: int = 0
    lost_count: int = 0
    learned: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    stop_intent: StopReason | None = None
    reports_steps: bool = False
    applied: frozenset[tuple[int, str]] = frozenset()
    last_exit_t: float | None = None
    admitted_free: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    admitted_estimate: Mapping[DeviceId, Resources | None] = field(default_factory=frozen_map)
    trusted: frozenset[DeviceId] = frozenset()
    admitted_observations: Mapping[DeviceId, int] = field(default_factory=frozen_map)
    stable_at: float | None = None
    starting_since: float | None = None
    warm_start: float | None = None
    usage: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    usage_samples: Mapping[DeviceId, tuple[tuple[float, Resources], ...]] = field(
        default_factory=frozen_map
    )
    peak: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    last_obs_t: Mapping[DeviceId, float] = field(default_factory=frozen_map)
    last_peak_growth_t: float | None = None
    steps_baseline: int | None = None
    steps_latest: int | None = None
    epoch_baseline: int | None = None
    epoch_latest: int | None = None
    reservation: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    # Order of the latest Start among all jobs (Rule V picks the newest as its victim).
    admit_seq: int | None = None
    # Stopped by Rule V: returns to PENDING after exit unless a user stop request wins.
    vacating: bool = False
    # Isolation retry (SPEC §5): set when the job OOMed before it was stable, cleared at
    # MarkStable; the order of that OOM, so the oldest one holds a device.
    isolate_seq: int | None = None
    # Per device, its sample counter (DeviceState.samples) when this attempt was last
    # observed there, or when it started.
    obs_sample: Mapping[DeviceId, int] = field(default_factory=frozen_map)

    @property
    def id(self) -> JobId:
        return self.job.id

    @property
    def steps_in_attempt(self) -> int | None:
        if self.steps_latest is None or self.steps_baseline is None:
            return 0 if self.reports_steps else None
        return self.steps_latest - self.steps_baseline

    @property
    def epoch_in_attempt(self) -> int | None:
        if self.epoch_latest is None or self.epoch_baseline is None:
            return None
        return self.epoch_latest - self.epoch_baseline

    def usage_at(self, device_id: DeviceId, now: float, window_s: float) -> Resources:
        """Max usage on a device over this attempt's samples with ``t >= now - window_s``."""
        out = Resources()
        for t, sample in self.usage_samples.get(device_id, ()):
            if t >= now - window_s:
                out = out.max(sample)
        return out

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
    # Applied DeviceObserved samples; a job not observed for missing_samples_warn of
    # them is warned about (JobNotObserved).
    samples: int = 0

    @property
    def capacity(self) -> Resources:
        return self.device.capacity

    @property
    def foreign(self) -> Resources:
        """Max foreign usage over the samples kept in the window."""
        out = Resources()
        for _, sample in self.foreign_samples:
            out = out.max(sample)
        return out

    def foreign_at(self, now: float, window_s: float) -> Resources:
        """``F_d`` at ``now``: max foreign usage over samples with ``t >= now - window_s``."""
        out = Resources()
        for t, sample in self.foreign_samples:
            if t >= now - window_s:
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
    completed_groups: frozenset[GroupId] = frozenset()
    warned: frozenset[str] = frozenset()
    active_warnings: Mapping[str, WarningKind] = field(default_factory=frozen_map)
    pause_reason: QueueStopped | None = None
    removed_devices: frozenset[DeviceId] = frozenset()
    # Counter for JobRuntime.admit_seq and isolate_seq.
    next_order: int = 0

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


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What ``decide`` sees: the state plus estimator results resolved beforehand.

    ``estimates[job][device]`` is the estimator's answer (None = unknown);
    ``decide`` combines it with the declared hint and the learned estimate.
    """

    state: State
    estimates: Mapping[JobId, Mapping[DeviceId, Resources | None]] = field(
        default_factory=frozen_map
    )
    trusted: frozenset[tuple[JobId, DeviceId]] = frozenset()
    # Complete observations behind a trusted estimate (rule A, SPEC §5); missing = 0.
    observations: Mapping[tuple[JobId, DeviceId], int] = field(default_factory=frozen_map)

    def is_trusted(self, rt: JobRuntime, device_id: DeviceId) -> bool:
        """SPEC §5: the estimator answered from the most specific fingerprint level and the
        job has not OOMed. Hints, learned values and fallback levels are untrusted."""
        return (rt.id, device_id) in self.trusted and rt.oom_count == 0

    def estimate(self, rt: JobRuntime, device_id: DeviceId) -> Resources | None:
        """``E_j,d = max(estimator result, declared hint, learned)``, or None if all unknown."""
        known = [
            e
            for e in (self.estimates.get(rt.id, {}).get(device_id), rt.prev_estimate(device_id))
            if e is not None
        ]
        if not known:
            return None
        out = known[0]
        for e in known[1:]:
            out = out.max(e)
        return out
