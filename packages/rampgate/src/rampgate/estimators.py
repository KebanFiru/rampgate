"""Estimators and history stores (SPEC §7)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rampgate.model import (
    TERMINAL_STATES,
    Device,
    DeviceId,
    HistoryRecord,
    Job,
    JobId,
    Snapshot,
    State,
    frozen_map,
)
from rampgate.resources import Resources

__all__ = [
    "EstimateResult",
    "Estimator",
    "HistoryEstimator",
    "HistoryStore",
    "InMemoryHistoryStore",
    "NullEstimator",
    "StaticEstimator",
    "apply_history",
    "default_device_kind",
    "resolve_estimates",
]


@dataclass(frozen=True, slots=True)
class EstimateResult:
    """An estimate and the index of the fingerprint key that produced it.

    ``level == 0`` is the most specific key (the exact template and config).
    ``complete``: the value includes a full observation of a ramp (a ``peak`` or
    ``running`` record), not only lower bounds from OOMs or stops. Only a complete
    level-0 estimate is *trusted* and lets a job skip probation (SPEC §5).
    """

    value: Resources
    level: int = 0
    complete: bool = True

    @property
    def trusted(self) -> bool:
        return self.level == 0 and self.complete


class Estimator(Protocol):
    """``estimate`` returns an `EstimateResult` (a bare `Resources` counts as untrusted)."""

    def estimate(self, job: Job, device: Device) -> EstimateResult | Resources | None: ...

    def record(
        self,
        job: Job,
        device: Device,
        peak: Resources,
        *,
        lower_bound: bool = False,
        record_id: str | None = None,
    ) -> None: ...


class HistoryStore(Protocol):
    """Durable peak history. ``put`` max-merges by ``(key, device_kind, record_id)``.

    A record's stored value is the per-dimension max of every put with its id:
    idempotent and order-independent. The Engine hands a history record to the
    estimator before the next ``dump``; after a crash in between, the restored
    engine hands it over again. A running record (Rule H) is also written again
    with a larger peak when the job's reservation grows.

    ``lower_bound=True`` means the value is a lower bound from a failure or a stop
    (an OOM, or a job stopped before it finished): the job needed *more than* this,
    not this. Complete observations have ``lower_bound=False``: ``peak`` records
    (SUCCESS) and ``running`` records (STABLE, Rule H; Rule S saw the ramp complete).
    `HistoryEstimator` puts lower bounds under ``key + LOWER_BOUND_SUFFIX``, apart
    from complete observations, and only a complete level-0 estimate is trusted
    (SPEC §5, §7).
    """

    def get(self, key: str, device_kind: str) -> Resources | None: ...

    def put(
        self,
        key: str,
        device_kind: str,
        peak: Resources,
        *,
        lower_bound: bool,
        record_id: str | None = None,
    ) -> None: ...


def default_device_kind(device: Device) -> str:
    return device.labels.get("kind", device.id)


class InMemoryHistoryStore:
    """Keeps the component-wise max of everything put per (key, device kind).

    Records with a ``record_id`` are max-merged per (key, device kind, record id);
    ``puts_applied`` counts the puts that changed something.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], Resources] = {}
        self._applied: dict[tuple[str, str, str], Resources] = {}
        self.puts_applied = 0

    def get(self, key: str, device_kind: str) -> Resources | None:
        return self._data.get((key, device_kind))

    def put(
        self,
        key: str,
        device_kind: str,
        peak: Resources,
        *,
        lower_bound: bool,
        record_id: str | None = None,
    ) -> None:
        if record_id is not None:
            seen = self._applied.get((key, device_kind, record_id))
            merged = peak if seen is None else seen.max(peak)
            if merged == seen:
                return
            self._applied[(key, device_kind, record_id)] = merged
        old = self._data.get((key, device_kind))
        self._data[(key, device_kind)] = peak if old is None else old.max(peak)
        self.puts_applied += 1

    def to_json(self) -> dict[str, Any]:
        """Peaks and applied records (id and merged value), sorted, for ``json.dumps``."""
        return {
            "peaks": [[k, kind, peak.to_json()] for (k, kind), peak in sorted(self._data.items())],
            "applied": [[*a, peak.to_json()] for a, peak in sorted(self._applied.items())],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> InMemoryHistoryStore:
        store = cls()
        for key, kind, peak in data["peaks"]:
            store._data[(str(key), str(kind))] = Resources.from_json(peak)
        for k, kind, r, *peak in data["applied"]:
            # Before max-merge (L3), only the ids were kept.
            value = Resources.from_json(peak[0]) if peak else Resources()
            store._applied[(str(k), str(kind), str(r))] = value
        return store


# Lower bounds (OOM or stop) are stored under ``key + LOWER_BOUND_SUFFIX``, apart from
# complete observations, so an estimate knows whether it is complete (SPEC §5, §7).
LOWER_BOUND_SUFFIX = "\x1flower_bound"


class HistoryEstimator:
    """Learns peaks per fingerprint key (Rule H).

    ``record`` writes to every key of ``job.fingerprint`` (lower bounds to the
    key plus `LOWER_BOUND_SUFFIX`); ``estimate`` returns, for the most specific
    key that has data, the max of its complete and lower-bound values, complete
    only if it has a complete record.
    """

    def __init__(
        self,
        store: HistoryStore,
        aggregate: Literal["max"] = "max",
        device_kind: Callable[[Device], str] = default_device_kind,
    ) -> None:
        if aggregate != "max":
            raise ValueError(f"unsupported aggregate {aggregate!r}; only 'max' exists for now")
        self.store = store
        self.aggregate = aggregate
        self.device_kind = device_kind

    def estimate(self, job: Job, device: Device) -> EstimateResult | None:
        kind = self.device_kind(device)
        for level, key in enumerate(job.fingerprint):
            full = self.store.get(key, kind)
            lower = self.store.get(key + LOWER_BOUND_SUFFIX, kind)
            if full is not None:
                return EstimateResult(full if lower is None else full.max(lower), level)
            if lower is not None:
                return EstimateResult(lower, level, complete=False)
        return None

    def record(
        self,
        job: Job,
        device: Device,
        peak: Resources,
        *,
        lower_bound: bool = False,
        record_id: str | None = None,
    ) -> None:
        kind = self.device_kind(device)
        for key in job.fingerprint:
            target = key + LOWER_BOUND_SUFFIX if lower_bound else key
            self.store.put(target, kind, peak, lower_bound=lower_bound, record_id=record_id)


class NullEstimator:
    """Always unknown."""

    def estimate(self, job: Job, device: Device) -> EstimateResult | None:
        return None

    def record(
        self,
        job: Job,
        device: Device,
        peak: Resources,
        *,
        lower_bound: bool = False,
        record_id: str | None = None,
    ) -> None:
        return None


class StaticEstimator:
    """Fixed estimates by job id and device id (for tests), all at fingerprint ``level``."""

    def __init__(
        self, mapping: Mapping[JobId, Mapping[DeviceId, Resources]], level: int = 0
    ) -> None:
        self.mapping = {j: dict(m) for j, m in mapping.items()}
        self.level = level

    def estimate(self, job: Job, device: Device) -> EstimateResult | None:
        value = self.mapping.get(job.id, {}).get(device.id)
        return None if value is None else EstimateResult(value, self.level)

    def record(
        self,
        job: Job,
        device: Device,
        peak: Resources,
        *,
        lower_bound: bool = False,
        record_id: str | None = None,
    ) -> None:
        return None


def resolve_estimates(state: State, estimator: Estimator) -> Snapshot:
    """Ask the estimator about every unfinished job, so ``decide`` can stay pure."""
    estimates: dict[JobId, Mapping[DeviceId, Resources | None]] = {}
    trusted: set[tuple[JobId, DeviceId]] = set()
    for rt in sorted(state.jobs.values(), key=lambda rt: rt.submit_seq):
        if rt.state in TERMINAL_STATES:
            continue
        per_device: dict[DeviceId, Resources | None] = {}
        for d in rt.job.devices:
            result = estimator.estimate(rt.job, state.devices[d].device)
            if isinstance(result, EstimateResult):
                per_device[d] = result.value
                if result.trusted:
                    trusted.add((rt.id, d))
            else:
                per_device[d] = result
        estimates[rt.id] = frozen_map(per_device)
    return Snapshot(state, frozen_map(estimates), frozenset(trusted))


def apply_history(estimator: Estimator, state: State, records: Iterable[HistoryRecord]) -> None:
    """Hand drained history records to the estimator (the only I/O step)."""
    for rec in records:
        rt = state.jobs.get(rec.job_id)
        ds = state.devices.get(rec.device_id)
        if rt is not None and ds is not None:
            estimator.record(
                rt.job, ds.device, rec.peak, lower_bound=rec.lower_bound, record_id=rec.record_id
            )
