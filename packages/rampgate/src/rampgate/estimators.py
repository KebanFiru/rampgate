"""Estimators and history stores (SPEC §7)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rampgate.config import Config
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
    "HistoryEntry",
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
    ``lower_bound_dims``: the dimensions whose value is a lower bound (no complete record,
    or a newer lower bound above it), where the job needs *more* than ``value``;
    `resolve_estimates` grows them like Rule O (SPEC §5).
    """

    value: Resources
    level: int = 0
    complete: bool = True
    # Complete observations behind the value (rule A, SPEC §5); 0 = unknown, the safe side.
    observations: int = 0
    lower_bound_dims: frozenset[str] = frozenset()

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


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """What a `HistoryStore` holds for one (key, device kind).

    - ``complete``: the per-dimension max of the complete records (``peak``,
      ``running``), None if there is none.
    - ``lower_bound``: the per-dimension max of the lower-bound records that are
      **newer** than the newest complete record (all of them if there is none),
      None if there is none. An older OOM never overrides a later observation.
    """

    complete: Resources | None
    lower_bound: Resources | None
    observation_count: int = 0

    @property
    def value(self) -> Resources:
        """``max(complete, lower_bound)``."""
        known = [r for r in (self.complete, self.lower_bound) if r is not None]
        out = known[0]
        for r in known[1:]:
            out = out.max(r)
        return out


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

    **Order.** The store numbers records by arrival: a put that changes a record's
    stored value (a new record, or a larger value for its id) gives it the next
    sequence number; a put that changes nothing leaves it as it was, so replaying
    a record never makes it newer. ``get`` uses that order to drop lower bounds
    older than the newest complete record (SPEC §5, §7).
    """

    def get(self, key: str, device_kind: str) -> HistoryEntry | None: ...

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


@dataclass(frozen=True, slots=True)
class _Stored:
    value: Resources
    lower_bound: bool
    seq: int


class InMemoryHistoryStore:
    """Keeps every record (max-merged by record id) with its arrival order.

    A put without a ``record_id`` is a record of its own. ``puts_applied`` counts
    the puts that changed something.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], dict[str, _Stored]] = {}
        self._seq = 0
        self.puts_applied = 0

    def get(self, key: str, device_kind: str) -> HistoryEntry | None:
        records = self._records.get((key, device_kind))
        if not records:
            return None
        complete = [r for r in records.values() if not r.lower_bound]
        newest = max((r.seq for r in complete), default=-1)
        lower = [r for r in records.values() if r.lower_bound and r.seq > newest]
        observations = {_observation(rid) for rid, r in records.items() if not r.lower_bound}
        return HistoryEntry(
            _max(r.value for r in complete), _max(r.value for r in lower), len(observations)
        )

    def put(
        self,
        key: str,
        device_kind: str,
        peak: Resources,
        *,
        lower_bound: bool,
        record_id: str | None = None,
    ) -> None:
        records = self._records.setdefault((key, device_kind), {})
        rid = record_id if record_id is not None else f"\x00anonymous:{self._seq}"
        seen = records.get(rid)
        merged = peak if seen is None else seen.value.max(peak)
        if seen is not None and merged == seen.value:
            return
        self._seq += 1
        records[rid] = _Stored(merged, lower_bound, self._seq)
        self.puts_applied += 1

    def to_json(self) -> dict[str, Any]:
        """Records (key, kind, id, lower bound, sequence, value), sorted, for ``json.dumps``."""
        return {
            "seq": self._seq,
            "records": [
                [key, kind, rid, r.lower_bound, r.seq, r.value.to_json()]
                for (key, kind), records in sorted(self._records.items())
                for rid, r in sorted(records.items())
            ],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> InMemoryHistoryStore:
        store = cls()
        if "records" not in data:
            return cls._from_legacy(data)
        store._seq = int(data["seq"])
        for key, kind, rid, lower, seq, value in data["records"]:
            records = store._records.setdefault((str(key), str(kind)), {})
            records[str(rid)] = _Stored(Resources.from_json(value), bool(lower), int(seq))
        return store

    @classmethod
    def _from_legacy(cls, data: Mapping[str, Any]) -> InMemoryHistoryStore:
        """Before the arrival order, all at sequence 0. Applied records keep their id
        (its last part is the kind, so ``lower_bound``); each merged peak is kept as a
        record too, a lower bound only if every applied record of its key is one."""
        store = cls()
        for k, kind, rid, *peak in data["applied"]:
            # Before max-merge (L3), only the ids were kept.
            value = Resources.from_json(peak[0]) if peak else Resources()
            lower = str(rid).endswith(":lower_bound")
            store._records.setdefault((str(k), str(kind)), {})[str(rid)] = _Stored(value, lower, 0)
        for key, kind, peak in data["peaks"]:
            records = store._records.setdefault((str(key), str(kind)), {})
            lower = bool(records) and all(r.lower_bound for r in records.values())
            records["\x00legacy"] = _Stored(Resources.from_json(peak), lower, 0)
        return store


def _observation(record_id: str) -> str:
    """``job:attempt:device:kind`` -> ``job:attempt:device``: a job's running and later peak
    records are one observation (rule A). Ids without a known kind count as themselves."""
    head, _, kind = record_id.rpartition(":")
    return head if kind in ("peak", "running", "lower_bound") and head else record_id


def _max(values: Iterable[Resources]) -> Resources | None:
    out: Resources | None = None
    for v in values:
        out = v if out is None else out.max(v)
    return out


class HistoryEstimator:
    """Learns peaks per fingerprint key (Rule H).

    ``record`` writes to every key of ``job.fingerprint``; ``estimate`` returns,
    for the most specific key that has data, ``max(complete, newer lower bound)``.
    The result is complete only if a complete record gives the value: a lower
    bound alone, or a newer lower bound above the complete value, says the job
    needs *more than* that, so the estimate is untrusted (SPEC §5).
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
            entry = self.store.get(key, kind)
            if entry is None:
                continue
            above: list[str] = []
            if entry.lower_bound is not None:
                above = (
                    list(entry.lower_bound)
                    if entry.complete is None
                    else entry.lower_bound.exceeds(entry.complete)
                )
            return EstimateResult(
                entry.value,
                level,
                complete=entry.complete is not None and not above,
                observations=entry.observation_count,
                lower_bound_dims=frozenset(above),
            )
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
            self.store.put(key, kind, peak, lower_bound=lower_bound, record_id=record_id)


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
    """Fixed estimates by job id and device id (for tests), all at fingerprint ``level``
    and backed by ``observations`` complete observations."""

    def __init__(
        self,
        mapping: Mapping[JobId, Mapping[DeviceId, Resources]],
        level: int = 0,
        observations: int = 0,
    ) -> None:
        self.mapping = {j: dict(m) for j, m in mapping.items()}
        self.level = level
        self.observations = observations

    def estimate(self, job: Job, device: Device) -> EstimateResult | None:
        value = self.mapping.get(job.id, {}).get(device.id)
        if value is None:
            return None
        return EstimateResult(value, self.level, observations=self.observations)

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


def resolve_estimates(state: State, estimator: Estimator, config: Config | None = None) -> Snapshot:
    """Ask the estimator about every unfinished job, so ``decide`` can stay pure.

    In the dimensions where the answer is a lower bound, it becomes
    ``min(value * oom_growth, capacity)`` (never below the value), the step Rule O gives
    the OOM'd job itself (SPEC §5).
    """
    growth = (config or Config()).oom_growth
    estimates: dict[JobId, Mapping[DeviceId, Resources | None]] = {}
    trusted: set[tuple[JobId, DeviceId]] = set()
    observations: dict[tuple[JobId, DeviceId], int] = {}
    for rt in sorted(state.jobs.values(), key=lambda rt: rt.submit_seq):
        if rt.state in TERMINAL_STATES:
            continue
        per_device: dict[DeviceId, Resources | None] = {}
        for d in rt.job.devices:
            result = estimator.estimate(rt.job, state.devices[d].device)
            if isinstance(result, EstimateResult):
                per_device[d] = result.value
                if result.lower_bound_dims:
                    per_device[d] = _grow(
                        result.value, result.lower_bound_dims, state.devices[d].capacity, growth
                    )
                if result.trusted:
                    trusted.add((rt.id, d))
                    observations[(rt.id, d)] = result.observations
            else:
                per_device[d] = result
        estimates[rt.id] = frozen_map(per_device)
    return Snapshot(state, frozen_map(estimates), frozenset(trusted), frozen_map(observations))


def _grow(value: Resources, dims: frozenset[str], capacity: Resources, growth: float) -> Resources:
    """Rule O's step for the lower-bound ``dims`` of ``value``: ``v * growth``, capped at
    capacity where it is positive, never below ``v``. Other dimensions are kept."""

    def grown(dim: str, v: float) -> float:
        if dim not in dims:
            return v
        return max(v, min(v * growth, capacity[dim]) if capacity[dim] > 0 else v * growth)

    return Resources({dim: grown(dim, v) for dim, v in value.items()})


def apply_history(estimator: Estimator, state: State, records: Iterable[HistoryRecord]) -> None:
    """Hand drained history records to the estimator (the only I/O step)."""
    for rec in records:
        rt = state.jobs.get(rec.job_id)
        ds = state.devices.get(rec.device_id)
        if rt is not None and ds is not None:
            estimator.record(
                rt.job, ds.device, rec.peak, lower_bound=rec.lower_bound, record_id=rec.record_id
            )
