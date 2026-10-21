"""Estimators and history stores (SPEC §7)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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


class Estimator(Protocol):
    def estimate(self, job: Job, device: Device) -> Resources | None: ...

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


class HistoryEstimator:
    """Learns peaks per fingerprint key (Rule H).

    ``record`` writes to every key of ``job.fingerprint``; ``estimate`` returns
    the value of the most specific key that has data.
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

    def estimate(self, job: Job, device: Device) -> Resources | None:
        kind = self.device_kind(device)
        for key in job.fingerprint:
            value = self.store.get(key, kind)
            if value is not None:
                return value
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

    def estimate(self, job: Job, device: Device) -> Resources | None:
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
    """Fixed estimates by job id and device id (for tests)."""

    def __init__(self, mapping: Mapping[JobId, Mapping[DeviceId, Resources]]) -> None:
        self.mapping = {j: dict(m) for j, m in mapping.items()}

    def estimate(self, job: Job, device: Device) -> Resources | None:
        return self.mapping.get(job.id, {}).get(device.id)

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
    estimates = {
        rt.id: frozen_map(
            {d: estimator.estimate(rt.job, state.devices[d].device) for d in rt.job.devices}
        )
        for rt in sorted(state.jobs.values(), key=lambda rt: rt.submit_seq)
        if rt.state not in TERMINAL_STATES
    }
    return Snapshot(state, frozen_map(estimates))


def apply_history(estimator: Estimator, state: State, records: Iterable[HistoryRecord]) -> None:
    """Hand drained history records to the estimator (the only I/O step)."""
    for rec in records:
        rt = state.jobs.get(rec.job_id)
        ds = state.devices.get(rec.device_id)
        if rt is not None and ds is not None:
            estimator.record(
                rt.job, ds.device, rec.peak, lower_bound=rec.lower_bound, record_id=rec.record_id
            )
