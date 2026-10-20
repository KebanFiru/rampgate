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
        self, job: Job, device: Device, peak: Resources, *, lower_bound: bool = False
    ) -> None: ...


class HistoryStore(Protocol):
    def get(self, key: str, device_kind: str) -> Resources | None: ...

    def put(self, key: str, device_kind: str, peak: Resources, *, lower_bound: bool) -> None: ...


def default_device_kind(device: Device) -> str:
    return device.labels.get("kind", device.id)


class InMemoryHistoryStore:
    """Keeps the component-wise max of everything put per (key, device kind)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], Resources] = {}

    def get(self, key: str, device_kind: str) -> Resources | None:
        return self._data.get((key, device_kind))

    def put(self, key: str, device_kind: str, peak: Resources, *, lower_bound: bool) -> None:
        old = self._data.get((key, device_kind))
        self._data[(key, device_kind)] = peak if old is None else old.max(peak)

    def to_json(self) -> list[list[Any]]:
        """``[[key, device_kind, peak], ...]``, sorted, for ``json.dumps``."""
        return [[k, kind, peak.to_json()] for (k, kind), peak in sorted(self._data.items())]

    @classmethod
    def from_json(cls, data: Iterable[Any]) -> InMemoryHistoryStore:
        store = cls()
        for key, kind, peak in data:
            store._data[(str(key), str(kind))] = Resources.from_json(peak)
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
        self, job: Job, device: Device, peak: Resources, *, lower_bound: bool = False
    ) -> None:
        kind = self.device_kind(device)
        for key in job.fingerprint:
            self.store.put(key, kind, peak, lower_bound=lower_bound)


class NullEstimator:
    """Always unknown."""

    def estimate(self, job: Job, device: Device) -> Resources | None:
        return None

    def record(
        self, job: Job, device: Device, peak: Resources, *, lower_bound: bool = False
    ) -> None:
        return None


class StaticEstimator:
    """Fixed estimates by job id and device id (for tests)."""

    def __init__(self, mapping: Mapping[JobId, Mapping[DeviceId, Resources]]) -> None:
        self.mapping = {j: dict(m) for j, m in mapping.items()}

    def estimate(self, job: Job, device: Device) -> Resources | None:
        return self.mapping.get(job.id, {}).get(device.id)

    def record(
        self, job: Job, device: Device, peak: Resources, *, lower_bound: bool = False
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
            estimator.record(rt.job, ds.device, rec.peak, lower_bound=rec.lower_bound)
