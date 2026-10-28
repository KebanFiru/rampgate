"""`ContainerUsageMapper`: probe process samples as `JobObserved` events of managed jobs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from rampgate import JobObserved, Resources
from rampgate_docker.executor import ATTEMPT_LABEL, JOB_LABEL, DockerClientLike

__all__ = ["ContainerUsageMapper", "ProcessSample"]


class ProcessSample(Protocol):
    """A process's usage on a device, e.g. ``rampgate_nvml.ProcessUsage``."""

    @property
    def pid(self) -> int: ...
    @property
    def device_id(self) -> str: ...
    @property
    def usage(self) -> Resources: ...


class ContainerUsageMapper:
    """Sums process usage per (job, attempt, device) of rampgate-labelled containers.

    ``container_of_pid`` maps a host pid to its container id (e.g.
    ``rampgate_nvml.pid_to_container_id``). Processes outside a container, or
    in a container without rampgate's labels, are left out: rampgate counts
    what no managed job explains as foreign usage. Labels are looked up once
    per container id.
    """

    def __init__(
        self,
        client: DockerClientLike,
        container_of_pid: Callable[[int], str | None],
    ) -> None:
        self.client = client
        self._container_of_pid = container_of_pid
        self._labels: dict[str, tuple[str, int] | None] = {}

    def map(self, samples: Iterable[ProcessSample], t: float) -> list[JobObserved]:
        totals: dict[tuple[str, int, str], Resources] = {}
        for sample in samples:
            container_id = self._container_of_pid(sample.pid)
            if container_id is None:
                continue
            owner = self._owner(container_id)
            if owner is None:
                continue
            key = (*owner, sample.device_id)
            totals[key] = totals.get(key, Resources()) + sample.usage
        return [
            JobObserved(job_id, device_id, t, usage, attempt=attempt)
            for (job_id, attempt, device_id), usage in sorted(totals.items())
        ]

    def _owner(self, container_id: str) -> tuple[str, int] | None:
        if container_id not in self._labels:
            try:
                labels = self.client.containers.get(container_id).labels
            except Exception:  # gone, or not visible: not a job we manage now
                return None
            job_id = labels.get(JOB_LABEL)
            attempt = labels.get(ATTEMPT_LABEL)
            self._labels[container_id] = (
                (job_id, int(attempt))
                if job_id is not None and attempt is not None and attempt.isdigit()
                else None
            )
        return self._labels[container_id]
