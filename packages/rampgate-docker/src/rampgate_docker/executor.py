"""`DockerExecutor`: rampgate's `Start`/`Stop` actions as Docker containers (SPEC §9)."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from rampgate import Action, Start, Stop
from rampgate.model import Job, JobId, StopReason

__all__ = [
    "ATTEMPT_LABEL",
    "JOB_LABEL",
    "ContainerLike",
    "ContainerSpec",
    "DockerClientLike",
    "DockerExecutor",
]

JOB_LABEL = "rampgate.job_id"
ATTEMPT_LABEL = "rampgate.attempt"


class ContainerLike(Protocol):
    """The part of ``docker.models.containers.Container`` rampgate uses."""

    @property
    def id(self) -> str | None: ...
    @property
    def labels(self) -> dict[str, str]: ...
    @property
    def attrs(self) -> dict[str, Any]: ...
    def start(self) -> None: ...
    def stop(self, *, timeout: float | None = None) -> None: ...
    def remove(self, *, v: bool = False, link: bool = False, force: bool = False) -> None: ...
    def logs(self, *, tail: int | Literal["all"] = "all") -> bytes: ...


class ContainerCollectionLike(Protocol):
    def create(self, image: str, command: str | list[str] | None = None, **kwargs: Any) -> Any: ...
    def get(self, container_id: str) -> Any: ...
    def list(self, *, all: bool = False, filters: dict[str, Any] | None = None) -> Any: ...


class DockerClientLike(Protocol):
    """The part of ``docker.DockerClient`` rampgate uses."""

    @property
    def containers(self) -> ContainerCollectionLike: ...
    def events(self, *, decode: Literal[True], filters: dict[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    """How to run one attempt of a job.

    ``gpus``: ``"all"`` or GPU indices/UUIDs, passed as a Docker device request
    (needs the NVIDIA container toolkit). ``labels`` are added to rampgate's
    own (which win). ``extra`` goes to ``containers.create`` as is
    (``volumes``, ``mem_limit``, ``security_opt``, ...), except ``labels`` and
    ``environment``, which have their own fields.
    """

    image: str
    command: str | Sequence[str] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    gpus: Literal["all"] | Sequence[str] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


# (event name, fields): diagnostics such as when a SIGTERM was actually sent.
ExecutorLog = Callable[[str, Mapping[str, Any]], None]


class DockerExecutor:
    """Applies `Start` and `Stop` actions; every other action is not the executor's.

    - ``Start``: first stops and removes every container of an **older attempt**
      of the same job (found by its ``rampgate.job_id`` label: it may survive a
      restart or a start timeout), then creates and starts a container labelled
      ``rampgate.job_id`` / ``rampgate.attempt``. The container's environment
      gets ``RAMPGATE_JOB_ID``, ``RAMPGATE_ATTEMPT`` and ``RAMPGATE_RESUME``
      (``1`` or ``0``).
    - ``Stop``: ``container.stop(timeout=grace_s)`` (SIGTERM, then SIGKILL after
      the grace period). It blocks until the container exits, so it runs on a
      worker thread and `apply` returns at once.

    `stop_reason` tells the exit classification which stop rampgate asked for.
    Calls to `apply` must come from one thread (the Engine's).
    """

    def __init__(
        self,
        spec_for: Callable[[Job, Start], ContainerSpec],
        client: DockerClientLike | None = None,
        *,
        log: ExecutorLog | None = None,
        clock: Callable[[], float] | None = None,
        stop_workers: int = 4,
    ) -> None:
        if client is None:
            import docker  # noqa: PLC0415 - only when no client is injected

            client = cast(DockerClientLike, docker.from_env())
        self.client: DockerClientLike = client
        self._spec_for = spec_for
        self._log = log
        self._clock = clock
        self._pool = ThreadPoolExecutor(stop_workers, thread_name_prefix="rampgate-stop")
        self._lock = threading.Lock()
        self._stops: dict[tuple[JobId, int], StopReason] = {}
        self._containers: dict[tuple[JobId, int], str] = {}
        self._pending: list[Future[None]] = []

    def apply(self, actions: Iterable[Action], jobs: Mapping[JobId, Job]) -> None:
        for action in actions:
            if isinstance(action, Start):
                self._start(jobs[action.job_id], action)
            elif isinstance(action, Stop):
                self._stop(action)

    def stop_reason(self, job_id: JobId, attempt: int) -> StopReason | None:
        """The stop rampgate requested for this attempt, if any."""
        with self._lock:
            return self._stops.get((job_id, attempt))

    def container_id(self, job_id: JobId, attempt: int) -> str | None:
        with self._lock:
            return self._containers.get((job_id, attempt))

    def wait_stops(self, timeout: float | None = None) -> None:
        """Wait for the stops in flight (tests, shutdown)."""
        with self._lock:
            pending, self._pending = self._pending, []
        for future in pending:
            future.result(timeout=timeout)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def _emit(self, name: str, **fields: Any) -> None:
        if self._log is not None:
            if self._clock is not None:
                fields["t"] = self._clock()
            self._log(name, fields)

    def _start(self, job: Job, start: Start) -> None:
        for old in self.client.containers.list(
            all=True, filters={"label": f"{JOB_LABEL}={job.id}"}
        ):
            if old.labels.get(ATTEMPT_LABEL) != str(start.attempt):
                old.remove(force=True)
                self._emit("removed_old_attempt", job_id=job.id, container=old.id)
        spec = self._spec_for(job, start)
        env = {
            **spec.environment,
            "RAMPGATE_JOB_ID": job.id,
            "RAMPGATE_ATTEMPT": str(start.attempt),
            "RAMPGATE_RESUME": "1" if start.resume else "0",
        }
        if {"labels", "environment"} & set(spec.extra):
            raise ValueError("ContainerSpec.extra must not set labels or environment")
        kwargs: dict[str, Any] = {
            "detach": True,
            **spec.extra,
            "labels": {**spec.labels, JOB_LABEL: job.id, ATTEMPT_LABEL: str(start.attempt)},
            "environment": env,
        }
        if spec.gpus is not None:
            kwargs["device_requests"] = [_gpu_request(spec.gpus)]
        command = (
            spec.command
            if spec.command is None or isinstance(spec.command, str)
            else list(spec.command)
        )
        container = self.client.containers.create(spec.image, command, **kwargs)
        with self._lock:
            self._containers[(job.id, start.attempt)] = container.id
        container.start()
        self._emit(
            "container_started", job_id=job.id, attempt=start.attempt, container=container.id
        )

    def _stop(self, stop: Stop) -> None:
        key = (stop.job_id, stop.attempt)
        with self._lock:
            self._stops[key] = stop.reason
            container_id = self._containers.get(key)
        timeout = math.ceil(stop.grace_s)

        def run() -> None:
            # A container this executor did not start (e.g. before a restart) is found by label.
            found = (
                [self.client.containers.get(container_id)]
                if container_id is not None
                else self._find(stop.job_id, stop.attempt)
            )
            if not found:
                self._emit("stop_no_container", job_id=stop.job_id, attempt=stop.attempt)
                return
            container = found[0]
            self._emit("stop_sent", job_id=stop.job_id, attempt=stop.attempt, grace_s=timeout)
            container.stop(timeout=timeout)
            self._emit("stop_returned", job_id=stop.job_id, attempt=stop.attempt)

        future = self._pool.submit(run)
        with self._lock:
            self._pending.append(future)

    def _find(self, job_id: JobId, attempt: int) -> list[Any]:
        labels = [f"{JOB_LABEL}={job_id}", f"{ATTEMPT_LABEL}={attempt}"]
        return list(self.client.containers.list(all=True, filters={"label": labels}))


def _gpu_request(gpus: Literal["all"] | Sequence[str]) -> Any:
    from docker.types import DeviceRequest  # noqa: PLC0415

    if gpus == "all":
        return DeviceRequest(count=-1, capabilities=[["gpu"]])
    return DeviceRequest(device_ids=list(gpus), capabilities=[["gpu"]])
