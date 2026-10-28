"""Against a real Docker daemon (`./dev integration`); skipped without one."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from rampgate import ExitKind, JobExited, JobStarted, Start, Stop
from rampgate.model import Job, StopReason
from rampgate_docker import ContainerSpec, DockerEventWatcher, DockerExecutor

IMAGE = "python:3.12-slim-bookworm"


def _client() -> Any:
    try:
        import docker  # noqa: PLC0415

        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception:
        return None
    return client


CLIENT = _client()
pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(CLIENT is None, reason=f"no Docker daemon, or no {IMAGE} image"),
]


@pytest.fixture
def cleanup() -> Iterator[None]:
    yield
    for c in CLIENT.containers.list(all=True, filters={"label": "rampgate.test=1"}):
        c.remove(force=True)


def _start(job_id: str) -> Start:
    return Start(job_id, 1, ("gpu0",), False, {}, 0.0, {"gpu0": None}, frozenset())


def _wait(
    watcher: DockerEventWatcher, seen: list[Any], done: Callable[[list[Any]], bool]
) -> list[Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        seen.extend(watcher.drain())
        if done(seen):
            return seen
        time.sleep(0.1)
    raise AssertionError(f"timed out; saw {seen}")


@pytest.mark.usefixtures("cleanup")
def test_oom_message_success_and_preempt() -> None:
    scripts = {
        "oom": "echo 'RuntimeError: CUDA out of memory.'; exit 1",
        "ok": "echo done",
        # A process that saves and exits 143 on SIGTERM, like the memory hog.
        "pre": "trap 'echo saved; exit 143' TERM; while true; do sleep 0.1; done",
    }

    def spec(job: Job, start: Start) -> ContainerSpec:
        return ContainerSpec(IMAGE, ["sh", "-c", scripts[job.id]], labels={"rampgate.test": "1"})

    executor = DockerExecutor(spec, CLIENT)
    t0 = time.monotonic()
    watcher = DockerEventWatcher(
        CLIENT, clock=lambda: time.monotonic() - t0, stop_reason=executor.stop_reason
    )
    watcher.start()
    try:
        jobs = {name: Job(name, {"gpu0": None}) for name in scripts}
        executor.apply([_start(name) for name in scripts], jobs)
        seen: list[Any] = []
        _wait(watcher, seen, lambda s: sum(isinstance(e, JobStarted) for e in s) == 3)
        executor.apply([Stop("pre", 1, StopReason.PREEMPT, 10)], jobs)
        exits = _wait(watcher, seen, lambda s: sum(isinstance(e, JobExited) for e in s) == 3)
        kinds = {e.job_id: e.kind for e in exits if isinstance(e, JobExited)}
        assert kinds == {"oom": ExitKind.OOM, "ok": ExitKind.SUCCESS, "pre": ExitKind.PREEMPTED}
        executor.wait_stops(timeout=30)
    finally:
        watcher.stop()
        executor.close()
