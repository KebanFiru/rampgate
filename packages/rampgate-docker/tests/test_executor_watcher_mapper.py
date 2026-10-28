"""`DockerExecutor`, `DockerEventWatcher` and `ContainerUsageMapper` on a fake client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from fakes import FakeClient, FakeContainer, docker_event
from rampgate import ExitKind, JobExited, JobObserved, JobStarted, Resources, Start, Stop
from rampgate.model import Job, StopReason
from rampgate_docker import (
    ATTEMPT_LABEL,
    JOB_LABEL,
    ContainerSpec,
    ContainerUsageMapper,
    DockerEventWatcher,
    DockerExecutor,
)

JOB = Job("j1", {"gpu0": None})


def _start(attempt: int = 1, resume: bool = False) -> Start:
    return Start(
        "j1",
        attempt,
        ("gpu0",),
        resume,
        admitted_free={},
        issued_at=0.0,
        estimate={"gpu0": None},
        trusted=frozenset(),
    )


def _spec(job: Job, start: Start) -> ContainerSpec:
    return ContainerSpec(
        "hog:1",
        ["python", "hog.py", job.id],
        environment={"PEAK": "1000"},
        gpus="all",
        extra={"security_opt": ["label=disable"]},
    )


def _executor(client: FakeClient, log: list[tuple[str, Any]] | None = None) -> DockerExecutor:
    sink = None if log is None else (lambda name, fields: log.append((name, dict(fields))))
    return DockerExecutor(_spec, client, log=sink, clock=lambda: 7.0)


def test_start_creates_labelled_container_with_env_and_gpus() -> None:
    client = FakeClient()
    executor = _executor(client)
    executor.apply([_start(attempt=2, resume=True)], {"j1": JOB})
    (container,) = client.containers.all
    assert container.started
    assert container.image == "hog:1"
    assert container.command == ["python", "hog.py", "j1"]
    assert container.labels == {JOB_LABEL: "j1", ATTEMPT_LABEL: "2"}
    env = container.kwargs["environment"]
    assert env == {
        "PEAK": "1000",
        "RAMPGATE_JOB_ID": "j1",
        "RAMPGATE_ATTEMPT": "2",
        "RAMPGATE_RESUME": "1",
    }
    assert container.kwargs["security_opt"] == ["label=disable"]
    (request,) = container.kwargs["device_requests"]
    assert request["Count"] == -1
    assert request["Capabilities"] == [["gpu"]]
    assert executor.container_id("j1", 2) == container.id


def test_spec_labels_merge_and_reserved_extra_keys() -> None:
    client = FakeClient()

    def labelled(job: Job, start: Start) -> ContainerSpec:
        return ContainerSpec("hog:1", labels={"team": "x", JOB_LABEL: "spoofed"})

    DockerExecutor(labelled, client).apply([_start()], {"j1": JOB})
    assert client.containers.all[0].labels == {"team": "x", JOB_LABEL: "j1", ATTEMPT_LABEL: "1"}

    def bad(job: Job, start: Start) -> ContainerSpec:
        return ContainerSpec("hog:1", extra={"labels": {}})

    with pytest.raises(ValueError, match="labels or environment"):
        DockerExecutor(bad, FakeClient()).apply([_start()], {"j1": JOB})


def test_specific_gpus_and_no_gpus() -> None:
    client = FakeClient()

    def by_index(job: Job, start: Start) -> ContainerSpec:
        return ContainerSpec("hog:1", "run", gpus=["0"])

    DockerExecutor(by_index, client).apply([_start()], {"j1": JOB})
    assert client.containers.all[0].kwargs["device_requests"][0]["DeviceIDs"] == ["0"]
    assert client.containers.all[0].command == "run"

    client = FakeClient()
    DockerExecutor(lambda j, s: ContainerSpec("hog:1"), client).apply([_start()], {"j1": JOB})
    assert "device_requests" not in client.containers.all[0].kwargs


def test_start_removes_containers_of_older_attempts_only() -> None:
    client = FakeClient()
    log: list[tuple[str, Any]] = []
    executor = _executor(client, log)
    executor.apply([_start(attempt=1)], {"j1": JOB})
    other = client.containers.create("x", labels={JOB_LABEL: "other", ATTEMPT_LABEL: "1"})
    executor.apply([_start(attempt=2)], {"j1": JOB})
    first, _, second = client.containers.all
    assert first.removed
    assert not other.removed
    assert not second.removed
    assert ("removed_old_attempt", {"job_id": "j1", "container": first.id, "t": 7.0}) in log


def test_stop_runs_in_the_background_with_the_grace_period() -> None:
    client = FakeClient()
    log: list[tuple[str, Any]] = []
    executor = _executor(client, log)
    executor.apply([_start()], {"j1": JOB})
    executor.apply([Stop("j1", 1, StopReason.PREEMPT, 9.5, vacate=True)], {"j1": JOB})
    executor.wait_stops(timeout=5)
    assert client.containers.all[0].stop_timeouts == [10]  # rounded up to whole seconds
    assert executor.stop_reason("j1", 1) is StopReason.PREEMPT
    assert executor.stop_reason("j1", 2) is None
    names = [name for name, _ in log]
    assert names == ["container_started", "stop_sent", "stop_returned"]
    executor.close()


def test_stop_finds_a_container_by_label_after_a_restart() -> None:
    client = FakeClient()
    survivor = client.containers.create("hog:1", labels={JOB_LABEL: "j1", ATTEMPT_LABEL: "3"})
    executor = _executor(client)
    executor.apply([Stop("j1", 3, StopReason.CANCEL, 1.0)], {"j1": JOB})
    executor.wait_stops(timeout=5)
    assert survivor.stop_timeouts == [1]


def test_stop_without_a_container_is_logged() -> None:
    client = FakeClient()
    log: list[tuple[str, Any]] = []
    executor = _executor(client, log)
    executor.apply([Stop("j1", 1, StopReason.CANCEL, 1.0)], {"j1": JOB})
    executor.wait_stops(timeout=5)
    assert log == [("stop_no_container", {"job_id": "j1", "attempt": 1, "t": 7.0})]


def _labelled(client: FakeClient, job: str = "j1", attempt: str = "1") -> FakeContainer:
    return client.containers.create("hog:1", labels={JOB_LABEL: job, ATTEMPT_LABEL: attempt})


def test_watcher_turns_start_and_die_into_engine_events() -> None:
    client = FakeClient()
    container = _labelled(client)
    container.state = {"ExitCode": 1, "OOMKilled": False}
    container.log = b"step 10\nRuntimeError: CUDA out of memory\n"
    exits: list[dict[str, Any]] = []
    watcher = DockerEventWatcher(client, clock=lambda: 4.0, on_exit=lambda r: exits.append(dict(r)))
    assert watcher.handle(docker_event("start", container)) == JobStarted("j1", 1, 4.0)
    assert watcher.handle(docker_event("die", container, exitCode="1")) == JobExited(
        "j1", 1, ExitKind.OOM, 4.0
    )
    assert watcher.drain() == [JobStarted("j1", 1, 4.0), JobExited("j1", 1, ExitKind.OOM, 4.0)]
    assert watcher.drain() == []
    (record,) = exits
    assert record["kind"] == "OOM"
    assert record["exit_code"] == 1
    assert "CUDA out of memory" in record["log_tail"]


def test_watcher_uses_the_requested_stop_and_memory_pressure() -> None:
    client = FakeClient()
    container = _labelled(client)
    container.state = {"ExitCode": 143}
    watcher = DockerEventWatcher(
        client, clock=lambda: 1.0, stop_reason=lambda job, attempt: StopReason.PREEMPT
    )
    event = watcher.handle(docker_event("die", container))
    assert isinstance(event, JobExited)
    assert event.kind is ExitKind.PREEMPTED

    container.state = {"ExitCode": 137}
    watcher = DockerEventWatcher(client, clock=lambda: 1.0, memory_pressure=lambda j, a: True)
    event = watcher.handle(docker_event("die", container))
    assert isinstance(event, JobExited)
    assert event.kind is ExitKind.OOM


def test_watcher_falls_back_to_the_event_exit_code_when_the_container_is_gone() -> None:
    client = FakeClient()
    container = _labelled(client)
    container.removed = True
    watcher = DockerEventWatcher(client, clock=lambda: 1.0)
    event = watcher.handle(docker_event("die", container, exitCode="0"))
    assert event == JobExited("j1", 1, ExitKind.SUCCESS, 1.0)
    assert len(watcher.errors) == 1


def test_watcher_ignores_other_events_and_unlabelled_containers() -> None:
    client = FakeClient()
    container = _labelled(client)
    watcher = DockerEventWatcher(client, clock=lambda: 1.0)
    assert watcher.handle(docker_event("create", container)) is None
    assert watcher.handle({"Type": "network", "Action": "connect"}) is None
    bare = client.containers.create("x")
    assert watcher.handle(docker_event("start", bare)) is None
    weird = _labelled(client, attempt="first")
    assert watcher.handle(docker_event("start", weird)) is None
    assert watcher.drain() == []


def test_watcher_thread_follows_the_stream() -> None:
    client = FakeClient()
    container = _labelled(client)
    client.stream = [docker_event("start", container)]
    watcher = DockerEventWatcher(client, clock=lambda: 2.0)
    watcher.start()
    watcher.stop()
    assert watcher.drain() == [JobStarted("j1", 1, 2.0)]
    assert client.event_filters == [{"type": "container", "label": [JOB_LABEL]}]


@dataclass(frozen=True)
class _Sample:
    pid: int
    device_id: str
    usage: Resources


def test_mapper_sums_per_job_attempt_and_device() -> None:
    client = FakeClient()
    a = _labelled(client, "a", "1")
    b = _labelled(client, "b", "2")
    plain = client.containers.create("x")
    where = {10: a.id, 11: a.id, 20: b.id, 30: plain.id, 40: None, 50: "f" * 64}
    mapper = ContainerUsageMapper(client, where.get)
    samples = [
        _Sample(10, "gpu0", Resources(vram=100)),
        _Sample(11, "gpu0", Resources(vram=50)),
        _Sample(10, "gpu1", Resources(vram=5)),
        _Sample(20, "gpu0", Resources(vram=70)),
        _Sample(30, "gpu0", Resources(vram=999)),  # unmanaged container
        _Sample(40, "gpu0", Resources(vram=999)),  # not in a container
        _Sample(50, "gpu0", Resources(vram=999)),  # container gone
    ]
    assert mapper.map(samples, 5.0) == [
        JobObserved("a", "gpu0", 5.0, Resources(vram=150), attempt=1),
        JobObserved("a", "gpu1", 5.0, Resources(vram=5), attempt=1),
        JobObserved("b", "gpu0", 5.0, Resources(vram=70), attempt=2),
    ]


def test_mapper_caches_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    a = _labelled(client, "a", "1")
    mapper = ContainerUsageMapper(client, lambda pid: a.id)
    mapper.map([_Sample(1, "gpu0", Resources(vram=1))], 0.0)
    a.removed = True  # labels are not looked up again
    assert mapper.map([_Sample(1, "gpu0", Resources(vram=2))], 1.0) == [
        JobObserved("a", "gpu0", 1.0, Resources(vram=2), attempt=1)
    ]
