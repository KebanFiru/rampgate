"""`DockerEventWatcher`: Docker container events as rampgate `JobStarted` / `JobExited`."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rampgate import ExitKind, JobExited, JobStarted
from rampgate.model import StopReason
from rampgate_docker.classify import DEFAULT_OOM_PATTERNS, classify_exit
from rampgate_docker.executor import ATTEMPT_LABEL, JOB_LABEL, DockerClientLike

__all__ = ["DockerEventWatcher", "ExitRecord"]

# Called on every classified exit with what the classification saw (for logs).
ExitRecord = Callable[[Mapping[str, Any]], None]


class DockerEventWatcher:
    """Turns the events of rampgate-labelled containers into Engine events.

    ``start`` follows Docker's event stream on a background thread; ``drain``
    returns the events received since the last call, for the Engine's thread to
    apply (the Engine is single-threaded). Events are timestamped with
    ``clock()`` when they are received, so they are on the caller's clock.

    A ``die`` event is classified with `classify_exit` from the container's
    ``State``, the last ``log_tail_lines`` lines of its log, the stop rampgate
    requested (``stop_reason``, e.g. `DockerExecutor.stop_reason`) and
    ``memory_pressure`` (a callable the caller answers, e.g. from its probe).
    """

    def __init__(
        self,
        client: DockerClientLike,
        clock: Callable[[], float],
        *,
        stop_reason: Callable[[str, int], StopReason | None] = lambda _job, _attempt: None,
        memory_pressure: Callable[[str, int], bool] = lambda _job, _attempt: False,
        patterns: Sequence[str] = DEFAULT_OOM_PATTERNS,
        log_tail_lines: int = 50,
        on_exit: ExitRecord | None = None,
    ) -> None:
        self.client = client
        self._clock = clock
        self._stop_reason = stop_reason
        self._memory_pressure = memory_pressure
        self._patterns = patterns
        self._tail = log_tail_lines
        self._on_exit = on_exit
        self._queue: queue.Queue[JobStarted | JobExited] = queue.Queue()
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self.errors: list[BaseException] = []

    def start(self) -> None:
        self._stream = self.client.events(
            decode=True, filters={"type": "container", "label": [JOB_LABEL]}
        )
        self._thread = threading.Thread(target=self._run, name="rampgate-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stream is not None and hasattr(self._stream, "close"):
            self._stream.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def drain(self) -> list[JobStarted | JobExited]:
        events: list[JobStarted | JobExited] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                return events

    def handle(self, event: Mapping[str, Any]) -> JobStarted | JobExited | None:
        """Translate one decoded Docker event (also queues it). Public for tests."""
        if event.get("Type", "container") != "container":
            return None
        action = event.get("Action") or event.get("status")
        attributes = event.get("Actor", {}).get("Attributes", {})
        job_id = attributes.get(JOB_LABEL)
        attempt_text = attributes.get(ATTEMPT_LABEL)
        if job_id is None or attempt_text is None or not attempt_text.isdigit():
            return None
        attempt = int(attempt_text)
        t = self._clock()
        out: JobStarted | JobExited
        if action == "start":
            out = JobStarted(job_id, attempt, t)
        elif action == "die":
            out = JobExited(job_id, attempt, self._classify(event, job_id, attempt), t)
        else:
            return None
        self._queue.put(out)
        return out

    def _classify(self, event: Mapping[str, Any], job_id: str, attempt: int) -> ExitKind:
        container_id = event.get("id") or event.get("Actor", {}).get("ID")
        state: dict[str, Any] = {}
        tail = ""
        try:
            container = self.client.containers.get(container_id)
            state = dict(container.attrs.get("State", {}))
            tail = container.logs(tail=self._tail).decode(errors="replace")
        except Exception as exc:  # the container may be gone (auto-remove)
            self.errors.append(exc)
        if "ExitCode" not in state:
            code = event.get("Actor", {}).get("Attributes", {}).get("exitCode")
            if code is not None and str(code).lstrip("-").isdigit():
                state["ExitCode"] = int(code)
        stop = self._stop_reason(job_id, attempt)
        pressure = self._memory_pressure(job_id, attempt)
        kind = classify_exit(
            state, tail, self._patterns, stop_reason=stop, memory_pressure=pressure
        )
        if self._on_exit is not None:
            self._on_exit(
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "container": container_id,
                    "exit_code": state.get("ExitCode"),
                    "oom_killed": bool(state.get("OOMKilled")),
                    "stop_reason": None if stop is None else stop.value,
                    "memory_pressure": pressure,
                    "kind": kind.value,
                    "log_tail": tail[-2000:],
                }
            )
        return kind

    def _run(self) -> None:
        try:
            for event in self._stream:
                self.handle(event)
        except Exception as exc:  # the stream was closed, or the daemon went away
            self.errors.append(exc)
