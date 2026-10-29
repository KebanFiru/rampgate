"""`classify_exit` decision table (docs/concepts.md exits while STOPPING, §9)."""

from __future__ import annotations

import pytest

from rampgate import ExitKind
from rampgate.model import StopReason
from rampgate_docker import DEFAULT_OOM_PATTERNS, classify_exit

PRE = StopReason.PREEMPT
CAN = StopReason.CANCEL


@pytest.mark.parametrize(
    ("state", "log", "stop", "pressure", "expected"),
    [
        ({"ExitCode": 0}, "", None, False, ExitKind.SUCCESS),
        ({"ExitCode": 1}, "Traceback ...\nValueError: bad", None, False, ExitKind.FAILURE),
        ({"ExitCode": 137, "OOMKilled": True}, "", None, False, ExitKind.OOM),
        # OOMKilled wins over a requested stop: the core records it as a lower bound.
        ({"ExitCode": 137, "OOMKilled": True}, "", PRE, False, ExitKind.OOM),
        ({"ExitCode": 1}, "RuntimeError: CUDA out of memory. Tried to", None, False, ExitKind.OOM),
        ({"ExitCode": 1}, "torch.OutOfMemoryError: CUDA out of memory", None, False, ExitKind.OOM),
        ({"ExitCode": 1}, "cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY", None, False, ExitKind.OOM),
        ({"ExitCode": 1}, "RESOURCE_EXHAUSTED: Out of memory while", None, False, ExitKind.OOM),
        (
            {"ExitCode": 134},
            "terminate called after throwing std::bad_alloc",
            None,
            False,
            ExitKind.OOM,
        ),
        ({"ExitCode": 1}, "Traceback\nMemoryError\n", None, False, ExitKind.OOM),
        ({"ExitCode": 1}, "cuda out of memory", None, False, ExitKind.OOM),  # case-insensitive
        # 137 alone is a SIGKILL; only with memory pressure is it the kernel's OOM killer.
        ({"ExitCode": 137}, "", None, False, ExitKind.FAILURE),
        ({"ExitCode": 137}, "", None, True, ExitKind.OOM),
        # After a requested stop: 143/137 are the stop, 0 is a finished job.
        ({"ExitCode": 143}, "", PRE, False, ExitKind.PREEMPTED),
        ({"ExitCode": 137}, "", PRE, True, ExitKind.PREEMPTED),
        ({"ExitCode": 143}, "", CAN, False, ExitKind.CANCELLED),
        ({"ExitCode": 137}, "", CAN, False, ExitKind.CANCELLED),
        ({"ExitCode": 0}, "", PRE, False, ExitKind.SUCCESS),
        # Another code after a stop is a FAILURE; the core turns it into PREEMPTED + UncleanStop.
        ({"ExitCode": 2}, "", PRE, False, ExitKind.FAILURE),
        ({}, "", None, False, ExitKind.FAILURE),  # no exit code at all
    ],
)
def test_table(
    state: dict[str, object],
    log: str,
    stop: StopReason | None,
    pressure: bool,
    expected: ExitKind,
) -> None:
    kind = classify_exit(state, log, stop_reason=stop, memory_pressure=pressure)
    assert kind is expected


def test_patterns_are_configurable() -> None:
    log = "my-framework: device allocation failed"
    assert classify_exit({"ExitCode": 3}, log) is ExitKind.FAILURE
    patterns = (*DEFAULT_OOM_PATTERNS, r"device allocation failed")
    assert classify_exit({"ExitCode": 3}, log, patterns) is ExitKind.OOM


def test_memory_error_must_start_a_line() -> None:
    # "MemoryError" inside a message is not Python's MemoryError.
    assert classify_exit({"ExitCode": 1}, "no MemoryError here") is ExitKind.FAILURE
