"""`classify_exit`: a container's exit as a rampgate `ExitKind` (docs/adapters.md)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rampgate import ExitKind
from rampgate.model import StopReason

__all__ = ["DEFAULT_OOM_PATTERNS", "classify_exit"]

# Messages of frameworks that ran out of device or host memory. Case-insensitive.
DEFAULT_OOM_PATTERNS: Sequence[str] = (
    r"CUDA out of memory",
    r"torch\.OutOfMemoryError",
    r"torch\.cuda\.OutOfMemoryError",
    r"CUDA_ERROR_OUT_OF_MEMORY",
    r"cudaErrorMemoryAllocation",
    r"CUBLAS_STATUS_ALLOC_FAILED",
    r"RESOURCE_EXHAUSTED: Out of memory",  # TensorFlow / XLA
    r"std::bad_alloc",
    r"^MemoryError\b",
)

# SIGTERM (143) and SIGKILL (137): what a stopped container usually exits with.
_STOP_CODES = frozenset({137, 143})


def classify_exit(
    container_state: Mapping[str, Any],
    log_tail: str = "",
    patterns: Sequence[str] = DEFAULT_OOM_PATTERNS,
    *,
    stop_reason: StopReason | None = None,
    memory_pressure: bool = False,
) -> ExitKind:
    """The `ExitKind` of an exited container.

    ``container_state`` is the ``State`` of ``docker inspect`` (``ExitCode``,
    ``OOMKilled``). ``stop_reason`` is set when rampgate asked the container to
    stop; ``memory_pressure`` when the caller saw the host or device near its
    capacity at the exit. In order:

    1. ``OOMKilled`` (the container's cgroup ran out of memory) → OOM.
    2. An OOM message in ``log_tail`` (``patterns``, case-insensitive, per line) → OOM.
    3. Exit code 137 (SIGKILL) under ``memory_pressure`` and without a requested
       stop → OOM (the kernel's OOM killer, outside the container's own limit).
    4. Exit code 0 → SUCCESS, even after a requested stop: the work finished.
    5. A requested stop and exit code 143/137 → PREEMPTED or CANCELLED.
    6. Anything else → FAILURE (after a requested stop, the core turns it into
       PREEMPTED/CANCELLED with ``Warn(UncleanStop)``).
    """
    code = int(container_state.get("ExitCode", -1))
    if container_state.get("OOMKilled"):
        return ExitKind.OOM
    if log_tail and any(
        re.search(p, log_tail, flags=re.IGNORECASE | re.MULTILINE) for p in patterns
    ):
        return ExitKind.OOM
    if code == 137 and memory_pressure and stop_reason is None:
        return ExitKind.OOM
    if code == 0:
        return ExitKind.SUCCESS
    if stop_reason is not None and code in _STOP_CODES:
        return ExitKind.PREEMPTED if stop_reason is StopReason.PREEMPT else ExitKind.CANCELLED
    # Any other code, also after a requested stop: the core then decides (a FAILURE
    # while STOPPING ends PREEMPTED/CANCELLED with Warn(UncleanStop), docs/concepts.md).
    return ExitKind.FAILURE
