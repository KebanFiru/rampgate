"""NVIDIA GPU probe for rampgate (NVML), with process-to-container mapping (SPEC §9).

- `NvmlProbe` reports each GPU as a rampgate device (`device`) and samples
  its total usage and the usage of every process on it (`sample`).
- `pid_to_container_id` maps a host pid to its container through
  ``/proc/<pid>/cgroup`` (cgroup v1 and v2; docker, containerd, CRI-O, podman).

NVML lists processes by **host** pid, and the cgroup lookup reads the host's
``/proc``. A probe running in a container therefore needs the host PID
namespace (``docker run --pid=host``).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from rampgate import Device, DeviceObserved, Resources

__version__ = "0.1.0"

__all__ = [
    "MEMORY_UNITS",
    "NvmlApi",
    "NvmlProbe",
    "ProcessUsage",
    "container_id_from_cgroup",
    "pid_to_container_id",
]

MEMORY_UNITS: Mapping[str, int] = {"bytes": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}


@dataclass(frozen=True, slots=True)
class ProcessUsage:
    """What one process uses on a device."""

    pid: int
    device_id: str
    usage: Resources


# --- process -> container ---------------------------------------------------------------

_ID = r"[0-9a-f]{64}"
# A scope named after its runtime (systemd cgroup driver), e.g. docker-<id>.scope.
_SCOPE = re.compile(rf"(?:docker|libpod|cri-containerd|crio)-({_ID})\.scope$")
# A path segment that is the id itself (cgroupfs driver), e.g. /docker/<id>, /kubepods/.../<id>.
_SEGMENT = re.compile(rf"^({_ID})$")


def container_id_from_cgroup(text: str) -> str | None:
    """The container id named in the contents of a ``/proc/<pid>/cgroup`` file, or None.

    Lines are ``hierarchy:controllers:path``; cgroup v2 has the single line
    ``0::path``. The innermost container wins (for nested containers, the last
    id in the path). podman's ``libpod-conmon-<id>.scope`` is the container
    monitor, not the container, and is ignored.
    """
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        found: str | None = None
        for segment in parts[2].split("/"):
            match = _SCOPE.search(segment) or _SEGMENT.match(segment)
            if match:
                found = match.group(1)
        if found is not None:
            return found
    return None


def pid_to_container_id(pid: int, proc_root: str | Path = "/proc") -> str | None:
    """The id of the container ``pid`` runs in, or None (not in a container, or gone)."""
    try:
        text = (Path(proc_root) / str(pid) / "cgroup").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return container_id_from_cgroup(text)


# --- NVML ------------------------------------------------------------------------------


class _MemoryInfo(Protocol):
    @property
    def total(self) -> int: ...
    @property
    def used(self) -> int: ...


class _ProcessInfo(Protocol):
    @property
    def pid(self) -> int: ...
    @property
    def usedGpuMemory(self) -> int | None: ...


class NvmlApi(Protocol):
    """The part of ``pynvml`` (``nvidia-ml-py``) the probe uses."""

    def nvmlInit(self) -> None: ...
    def nvmlShutdown(self) -> None: ...
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _MemoryInfo: ...
    def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> Sequence[_ProcessInfo]: ...
    def nvmlDeviceGetGraphicsRunningProcesses(self, handle: object) -> Sequence[_ProcessInfo]: ...


def _load_pynvml() -> NvmlApi:
    return cast(NvmlApi, importlib.import_module("pynvml"))


class NvmlProbe:
    """GPU memory of the devices in ``index_to_device_id``, in ``dim`` (unit ``unit``).

    ``sample`` returns a `DeviceObserved` per GPU (NVML's ``used``, which
    includes the driver's own share and every process, managed or not) and a
    `ProcessUsage` per process NVML lists. A process whose usage NVML cannot
    report (``usedGpuMemory`` None, e.g. without permission) is left out.
    Compute processes are always listed; graphics processes only with
    ``include_graphics``.
    """

    def __init__(
        self,
        index_to_device_id: Mapping[int, str],
        dim: str = "vram",
        *,
        unit: str = "bytes",
        include_graphics: bool = False,
        stable_min_usage_mib: float = 256.0,
        nvml: NvmlApi | None = None,
    ) -> None:
        if not index_to_device_id:
            raise ValueError("index_to_device_id must name at least one GPU")
        if unit not in MEMORY_UNITS:
            raise ValueError(f"unknown unit {unit!r}; expected one of {', '.join(MEMORY_UNITS)}")
        self._nvml = nvml if nvml is not None else _load_pynvml()
        self._nvml.nvmlInit()
        self.dim = dim
        self.unit = unit
        self.include_graphics = include_graphics
        self.stable_min_usage_mib = stable_min_usage_mib
        self._handles = {
            device_id: self._nvml.nvmlDeviceGetHandleByIndex(index)
            for index, device_id in index_to_device_id.items()
        }

    def close(self) -> None:
        self._nvml.nvmlShutdown()

    def __enter__(self) -> NvmlProbe:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def devices(self) -> list[Device]:
        """Every probed GPU as a `Device` whose capacity is its total memory.

        Its ``stable_min_usage`` (Rule S usage floor) is ``stable_min_usage_mib``
        (default 256 MiB, in this probe's unit): a CUDA context alone was measured at
        about 82 MiB on the L5 hardware, so a job below the floor has not started
        using the GPU yet. ``stable_min_usage_mib=0`` turns the floor off.
        """
        floor = self._resources(int(self.stable_min_usage_mib * MEMORY_UNITS["MiB"]))
        return [
            Device(
                device_id,
                self._resources(self._nvml.nvmlDeviceGetMemoryInfo(handle).total),
                stable_min_usage=floor,
            )
            for device_id, handle in self._handles.items()
        ]

    def sample(self, t: float) -> tuple[list[DeviceObserved], list[ProcessUsage]]:
        observed: list[DeviceObserved] = []
        processes: list[ProcessUsage] = []
        for device_id, handle in self._handles.items():
            used = self._nvml.nvmlDeviceGetMemoryInfo(handle).used
            observed.append(DeviceObserved(device_id, t, self._resources(used)))
            infos = list(self._nvml.nvmlDeviceGetComputeRunningProcesses(handle))
            if self.include_graphics:
                infos.extend(self._nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            per_pid: dict[int, int] = {}
            for info in infos:
                if info.usedGpuMemory is None:
                    continue
                # A process listed as both compute and graphics holds one allocation.
                per_pid[info.pid] = max(per_pid.get(info.pid, 0), info.usedGpuMemory)
            processes.extend(
                ProcessUsage(pid, device_id, self._resources(mem))
                for pid, mem in sorted(per_pid.items())
            )
        return observed, processes

    def _resources(self, nbytes: int) -> Resources:
        return Resources({self.dim: nbytes / MEMORY_UNITS[self.unit]})
