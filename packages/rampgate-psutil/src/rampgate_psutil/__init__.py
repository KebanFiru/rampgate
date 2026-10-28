"""Host RAM/CPU probe for rampgate (psutil), SPEC §9.

- `HostProbe` reports the host as one rampgate device: its capacity
  (`device`), its total usage (`sample`, a `DeviceObserved`) and the usage of
  a process tree (`process_tree_usage`: RSS sum and CPU cores used).
- ``dims`` maps a dimension name to a unit. Memory units are ``bytes``,
  ``KiB``, ``MiB`` and ``GiB``; CPU is ``cores``.

CPU usage is measured between two calls (psutil's ``cpu_percent`` with no
interval), so the first sample of the host or of a process reports 0 cores.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import psutil

from rampgate import Device, DeviceObserved, Resources

__version__ = "0.1.0.dev0"

__all__ = ["MEMORY_UNITS", "HostProbe", "ProcessUsage"]

MEMORY_UNITS: Mapping[str, int] = {"bytes": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
_CPU_UNIT = "cores"


@dataclass(frozen=True, slots=True)
class ProcessUsage:
    """What one process tree uses on a device (the root ``pid`` and its descendants)."""

    pid: int
    device_id: str
    usage: Resources


class HostProbe:
    """The host's RAM and CPU as one rampgate device."""

    def __init__(
        self,
        device_id: str = "cpu",
        dims: Mapping[str, str] | None = None,
    ) -> None:
        dims = dict(dims) if dims is not None else {"ram": "bytes", "cpu": "cores"}
        if not dims:
            raise ValueError("dims must name at least one dimension")
        for dim, unit in dims.items():
            if unit != _CPU_UNIT and unit not in MEMORY_UNITS:
                units = ", ".join([*MEMORY_UNITS, _CPU_UNIT])
                raise ValueError(f"unknown unit {unit!r} for {dim!r}; expected one of {units}")
        self.device_id = device_id
        self.dims: Mapping[str, str] = dims
        self._procs: dict[int, psutil.Process] = {}
        psutil.cpu_percent(interval=None)  # start the host's CPU measurement window

    def device(self) -> Device:
        """The host as a `Device`: total RAM and the logical CPU count."""
        total = psutil.virtual_memory().total
        cores = psutil.cpu_count(logical=True) or 1
        return Device(self.device_id, self._resources(total, float(cores)))

    def sample(self, t: float) -> DeviceObserved:
        """Everything the host uses right now (``total - available`` RAM, busy cores)."""
        vm = psutil.virtual_memory()
        cores = psutil.cpu_count(logical=True) or 1
        busy = psutil.cpu_percent(interval=None) / 100 * cores
        return DeviceObserved(self.device_id, t, self._resources(vm.total - vm.available, busy))

    def process_tree_usage(self, pid: int) -> ProcessUsage:
        """RSS sum and CPU cores of ``pid`` and its descendants.

        Processes that exit while they are being read are skipped. Raises
        ``psutil.NoSuchProcess`` if ``pid`` itself does not exist.
        """
        rss = 0
        cores = 0.0
        seen: set[int] = set()
        for proc in self._tree(pid):
            try:
                with proc.oneshot():
                    rss += proc.memory_info().rss
                    cores += proc.cpu_percent(interval=None) / 100
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            seen.add(proc.pid)
        # Forget processes that are gone, so a reused pid starts a fresh CPU window.
        for gone in set(self._procs) - seen - {pid}:
            del self._procs[gone]
        return ProcessUsage(pid, self.device_id, self._resources(rss, cores))

    def _tree(self, pid: int) -> Iterator[psutil.Process]:
        root = self._process(pid)
        yield root
        try:
            children = root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        for child in children:
            yield self._process(child.pid, child)

    def _process(self, pid: int, proc: psutil.Process | None = None) -> psutil.Process:
        # The same Process object across calls keeps cpu_percent's window per process.
        cached = self._procs.get(pid)
        if cached is not None and cached.is_running():
            return cached
        proc = proc if proc is not None else psutil.Process(pid)
        self._procs[pid] = proc
        return proc

    def _resources(self, ram_bytes: float, cores: float) -> Resources:
        values: dict[str, float] = {}
        for dim, unit in self.dims.items():
            if unit == _CPU_UNIT:
                values[dim] = max(cores, 0.0)
            else:
                values[dim] = max(ram_bytes, 0) / MEMORY_UNITS[unit]
        return Resources(values)
