"""`NvmlProbe` against a fake NVML, plus a live test skipped without a GPU."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from rampgate import Resources
from rampgate_nvml import NvmlProbe, ProcessUsage

MiB = 1 << 20


@dataclass
class _Mem:
    total: int
    used: int


@dataclass
class _Proc:
    pid: int
    usedGpuMemory: int | None


@dataclass
class FakeNvml:
    mem: dict[int, _Mem]
    compute: dict[int, list[_Proc]] = field(default_factory=dict)
    graphics: dict[int, list[_Proc]] = field(default_factory=dict)
    inits: int = 0
    shutdowns: int = 0

    def nvmlInit(self) -> None:
        self.inits += 1

    def nvmlShutdown(self) -> None:
        self.shutdowns += 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _Mem:
        assert isinstance(handle, int)
        return self.mem[handle]

    def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> Sequence[_Proc]:
        assert isinstance(handle, int)
        return self.compute.get(handle, [])

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle: object) -> Sequence[_Proc]:
        assert isinstance(handle, int)
        return self.graphics.get(handle, [])


def _fake() -> FakeNvml:
    return FakeNvml(
        mem={0: _Mem(6144 * MiB, 1000 * MiB), 1: _Mem(8192 * MiB, 0)},
        compute={0: [_Proc(11, 300 * MiB), _Proc(12, 600 * MiB), _Proc(13, None)]},
        graphics={0: [_Proc(12, 20 * MiB), _Proc(14, 7 * MiB)]},
    )


def test_devices_have_total_memory_as_capacity() -> None:
    probe = NvmlProbe({0: "gpu0", 1: "gpu1"}, unit="MiB", nvml=_fake())
    devices = probe.devices()
    assert [d.id for d in devices] == ["gpu0", "gpu1"]
    assert devices[0].capacity == Resources(vram=6144)
    assert devices[1].capacity == Resources(vram=8192)


def test_sample_reports_device_totals_and_compute_processes() -> None:
    probe = NvmlProbe({0: "gpu0", 1: "gpu1"}, unit="MiB", nvml=_fake())
    observed, processes = probe.sample(3.0)
    assert [(o.device_id, o.t, o.used_total) for o in observed] == [
        ("gpu0", 3.0, Resources(vram=1000)),
        ("gpu1", 3.0, Resources()),
    ]
    # pid 13 has no usage NVML can report and is left out; graphics are off by default.
    assert processes == [
        ProcessUsage(11, "gpu0", Resources(vram=300)),
        ProcessUsage(12, "gpu0", Resources(vram=600)),
    ]


def test_graphics_processes_on_request_without_double_counting() -> None:
    probe = NvmlProbe({0: "gpu0"}, "mem", unit="MiB", include_graphics=True, nvml=_fake())
    _, processes = probe.sample(0.0)
    assert processes == [
        ProcessUsage(11, "gpu0", Resources(mem=300)),
        ProcessUsage(12, "gpu0", Resources(mem=600)),  # max of compute and graphics, not sum
        ProcessUsage(14, "gpu0", Resources(mem=7)),
    ]


def test_bytes_by_default() -> None:
    probe = NvmlProbe({0: "gpu0"}, nvml=_fake())
    assert probe.devices()[0].capacity == Resources(vram=6144 * MiB)


def test_init_shutdown_and_context_manager() -> None:
    fake = _fake()
    with NvmlProbe({0: "gpu0"}, nvml=fake) as probe:
        assert fake.inits == 1
        probe.sample(0.0)
    assert fake.shutdowns == 1


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least one GPU"):
        NvmlProbe({}, nvml=_fake())
    with pytest.raises(ValueError, match="unknown unit"):
        NvmlProbe({0: "gpu0"}, unit="MB", nvml=_fake())


def _gpu_available() -> bool:
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
    except Exception:
        return False
    try:
        return bool(pynvml.nvmlDeviceGetCount() > 0)
    finally:
        pynvml.nvmlShutdown()


@pytest.mark.skipif(not _gpu_available(), reason="no NVIDIA GPU / NVML available")
def test_live_gpu() -> None:
    with NvmlProbe({0: "gpu0"}, unit="MiB") as probe:
        (device,) = probe.devices()
        observed, processes = probe.sample(0.0)
    assert device.capacity["vram"] > 0
    assert 0 <= observed[0].used_total["vram"] <= device.capacity["vram"]
    assert all(p.device_id == "gpu0" and p.usage["vram"] >= 0 for p in processes)
