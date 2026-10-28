"""`HostProbe` against the real host (psutil works everywhere)."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator

import psutil
import pytest

from rampgate_psutil import MEMORY_UNITS, HostProbe

# A child that holds ~64 MiB and spawns a grandchild holding ~32 MiB.
_HOG = """
import subprocess, sys, time
block = bytearray(64 << 20)
child = subprocess.Popen([sys.executable, "-c",
    "import time; b = bytearray(32 << 20); print('ok', flush=True); time.sleep(60)"],
    stdout=subprocess.PIPE, text=True)
child.stdout.readline()
print("ready", flush=True)
time.sleep(60)
"""


@pytest.fixture
def hog() -> Iterator[subprocess.Popen[str]]:
    proc = subprocess.Popen([sys.executable, "-c", _HOG], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    yield proc
    for child in psutil.Process(proc.pid).children(recursive=True):
        child.kill()
    proc.kill()
    proc.wait()


def test_device_reports_total_ram_and_cores() -> None:
    probe = HostProbe()
    device = probe.device()
    assert device.id == "cpu"
    assert device.capacity["ram"] == psutil.virtual_memory().total
    assert device.capacity["cpu"] == psutil.cpu_count(logical=True)


def test_units_and_dimension_names() -> None:
    probe = HostProbe("host", {"mem": "MiB"})
    capacity = probe.device().capacity
    assert set(capacity) == {"mem"}
    assert capacity["mem"] == pytest.approx(psutil.virtual_memory().total / MEMORY_UNITS["MiB"])


def test_unknown_unit_and_empty_dims_raise() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        HostProbe(dims={"ram": "megabytes"})
    with pytest.raises(ValueError, match="at least one"):
        HostProbe(dims={})


def test_sample_is_a_device_observation_within_capacity() -> None:
    probe = HostProbe()
    obs = probe.sample(12.5)
    assert obs.device_id == "cpu"
    assert obs.t == 12.5
    capacity = probe.device().capacity
    assert 0 < obs.used_total["ram"] <= capacity["ram"]
    assert 0 <= obs.used_total["cpu"] <= capacity["cpu"]


def test_process_tree_sums_rss_of_descendants(hog: subprocess.Popen[str]) -> None:
    probe = HostProbe(dims={"ram": "MiB"})
    usage = probe.process_tree_usage(hog.pid)
    assert usage.pid == hog.pid
    assert usage.device_id == "cpu"
    # The root's own RSS alone is ~64 MiB + interpreter; the tree adds the grandchild.
    root_only = psutil.Process(hog.pid).memory_info().rss / MEMORY_UNITS["MiB"]
    assert usage.usage["ram"] >= root_only + 32


def test_cpu_cores_measured_between_calls() -> None:
    probe = HostProbe(dims={"cpu": "cores"})
    busy = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        first = probe.process_tree_usage(busy.pid)
        assert first.usage["cpu"] == 0  # the first call opens the window
        time.sleep(0.5)
        second = probe.process_tree_usage(busy.pid)
        assert second.usage["cpu"] > 0.2
    finally:
        busy.kill()
        busy.wait()


def test_missing_pid_raises() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    with pytest.raises(psutil.NoSuchProcess):
        HostProbe().process_tree_usage(proc.pid)
