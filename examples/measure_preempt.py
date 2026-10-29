"""Measure start-up and preemption latency of a GPU memory hog
(docs/real-run/assumptions.md, measured items).

Usage (inside ``./dev integration``)::

    ./dev integration python examples/measure_preempt.py --out docs/real-run/preempt.json

For each sample, a hog container allocates ``size`` MiB, then gets SIGTERM
(``container.kill("SIGTERM")``, so the signal time is exact). Polling NVML and
Docker every ``--poll`` seconds, it records:

- start-up: ``create+start`` call → the hog's process appears in NVML →
  NVML shows the full allocation;
- preemption: SIGTERM → the container exits (``container.wait``), and SIGTERM →
  the memory is freed in NVML (the process is gone from NVML's list, and the
  GPU's used total is back within 64 MiB of the baseline before the start).

The hog saves its checkpoint on SIGTERM for ``HOG_CKPT_SAVE_S`` seconds; the
samples run with the default (2 s) and with 0 s, to separate the save from the
runtime's own latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import docker

from rampgate_nvml import NvmlProbe, pid_to_container_id

IMAGE = "rampgate-hog:dev"


def _wait_for(pred: Any, timeout: float, poll: float) -> float | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return time.monotonic()
        time.sleep(poll)
    return None


def sample(client: Any, probe: NvmlProbe, size: int, save_s: float, poll: float) -> dict[str, Any]:
    def used() -> float:
        observed, _ = probe.sample(0.0)
        return float(observed[0].used_total["vram"])

    def procs_of(container_id: str) -> list[float]:
        _, processes = probe.sample(0.0)
        return [p.usage["vram"] for p in processes if pid_to_container_id(p.pid) == container_id]

    baseline = used()
    t_create = time.monotonic()
    container = client.containers.create(
        IMAGE,
        environment={
            "HOG_STEPS": json.dumps([[0, size]]),
            "HOG_RATE": "8192",
            "HOG_DURATION": "600",
            "HOG_CKPT_SAVE_S": str(save_s),
            "HOG_CKPT_DIR": "/tmp",
            "RAMPGATE_JOB_ID": "measure",
        },
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        labels={"rampgate.measure": "1"},
        detach=True,
    )
    container.start()
    t_started = time.monotonic()
    cid = container.id
    t_visible = _wait_for(lambda: bool(procs_of(cid)), 60, poll)
    t_full = _wait_for(lambda: sum(procs_of(cid)) >= size, 60, poll)
    held = sum(procs_of(cid))

    exited: dict[str, float] = {}

    def wait() -> None:
        container.wait()
        exited["t"] = time.monotonic()

    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    t_sig = time.monotonic()
    container.kill(signal="SIGTERM")
    t_proc_gone = _wait_for(lambda: not procs_of(cid), 60, poll)
    t_freed = _wait_for(lambda: used() <= baseline + 64, 60, poll)
    waiter.join(timeout=60)
    container.reload()
    exit_code = container.attrs["State"]["ExitCode"]
    container.remove(force=True)

    def rel(t: float | None, base: float) -> float | None:
        return None if t is None else round(t - base, 3)

    return {
        "size_mib": size,
        "ckpt_save_s": save_s,
        "baseline_used_mib": baseline,
        "nvml_process_mib_at_full": held,
        "context_overhead_mib": round(held - size, 1),
        "create_to_started_s": rel(t_started, t_create),
        "create_to_nvml_visible_s": rel(t_visible, t_create),
        "create_to_full_allocation_s": rel(t_full, t_create),
        "sigterm_to_exit_s": rel(exited.get("t"), t_sig),
        "sigterm_to_nvml_process_gone_s": rel(t_proc_gone, t_sig),
        "sigterm_to_memory_freed_s": rel(t_freed, t_sig),
        "exit_code": exit_code,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 3072])
    parser.add_argument("--save", type=float, nargs="+", default=[2.0, 0.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--poll", type=float, default=0.02)
    args = parser.parse_args(argv)

    client = docker.from_env()
    samples: list[dict[str, Any]] = []
    with NvmlProbe({0: "gpu0"}, unit="MiB") as probe:
        for save_s in args.save:
            for size in args.sizes:
                for i in range(args.repeats):
                    s = sample(client, probe, size, save_s, args.poll)
                    s["repeat"] = i
                    samples.append(s)
                    print(json.dumps(s), flush=True)
                    time.sleep(1)
    keys = [
        "create_to_nvml_visible_s",
        "create_to_full_allocation_s",
        "sigterm_to_exit_s",
        "sigterm_to_nvml_process_gone_s",
        "sigterm_to_memory_freed_s",
        "context_overhead_mib",
    ]
    summary = []
    for save_s in args.save:
        for size in args.sizes:
            group = [s for s in samples if s["size_mib"] == size and s["ckpt_save_s"] == save_s]
            row: dict[str, Any] = {"size_mib": size, "ckpt_save_s": save_s, "n": len(group)}
            for k in keys:
                values = [s[k] for s in group if s[k] is not None]
                row[k] = (
                    {"median": statistics.median(values), "min": min(values), "max": max(values)}
                    if values
                    else None
                )
            summary.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"poll_s": args.poll, "samples": samples, "summary": summary}, indent=1) + "\n"
    )
    for row in summary:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
