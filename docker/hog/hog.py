"""GPU memory hog: allocates device memory on a schedule, like a training job's ramp.

Configured by environment variables (all memory in MiB, all times in seconds):

- ``HOG_STEPS``: JSON list ``[[t, mib], ...]``: from ``t`` seconds after this
  attempt started, hold ``mib`` MiB (allocated or freed in 64 MiB chunks, at
  most ``HOG_RATE`` MiB/s). Default ``[[0, 512]]``.
- ``HOG_STEPS_RESUME``: optional, the same for attempts that resume from a
  checkpoint (``RAMPGATE_RESUME=1``); default: ``HOG_STEPS``.
- ``HOG_SPIKE``: optional JSON ``{"every_s": 30, "mib": 400, "hold_s": 5}``:
  every ``every_s`` seconds, allocate ``mib`` more for ``hold_s`` seconds, then
  free it again (an end-of-epoch evaluation, or a framework's transient buffers).
- ``HOG_DURATION``: seconds of work in total; progress survives a preemption
  through the checkpoint. Default 120.
- ``HOG_RATE``: allocation speed, MiB/s. Default 256.
- ``HOG_CKPT_DIR``: where the checkpoint lives (a volume). Default ``/ckpt``.
- ``HOG_CKPT_SAVE_S``: how long saving the checkpoint takes on SIGTERM. Default 2.

rampgate's executor sets ``RAMPGATE_JOB_ID``, ``RAMPGATE_ATTEMPT`` and
``RAMPGATE_RESUME``. On SIGTERM the hog saves its progress and exits 143. If an
allocation fails it prints ``CUDA out of memory`` and exits 1. Every second it
prints a JSON line with what it holds, for comparison with NVML.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path

CHUNK_MIB = 64
CUDA_ERROR_OUT_OF_MEMORY = 2


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._check(self.lib.cuInit(0), "cuInit")
        device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, device), "cuCtxCreate")

    def alloc(self, mib: int) -> int | None:
        ptr = ctypes.c_uint64()
        status = self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(mib << 20))
        if status == CUDA_ERROR_OUT_OF_MEMORY:
            return None
        self._check(status, "cuMemAlloc")
        # Touch it, so the memory is really in use (not only reserved).
        self._check(
            self.lib.cuMemsetD8_v2(ptr, ctypes.c_ubyte(1), ctypes.c_size_t(mib << 20)), "cuMemset"
        )
        return ptr.value

    def free(self, ptr: int) -> None:
        self._check(self.lib.cuMemFree_v2(ctypes.c_uint64(ptr)), "cuMemFree")

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} failed with CUDA error {status}")


def log(**fields: object) -> None:
    print(json.dumps({"wall": round(time.time(), 3), **fields}), flush=True)


def target_at(steps: list[list[float]], t: float) -> int:
    mib = 0
    for start, value in steps:
        if t >= start:
            mib = int(value)
    return mib


def main() -> int:  # noqa: PLR0915 - one control loop, read top to bottom
    job = os.environ.get("RAMPGATE_JOB_ID", "hog")
    attempt = os.environ.get("RAMPGATE_ATTEMPT", "1")
    steps = json.loads(os.environ.get("HOG_STEPS", "[[0, 512]]"))
    spike = json.loads(os.environ.get("HOG_SPIKE", "null"))
    duration = float(os.environ.get("HOG_DURATION", "120"))
    rate = float(os.environ.get("HOG_RATE", "256"))
    save_s = float(os.environ.get("HOG_CKPT_SAVE_S", "2"))
    ckpt = Path(os.environ.get("HOG_CKPT_DIR", "/ckpt")) / f"{job}.json"

    done = 0.0
    resume = os.environ.get("RAMPGATE_RESUME") == "1"
    if resume and ckpt.exists():
        done = float(json.loads(ckpt.read_text())["done"])
    if resume and "HOG_STEPS_RESUME" in os.environ:
        steps = json.loads(os.environ["HOG_STEPS_RESUME"])
    log(event="start", job=job, attempt=attempt, resume_from=done)

    stopping = False

    def on_term(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_term)

    cuda = Cuda()
    log(event="cuda_ready", job=job, attempt=attempt)
    held: list[int] = []
    spike_held: list[int] = []
    t0 = time.monotonic()
    last_tick = t0
    last_log = 0.0
    budget = 0.0  # MiB the rate limit allows to allocate now

    def oom(mib: int, kind: str) -> int:
        total = (len(held) + len(spike_held)) * CHUNK_MIB
        print(
            f"RuntimeError: CUDA out of memory: cuMemAlloc of {mib} MiB failed "
            f"(CUDA_ERROR_OUT_OF_MEMORY) while holding {total} MiB ({kind})",
            flush=True,
        )
        log(event="oom", job=job, attempt=attempt, holding_mib=total, kind=kind)
        return 1

    while True:
        now = time.monotonic()
        elapsed = now - t0
        done += now - last_tick
        budget = min(budget + (now - last_tick) * rate, max(rate, CHUNK_MIB))
        last_tick = now
        if stopping:
            log(event="sigterm", job=job, attempt=attempt, done=round(done, 1))
            time.sleep(save_s)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps({"done": done}))
            log(event="checkpoint_saved", job=job, attempt=attempt, done=round(done, 1))
            return 143
        if done >= duration:
            log(event="finished", job=job, attempt=attempt, done=round(done, 1))
            return 0

        want = target_at(steps, elapsed) // CHUNK_MIB
        while len(held) > want:
            cuda.free(held.pop())
        while len(held) < want and budget >= CHUNK_MIB:
            ptr = cuda.alloc(CHUNK_MIB)
            if ptr is None:
                return oom(CHUNK_MIB, "steps")
            held.append(ptr)
            budget -= CHUNK_MIB

        if spike:
            phase = elapsed % float(spike["every_s"])
            in_spike = elapsed >= float(spike["every_s"]) and phase < float(spike["hold_s"])
            if in_spike and not spike_held:
                for _ in range(int(spike["mib"]) // CHUNK_MIB):
                    ptr = cuda.alloc(CHUNK_MIB)
                    if ptr is None:
                        return oom(CHUNK_MIB, "spike")
                    spike_held.append(ptr)
                log(event="spike_on", job=job, attempt=attempt, mib=len(spike_held) * CHUNK_MIB)
            elif not in_spike and spike_held:
                while spike_held:
                    cuda.free(spike_held.pop())
                log(event="spike_off", job=job, attempt=attempt)

        if now - last_log >= 1.0:
            last_log = now
            log(
                event="hold",
                job=job,
                attempt=attempt,
                t=round(elapsed, 1),
                mib=(len(held) + len(spike_held)) * CHUNK_MIB,
            )
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
