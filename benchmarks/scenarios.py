"""Benchmark scenarios (SPEC §8): ``make(name, seed) -> Case``.

A seed changes the scenario itself (which templates the jobs use, per-job size
jitter, durations, submit times, noise and hint errors), not only the
executor's noise. Templates (a model's base memory profile) are fixed per
scenario, like a team's set of models. Fingerprints name a job's template (and
config), never its id, so history learned from one seed applies to another.
GPUs share the device kind ``"gpu"`` for the same reason.

Units: GPU memory ``vram`` and host ``ram`` in GiB, ``cpu`` in cores, time in
seconds. Jobs carry no hints except in (d).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rampgate import Device, FifoPolicy, Job, OrderingPolicy, Resources, StagePolicy
from rampgate.sim import EpochEval, RampPlateau, Scenario, SimJob

__all__ = ["NAMES", "SIM_KWARGS", "Case", "make"]

# Simulator settings shared by every scenario and strategy. A (re)start costs
# ``start_delay_s`` of process start-up (container, CUDA init, imports) before
# the job reports JobStarted; an OOM'd job pays it again on every retry.
SIM_KWARGS: dict[str, Any] = {
    "dt": 1.0,
    "start_delay_s": 15.0,
    "stop_delay_s": 1.0,
    "ckpt_save_s": 10.0,
}


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    seed: int
    scenario: Scenario
    policy: OrderingPolicy
    max_time: float
    params: dict[str, Any]


def _gpus(n: int, vram: float = 80.0) -> tuple[Device, ...]:
    return tuple(Device(f"gpu:{i}", Resources(vram=vram), {"kind": "gpu"}) for i in range(n))


def _ramp(rng: random.Random, plateau: Resources, duration: float) -> RampPlateau:
    return RampPlateau(
        load_s=rng.uniform(10, 30),
        ramp_s=rng.uniform(20, 60),
        plateau=plateau,
        duration_s=duration,
        noise=rng.uniform(0.0, 0.02),
        checkpoint_every_s=120.0,
    )


def _templates(scenario: str) -> random.Random:
    """The fixed generator for a scenario's templates (independent of the seed)."""
    return random.Random(f"templates:{scenario}")


def _jitter(rng: random.Random, base: float, rel: float = 0.05) -> float:
    return base * rng.uniform(1 - rel, 1 + rel)


def _small(seed: int) -> Case:
    """(a) Many small jobs: 40 jobs of 2-7 GiB on two 80 GiB GPUs, submitted in a burst."""
    rng = random.Random(seed)
    devices = _gpus(2)
    fixed = _templates("a")
    templates = {f"s{i}": fixed.uniform(2, 7) for i in range(5)}
    jobs = []
    for i in range(40):
        name = rng.choice(sorted(templates))
        vram = _jitter(rng, templates[name])
        job = Job(f"a{i:02d}", {devices[i % 2].id: None}, fingerprint=(f"tmpl:{name}",))
        workload = _ramp(rng, Resources(vram=vram), rng.uniform(300, 900))
        jobs.append(SimJob(job, workload, submit_at=round(rng.uniform(0, 60))))
    return Case("a_small", seed, Scenario(devices, tuple(jobs), seed=seed), _fifo(), 30_000, {})


def _mixed_jobs(rng: random.Random, devices: tuple[Device, ...], prefix: str) -> list[SimJob]:
    fixed = _templates("b")
    templates = {f"s{i}": fixed.uniform(3, 8) for i in range(3)}
    templates |= {f"l{i}": fixed.uniform(30, 55) for i in range(2)}
    jobs = []
    for i in range(22):
        name = rng.choice(sorted(templates)) if i >= 2 else f"l{i}"
        large = name.startswith("l")
        vram = _jitter(rng, templates[name])
        duration = rng.uniform(900, 1800) if large else rng.uniform(300, 900)
        job = Job(f"{prefix}{i:02d}", {devices[i % 2].id: None}, fingerprint=(f"tmpl:{name}",))
        jobs.append(
            SimJob(
                job,
                _ramp(rng, Resources(vram=vram), duration),
                submit_at=round(rng.uniform(0, 600)),
            )
        )
    return jobs


def _mixed(seed: int) -> Case:
    """(b) Mixed: 22 jobs, small (3-8 GiB) and large (30-55 GiB), arriving over 10 minutes."""
    rng = random.Random(seed)
    devices = _gpus(2)
    jobs = _mixed_jobs(rng, devices, "b")
    return Case("b_mixed", seed, Scenario(devices, tuple(jobs), seed=seed), _fifo(), 40_000, {})


def _late_spike(seed: int) -> Case:
    """(c) Late spikes: training jobs whose end-of-epoch evaluation adds 30-60 % memory."""
    rng = random.Random(seed)
    devices = _gpus(2)
    fixed = _templates("c")
    templates = {f"e{i}": (fixed.uniform(8, 20), fixed.uniform(0.3, 0.6)) for i in range(4)}
    jobs = []
    for i in range(20):
        name = rng.choice(sorted(templates))
        base, spike = templates[name]
        base = _jitter(rng, base)
        epoch_s = rng.uniform(120, 240)
        workload = EpochEval(
            base=Resources(vram=base),
            eval_spike=Resources(vram=base * spike),
            epoch_s=epoch_s,
            eval_s=rng.uniform(15, 30),
            load_s=rng.uniform(10, 30),
            duration_s=epoch_s * rng.randint(3, 6),
            noise=rng.uniform(0.0, 0.02),
            checkpoint_every_s=epoch_s,
        )
        job = Job(f"c{i:02d}", {devices[i % 2].id: None}, fingerprint=(f"tmpl:{name}",))
        jobs.append(SimJob(job, workload, submit_at=round(rng.uniform(0, 120))))
    return Case(
        "c_late_spike", seed, Scenario(devices, tuple(jobs), seed=seed), _fifo(), 40_000, {}
    )


def _hints(seed: int, *, under: bool) -> Case:
    """(d) Declared hints: the mixed workload of (b), hints off by a random factor."""
    rng = random.Random(seed)
    devices = _gpus(2)
    lo, hi = (0.4, 0.7) if under else (1.5, 2.5)
    jobs = []
    for sj in _mixed_jobs(rng, devices, "d"):
        assert isinstance(sj.workload, RampPlateau)
        (device,) = sj.job.placement
        # Nobody declares more than the device has: hints are capped at capacity.
        capacity = devices[0].capacity
        hint = sj.workload.plateau.scale(rng.uniform(lo, hi)).min(capacity)
        job = Job(sj.job.id, {device: hint}, fingerprint=sj.job.fingerprint)
        jobs.append(SimJob(job, sj.workload, sj.submit_at))
    name = "d_under_declared" if under else "d_over_declared"
    return Case(
        name,
        seed,
        Scenario(devices, tuple(jobs), seed=seed),
        _fifo(),
        40_000,
        {"hint_factor": [lo, hi]},
    )


def _trainq(seed: int) -> Case:
    """(e) TrainQ-like: stage barriers, a sweep of templates x configs, on_finish tests.

    Stages: ``prep`` (CPU-only data preparation), then ``sweep-1`` and
    ``sweep-2``, each running every template with two of its four configs
    (``sweep-2`` reuses what ``sweep-1`` taught the history). Every training job
    has an on_finish test job (``after=``, no group) on the GPU it trained on.
    Devices: two 80 GiB GPUs and a CPU host (256 GiB, 32 cores).
    """
    rng = random.Random(seed)
    gpus = _gpus(2)
    cpu = Device("cpu:0", Resources(ram=256, cpu=32), {"kind": "cpu"})
    devices = (*gpus, cpu)
    jobs: list[SimJob] = []
    for i in range(4):
        ram, cores = rng.uniform(16, 48), rng.randint(4, 8)
        prep = Job(f"prep{i}", {cpu.id: None}, group="prep", fingerprint=("tmpl:prep",))
        workload = _ramp(rng, Resources(ram=ram, cpu=cores), rng.uniform(120, 300))
        jobs.append(SimJob(prep, workload))
    fixed = _templates("e")
    templates = {f"t{i}": (fixed.uniform(6, 30), fixed.uniform(0.2, 0.5)) for i in range(3)}
    n = 0
    for stage, configs in (("sweep-1", (0, 1)), ("sweep-2", (2, 3))):
        for name in sorted(templates):
            base, spike = templates[name]
            for c in configs:
                scale = 1 + 0.15 * c  # bigger configs use more memory
                vram = _jitter(rng, base * scale, 0.03)
                epoch_s = rng.uniform(90, 180)
                device = gpus[n % 2].id
                n += 1
                train = Job(
                    f"{name}-c{c}",
                    {device: None},
                    group=stage,
                    position=c,
                    fingerprint=(f"tmpl:{name}:c{c}", f"tmpl:{name}"),
                )
                workload = EpochEval(
                    base=Resources(vram=vram),
                    eval_spike=Resources(vram=vram * spike),
                    epoch_s=epoch_s,
                    eval_s=rng.uniform(10, 25),
                    load_s=rng.uniform(10, 30),
                    duration_s=epoch_s * rng.randint(3, 5),
                    noise=rng.uniform(0.0, 0.02),
                    checkpoint_every_s=epoch_s,
                )
                jobs.append(SimJob(train, workload))
                test = Job(f"{name}-c{c}-test", {device: None}, fingerprint=("tmpl:test",))
                test_load = _ramp(rng, Resources(vram=rng.uniform(2, 5)), rng.uniform(60, 120))
                jobs.append(SimJob(test, test_load, after=train.id))
    policy = StagePolicy(("prep", "sweep-1", "sweep-2"))
    return Case("e_trainq", seed, Scenario(devices, tuple(jobs), seed=seed), policy, 40_000, {})


def _fifo() -> OrderingPolicy:
    return FifoPolicy(strict=False)


_MAKERS: dict[str, Callable[[int], Case]] = {
    "a_small": _small,
    "b_mixed": _mixed,
    "c_late_spike": _late_spike,
    "d_under_declared": lambda seed: _hints(seed, under=True),
    "d_over_declared": lambda seed: _hints(seed, under=False),
    "e_trainq": _trainq,
}
NAMES = tuple(_MAKERS)


def make(name: str, seed: int) -> Case:
    return _MAKERS[name](seed)
