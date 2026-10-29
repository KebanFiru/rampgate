# rampgate

> *ramp* + *gate*: hold each job's warm-up ramp at the gate, measure what it really uses, then reserve exactly that.

rampgate is an **observation-based admission scheduler** packaged as a small, embeddable, pure-Python library
(zero runtime dependencies in the core). It decides **when** a job may start on a device the caller chose:

- **Probation:** at most one warming job per device, so ramp-ups never race.
- **Observed-peak reservation:** reserve `observed_peak × safety_margin`, growing if the peak grows.
- **OOM feedback:** an OOM grows the estimate and re-queues the job to resume from its checkpoint.

**Status:** pre-alpha. L4 gates: failed on the first held-out evaluation (freeze-1, 6 OOMs); met on the second held-out set (freeze-2, seeds 2000–2019) after the fixes listed in [Methodology](docs/benchmark.md#methodology). Not on PyPI yet.

## Install

```sh
pip install rampgate            # the core: stdlib only
pip install rampgate-nvml       # NVIDIA GPU probe (NVML)
pip install rampgate-psutil     # host RAM/CPU probe
pip install rampgate-docker     # Docker executor and event watcher
```

## Quickstart

```python
from rampgate import Device, FifoPolicy, Job, MarkStable, Resources, Start
from rampgate.sim import RampPlateau, Scenario, SimJob, simulate

gpu = Device("gpu0", Resources(vram=24))  # GB here; any unit works
ramp = RampPlateau(load_s=20, ramp_s=40, plateau=Resources(vram=9), duration_s=300)
jobs = tuple(SimJob(Job(f"train-{i}", {"gpu0": None}), ramp, submit_at=0) for i in range(3))

trace = simulate(Scenario((gpu,), jobs), FifoPolicy(), max_time=3000)
for tick in trace.ticks:
    for action in tick.decision.actions:
        if isinstance(action, Start | MarkStable):
            print(f"t={tick.now:5.0f}  {type(action).__name__:10} {action.job_id}")
print("OOM kills:", trace.world.oom_count, " finished at t =", trace.ticks[-1].now)
```

Three jobs ramp one at a time (probation) and are reserved at their observed peaks. The third has no history, so it starts in the room that is left, is OOM-killed once (a first-contact OOM), and resumes from its checkpoint once it really fits. With real devices, drive the same `Engine` from your own loop: see [docs/integrating.md](docs/integrating.md) and `examples/docker_gpu_queue.py`.

## Results (simulated)

![makespan vs OOM kills per run, freeze 2](docs/benchmark/freeze-2/heldout/pareto_ooms.png)

In six simulated scenarios (freeze 2, held-out seeds 2000–2019), rampgate had no OOMs except a few first-contact ones (a median of 0–1 per run, all in cold runs) and finished in 0.12–0.75× the makespan of `Serial`, the other strategy that is OOM-free in every scenario. The price is makespan: warm rampgate took 1.07–1.59× as long as `GreedyFree` (per-seed medians), which OOMs hundreds of times per run in five of the six scenarios, and it missed its own ≤ 1.2× target (ratio of medians) in four of six. In the TrainQ-like scenario (e) no strategy OOMs, so rampgate's extra makespan (1.20× warm, 1.50× cold) buys nothing there; this scenario likely has too little memory pressure.

All benchmark results are from rampgate's deterministic simulator. Demonstrated (not benchmarked) on one laptop GPU (RTX 4050, 6 GB) with synthetic memory-hog containers: probation, a Rule V vacate and an OOM→Rule O cycle, checked by script. Not yet tested: multiple GPUs, gang jobs, CPU devices, a real framework's caching allocator, orchestrator restart, sub-second spikes.

Full tables, the protocol and both held-out evaluations: [docs/benchmark.md](docs/benchmark.md). What the real runs confirmed and contradicted: [docs/real-run/assumptions.md](docs/real-run/assumptions.md).

## Documentation

- [Concepts](docs/concepts.md): resources, devices, jobs, the lifecycle, events and actions
- [Algorithm](docs/algorithm.md): Rules P, A, S, O and V, the isolation retry, foreign usage
- [Invariants](docs/invariants.md): the properties the test suite checks
- [Policies](docs/policies.md) · [Estimators](docs/estimators.md) · [Adapters](docs/adapters.md) · [Simulator](docs/simulator.md)
- [Integrating rampgate in your scheduler](docs/integrating.md) · [Restarts and reconciliation](docs/reconciliation.md)
- [Benchmark](docs/benchmark.md) · [Real hardware](docs/real-run/assumptions.md) · [Prior art](docs/prior-art.md)

## Packages

| Distribution | Purpose |
|---|---|
| `rampgate` | Core: model, reducer, decide, policies, estimators, Engine, simulator. Stdlib only. |
| `rampgate-nvml` | NVIDIA GPU probe; process → container mapping via `/proc/<pid>/cgroup` |
| `rampgate-psutil` | Host RAM/CPU probe; process-tree usage |
| `rampgate-docker` | Docker executor, exit classification, event watcher, usage mapper |

## Development

Everything runs in Docker; nothing needs to be installed on the host.

```sh
./dev build          # build the dev image (uv + CPython 3.11–3.13)
./dev ci             # ruff, mypy --strict, pytest+coverage on 3.11, 3.12, 3.13
./dev ci 3.12        # a single version
./dev zero-deps      # verify the core wheel has no runtime dependencies
./dev shell          # interactive shell with a synced venv
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
