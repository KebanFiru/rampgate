# rampgate

> *ramp* + *gate*: hold each job's warm-up ramp at the gate, measure what it really uses, then reserve exactly that.

rampgate is an **observation-based admission scheduler**: a small, embeddable, pure-Python library (zero runtime dependencies in the core). It decides **when** a job may start on a device that the caller already chose:

- **Probation:** at most one warming job without a trusted estimate per device, so ramp-ups never race.
- **Observed-peak reservation:** once a job is stable, reserve `observed_peak × safety_margin`, and grow the reservation if the peak grows.
- **OOM feedback:** an OOM grows the job's estimate and re-queues it to resume from its checkpoint.

It is not a cluster manager. You keep your executor (Docker, a process pool, a batch system), and rampgate tells it when to start and stop what.

## Install

```sh
pip install rampgate                 # the core: stdlib only
pip install rampgate-nvml            # NVIDIA GPU probe
pip install rampgate-psutil          # host RAM/CPU probe
pip install rampgate-docker          # Docker executor, exit classification, events
```

## Quickstart

Three jobs that ramp to 9 GB each, on one 24 GB GPU, in rampgate's deterministic simulator:

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

The jobs ramp one at a time (probation). Each is reserved at its observed peak once it is stable. The third job has no history and starts in the room that is left. Its ramp does not fit, so it is OOM-killed once (a *first-contact* OOM), and its estimate grows. It then waits until it really fits and resumes from its checkpoint.

With real devices you drive the same `Engine` from your own loop: see [Integrating rampgate](integrating.md).

## Where to go next

- [Concepts](concepts.md): devices, jobs, the lifecycle, events, actions and explanations.
- [Algorithm](algorithm.md): the admission rules.
- [Benchmark](benchmark.md): simulated results against baselines, with the price of safety.
- [Real hardware](real-run/assumptions.md): what a run on one laptop GPU confirmed and contradicted.
