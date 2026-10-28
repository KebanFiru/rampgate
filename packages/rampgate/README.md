# rampgate

> *ramp* + *gate*: hold each job's warm-up ramp at the gate, measure what it really uses, then reserve exactly that.

rampgate is an **observation-based admission scheduler**: a small, embeddable, pure-Python library with **zero runtime dependencies**. It decides **when** a job may start on a device the caller chose:

- **Probation:** at most one warming job without a trusted estimate per device, so ramp-ups never race.
- **Observed-peak reservation:** reserve `observed_peak × safety_margin`, growing if the peak grows.
- **OOM feedback:** an OOM grows the estimate and re-queues the job to resume from its checkpoint.

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

Adapters for real machines: [`rampgate-nvml`](https://pypi.org/project/rampgate-nvml/) (NVIDIA GPUs), [`rampgate-psutil`](https://pypi.org/project/rampgate-psutil/) (host RAM/CPU), [`rampgate-docker`](https://pypi.org/project/rampgate-docker/) (Docker executor).

Status: alpha. All benchmark results come from rampgate's deterministic simulator; a demonstration on one laptop GPU is described in the [repository](https://github.com/KebanFiru/rampgate), along with the benchmark, its methodology and the limits of what was tested.
