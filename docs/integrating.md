# Integrating rampgate in your scheduler

rampgate is a library you call from your own control loop. The loop owns the clock, the executor and the probes; the `Engine` owns the scheduling state. The repository's `examples/docker_gpu_queue.py` is a complete version of this loop on a real GPU.

## The loop

```python
import time
from rampgate import Engine, FifoPolicy, HistoryEstimator, JobSample
from rampgate_docker import ContainerSpec, ContainerUsageMapper, DockerEventWatcher, DockerExecutor
from rampgate_nvml import NvmlProbe, pid_to_container_id

t0 = time.monotonic()
clock = lambda: time.monotonic() - t0
probe = NvmlProbe({0: "gpu0"}, unit="MiB")
engine = Engine(probe.devices(), FifoPolicy(), HistoryEstimator(my_durable_store))
executor = DockerExecutor(lambda job, start: ContainerSpec("my-trainer:1", gpus="all"))
watcher = DockerEventWatcher(executor.client, clock, stop_reason=executor.stop_reason)
mapper = ContainerUsageMapper(executor.client, pid_to_container_id)
watcher.start()

while True:
    now = clock()
    for event in watcher.drain():              # JobStarted / JobExited
        engine.apply(event)
    observed, processes = probe.sample(now)
    jobs = mapper.map(processes, now)
    for dev in observed:                       # one atomic sample per device
        engine.observe_sample(dev.device_id, now, dev.used_total, {
            o.job_id: JobSample(o.attempt, o.usage) for o in jobs if o.device_id == dev.device_id
        })
    for job in new_submissions():
        engine.submit(job, t=now)
    decision = engine.tick(now)
    executor.apply(decision.actions, {j: rt.job for j, rt in engine.state.jobs.items()})
    log(decision)                              # actions, explanations, warnings
    time.sleep(1.0)
```

## Sampling

Use `observe_sample` for each device sample. It applies the jobs' usage **before** the device total. rampgate infers foreign usage from the device total minus the jobs' latest usage. If you use the per-event `observe` instead, feed each sample's `JobObserved` events before its `DeviceObserved`. Otherwise a growing job's growth since its previous sample counts as foreign. In the L5 real run this put foreign usage at 679–807 MiB instead of 423 MiB, and moved a vacate.

A running job missing from a sample, e.g. because the probe did not see its process this time, keeps its last known usage. Nothing else about it moves. After `Config.missing_samples_warn` (5) missing samples in a row, you get `Warn(JobNotObserved)`.

## Steps and epochs

Report `steps` (and `epoch`, incremented **after** the end-of-epoch evaluation) in `JobSample` when your training script exposes them. Rule S then waits for real progress, not just quiet memory. Without steps, set a usage floor (`Device.stable_min_usage`; `rampgate-nvml` sets 256 MiB), so a job loading data before touching the GPU is not marked stable at zero.

## Exits

Your training script should:

- save a checkpoint on SIGTERM and exit **143**;
- resume when `RAMPGATE_RESUME=1`;
- let the framework's OOM error reach its log, so `classify_exit` recognizes it.

## Persistence

Dump the engine (`engine.dump()`, JSON) after each tick or periodically. After a restart, `Engine.load(...)` and reconcile the running jobs from your executor: see [Restarts and reconciliation](reconciliation.md). The history store must be durable on its own.
