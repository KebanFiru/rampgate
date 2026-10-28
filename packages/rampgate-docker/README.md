# rampgate-docker

Docker executor, exit classification and event watcher for rampgate.

Part of the [rampgate](../../README.md) monorepo.

```python
from rampgate_docker import ContainerSpec, ContainerUsageMapper, DockerEventWatcher, DockerExecutor
from rampgate_nvml import NvmlProbe, pid_to_container_id

executor = DockerExecutor(lambda job, start: ContainerSpec("my-trainer:1", gpus="all"))
watcher = DockerEventWatcher(executor.client, clock, stop_reason=executor.stop_reason)
mapper = ContainerUsageMapper(executor.client, pid_to_container_id)
watcher.start()
while True:
    now = clock()
    for event in watcher.drain():                   # JobStarted / JobExited
        engine.apply(event)
    observed, processes = probe.sample(now)
    for event in [*observed, *mapper.map(processes, now)]:
        engine.observe(event)
    decision = engine.tick(now)
    executor.apply(decision.actions, {j: rt.job for j, rt in engine.state.jobs.items()})
```

- **`DockerExecutor`**: `Start` removes containers of older attempts of the job, then creates and starts a container labelled `rampgate.job_id` / `rampgate.attempt`, with `RAMPGATE_JOB_ID`, `RAMPGATE_ATTEMPT` and `RAMPGATE_RESUME` in its environment. `Stop` runs `container.stop(timeout=grace_s)` on a worker thread.
- **`classify_exit`**: `OOMKilled`, an OOM message in the log tail (`CUDA out of memory`, `torch.OutOfMemoryError`, …; configurable), or 137 under memory pressure → `OOM`; 0 → `SUCCESS`; 143/137 after a requested stop → `PREEMPTED`/`CANCELLED`; anything else → `FAILURE`. A script should exit 143 (not 0) when it stops on SIGTERM; 0 means the job finished.
- **`DockerEventWatcher`**: follows Docker's event stream on a thread; `drain()` hands the events to the Engine's thread.
- **`ContainerUsageMapper`**: probe process samples (e.g. from `rampgate-nvml`) → `JobObserved` per job, attempt and device. Usage outside managed containers is left to rampgate's foreign-usage accounting.
