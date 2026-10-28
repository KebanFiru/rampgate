# Adapters

The core does no I/O. Three separate distributions connect it to real machines.

## rampgate-nvml

```python
from rampgate_nvml import NvmlProbe, pid_to_container_id

with NvmlProbe({0: "gpu0"}, unit="MiB") as probe:
    devices = probe.devices()               # capacity: total memory; stable_min_usage: 256 MiB
    observed, processes = probe.sample(t)   # DeviceObserved per GPU, ProcessUsage per process
```

- `DeviceObserved.used_total` is NVML's `used`. It includes the driver's reservation: 371 MiB on the RTX 4050 we measured. rampgate counts that as foreign usage.
- Each process's usage includes its CUDA context, measured at about 82 MiB.
- `pid_to_container_id(pid)` reads `/proc/<pid>/cgroup`. It handles cgroup v1 and v2 paths of docker (systemd and cgroupfs drivers), containerd, CRI-O and podman, including paths relative to a private cgroup namespace.
- NVML lists **host** pids: a probe in a container needs `--pid=host`.

## rampgate-psutil

```python
from rampgate_psutil import HostProbe

probe = HostProbe("cpu", {"ram": "GiB", "cpu": "cores"})
device = probe.device()                      # total RAM, logical cores
obs = probe.sample(t)                        # DeviceObserved: used RAM, busy cores
usage = probe.process_tree_usage(pid)        # RSS sum and cores of a process tree
```

## rampgate-docker

- **`DockerExecutor(spec_for)`**:
  - On `Start`, it removes containers of older attempts, then starts a container labelled `rampgate.job_id` / `rampgate.attempt`. The container gets `RAMPGATE_JOB_ID`, `RAMPGATE_ATTEMPT` and `RAMPGATE_RESUME` in its environment.
  - On `Stop`, it runs `stop(timeout=grace_s)` on a worker thread.
- **`classify_exit`**:
  - `OOMKilled`, an OOM message in the log tail (`CUDA out of memory`, `torch.OutOfMemoryError`, …; configurable), or 137 under memory pressure → OOM.
  - 0 → SUCCESS.
  - 143/137 after a requested stop → PREEMPTED or CANCELLED.
  - Anything else → FAILURE.

  A GPU OOM has no `OOMKilled` flag, so a job must print a recognisable message. A job stopped with SIGTERM should exit 143, not 0: 0 means it finished.
- **`DockerEventWatcher`**: follows Docker's events on a thread. `drain()` hands `JobStarted` / `JobExited` to the Engine's thread.
- **`ContainerUsageMapper`**: turns probe process samples into per-job usage, summed per job, attempt and device. Processes outside managed containers are left to foreign usage.

The Docker socket is root-equivalent. The repository's `./dev integration` mounts it for integration tests only.
