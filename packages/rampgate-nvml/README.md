# rampgate-nvml

NVIDIA GPU probe for rampgate (NVML), with process-to-container mapping.

Part of the [rampgate](../../README.md) monorepo.

```python
from rampgate_nvml import NvmlProbe, pid_to_container_id

with NvmlProbe({0: "gpu0"}, unit="MiB") as probe:
    for device in probe.devices():            # capacity: total GPU memory
        engine.add_device(device)
    observed, processes = probe.sample(t)     # DeviceObserved per GPU, ProcessUsage per process
    for p in processes:
        container = pid_to_container_id(p.pid)  # via /proc/<pid>/cgroup; None outside a container
```

- `DeviceObserved.used_total` is NVML's `used`, which includes the driver's own share and every process, managed or not. rampgate treats what it cannot attribute to a managed job as foreign usage.
- `pid_to_container_id` understands cgroup v1 and v2 paths of docker (systemd and cgroupfs drivers), containerd, CRI-O and podman. Paths relative to a private cgroup namespace (`0::/../docker-<id>.scope`) work too.
- **Host PID namespace required.** NVML lists processes by host pid, so a probe running in a container needs `docker run --pid=host` (and `--gpus all`).
