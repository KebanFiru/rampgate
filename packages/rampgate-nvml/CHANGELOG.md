# Changelog — rampgate-nvml

## Unreleased

### Added
- `NvmlProbe.devices()` sets each GPU's `stable_min_usage` (the Rule S usage floor) to `stable_min_usage_mib` (default 256 MiB, in the probe's unit; 0 turns it off) (L6).
- `NvmlProbe`: GPUs as devices (`devices`), and `sample` → a `DeviceObserved` per GPU plus a `ProcessUsage` per process (compute, optionally graphics), in a configurable dimension and unit (L5).
- `pid_to_container_id` / `container_id_from_cgroup`: container id from `/proc/<pid>/cgroup`, cgroup v1 and v2, docker, containerd, CRI-O and podman, tested on fixtures (L5).
- Package skeleton (L0).
