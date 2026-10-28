# Changelog — rampgate-nvml

## Unreleased

### Added
- `NvmlProbe`: GPUs as devices (`devices`), and `sample` → a `DeviceObserved` per GPU plus a `ProcessUsage` per process (compute, optionally graphics), in a configurable dimension and unit (L5).
- `pid_to_container_id` / `container_id_from_cgroup`: container id from `/proc/<pid>/cgroup`, cgroup v1 and v2, docker, containerd, CRI-O and podman, tested on fixtures (L5).
- Package skeleton (L0).
