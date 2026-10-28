# Changelog — rampgate-docker

## Unreleased

### Added
- `DockerExecutor`: `Start` → labelled container (older attempts removed first), `Stop` → `stop(timeout=grace_s)` on a worker thread; `ContainerSpec` with GPUs, labels and extra create arguments (L5).
- `classify_exit` with configurable OOM patterns, memory-pressure 137, and stop-aware 143/137 (L5).
- `DockerEventWatcher`: Docker events → `JobStarted` / `JobExited` (L5).
- `ContainerUsageMapper`: probe process samples → `JobObserved` per job, attempt and device (L5).
- Package skeleton (L0).
