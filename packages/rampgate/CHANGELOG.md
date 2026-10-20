# Changelog — rampgate

## Unreleased

### Added
- Package skeleton (L0).
- `Resources`: immutable non-negative resource vector with `+`, `-` (raises on underflow, eps-snapping), `sub(clamp=True)`, `scale`, `max`, `min`, `fits_in`, `exceeds`, `is_zero`, JSON round-trip; `KiB`/`MiB`/`GiB`/`TiB` constants (L0).
- `Config` with every SPEC §5 threshold, plus `max_lost_retries` and `foreign_tolerance` (L1).
- Model: `Device`, `Job`, `JobState`, `ExitKind`, `StopReason`, and the immutable reducer `State` / `JobRuntime` / `DeviceState` (L1).
- Events (SPEC §3.5) including `JobExited.oom_dims` and an optional `JobObserved.attempt` (L1).
- Actions (SPEC §3.6) with deterministic `action_id`; `Start.admitted_free`, `UncleanStop`, `EXCEEDS_DEVICE_CAPACITY`, `LOST_RETRIES_EXHAUSTED` (L1).
- Pure `reduce(state, message, config)`: the full lifecycle, Rule S helpers (`is_stable`, `is_peak_growth`), steps baselining, monotone reservations, Rule O with `admitted_free`, LOST handling, the STOPPING exit table, stale/duplicate/out-of-order handling with `ignored_events`, the action outbox, and `drain` (L1).
- `decide(snapshot, now, config, policy) -> Decision`: Rules P and A (compared directly against capacity), gang all-or-nothing, per-device head-of-line blocking, start timeouts (`Start.issued_at`), `Stop` emission, `MarkStable`, warnings with dedup (`CapacityExceeded`, `Overcommitted`, `ForeignUsageHigh`, `QueueStopped`), `GroupCompleted`, and explanations for every waiting job (L2).
- Explanations: `WaitingForWarmup`, `InsufficientResources`, `GroupBarrier`, `EnginePaused`, `UnknownEstimateBlocked`, `GangPartiallyBlocked`, `DeviceHeldFor`, `AwaitingResume` (L2).
- Policies: `StagePolicy` (backfill, `STOP_QUEUE`), `FifoPolicy`, `PriorityPolicy` (aging, blocking), `BasePolicy` (L2).
- Estimators: `HistoryEstimator`, `InMemoryHistoryStore`, `NullEstimator`, `StaticEstimator`, `resolve_estimates`, `apply_history` (L2).
- Rule S epoch gate (`JobObserved.epoch`, `Config.stable_requires_epoch`); `JobSubmitted.t` for aging (L2).
- `rampgate.sim`: workloads (`RampPlateau`, `StepProfile`, `Spiky`, `EpochEval`), `SimExecutor` with the OOM model, `Scenario` and `simulate` (L2).

### Changed
- `max_lost_retries` now counts retries, like `max_oom_retries` (fails on the 4th consecutive LOST by default) (L2).
- `DeviceRemoved` of an already removed device is ignored as a duplicate (L2).

