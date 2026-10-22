# Changelog — rampgate

## Unreleased

### Added
- `Config.trusted_margin` (default 1.05): a trusted STARTING/WARMING job counts as `max(E × trusted_margin, usage)`; `safety_margin` (1.15) stays for reservations from one observation (SPEC §5; changed after the first benchmark results).
- `Report.ooms_per_job`, `max_job_ooms`, `repeat_device_ooms`; `TimelineEntry.device` names the device of an OOM kill.
- `Report.first_contact_unknown` / `first_contact_hint`: first-contact means no observation-based estimate (no history record at any level, first attempt); a job admitted on its declared hint alone counts (SPEC §12 L4 clarification).
- `Warn(HintExceeded(job_id, attempt, device_id, hint, observed))`: edge-triggered, raised once per attempt when a running job's observed peak exceeds its declared hint, cleared when the attempt ends. Informational; admission ignores it (SPEC §3.6, I16).
- Simulator & benchmark (L4): `Report` (with first-contact OOMs), `run`, `quantile`/`summarize`, baselines `GreedyFree`, `DeclaredReservation`, `Serial` and the ablations as `Strategy` rows (`MAIN`, `ABLATIONS`, `WARM`), `Admission` / `PROBATION` / `measured_free`, `RejectedByAdmission`, executor ground truth (`WorldStats`: lost work, usage integrals, overcommit, timeline), `ckpt_save_s`, foreign-load schedules, `SimJob.after`. `Resources` arithmetic skips re-validation (about 3x faster simulations, identical results).
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

- `Engine` (L3): the SPEC §4 API, `tick` with estimator I/O, `apply`, `unreconciled()`, versioned `dump`/`load` (L3).
- `dump_state` / `load_state`: lossless, allowlisted, versioned JSON for the whole state, with a migration table and `UnsupportedSchema` (L3).
- `InMemoryHistoryStore.to_json` / `from_json`; `simulate(..., restart_at=...)` runs through the Engine and can restart mid-run (L3).
- `Start.estimate`; `unknown_min_free` as a fraction of capacity (default 10 %); edge-triggered warnings with `WarnCleared`; `BlockedByForeignUsage` (L2 follow-up).

- `Engine.confirm`, `Config.reconcile_grace_s` and the edge-triggered `Warn(Unreconciled)`; `HistoryRecord.record_id` and idempotent `HistoryStore.put`; `simulate(restart_at=[...], downtime=[...])` (L3 follow-up).

### Changed
- **Breaking (`HistoryStore` protocol):** `get` returns `HistoryEntry(complete, lower_bound)`; stores keep each record's `lower_bound` flag and an arrival order (bumped only when a put changes a record). A lower bound counts only if it is newer than the newest complete record of its key; if it exceeds the complete value, the estimate is untrusted. `LOWER_BOUND_SUFFIX` is gone. `InMemoryHistoryStore` JSON has a new format; the old one still loads (SPEC §7).
- An estimate is trusted only if its level-0 key has a complete observation (`peak` or `running` record); OOM/stop lower bounds alone give the same value, untrusted (`EstimateResult.complete`). `HistoryEstimator` stores lower bounds under `key + LOWER_BOUND_SUFFIX`; Rule H running records are written with `lower_bound=False` (SPEC §5, §7; changed after the first benchmark results).
- Rule P applies only to jobs without a trusted estimate (`Config.probation = "untrusted_only"`, default; `"all"` keeps the old behavior). An estimate is trusted when the estimator answered from the most specific fingerprint level (`EstimateResult(value, level)`, level 0) and the job has not OOMed. A trusted warming job takes no probation slot and counts as `max(E × safety_margin, usage)`; an untrusted one as `max(E, usage)`. `Start.trusted`, `JobRuntime.trusted`; `Estimator.estimate` returns `EstimateResult`; `Snapshot.trusted`. A job marked STABLE keeps its probation slot until the next tick (`MarkStable.issued_at`, `JobRuntime.stable_at`), so a sibling sees its running history record first (L4).
- Rule H also writes a `running` history record (lower bound) when a job becomes STABLE, and rewrites it with the same `record_id` when its reservation grows. `HistoryStore.put` max-merges by `record_id` (idempotent and order-independent); `InMemoryHistoryStore` keeps the merged value per record and still reads the L3 JSON format (L4).
- Rule O uses the room the device had left at OOM time (`C − F − Σ neighbours' usage`, max over `foreign_window_s`) instead of `admitted_free`, both for the lower bound and for `Fail(EXCEEDS_DEVICE_CAPACITY)`. A STABLE job that spikes next to a later neighbour now retries with growth instead of failing. `JobRuntime.usage_samples` keeps the window; state schema v2, with a migration from v1 (L4).
- `Fail(EXCEEDS_DEVICE_CAPACITY)` is decided from evidence only: the job OOMed while it had effectively the whole device free (L3 follow-up).
- Only observations naming the current attempt reconcile a job after a load (L3 follow-up).
- Unknown-estimate jobs use Rule A's direct comparison against `unknown_min_free` and are refused on overcommitted devices (L2 follow-up).
- `CapacityExceeded`, `Overcommitted` and `ForeignUsageHigh` are edge-triggered; `ForeignUsageHigh` is keyed by device and lists the blocked jobs (L2 follow-up).
- `max_lost_retries` now counts retries, like `max_oom_retries` (fails on the 4th consecutive LOST by default) (L2).
- `DeviceRemoved` of an already removed device is ignored as a duplicate (L2).

