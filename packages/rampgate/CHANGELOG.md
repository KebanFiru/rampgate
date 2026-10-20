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
