# Simulator

`rampgate.sim` is a deterministic, discrete-event simulator with a virtual clock and seeded randomness. It is in the core and still has zero dependencies. The whole test suite and the benchmark run on it.

## Workloads

Each workload has `duration_s`, `steps_per_s`, `checkpoint_every_s`, `noise` and `reports_steps`.

- `RampPlateau(load_s, ramp_s, plateau)`: model loading (up to 30 % of the plateau), a linear ramp, then flat.
- `StepProfile(steps=((t, usage), ...))`: piecewise constant.
- `Spiky(base, spike, every_s, spike_s)`: periodic spikes.
- `EpochEval`: training with an end-of-epoch evaluation spike.

## Model

- **OOM:** when a device overflows, the job whose usage grew most recently is killed.
- **Preemption:** `Stop(PREEMPT)` saves a checkpoint for `ckpt_save_s`, then exits. No work is lost.
- **Start-up:** `JobStarted` arrives `start_delay_s` after `Start`.
- **Lost work:** an OOM rolls a job back to its last checkpoint.
- **Foreign load:** constant, or a step schedule, per device.

On real hardware, OOMs, sampling and latency behave somewhat differently; see [Real hardware](real-run/assumptions.md).

## Running

- `simulate(scenario, policy, config=..., estimator=..., admission=..., max_time=...)` returns a `Trace` with every tick's events, snapshot, decision and state, and the executor's ground truth. `restart_at` and `downtime` test dump/load and reconciliation.
- `run(scenario, strategy, policy=...)` returns a `Report`: makespan, OOM counts (first-contact separately), lost work, utilization, queue waits.
- The baselines `GreedyFree`, `DeclaredReservation` and `Serial` run on the same Engine and differ only in admission.
