# Algorithm

`decide(snapshot, now)` is pure: the same state and time always give the same decision. The full definition is in the repository's `SPEC.md` §5; this page is the short version. Every threshold is a field of `Config`.

## Estimates

For job *j* on device *d*, the estimate is `E = max(estimator result, declared hint, what Rule O learned)`, or unknown. An estimate is **trusted** only when it comes from complete observations of this job's template (its fingerprint), not from a hint or a lower bound.

## Rule P: probation

At most one job **without a trusted estimate** is STARTING or WARMING on a device (`probation="untrusted_only"`, the default; `"all"` serializes every ramp). While it warms, it holds the device against every start, trusted jobs too, because its peak is unknown.

## Rule A: admission

A candidate is admitted on *d* if, in every dimension:

```
F + Σ reservations of STABLE jobs + Σ W of warming jobs + E' ≤ capacity
```

- `F` is the foreign usage (see below).
- `W` is a warming job's estimate with a margin, or its current usage when that is larger.
- `E'` is the candidate's estimate × `trusted_margin` (1.05) when the estimate is trusted and backed by `min_trusted_observations` (3), else × `safety_margin` (1.15).
- An unknown estimate needs `unknown_min_free` (10 % of capacity by default) to be free.

Gang jobs are admitted on all their devices or none.

## Rule S: stability

A WARMING job becomes STABLE once it has warmed for `min_warmup_s`, and either:

- its peak has not grown (by more than `growth_tol`) for `stable_window_s`, its step count reached `stable_min_steps` (if it reports steps), and it completed an epoch (if it reports epochs); or
- it has warmed for `max_warmup_s`.

At that point it is reserved at `peak × safety_margin`.

**Usage floor.** A job that reports no steps does not become STABLE on a device before its usage there exceeded the device's `stable_min_usage`. Not even `max_warmup_s` overrides this. Without the floor, a job that spends its first minute loading data, with no GPU memory yet, would be "stable" at zero and reserved nothing. At `max_warmup_s` such a job raises `NoUsageObserved` instead.

## Rule O: OOM feedback

An OOM exit grows the estimate from the room the job actually had at the OOM: by `oom_growth` (1.25), capped at capacity. The job is re-queued to resume from its checkpoint (`Requeue(OOM)`), and gets at most `max_oom_retries` retries. A job is failed as too big for the device only if it OOMed while it had the whole device.

## Rule V: vacate

When a `MarkStable` or `UpdateReservation` pushes a device's managed load above `capacity × (1 − vacate_headroom)`, rampgate stops the most recently admitted **STABLE** job (never the last job on the device) and re-queues it (`Requeue(VACATED)`). Preempting a STABLE job is cheap: it has finished growing. On real hardware, its memory was back about 0.15 s after it exited.

## Isolation retry

A job that OOMed before it became stable retries alone: it holds the devices of its placement until no other managed job is active there. A device is held by at most one such job, the one with the oldest OOM. An isolation hold takes precedence over head-of-line holds.

## Foreign usage

`F` is what the device uses beyond the managed jobs' latest usage: other processes, the driver's own reservation. It is the maximum over the last `foreign_window_s`. Foreign usage blocks admission (`BlockedByForeignUsage`) but never triggers Rule V on its own.

## Ordering

The [policy](policies.md) proposes candidates in order. With head-of-line blocking, a candidate that is not admitted on *d* closes *d* for later candidates in the same tick.
