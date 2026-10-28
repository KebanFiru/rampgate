# Ordering policies

A policy decides the **order** in which PENDING jobs are tried. The rules decide whether each one fits.

| Policy | Order | Head-of-line blocking |
|---|---|---|
| `FifoPolicy(strict=True)` | submission order | yes |
| `FifoPolicy(strict=False)` | submission order | no: a job that does not fit is skipped |
| `PriorityPolicy(aging_per_s=0.0, head_of_line_blocking=True)` | highest `priority` first; waiting adds `aging_per_s` per second | configurable |
| `StagePolicy(groups_order, backfill=False)` | group by group: a group starts only after the earlier ones completed (with `backfill`, later groups may use idle devices) | no |

**Head-of-line blocking (per device).** A candidate that is not admitted on device *d* closes *d* for later candidates in the same tick. Other devices stay open. A job skipped this way is explained with `DeviceHeldFor(d, holder)`. An isolation retry's hold takes precedence: a candidate waiting for a device's isolation holder never closes the device for that holder.

`StagePolicy` groups can stop the queue when a job fails (`QueueStopped` pauses the engine until you `unpause`).

Custom policies implement `OrderingPolicy`: `candidates(state, now)` and `explain_blocked(state, job_id)`.
