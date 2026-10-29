# Restarting and reconciling

rampgate's `Engine` keeps all scheduling state in memory. To survive a restart of
your scheduler process, persist `engine.dump()` and restore it with `Engine.load`.
While your process was down, the world kept moving: jobs finished, crashed or
were killed. **Reconciliation** is how you tell the engine what happened.

## 1. Persist

```python
import json

with open("rampgate.json", "w") as f:
    json.dump(engine.dump(), f)
```

Persist **after** your executor has carried out the actions of a tick. `tick`
already counts its actions as applied (a `Start` makes the job STARTING, a `Stop`
makes it STOPPING), so a dump taken between `tick` and executing its actions
records work that may never have happened. Section 3 covers the one case where
that matters.

The dump is versioned (`"format": "rampgate.engine"`, `"schema_version"`) and
contains the whole state, including actions and history records that were not
emitted yet. A history record can therefore reach your `HistoryStore` twice (handed
over by a tick, then the process dies before the next dump): make `put` idempotent
by its `record_id`. It does **not** contain the policy, the estimator or the config: you
pass those again on load. Your `HistoryStore` is yours to persist (TrainQ keeps it
in SQLite); rampgate assumes it is durable.

## 2. Load

```python
with open("rampgate.json") as f:
    engine = Engine.load(json.load(f), policy=policy, estimator=estimator, config=config)
```

A dump written by a newer rampgate raises `UnsupportedSchema`; older dumps are
upgraded automatically.

## 3. Reconcile

Every job that was STARTING, WARMING, STABLE or STOPPING when the dump was taken
is *unreconciled*: the engine does not know whether its process still exists.

```python
for job_id in engine.unreconciled():
    rt = engine.state.jobs[job_id]
    container = find_container(job_id, rt.attempt)    # e.g. by the rampgate.* labels
    if container is None:
        engine.on_exited(job_id, rt.attempt, ExitKind.LOST, t=now)
    elif container.exited:
        engine.on_exited(job_id, rt.attempt, classify_exit(container), t=container.finished_at)
    elif container.running:
        engine.on_started(job_id, rt.attempt, t=container.started_at)  # no-op if already known
        engine.confirm(job_id, rt.attempt)
        if rt.state is JobState.STOPPING:
            # The Stop may never have reached the container: send it again.
            executor.apply(Stop(job_id, rt.attempt, rt.stop_intent, config.preempt_grace_s))
```

**STOPPING jobs need care.** The engine never emits a `Stop` twice: once it is
reduced, the job is STOPPING and waits for its exit. If your process died after
`tick` but before the executor sent the stop, the container keeps running. So for
an unreconciled STOPPING job whose container still runs, send the stop again
yourself, as above. Its `action_id` is deterministic, so an executor that applies
actions idempotently can safely see it twice.

A job leaves `unreconciled()` when it is reconciled: an `on_started` or
`on_exited` (including `LOST`) the engine did not already have, a `JobObserved`
that names the job's current `attempt`, or `engine.confirm(job_id, attempt)`.
Facts the engine already had are ignored as duplicates and do not count.

If a job is still unreconciled `reconcile_grace_s` (default 60 s) after the first
tick following the load, `tick` emits `Warn(Unreconciled(job_id, attempt))`, and
`WarnCleared` once it is reconciled. rampgate never declares a job LOST by itself:
only you know whether the process is gone.

What the engine does with each answer:

| You report | Result |
|---|---|
| `on_exited(..., LOST)` | re-queued with `resume=True` (no OOM accounting; bounded by `max_lost_retries`) |
| `on_exited(..., SUCCESS / FAILURE / OOM)` | the normal lifecycle rules ([Lifecycle](concepts.md#lifecycle), Rule O) |
| an observation of the current attempt | the job simply continues |
| nothing, for a job that was STARTING | after `start_timeout_s` it gets `Requeue(START_TIMEOUT)` |

A stop *request* (`preempt` / `cancel`) that no tick had turned into a `Stop` yet
survives the restart: the next `tick` emits the `Stop`. Exits of stopping jobs
still follow the STOPPING table.

## 4. Continue

Call `engine.tick(now)` as usual. `now` must not be earlier than the last tick
before the dump (`engine.now`); use the same clock you used before the restart.

## Example

```python
engine = Engine.load(data, policy=FifoPolicy())
print(engine.unreconciled())            # ['a', 'b']
engine.confirm("a", 1)                  # or an observation naming attempt 1
engine.on_exited("b", 1, ExitKind.LOST, t=103.0)
assert engine.unreconciled() == []
decision = engine.tick(104.0)           # Requeue(b, LOST) and a Start(b, attempt=2, resume=True)
```
