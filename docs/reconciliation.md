# Restarting and reconciling

rampgate's `Engine` keeps all scheduling state in memory. To survive a restart of
your scheduler process, persist `engine.dump()` and restore it with `Engine.load`.
While your process was down, the world kept moving: jobs finished, crashed or
were killed. **Reconciliation** is how you tell the engine what happened.

## 1. Persist

```python
import json

with open("rampgate.json", "w") as f:
    json.dump(engine.dump(), f)        # call between ticks, as often as you like
```

The dump is versioned (`"format": "rampgate.engine"`, `"schema_version"`) and
contains the whole state, including actions and history records that were not
emitted yet. It does **not** contain the policy, the estimator or the config: you
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
        # usage samples (engine.observe) will follow from your probes
```

A job leaves `unreconciled()` as soon as any fact about it is applied: an
`on_started`, an `on_exited` (including `LOST`), or a usage observation for its
current attempt. Facts the engine already had are ignored as duplicates and do not
count, so for a job that was already WARMING, report an observation or its exit.

What the engine does with each answer:

| You report | Result |
|---|---|
| `on_exited(..., LOST)` | re-queued with `resume=True` (no OOM accounting; bounded by `max_lost_retries`) |
| `on_exited(..., SUCCESS / FAILURE / OOM)` | the normal lifecycle rules (SPEC §3.4, Rule O) |
| an observation of the current attempt | the job simply continues |
| nothing, for a job that was STARTING | after `start_timeout_s` it gets `Requeue(START_TIMEOUT)` |

A pending `Stop` request survives the restart: the next `tick` emits the `Stop`
again if it had not been reduced yet, and exits still follow the STOPPING table.

## 4. Continue

Call `engine.tick(now)` as usual. `now` must not be earlier than the last tick
before the dump (`engine.now`); use the same clock you used before the restart.

## Example

```python
engine = Engine.load(data, policy=FifoPolicy())
print(engine.unreconciled())            # ['a', 'b']
engine.observe(JobObserved("a", "gpu:0", t=103.0, usage=Resources(vram=40e9), attempt=1))
engine.on_exited("b", 1, ExitKind.LOST, t=103.0)
assert engine.unreconciled() == []
decision = engine.tick(104.0)           # Requeue(b, LOST) and a Start(b, attempt=2, resume=True)
```
