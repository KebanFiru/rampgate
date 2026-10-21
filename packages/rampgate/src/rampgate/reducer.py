"""The pure transition function ``reduce(state, message, config) -> state`` (SPEC §3.4, §4, §5).

``reduce`` applies both input events and the actions a decision emitted:

- **Events** record facts. Consequences that follow directly from a fact
  (``Requeue``, ``Fail``, ``UpdateReservation``, ``Warn(UncleanStop)``) are
  put in ``state.outbox``; peaks to remember go in ``state.history_records``.
- **Transition actions** (``Start``, ``Stop``, ``MarkStable``,
  ``Requeue(START_TIMEOUT)``) move a job. Reducing one a second time is a no-op.
- **Clear-only actions** (everything else) only remove themselves from the
  outbox once ``decide`` has emitted them.

Stale, duplicate and out-of-order events are no-ops recorded in
``state.ignored_events``. Impossible ones raise `InvalidTransition`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, TypeAlias

from rampgate.actions import (
    Action,
    Fail,
    FailReason,
    GroupCompleted,
    MarkStable,
    QueueStopped,
    Requeue,
    RequeueReason,
    Start,
    Stop,
    UncleanStop,
    UpdateReservation,
    Warn,
    WarnCleared,
    is_edge_triggered,
    key_id,
    warning_key,
)
from rampgate.config import Config
from rampgate.events import (
    UNSET,
    CancelRequested,
    DeviceAdded,
    DeviceObserved,
    DeviceRemoved,
    Event,
    JobExited,
    JobObserved,
    JobStarted,
    JobSubmitted,
    JobUpdated,
    Paused,
    PreemptRequested,
    ResumeRequested,
    Unpaused,
)
from rampgate.model import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    DeviceId,
    DeviceState,
    ExitKind,
    GroupId,
    HistoryRecord,
    JobId,
    JobRuntime,
    JobState,
    State,
    StopReason,
    frozen_map,
)
from rampgate.resources import Resources
from rampgate.stability import is_peak_growth

__all__ = ["IgnoreReason", "IgnoredEvent", "InvalidTransition", "Message", "drain", "reduce"]

Message: TypeAlias = Event | Action

_STARTED = "started"
_EXITED = "exited"
_START = "start"
_STABLE = "stable"


class InvalidTransition(Exception):
    """An event or action that is impossible for the job's current attempt and state."""


class IgnoreReason(StrEnum):
    STALE_ATTEMPT = "STALE_ATTEMPT"
    DUPLICATE = "DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"


@dataclass(frozen=True, slots=True)
class IgnoredEvent:
    event: Event
    reason: IgnoreReason


def reduce(state: State, msg: Message, config: Config) -> State:
    """Apply one event or action. Pure: never mutates ``state``."""
    match msg:
        case JobSubmitted():
            return _job_submitted(state, msg)
        case JobUpdated():
            return _job_updated(state, msg)
        case JobStarted():
            return _job_started(state, msg)
        case JobObserved():
            return _job_observed(state, msg, config)
        case JobExited():
            return _job_exited(state, msg, config)
        case DeviceObserved():
            return _device_observed(state, msg, config)
        case PreemptRequested():
            return _preempt_requested(state, msg)
        case CancelRequested():
            return _cancel_requested(state, msg)
        case ResumeRequested():
            return _resume_requested(state, msg)
        case Paused() | Unpaused():
            paused = isinstance(msg, Paused)
            if state.paused == paused:
                return _ignore(state, msg, IgnoreReason.DUPLICATE)
            return replace(state, paused=paused, pause_reason=None)
        case DeviceAdded():
            return _device_added(state, msg)
        case DeviceRemoved():
            return _device_removed(state, msg)
        case Start():
            return _start(state, msg, config)
        case Stop():
            return _stop(state, msg)
        case MarkStable():
            return _mark_stable(state, msg)
        case Requeue(reason=RequeueReason.START_TIMEOUT):
            return _start_timeout(state, msg)
        case GroupCompleted():
            return _group_completed(state, msg)
        case Warn():
            return _warn(state, msg)
        case WarnCleared():
            return _warn_cleared(state, msg)
        case Requeue() | Fail() | UpdateReservation():
            return _clear(state, msg)
    raise TypeError(f"not an event or action: {msg!r}")


def drain(state: State) -> tuple[State, tuple[IgnoredEvent, ...], tuple[HistoryRecord, ...]]:
    """Take the ignored-event log and pending history records out of the state."""
    return (
        replace(state, ignored_events=(), history_records=()),
        state.ignored_events,
        state.history_records,
    )


def _ignore(state: State, event: Event, reason: IgnoreReason) -> State:
    return replace(state, ignored_events=(*state.ignored_events, IgnoredEvent(event, reason)))


def _job(state: State, job_id: JobId) -> JobRuntime:
    rt = state.jobs.get(job_id)
    if rt is None:
        raise InvalidTransition(f"unknown job {job_id!r}")
    return rt


def _put(state: State, rt: JobRuntime) -> State:
    return replace(state, jobs=frozen_map({**state.jobs, rt.id: rt}))


def _emit(state: State, *actions: Action) -> State:
    return replace(state, outbox=(*state.outbox, *actions))


def _record(state: State, records: Iterable[HistoryRecord]) -> State:
    return replace(state, history_records=(*state.history_records, *records))


def _clear(state: State, action: Action) -> State:
    action_id = action.action_id
    return replace(state, outbox=tuple(a for a in state.outbox if a.action_id != action_id))


def _bad(rt: JobRuntime, what: str) -> InvalidTransition:
    return InvalidTransition(f"job {rt.id!r} (attempt {rt.attempt}, {rt.state}): {what}")


def _fresh_attempt() -> dict[str, Any]:
    """Field values that ``Start`` resets for a new attempt."""
    return {
        "admitted_free": frozen_map(),
        "admitted_estimate": frozen_map(),
        "starting_since": None,
        "warm_start": None,
        "usage": frozen_map(),
        "usage_samples": frozen_map(),
        "peak": frozen_map(),
        "last_obs_t": frozen_map(),
        "last_peak_growth_t": None,
        "steps_baseline": None,
        "steps_latest": None,
        "epoch_baseline": None,
        "epoch_latest": None,
        "reservation": frozen_map(),
    }


def _end_attempt(rt: JobRuntime, new_state: JobState, **changes: Any) -> JobRuntime:
    """Leave the active states: release usage and reservations, forget the stop intent."""
    return replace(
        rt,
        state=new_state,
        stop_intent=None,
        usage=frozen_map(),
        usage_samples=frozen_map(),
        reservation=frozen_map(),
        **changes,
    )


def _requeue(rt: JobRuntime, **changes: Any) -> JobRuntime:
    """Back to PENDING for the next attempt, keeping group and position."""
    return _end_attempt(rt, JobState.PENDING, attempt=rt.attempt + 1, **changes)


def _estimates(rt: JobRuntime) -> Mapping[DeviceId, Resources | None]:
    return frozen_map({d: rt.prev_estimate(d) for d in rt.job.devices})


def _observed_peaks(rt: JobRuntime, *, lower_bound: bool) -> list[HistoryRecord]:
    """History records for every device that was actually observed in this attempt."""
    return [
        HistoryRecord(rt.id, d, rt.peak.get(d, Resources()), lower_bound, rt.attempt)
        for d in rt.job.devices
        if d in rt.last_obs_t
    ]


def _job_submitted(state: State, ev: JobSubmitted) -> State:
    job = ev.job
    existing = state.jobs.get(job.id)
    if existing is not None:
        if existing.job == job:
            return _ignore(state, ev, IgnoreReason.DUPLICATE)
        raise InvalidTransition(f"job {job.id!r} was already submitted with different content")
    unknown = sorted(d for d in job.placement if d not in state.devices)
    if unknown:
        raise InvalidTransition(f"job {job.id!r} is placed on unknown devices {unknown}")
    rt = JobRuntime(
        job=job, state=JobState.PENDING, attempt=1, submit_seq=state.next_seq, submit_t=ev.t
    )
    state = _reopen_group(state, job.group)
    return replace(_put(state, rt), next_seq=state.next_seq + 1)


def _reopen_group(state: State, group: GroupId | None) -> State:
    if group is None or group not in state.completed_groups:
        return state
    return replace(state, completed_groups=state.completed_groups - {group})


def _job_updated(state: State, ev: JobUpdated) -> State:
    rt = _job(state, ev.job_id)
    if rt.state in TERMINAL_STATES:
        raise _bad(rt, "cannot update a finished job")
    changes: dict[str, Any] = {}
    for name in ("group", "position", "priority"):
        value = getattr(ev, name)
        if value is not UNSET:
            changes[name] = value
    for name in ("placement", "force"):
        value = getattr(ev, name)
        if value is not UNSET:
            if rt.state not in (JobState.PENDING, JobState.PREEMPTED):
                raise _bad(rt, f"{name} can only change while PENDING or PREEMPTED")
            changes[name] = value
    if not changes:
        return state
    job = replace(rt.job, **changes)
    unknown = sorted(d for d in job.placement if d not in state.devices)
    if unknown:
        raise _bad(rt, f"placement names unknown devices {unknown}")
    learned = frozen_map({d: r for d, r in rt.learned.items() if d in job.placement})
    state = _reopen_group(state, job.group)
    return _put(state, replace(rt, job=job, learned=learned))


def _job_started(state: State, ev: JobStarted) -> State:
    rt = _job(state, ev.job_id)
    if ev.attempt < rt.attempt:
        return _ignore(state, ev, IgnoreReason.STALE_ATTEMPT)
    if ev.attempt > rt.attempt:
        raise _bad(rt, f"started for future attempt {ev.attempt}")
    if (ev.attempt, _STARTED) in rt.applied:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if (ev.attempt, _EXITED) in rt.applied:
        return _ignore(state, ev, IgnoreReason.OUT_OF_ORDER)
    if rt.state not in (JobState.STARTING, JobState.STOPPING):
        raise _bad(rt, "started but it was never told to start")
    rt = replace(
        rt,
        state=JobState.WARMING if rt.state is JobState.STARTING else JobState.STOPPING,
        warm_start=ev.t,
        has_run=True,
        applied=rt.applied | {(ev.attempt, _STARTED)},
    )
    return _put(state, rt)


def _job_observed(state: State, ev: JobObserved, config: Config) -> State:
    rt = _job(state, ev.job_id)
    attempt = rt.attempt if ev.attempt is None else ev.attempt
    if attempt < rt.attempt:
        return _ignore(state, ev, IgnoreReason.STALE_ATTEMPT)
    if attempt > rt.attempt:
        raise _bad(rt, f"observed for future attempt {attempt}")
    if ev.device_id not in rt.job.placement:
        raise _bad(rt, f"observed on {ev.device_id!r}, which is not in its placement")
    if rt.state not in ACTIVE_STATES:
        if (attempt, _EXITED) in rt.applied or (ev.attempt is None and rt.has_run):
            return _ignore(state, ev, IgnoreReason.STALE_ATTEMPT)
        raise _bad(rt, "observed but it was never started")
    if ev.attempt is None and rt.last_exit_t is not None and ev.t <= rt.last_exit_t:
        return _ignore(state, ev, IgnoreReason.STALE_ATTEMPT)
    last = rt.last_obs_t.get(ev.device_id)
    if last is not None and ev.t == last:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if last is not None and ev.t < last:
        return _ignore(state, ev, IgnoreReason.OUT_OF_ORDER)

    dev = ev.device_id
    old_peak = rt.peak.get(dev, Resources())
    new_peak = old_peak.max(ev.usage)
    growth_t = rt.last_peak_growth_t
    if is_peak_growth(old_peak, new_peak, config.growth_tol):
        growth_t = ev.t if growth_t is None else max(growth_t, ev.t)

    baseline, latest = _rebaseline(rt.steps_baseline, rt.steps_latest, ev.steps)
    epoch_baseline, epoch_latest = _rebaseline(rt.epoch_baseline, rt.epoch_latest, ev.epoch)

    window_start = ev.t - config.foreign_window_s
    kept = tuple(x for x in rt.usage_samples.get(dev, ()) if x[0] >= window_start)
    rt = replace(
        rt,
        usage=frozen_map({**rt.usage, dev: ev.usage}),
        usage_samples=frozen_map({**rt.usage_samples, dev: (*kept, (ev.t, ev.usage))}),
        peak=frozen_map({**rt.peak, dev: new_peak}),
        last_obs_t=frozen_map({**rt.last_obs_t, dev: ev.t}),
        last_peak_growth_t=growth_t,
        steps_baseline=baseline,
        steps_latest=latest,
        reports_steps=rt.reports_steps or ev.steps is not None,
        epoch_baseline=epoch_baseline,
        epoch_latest=epoch_latest,
    )

    reserved = rt.reservation.get(dev)
    if reserved is not None and new_peak.exceeds(reserved):
        grown = reserved.max(new_peak.scale(config.safety_margin))
        rt = replace(rt, reservation=frozen_map({**rt.reservation, dev: grown}))
        update = UpdateReservation(rt.id, rt.attempt, rt.reservation)
        state = _record(state, _running_records(rt, [dev]))
        outbox = tuple(
            a
            for a in state.outbox
            if not (
                isinstance(a, UpdateReservation) and (a.job_id, a.attempt) == (rt.id, rt.attempt)
            )
        )
        state = replace(state, outbox=(*outbox, update))
    return _put(state, rt)


def _rebaseline(
    baseline: int | None, latest: int | None, value: int | None
) -> tuple[int | None, int | None]:
    """Track a cumulative counter per attempt (SPEC §3.5): first value and decreases re-baseline."""
    if value is None:
        return baseline, latest
    if latest is None or value < latest:
        baseline = value
    return baseline, value


def _job_exited(state: State, ev: JobExited, config: Config) -> State:
    rt = _job(state, ev.job_id)
    if ev.attempt < rt.attempt:
        return _ignore(state, ev, IgnoreReason.STALE_ATTEMPT)
    if ev.attempt > rt.attempt:
        raise _bad(rt, f"exited for future attempt {ev.attempt}")
    if (ev.attempt, _EXITED) in rt.applied:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if rt.state not in ACTIVE_STATES:
        raise _bad(rt, "exited but it was never started")
    last_exit = ev.t if rt.last_exit_t is None else max(rt.last_exit_t, ev.t)
    rt = replace(
        rt, applied=rt.applied | {(ev.attempt, _EXITED)}, has_run=True, last_exit_t=last_exit
    )
    if rt.stop_intent is not None:
        return _exit_while_stopping(state, rt, ev, rt.stop_intent, config)

    match ev.kind:
        case ExitKind.SUCCESS:
            state = _record(state, _observed_peaks(rt, lower_bound=False))
            return _put(state, _end_attempt(rt, JobState.SUCCEEDED))
        case ExitKind.FAILURE:
            state = _emit(state, Fail(rt.id, rt.attempt, FailReason.EXITED_FAILURE))
            return _put(state, _end_attempt(rt, JobState.FAILED))
        case ExitKind.OOM:
            return _oom(state, rt, ev, config)
        case ExitKind.LOST:
            return _lost(state, rt, config)
        case ExitKind.PREEMPTED:
            return _put(state, _end_attempt(rt, JobState.PREEMPTED))
        case ExitKind.CANCELLED:
            return _put(state, _end_attempt(rt, JobState.CANCELLED))
    raise AssertionError(ev.kind)  # pragma: no cover


def _exit_while_stopping(
    state: State, rt: JobRuntime, ev: JobExited, intent: StopReason, config: Config
) -> State:
    """SPEC §3.4: the real outcome decides what we learn, the stop intent what happens next.

    Applies whenever a stop was requested, whether or not the ``Stop`` was sent yet.
    A stopped job is never re-queued and never FAILED (I13).
    """
    on_stop = JobState.PREEMPTED if intent is StopReason.PREEMPT else JobState.CANCELLED
    match ev.kind:
        case ExitKind.SUCCESS:
            state = _record(state, _observed_peaks(rt, lower_bound=False))
            return _put(state, _end_attempt(rt, JobState.SUCCEEDED))
        case ExitKind.OOM if intent is StopReason.PREEMPT:
            learned, records, _ = _learn_from_oom(state, rt, ev, config)
            rt = replace(rt, oom_count=rt.oom_count + 1, learned=learned)
            return _put(_record(state, records), _end_attempt(rt, JobState.PREEMPTED))
        case ExitKind.OOM:
            state = _record(state, _observed_peaks(rt, lower_bound=True))
            return _put(state, _end_attempt(rt, JobState.CANCELLED))
        case ExitKind.FAILURE:
            if intent is StopReason.PREEMPT:
                state = _emit(state, Warn(UncleanStop(rt.id, rt.attempt)))
            return _put(state, _end_attempt(rt, on_stop))
        case ExitKind.PREEMPTED:
            return _put(state, _end_attempt(rt, JobState.PREEMPTED))
        case ExitKind.CANCELLED:
            return _put(state, _end_attempt(rt, JobState.CANCELLED))
        case ExitKind.LOST:
            return _put(state, _end_attempt(rt, on_stop))
    raise AssertionError(ev.kind)  # pragma: no cover


def _room_at_oom(
    state: State, rt: JobRuntime, d: DeviceId, t: float, config: Config
) -> tuple[Resources, bool]:
    """``C - F - sum(usage of the other active managed jobs)`` on ``d`` when ``rt`` died.

    ``F`` and each neighbour's usage are the max over the last ``foreign_window_s``,
    so a spike between two samples still counts against the room. The flag says
    whether the room is *known*: False if an active neighbour has no usage sample
    in the window (e.g. it began in the same instant), whose usage then counts as 0.
    """
    window = config.foreign_window_s
    taken = state.devices[d].foreign_at(t, window)
    known = True
    for other in state.jobs_on(d):
        if other.id != rt.id and other.state in ACTIVE_STATES:
            taken = taken + other.usage_at(d, t, window)
            samples = other.usage_samples.get(d, ())
            known = known and any(s_t >= t - window for s_t, _ in samples)
    return state.devices[d].capacity.sub(taken, clamp=True), known


def _learn_from_oom(
    state: State, rt: JobRuntime, ev: JobExited, config: Config
) -> tuple[Mapping[DeviceId, Resources], list[HistoryRecord], bool]:
    """Rule O, per placement device and affected dimension.

    ``lower_bound = max(peak, previous estimate, room_at_oom)``, plus
    ``unknown_min_free`` on devices admitted without an estimate, and
    ``new_estimate = min(lower_bound * oom_growth, capacity)``, never below the
    previous estimate. Unaffected dimensions keep ``max(peak, previous estimate)``.
    ``room_at_oom`` is what the device had left for the job when it died (see
    `_room_at_oom`); ``admitted_free`` is diagnostic only.

    Returns the new learned estimates, lower-bound history records, and whether
    the evidence says the job cannot fit the device: it OOMed in a dimension in
    which it had (almost) the whole device to itself at that moment,
    ``room_at_oom >= capacity * (1 - foreign_tolerance)``, and the room is known
    (no active neighbour without a recent usage sample). Where the estimate came
    from does not matter for that decision.
    """
    learned: dict[DeviceId, Resources] = {}
    records: list[HistoryRecord] = []
    exceeds_capacity = False
    for d in rt.job.devices:
        cap = state.devices[d].capacity
        prev = rt.prev_estimate(d) or Resources()
        admitted = rt.admitted_estimate.get(d)
        if admitted is not None:
            prev = prev.max(admitted)
        base = rt.peak.get(d, Resources()).max(prev)
        room, room_known = _room_at_oom(state, rt, d, ev.t, config)
        full_lb = base.max(room)
        if d in rt.admitted_estimate and rt.admitted_estimate[d] is None:
            full_lb = full_lb.max(config.unknown_min_free_for(cap))
        dims = ev.oom_dims if ev.oom_dims is not None else sorted(full_lb.keys() | cap.keys())
        new = dict(base)
        lower = dict(base)
        for dim in dims:
            lower[dim] = full_lb[dim]
            grown = full_lb[dim] * config.oom_growth
            if cap[dim] > 0:
                grown = min(grown, cap[dim])
            new[dim] = max(prev[dim], grown)
            whole = room[dim] >= cap[dim] * (1 - config.foreign_tolerance)
            if cap[dim] > 0 and whole and room_known:
                exceeds_capacity = True
        learned[d] = Resources(new)
        if any(lower.values()):
            records.append(HistoryRecord(rt.id, d, Resources(lower), True, rt.attempt))
    return frozen_map(learned), records, exceeds_capacity


def _oom(state: State, rt: JobRuntime, ev: JobExited, config: Config) -> State:
    learned, records, exceeds_capacity = _learn_from_oom(state, rt, ev, config)
    rt = replace(rt, oom_count=rt.oom_count + 1, learned=learned)
    state = _record(state, records)
    if exceeds_capacity:
        state = _emit(state, Fail(rt.id, rt.attempt, FailReason.EXCEEDS_DEVICE_CAPACITY))
        return _put(state, _end_attempt(rt, JobState.FAILED))
    if rt.oom_count > config.max_oom_retries:
        state = _emit(state, Fail(rt.id, rt.attempt, FailReason.OOM_RETRIES_EXHAUSTED))
        return _put(state, _end_attempt(rt, JobState.FAILED))
    rt = _requeue(rt)
    state = _emit(state, Requeue(rt.id, rt.attempt, _estimates(rt), RequeueReason.OOM))
    return _put(state, rt)


def _lost(state: State, rt: JobRuntime, config: Config) -> State:
    """An involuntary preemption: no OOM accounting, no history (the peak is incomplete)."""
    rt = replace(rt, lost_count=rt.lost_count + 1)
    if rt.lost_count > config.max_lost_retries:
        state = _emit(state, Fail(rt.id, rt.attempt, FailReason.LOST_RETRIES_EXHAUSTED))
        return _put(state, _end_attempt(rt, JobState.FAILED))
    rt = _requeue(rt)
    state = _emit(state, Requeue(rt.id, rt.attempt, _estimates(rt), RequeueReason.LOST))
    return _put(state, rt)


def _preempt_requested(state: State, ev: PreemptRequested) -> State:
    rt = _job(state, ev.job_id)
    if rt.state is JobState.PENDING:
        return _put(state, replace(rt, state=JobState.PREEMPTED))
    if rt.state is JobState.PREEMPTED or rt.stop_intent is not None:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if rt.state in ACTIVE_STATES:
        return _put(state, replace(rt, stop_intent=StopReason.PREEMPT))
    raise _bad(rt, "cannot preempt a finished job")


def _cancel_requested(state: State, ev: CancelRequested) -> State:
    rt = _job(state, ev.job_id)
    if rt.state in (JobState.PENDING, JobState.PREEMPTED):
        return _put(state, replace(rt, state=JobState.CANCELLED))
    if rt.state is JobState.CANCELLED or rt.stop_intent is StopReason.CANCEL:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if rt.state in ACTIVE_STATES:
        return _put(state, replace(rt, stop_intent=StopReason.CANCEL))
    raise _bad(rt, "cannot cancel a finished job")


def _resume_requested(state: State, ev: ResumeRequested) -> State:
    rt = _job(state, ev.job_id)
    if rt.state is JobState.PENDING:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if rt.state is not JobState.PREEMPTED:
        raise _bad(rt, "only a PREEMPTED job can be resumed")
    used = (rt.attempt, _EXITED) in rt.applied
    return _put(state, replace(rt, state=JobState.PENDING, attempt=rt.attempt + int(used)))


def _device_added(state: State, ev: DeviceAdded) -> State:
    existing = state.devices.get(ev.device.id)
    if existing is not None:
        if existing.device == ev.device:
            return _ignore(state, ev, IgnoreReason.DUPLICATE)
        raise InvalidTransition(f"device {ev.device.id!r} already exists with different content")
    return replace(
        state,
        devices=frozen_map({**state.devices, ev.device.id: DeviceState(ev.device)}),
        removed_devices=state.removed_devices - {ev.device.id},
    )


def _device_removed(state: State, ev: DeviceRemoved) -> State:
    if ev.device_id in state.removed_devices:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if ev.device_id not in state.devices:
        raise InvalidTransition(f"unknown device {ev.device_id!r}")
    users = [rt.id for rt in state.jobs_on(ev.device_id) if rt.state not in TERMINAL_STATES]
    if users:
        raise InvalidTransition(
            f"device {ev.device_id!r} is still used by unfinished jobs {users}; "
            "cancel them or move their placement first"
        )
    return replace(
        state,
        devices=frozen_map({k: v for k, v in state.devices.items() if k != ev.device_id}),
        removed_devices=state.removed_devices | {ev.device_id},
    )


def _device_observed(state: State, ev: DeviceObserved, config: Config) -> State:
    ds = state.devices.get(ev.device_id)
    if ds is None:
        raise InvalidTransition(f"unknown device {ev.device_id!r}")
    if ds.last_obs_t is not None and ev.t == ds.last_obs_t:
        return _ignore(state, ev, IgnoreReason.DUPLICATE)
    if ds.last_obs_t is not None and ev.t < ds.last_obs_t:
        return _ignore(state, ev, IgnoreReason.OUT_OF_ORDER)
    managed = Resources()
    for rt in state.jobs_on(ev.device_id):
        if rt.state in ACTIVE_STATES:
            managed = managed + rt.usage.get(ev.device_id, Resources())
    foreign = ev.used_total.sub(managed, clamp=True)
    window_start = ev.t - config.foreign_window_s
    kept = tuple(s for s in ds.foreign_samples if s[0] >= window_start)
    samples = (*kept, (ev.t, foreign))
    ds = replace(ds, used_total=ev.used_total, last_obs_t=ev.t, foreign_samples=samples)
    return replace(state, devices=frozen_map({**state.devices, ev.device_id: ds}))


def _start(state: State, a: Start, config: Config) -> State:
    rt = _job(state, a.job_id)
    if a.attempt < rt.attempt or (a.attempt, _START) in rt.applied:
        return state
    if a.attempt > rt.attempt:
        raise _bad(rt, f"Start for future attempt {a.attempt}")
    if rt.state is not JobState.PENDING:
        raise _bad(rt, "only a PENDING job can start")
    if set(a.placement) != set(rt.job.placement):
        raise _bad(rt, f"Start placement {list(a.placement)} differs from the job's")
    fields = _fresh_attempt() | {
        "state": JobState.STARTING,
        "admitted_free": frozen_map(a.admitted_free),
        "starting_since": a.issued_at,
        "admitted_estimate": frozen_map(a.estimate),
        "applied": rt.applied | {(a.attempt, _START)},
    }
    rt = replace(rt, **fields)
    return _put(state, rt)


def _stop(state: State, a: Stop) -> State:
    rt = _job(state, a.job_id)
    if a.attempt < rt.attempt or (a.attempt == rt.attempt and rt.state is JobState.STOPPING):
        return state
    if a.attempt > rt.attempt:
        raise _bad(rt, f"Stop for future attempt {a.attempt}")
    if (a.attempt, _EXITED) in rt.applied:
        return state
    if rt.state not in ACTIVE_STATES or rt.stop_intent is not a.reason:
        raise _bad(rt, f"Stop({a.reason}) without a matching request")
    return _put(state, replace(rt, state=JobState.STOPPING))


def _mark_stable(state: State, a: MarkStable) -> State:
    rt = _job(state, a.job_id)
    if a.attempt < rt.attempt or (a.attempt, _STABLE) in rt.applied:
        return state
    if a.attempt > rt.attempt:
        raise _bad(rt, f"MarkStable for future attempt {a.attempt}")
    if rt.state is not JobState.WARMING:
        raise _bad(rt, "only a WARMING job can become STABLE")
    if set(a.reservation) != set(rt.job.placement):
        raise _bad(rt, "MarkStable reservation must cover exactly the placement devices")
    rt = replace(
        rt,
        state=JobState.STABLE,
        reservation=frozen_map(a.reservation),
        lost_count=0,
        applied=rt.applied | {(a.attempt, _STABLE)},
    )
    state = _record(state, _running_records(rt, rt.job.devices))
    return _put(_clear(state, a), rt)


def _running_records(rt: JobRuntime, devices: Iterable[DeviceId]) -> list[HistoryRecord]:
    """Rule H: a STABLE job's observed peak, so siblings get an estimate before it finishes.

    Written again with the same ``record_id`` when the reservation grows; stores
    keep the per-dimension max of a record id.
    """
    return [
        HistoryRecord(rt.id, d, rt.peak[d], True, rt.attempt, running=True)
        for d in devices
        if d in rt.last_obs_t and not rt.peak[d].is_zero
    ]


def _start_timeout(state: State, a: Requeue) -> State:
    rt = _job(state, a.job_id)
    if a.next_attempt <= rt.attempt:
        return state
    if a.next_attempt != rt.attempt + 1 or rt.state is not JobState.STARTING:
        raise _bad(rt, f"Requeue(START_TIMEOUT) to attempt {a.next_attempt}")
    intent = rt.stop_intent
    rt = _requeue(rt)
    if intent is StopReason.PREEMPT:
        rt = replace(rt, state=JobState.PREEMPTED)
    elif intent is StopReason.CANCEL:
        rt = replace(rt, state=JobState.CANCELLED)
    return _put(_clear(state, a), rt)


def _group_completed(state: State, a: GroupCompleted) -> State:
    return replace(_clear(state, a), completed_groups=state.completed_groups | {a.group})


def _warn(state: State, a: Warn) -> State:
    state = _clear(state, a)
    kind, key = warning_key(a.warning)
    if is_edge_triggered(a.warning):
        active = {**state.active_warnings, key_id(kind, key): a.warning}
        state = replace(state, active_warnings=frozen_map(active))
    else:
        state = replace(state, warned=state.warned | {key_id(kind, key)})
    if isinstance(a.warning, QueueStopped):
        state = replace(state, paused=True, pause_reason=a.warning)
    return state


def _warn_cleared(state: State, a: WarnCleared) -> State:
    k = key_id(a.kind, a.key)
    active = {i: w for i, w in state.active_warnings.items() if i != k}
    return replace(state, active_warnings=frozen_map(active))
