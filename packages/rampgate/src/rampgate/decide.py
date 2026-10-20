"""``decide(snapshot, now, config, policy) -> Decision``: pure admission (SPEC §5).

Work is done in a fixed order on a local working copy of the state, so later
steps see the effect of earlier ones within the same tick:

1. actions waiting in the outbox;
2. ``Requeue(START_TIMEOUT)`` for jobs stuck in STARTING;
3. ``Stop`` for jobs with a stop request;
4. ``MarkStable`` (Rule S);
5. ``Warn(QueueStopped)`` for STOP_QUEUE failures (it pauses admission);
6. admission: Rules P and A in policy order, with per-device head-of-line
   blocking, plus the candidate warnings;
7. ``Warn(Overcommitted)``;
8. ``GroupCompleted``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from rampgate.actions import (
    Action,
    CapacityExceeded,
    ForeignUsageHigh,
    GroupCompleted,
    MarkStable,
    Overcommitted,
    QueueStopped,
    Requeue,
    RequeueReason,
    Start,
    Stop,
    Warn,
    WarningKind,
    warning_key,
)
from rampgate.config import Config
from rampgate.explain import (
    AwaitingResume,
    DeviceHeldFor,
    EnginePaused,
    Explanation,
    GangPartiallyBlocked,
    InsufficientResources,
    UnknownEstimateBlocked,
    WaitingForWarmup,
)
from rampgate.model import (
    DeviceId,
    JobId,
    JobRuntime,
    JobState,
    Snapshot,
    State,
    frozen_map,
)
from rampgate.policies.base import OrderingPolicy
from rampgate.reducer import IgnoredEvent, reduce
from rampgate.resources import Resources
from rampgate.stability import is_stable

__all__ = ["Decision", "DeviceLoad", "decide", "device_loads"]

_WARMING_LIKE = (JobState.STARTING, JobState.WARMING)


@dataclass(frozen=True, slots=True)
class Decision:
    actions: tuple[Action, ...]
    explanations: Mapping[JobId, Explanation] = field(default_factory=frozen_map)
    ignored_events: tuple[IgnoredEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class DeviceLoad:
    """The accounting of SPEC §5 for one device at ``now``."""

    capacity: Resources
    foreign: Resources
    reserved: Resources
    warming_usage: Resources
    warming_job: JobId | None

    @property
    def available(self) -> Resources:
        """``C - F``, clamped at 0."""
        return self.capacity.sub(self.foreign, clamp=True)

    @property
    def committed(self) -> Resources:
        """``F + sum(R stable) + sum(usage warming)``."""
        return self.foreign + self.reserved + self.warming_usage

    @property
    def free(self) -> Resources:
        """``free_d = C - F - sum(R stable) - sum(usage warming)``, clamped at 0."""
        return self.capacity.sub(self.committed, clamp=True)

    def admits(self, estimate: Resources, *, ignore_foreign: bool = False) -> bool:
        """Rule A for a known estimate, compared directly (never via a clamped ``free_d``).

        On an overcommitted device even a zero estimate is refused.
        """
        committed = self.committed if not ignore_foreign else self.reserved + self.warming_usage
        return (committed + estimate).fits_in(self.capacity)


def _is_warming_like(rt: JobRuntime) -> bool:
    """Occupies the probation slot: STARTING, WARMING, or STOPPING before it became stable."""
    return rt.state in _WARMING_LIKE or (rt.state is JobState.STOPPING and not rt.reservation)


def device_loads(state: State, now: float, config: Config) -> dict[DeviceId, DeviceLoad]:
    loads: dict[DeviceId, DeviceLoad] = {}
    for d, ds in sorted(state.devices.items()):
        reserved = Resources()
        warming_usage = Resources()
        warming_job: JobId | None = None
        for rt in state.jobs_on(d):
            if _is_warming_like(rt):
                warming_usage = warming_usage + rt.usage.get(d, Resources())
                warming_job = warming_job or rt.id
            elif rt.state in (JobState.STABLE, JobState.STOPPING):
                reserved = reserved + rt.reservation.get(d, Resources())
        loads[d] = DeviceLoad(
            capacity=ds.capacity,
            foreign=ds.foreign_at(now, config.foreign_window_s),
            reserved=reserved,
            warming_usage=warming_usage,
            warming_job=warming_job,
        )
    return loads


class _Tick:
    """Mutable scratch space for one call of ``decide``; never escapes it."""

    def __init__(self, snapshot: Snapshot, now: float, config: Config) -> None:
        self.snapshot = snapshot
        self.now = now
        self.config = config
        self.state = snapshot.state
        self.actions: list[Action] = []
        self.explanations: dict[JobId, Explanation] = {}

    def emit(self, action: Action) -> None:
        self.actions.append(action)
        self.state = reduce(self.state, action, self.config)

    def warn(self, warning: WarningKind) -> None:
        if warning_key(warning) not in self.state.warned:
            self.emit(Warn(warning))

    def jobs(self, *states: JobState) -> list[JobRuntime]:
        return sorted(
            (rt for rt in self.state.jobs.values() if rt.state in states),
            key=lambda rt: rt.submit_seq,
        )


def decide(snapshot: Snapshot, now: float, config: Config, policy: OrderingPolicy) -> Decision:
    """Pure: the same snapshot, ``now``, config and policy always give the same decision."""
    tick = _Tick(snapshot, now, config)
    for action in snapshot.state.outbox:
        tick.emit(action)
    _start_timeouts(tick)
    _stops(tick)
    _mark_stable(tick)
    for group, job_id in policy.queue_stops(tick.state):
        tick.warn(QueueStopped(group, job_id))
    _admit(tick, policy)
    _overcommit(tick)
    for group, outcome in policy.completed_groups(tick.state):
        if group not in tick.state.completed_groups:
            tick.emit(GroupCompleted(group, outcome))
    _explain_waiting(tick, policy)
    return Decision(
        tuple(tick.actions),
        frozen_map(dict(sorted(tick.explanations.items()))),
        snapshot.state.ignored_events,
    )


def _start_timeouts(tick: _Tick) -> None:
    for rt in tick.jobs(JobState.STARTING):
        since = rt.starting_since
        if since is not None and tick.now - since >= tick.config.start_timeout_s:
            estimates = {d: rt.prev_estimate(d) for d in rt.job.devices}
            tick.emit(Requeue(rt.id, rt.attempt + 1, estimates, RequeueReason.START_TIMEOUT))


def _stops(tick: _Tick) -> None:
    for rt in tick.jobs(JobState.STARTING, JobState.WARMING, JobState.STABLE):
        if rt.stop_intent is not None:
            tick.emit(Stop(rt.id, rt.attempt, rt.stop_intent, tick.config.preempt_grace_s))


def _mark_stable(tick: _Tick) -> None:
    margin = tick.config.safety_margin
    for rt in tick.jobs(JobState.WARMING):
        if not is_stable(rt, tick.now, tick.config):
            continue
        reservation = {}
        for d in rt.job.devices:
            if d in rt.last_obs_t:
                reservation[d] = rt.peak.get(d, Resources()).scale(margin)
            else:
                estimate = tick.snapshot.estimate(rt, d)
                reservation[d] = Resources() if estimate is None else estimate.scale(margin)
        tick.emit(MarkStable(rt.id, rt.attempt, reservation))


def _admit(tick: _Tick, policy: OrderingPolicy) -> None:
    if tick.state.paused:
        return
    loads = device_loads(tick.state, tick.now, tick.config)
    warming = {d: load.warming_job for d, load in loads.items() if load.warming_job is not None}
    held: dict[DeviceId, JobId] = {}
    blocking = policy.head_of_line_blocking

    for job_id in list(policy.candidates(tick.state, tick.now)):
        rt = tick.state.jobs.get(job_id)
        if rt is None or rt.state is not JobState.PENDING:
            continue
        devices = rt.job.devices
        held_devices = [d for d in devices if d in held]
        if held_devices:
            d = held_devices[0]
            tick.explanations[job_id] = DeviceHeldFor(d, held[d])
            for other in devices:
                held.setdefault(other, rt.id)
            continue

        estimates = {d: tick.snapshot.estimate(rt, d) for d in devices}
        probation = [d for d in devices if d in warming]
        no_fit = [
            d for d in devices if not rt.job.force and not _fits(tick, estimates[d], loads[d])
        ]

        if not probation and not no_fit:
            start = Start(
                rt.id,
                rt.attempt,
                devices,
                rt.has_run,
                {d: loads[d].free for d in devices},
                tick.now,
            )
            tick.emit(start)
            for d in devices:
                warming[d] = rt.id
                extra = loads[d].warming_usage + (estimates[d] or Resources())
                loads[d] = replace(loads[d], warming_usage=extra, warming_job=rt.id)
            continue

        failing = sorted(set(probation) | set(no_fit))
        tick.explanations[job_id] = _blocked_explanation(
            rt,
            failing=failing,
            probation=probation,
            estimates=estimates,
            loads=loads,
            warming=warming,
        )
        for d in no_fit:
            _candidate_warnings(tick, rt, d, estimates[d], loads[d])
        if blocking:
            for d in devices if rt.job.is_gang else failing:
                held.setdefault(d, rt.id)


def _fits(tick: _Tick, estimate: Resources | None, load: DeviceLoad) -> bool:
    if estimate is None:
        return tick.config.unknown_min_free.fits_in(load.free)
    return load.admits(estimate)


def _blocked_explanation(
    rt: JobRuntime,
    *,
    failing: list[DeviceId],
    probation: list[DeviceId],
    estimates: Mapping[DeviceId, Resources | None],
    loads: Mapping[DeviceId, DeviceLoad],
    warming: Mapping[DeviceId, JobId],
) -> Explanation:
    if rt.job.is_gang and len(failing) < len(rt.job.placement):
        return GangPartiallyBlocked(tuple(failing))
    if probation:
        return WaitingForWarmup(probation[0], warming[probation[0]])
    d = failing[0]
    estimate = estimates[d]
    if estimate is None:
        return UnknownEstimateBlocked(d)
    load = loads[d]
    dim = (load.committed + estimate).exceeds(load.capacity)[0]
    return InsufficientResources(d, dim, estimate[dim], load.free[dim])


def _candidate_warnings(
    tick: _Tick, rt: JobRuntime, d: DeviceId, estimate: Resources | None, load: DeviceLoad
) -> None:
    if estimate is None:
        return
    if not estimate.fits_in(load.capacity):
        tick.warn(CapacityExceeded(rt.job.group, d, estimate, load.capacity, rt.id))
    elif not load.foreign.is_zero and load.admits(estimate, ignore_foreign=True):
        tick.warn(ForeignUsageHigh(d, load.foreign, rt.id))


def _overcommit(tick: _Tick) -> None:
    for d, load in device_loads(tick.state, tick.now, tick.config).items():
        if (load.reserved + load.foreign).exceeds(load.capacity):
            tick.warn(Overcommitted(d, load.reserved, load.available))


def _explain_waiting(tick: _Tick, policy: OrderingPolicy) -> None:
    state = tick.state
    for rt in tick.jobs(JobState.PENDING, JobState.PREEMPTED):
        if state.paused:
            tick.explanations[rt.id] = EnginePaused(state.pause_reason)
        elif rt.state is JobState.PREEMPTED:
            tick.explanations[rt.id] = AwaitingResume()
        elif rt.id not in tick.explanations:
            reason = policy.explain_blocked(state, rt.id)
            if reason is not None:
                tick.explanations[rt.id] = reason
