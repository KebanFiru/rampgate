"""Scenarios and the minimal run loop of the simulator (SPEC §8)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rampgate.config import Config
from rampgate.decide import Decision
from rampgate.engine import Engine
from rampgate.estimators import Estimator, NullEstimator
from rampgate.events import Event, JobExited, JobStarted, JobSubmitted
from rampgate.model import (
    TERMINAL_STATES,
    Device,
    DeviceId,
    ExitKind,
    Job,
    JobState,
    Snapshot,
    State,
    frozen_map,
)
from rampgate.policies.base import OrderingPolicy
from rampgate.reducer import InvalidTransition
from rampgate.resources import Resources
from rampgate.sim.clock import VirtualClock
from rampgate.sim.executor import SimExecutor
from rampgate.sim.workloads import Workload

__all__ = ["Command", "Scenario", "SimJob", "Tick", "Trace", "simulate"]


@dataclass(frozen=True, slots=True)
class SimJob:
    job: Job
    workload: Workload
    submit_at: float = 0.0


@dataclass(frozen=True, slots=True)
class Command:
    """A caller request (preempt, resume, cancel, pause, ...) issued at ``at``."""

    at: float
    event: Event


@dataclass(frozen=True, slots=True)
class Scenario:
    devices: tuple[Device, ...]
    jobs: tuple[SimJob, ...]
    commands: tuple[Command, ...] = ()
    foreign: Mapping[DeviceId, Resources] = field(default_factory=frozen_map)
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Tick:
    """One step of the loop: events reduced, then ``decide`` on ``snapshot``."""

    now: float
    events: tuple[Event, ...]
    rejected: tuple[Event, ...]
    snapshot: Snapshot
    decision: Decision
    after: State


@dataclass(frozen=True, slots=True)
class Trace:
    ticks: tuple[Tick, ...]
    dumps: tuple[str, ...] = ()

    @property
    def final(self) -> State:
        return self.ticks[-1].after


def _times(value: float | Sequence[float] | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    return sorted(float(v) for v in value)


def _reconcile(engine: Engine, executor: SimExecutor) -> list[Event]:
    """Report ground truth for every unreconciled job, as a real caller would after downtime."""
    reported: list[Event] = []
    for job_id in engine.unreconciled():
        rt = engine.state.jobs[job_id]
        status, began, exit_event = executor.truth(job_id, rt.attempt)
        if status == "running":
            assert began is not None
            if rt.state is JobState.STARTING:
                reported.append(JobStarted(job_id, rt.attempt, began))
                engine.on_started(job_id, rt.attempt, began)
            engine.confirm(job_id, rt.attempt)
        elif status == "exited":
            assert exit_event is not None
            reported.append(exit_event)
            engine.apply(exit_event)
        elif status == "unknown":
            lost = JobExited(job_id, rt.attempt, ExitKind.LOST, engine.now or 0.0)
            reported.append(lost)
            engine.apply(lost)
    return reported


def _feed(
    engine: Engine, now: float, pending_jobs: list[SimJob], pending_cmds: list[Command]
) -> tuple[list[Event], list[Event], list[Event]]:
    """Submit due jobs and apply due commands; returns (submitted, commands, rejected)."""
    submitted: list[Event] = []
    while pending_jobs and pending_jobs[0].submit_at <= now:
        job = pending_jobs.pop(0).job
        engine.submit(job, t=now)
        submitted.append(JobSubmitted(job, t=now))
    commands: list[Event] = []
    rejected: list[Event] = []
    while pending_cmds and pending_cmds[0].at <= now:
        command = pending_cmds.pop(0).event
        commands.append(command)
        try:
            engine.apply(command)
        except InvalidTransition:
            rejected.append(command)
    return submitted, commands, rejected


def simulate(
    scenario: Scenario,
    policy: OrderingPolicy,
    *,
    config: Config | None = None,
    estimator: Estimator | None = None,
    dt: float = 1.0,
    max_time: float = 10_000.0,
    start_delay_s: float = 1.0,
    stop_delay_s: float = 1.0,
    reset_counters_on_resume: bool = False,
    restart_at: float | Sequence[float] | None = None,
    downtime: Sequence[tuple[float, float]] = (),
) -> Trace:
    """Run ``scenario`` until every job is finished or ``max_time`` is reached.

    Each tick: the executor reports what happened, submissions and commands due
    are added, the engine ticks, and its actions go to the executor. Commands
    that are invalid when they fire are recorded in ``Tick.rejected``.

    Restarts (the outside world and the estimator's store carry on throughout):

    - ``restart_at``: at each of these times the engine is dumped to a JSON
      string *after* the tick's events were applied and *before* it ticks
      (so pending outbox actions and history records are in the dump), then
      loaded again. Every dump string is kept in ``Trace.dumps``.
    - ``downtime``: ``(start, end)`` windows during which the engine is gone.
      It is dumped at ``start``; events in the window are lost; submissions and
      commands wait; at ``end`` it is loaded and reconciled from the executor's
      ground truth (the reported events open that tick's ``events``).
    """
    config = config or Config()
    estimator = estimator or NullEstimator()
    executor = SimExecutor(
        scenario.devices,
        {sj.job.id: sj.workload for sj in scenario.jobs},
        foreign=scenario.foreign,
        start_delay_s=start_delay_s,
        stop_delay_s=stop_delay_s,
        reset_counters_on_resume=reset_counters_on_resume,
        seed=scenario.seed,
    )
    clock = VirtualClock(dt)
    engine: Engine | None = Engine(scenario.devices, policy, estimator, config)
    pending_jobs = sorted(scenario.jobs, key=lambda sj: (sj.submit_at, sj.job.id))
    pending_cmds = sorted(scenario.commands, key=lambda c: c.at)
    restarts = _times(restart_at)
    windows = sorted(downtime)
    dumps: list[str] = []
    saved: str | None = None
    ticks: list[Tick] = []
    now = clock.now

    def load(data: str) -> Engine:
        return Engine.load(json.loads(data), policy=policy, estimator=estimator, config=config)

    while True:
        if windows and engine is not None and now >= windows[0][0]:
            saved = json.dumps(engine.dump())
            dumps.append(saved)
            engine = None
        if engine is None:
            executor.advance(now)  # nobody listens: these events are lost
            assert saved is not None
            if now < windows[0][1] and now < max_time:
                now = clock.advance()
                continue
            windows.pop(0)
            engine = load(saved)
            facts: list[Event] = _reconcile(engine, executor)
        else:
            facts = []
            facts.extend(executor.advance(now))
            for event in facts:
                engine.apply(event)

        submitted, commands, rejected = _feed(engine, now, pending_jobs, pending_cmds)

        while restarts and now >= restarts[0]:
            restarts.pop(0)
            data = json.dumps(engine.dump())
            dumps.append(data)
            engine = load(data)

        decision = engine.tick(now)
        state = engine.state
        executor.apply(decision.actions, now, {j: rt.job for j, rt in state.jobs.items()})
        snapshot = engine.last_snapshot
        assert snapshot is not None
        ticks.append(
            Tick(now, (*facts, *submitted, *commands), tuple(rejected), snapshot, decision, state)
        )

        finished = all(rt.state in TERMINAL_STATES for rt in state.jobs.values())
        if (finished and not pending_jobs and not pending_cmds) or now >= max_time:
            break
        now = clock.advance()
    return Trace(tuple(ticks), tuple(dumps))
