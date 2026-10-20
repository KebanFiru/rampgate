"""Scenarios and the minimal run loop of the simulator (SPEC §8)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from rampgate.config import Config
from rampgate.decide import Decision
from rampgate.engine import Engine
from rampgate.estimators import Estimator, NullEstimator
from rampgate.events import Event, JobSubmitted
from rampgate.model import TERMINAL_STATES, Device, DeviceId, Job, Snapshot, State, frozen_map
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

    @property
    def final(self) -> State:
        return self.ticks[-1].after


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
    restart_at: float | None = None,
) -> Trace:
    """Run ``scenario`` until every job is finished or ``max_time`` is reached.

    Each tick: the executor reports what happened, submissions and commands due
    are added, events are reduced, estimates are resolved, ``decide`` runs, its
    actions are reduced and handed to the executor. Commands that are invalid at
    the time they fire are recorded in ``Tick.rejected`` instead of raising.

    With ``restart_at``, the engine is dumped to JSON and loaded again at the
    first tick at or after that time; the executor and the estimator carry on,
    as the outside world and a durable history store would.
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
    engine = Engine(scenario.devices, policy, estimator, config)
    pending_jobs = sorted(scenario.jobs, key=lambda sj: (sj.submit_at, sj.job.id))
    pending_cmds = sorted(scenario.commands, key=lambda c: c.at)
    ticks: list[Tick] = []
    restarted = False
    now = clock.now
    while True:
        if restart_at is not None and not restarted and now >= restart_at:
            data = json.loads(json.dumps(engine.dump()))
            engine = Engine.load(data, policy=policy, estimator=estimator, config=config)
            restarted = True

        facts: list[Event] = list(executor.advance(now))
        for event in facts:
            engine.apply(event)
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
    return Trace(tuple(ticks))
