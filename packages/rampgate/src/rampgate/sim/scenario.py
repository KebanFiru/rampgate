"""Scenarios and the minimal run loop of the simulator (SPEC §8)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from rampgate.config import Config
from rampgate.decide import Decision, decide
from rampgate.estimators import Estimator, NullEstimator, apply_history, resolve_estimates
from rampgate.events import Event, JobSubmitted
from rampgate.model import TERMINAL_STATES, Device, DeviceId, Job, Snapshot, State, frozen_map
from rampgate.policies.base import OrderingPolicy
from rampgate.reducer import InvalidTransition, drain, reduce
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
) -> Trace:
    """Run ``scenario`` until every job is finished or ``max_time`` is reached.

    Each tick: the executor reports what happened, submissions and commands due
    are added, events are reduced, estimates are resolved, ``decide`` runs, its
    actions are reduced and handed to the executor. Commands that are invalid at
    the time they fire are recorded in ``Tick.rejected`` instead of raising.
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
    state = State.initial(scenario.devices)
    pending_jobs = sorted(scenario.jobs, key=lambda sj: (sj.submit_at, sj.job.id))
    pending_cmds = sorted(scenario.commands, key=lambda c: c.at)
    ticks: list[Tick] = []
    now = clock.now
    while True:
        facts: list[Event] = list(executor.advance(now))
        while pending_jobs and pending_jobs[0].submit_at <= now:
            facts.append(JobSubmitted(pending_jobs.pop(0).job, t=now))
        commands: list[Event] = []
        while pending_cmds and pending_cmds[0].at <= now:
            commands.append(pending_cmds.pop(0).event)

        for event in facts:
            state = reduce(state, event, config)
        rejected: list[Event] = []
        for event in commands:
            try:
                state = reduce(state, event, config)
            except InvalidTransition:
                rejected.append(event)

        state, _, history = drain(state)
        apply_history(estimator, state, history)
        snapshot = resolve_estimates(state, estimator)
        decision = decide(snapshot, now, config, policy)
        for action in decision.actions:
            state = reduce(state, action, config)
        executor.apply(decision.actions, now, {j: rt.job for j, rt in state.jobs.items()})
        ticks.append(Tick(now, (*facts, *commands), tuple(rejected), snapshot, decision, state))

        finished = all(rt.state in TERMINAL_STATES for rt in state.jobs.values())
        if (finished and not pending_jobs and not pending_cmds) or now >= max_time:
            break
        now = clock.advance()
    return Trace(tuple(ticks))
