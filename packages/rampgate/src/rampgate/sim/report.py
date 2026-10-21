"""`Report`: what a simulated run achieved (SPEC §8), and `run` for one strategy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rampgate.actions import Start
from rampgate.estimators import Estimator, HistoryEstimator, InMemoryHistoryStore, NullEstimator
from rampgate.events import JobSubmitted
from rampgate.model import DeviceId, JobId, JobState
from rampgate.policies.base import OrderingPolicy
from rampgate.sim.baselines import Strategy
from rampgate.sim.executor import TimelineEntry
from rampgate.sim.scenario import Scenario, Trace, simulate

__all__ = ["Report", "quantile", "report", "run", "summarize"]


@dataclass(frozen=True, slots=True)
class Report:
    """Metrics of one run.

    - ``makespan``: time from the first tick until every job is finished, or
      None when the run was cut at ``max_time`` (censored).
    - ``completed`` / ``total``: SUCCEEDED jobs / all jobs of the scenario
      (including jobs never submitted because the job they waited for failed).
    - ``oom_count``, ``lost_work_step_s``, ``max_overcommit``: the executor's
      ground truth (`WorldStats`), the overcommit as the max over devices.
    - ``utilization``: per device and dimension, mean job usage / capacity over
      the run (up to ``max_time`` when censored).
    - ``wait_p50`` / ``wait_p95``: queue wait, from a job's submission to its
      first ``Start``. A job never started counts as infinitely long; a
      quantile that reaches it is ``math.inf``.
    """

    makespan: float | None
    completed: int
    total: int
    oom_count: int
    lost_work_step_s: float
    utilization: Mapping[DeviceId, Mapping[str, float]]
    wait_p50: float
    wait_p95: float
    max_overcommit: float
    timeline: tuple[TimelineEntry, ...]

    @property
    def censored(self) -> bool:
        return self.makespan is None

    def metrics(self) -> dict[str, Any]:
        """The scalar metrics, JSON-ready (``inf`` / censored become None)."""

        def finite(x: float | None) -> float | None:
            return None if x is None or math.isinf(x) else x

        return {
            "makespan": self.makespan,
            "completed": self.completed,
            "total": self.total,
            "oom_count": self.oom_count,
            "lost_work_step_s": self.lost_work_step_s,
            "utilization": self.mean_utilization,
            "wait_p50": finite(self.wait_p50),
            "wait_p95": finite(self.wait_p95),
            "max_overcommit": self.max_overcommit,
        }

    @property
    def mean_utilization(self) -> float:
        """Mean over every device and dimension of `utilization`."""
        values = [v for dims in self.utilization.values() for v in dims.values()]
        return sum(values) / len(values) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (like ``numpy.quantile``'s default).

    ``inf`` values sort last; a quantile that interpolates with one is ``inf``.
    """
    if not values:
        raise ValueError("quantile of no values")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if math.isinf(xs[hi]) or math.isinf(xs[lo]):
        return math.inf
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> tuple[float, float, float]:
    """``(q1, median, q3)``."""
    return quantile(values, 0.25), quantile(values, 0.5), quantile(values, 0.75)


def report(trace: Trace, scenario: Scenario) -> Report:
    world = trace.world
    assert world is not None, "trace without world stats"
    start = trace.ticks[0].now
    end = trace.ticks[-1].now
    makespan = end - start if trace.finished else None
    horizon = end - start

    submitted: dict[JobId, float] = {}
    started: dict[JobId, float] = {}
    for tick in trace.ticks:
        for event in tick.events:
            if isinstance(event, JobSubmitted):
                submitted.setdefault(event.job.id, tick.now)
        for action in tick.decision.actions:
            if isinstance(action, Start):
                started.setdefault(action.job_id, tick.now)
    final = trace.final.jobs
    waits = [
        started[j] - t if j in started else math.inf
        for j, t in submitted.items()
        if j in started or final[j].state is not JobState.CANCELLED
    ]

    capacity = {d.id: d.capacity for d in scenario.devices}
    utilization = {
        d: {
            dim: (world.usage_s[d][dim] / (horizon * cap) if horizon > 0 else 0.0)
            for dim, cap in capacity[d].items()
            if cap > 0
        }
        for d in sorted(capacity)
    }
    return Report(
        makespan=makespan,
        completed=sum(1 for rt in final.values() if rt.state is JobState.SUCCEEDED),
        total=len(scenario.jobs),
        oom_count=world.oom_count,
        lost_work_step_s=world.lost_work_step_s,
        utilization=utilization,
        wait_p50=quantile(waits, 0.5) if waits else 0.0,
        wait_p95=quantile(waits, 0.95) if waits else 0.0,
        max_overcommit=max(world.max_overcommit.values(), default=0.0),
        timeline=world.timeline,
    )


def run(
    scenario: Scenario,
    strategy: Strategy,
    *,
    policy: OrderingPolicy,
    history_store: InMemoryHistoryStore | None = None,
    **sim_kwargs: Any,
) -> Report:
    """Simulate ``scenario`` under ``strategy`` and report.

    A ``"history"`` strategy uses ``history_store`` (a fresh empty store when
    None, i.e. a cold run); pass a filled store for a warm run. The store is
    written to.
    """
    estimator: Estimator
    if strategy.estimator == "history":
        estimator = HistoryEstimator(
            history_store if history_store is not None else InMemoryHistoryStore()
        )
    else:
        estimator = NullEstimator()
    trace = simulate(
        scenario,
        policy,
        config=strategy.config,
        estimator=estimator,
        admission=strategy.admission,
        **sim_kwargs,
    )
    return report(trace, scenario)
