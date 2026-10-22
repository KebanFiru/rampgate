"""Baseline and ablation strategies for the benchmark (SPEC §8).

Every strategy runs on the same Engine and reducer; only the admission step of
``decide`` (and the config) differs. An OOM'd job is requeued with
``resume=True`` and rolls back to its last checkpoint under every strategy.
Baselines do not learn (they ignore estimates) and are not limited by
``max_oom_retries``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rampgate.config import Config
from rampgate.decide import PROBATION, Admission, DeviceLoad, ProbationAdmission, measured_free
from rampgate.model import ACTIVE_STATES, DeviceId, JobRuntime, State
from rampgate.resources import Resources

__all__ = [
    "ABLATIONS",
    "BASELINES",
    "MAIN",
    "MARGIN_SWEEP",
    "PROBATION_ALL",
    "PROBATION_COLD",
    "PROBATION_WARM",
    "STRATEGIES",
    "TRUSTED_MARGINS",
    "UNBOUNDED_RETRIES",
    "WARM",
    "DeclaredReservation",
    "GreedyFree",
    "GreedyFreeRuleO",
    "Serial",
    "Strategy",
]

# Retries never run out: a baseline that keeps OOMing shows up as censored
# makespan, OOMs and lost work instead of as failed jobs.
UNBOUNDED_RETRIES = Config(max_oom_retries=1_000_000)


def _others(state: State, rt: JobRuntime, device_id: DeviceId) -> list[JobRuntime]:
    return [o for o in state.jobs_on(device_id) if o.id != rt.id and o.state in ACTIVE_STATES]


def _positive(free: Resources, capacity: Resources) -> bool:
    return all(free[dim] > 0 for dim, cap in capacity.items() if cap > 0)


@dataclass(frozen=True, slots=True)
class GreedyFree:
    """Start whenever the measured free capacity is above 0 in every dimension."""

    name: str = "GreedyFree"
    probation: bool = False

    def fits(
        self,
        rt: JobRuntime,
        device_id: DeviceId,
        *,
        estimate: Resources | None,
        load: DeviceLoad,
        state: State,
        now: float,
        config: Config,
    ) -> bool:
        return _positive(measured_free(state, device_id, now, config), load.capacity)


@dataclass(frozen=True, slots=True)
class DeclaredReservation:
    """Reserve exactly the declared hint (a missing hint reserves nothing)."""

    name: str = "DeclaredReservation"
    probation: bool = False

    def fits(
        self,
        rt: JobRuntime,
        device_id: DeviceId,
        *,
        estimate: Resources | None,
        load: DeviceLoad,
        state: State,
        now: float,
        config: Config,
    ) -> bool:
        taken = load.foreign
        for other in _others(state, rt, device_id):
            taken = taken + (other.job.placement.get(device_id) or Resources())
        hint = rt.job.placement.get(device_id) or Resources()
        return (taken + hint).fits_in(load.capacity)


@dataclass(frozen=True, slots=True)
class Serial:
    """One job per device at a time."""

    name: str = "Serial"
    probation: bool = False

    def fits(
        self,
        rt: JobRuntime,
        device_id: DeviceId,
        *,
        estimate: Resources | None,
        load: DeviceLoad,
        state: State,
        now: float,
        config: Config,
    ) -> bool:
        return not _others(state, rt, device_id)


@dataclass(frozen=True, slots=True)
class GreedyFreeRuleO:
    """Ablation: GreedyFree that also uses what Rule O learned (no probation, no history)."""

    name: str = "GreedyFree + Rule O"
    probation: bool = False

    def fits(
        self,
        rt: JobRuntime,
        device_id: DeviceId,
        *,
        estimate: Resources | None,
        load: DeviceLoad,
        state: State,
        now: float,
        config: Config,
    ) -> bool:
        free = measured_free(state, device_id, now, config)
        learned = rt.learned.get(device_id)
        if learned is None:
            return _positive(free, load.capacity)
        return learned.fits_in(free)


@dataclass(frozen=True, slots=True)
class Strategy:
    """A benchmark row: an admission, its config, and where its estimates come from.

    ``estimator``: ``"history"`` uses a `HistoryEstimator` over the run's store
    (empty for a cold run); ``"null"`` uses `NullEstimator`. ``warm``: the store
    is filled first by an independent run of the same scenario with another seed.
    """

    name: str
    admission: Admission = PROBATION
    config: Config = field(default_factory=Config)
    estimator: Literal["history", "null"] = "history"
    warm: bool = False


PROBATION_COLD = Strategy("Probation")
PROBATION_ALL = Strategy("Probation (all)", config=Config(probation="all"))
PROBATION_WARM = Strategy("Probation (warm)", warm=True)
WARM = (
    PROBATION_WARM,
    Strategy("Probation (all, warm)", config=Config(probation="all"), warm=True),
)
BASELINES = (
    Strategy("GreedyFree", GreedyFree(), UNBOUNDED_RETRIES, "null"),
    Strategy("DeclaredReservation", DeclaredReservation(), UNBOUNDED_RETRIES, "null"),
    Strategy("Serial", Serial(), UNBOUNDED_RETRIES, "null"),
)
MAIN = (PROBATION_COLD, PROBATION_ALL, *BASELINES)
ABLATIONS = (
    Strategy("GreedyFree + Rule O", GreedyFreeRuleO(), UNBOUNDED_RETRIES, "null"),
    Strategy("rampgate - Rule P", ProbationAdmission("rampgate - Rule P", probation=False)),
    Strategy("rampgate - history", estimator="null"),
)
TRUSTED_MARGINS = (1.00, 1.05, 1.10, 1.15)
# Reported only (SPEC §5): the default trusted_margin is not picked from this sweep.
MARGIN_SWEEP = tuple(
    Strategy(
        f"Probation{suffix} [trusted_margin={m:.2f}]", config=Config(trusted_margin=m), warm=warm
    )
    for m in TRUSTED_MARGINS
    for suffix, warm in (("", False), (" (warm)", True))
)
STRATEGIES = {s.name: s for s in (*MAIN, *ABLATIONS, *WARM, *MARGIN_SWEEP)}
