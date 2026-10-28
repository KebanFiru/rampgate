"""`Config`: every tunable threshold of the algorithm (SPEC §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

from rampgate.resources import Resources

__all__ = ["Config"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Algorithm thresholds. Times are in seconds of the caller's clock."""

    # Reservations based on one observation (a STABLE job's peak).
    safety_margin: float = 1.15
    # A trusted STARTING/WARMING job: its estimate is already a max of observations.
    trusted_margin: float = 1.05
    # ...but only with this many complete observations of its level-0 key (rule A).
    min_trusted_observations: int = 3
    min_warmup_s: float = 30
    stable_window_s: float = 60
    stable_min_steps: int = 20
    stable_requires_epoch: bool = True
    growth_tol: float = 0.02
    max_warmup_s: float = 600
    oom_growth: float = 1.25
    max_oom_retries: int = 2
    max_lost_retries: int = 3
    preempt_grace_s: float = 120
    foreign_window_s: float = 10
    foreign_tolerance: float = 0.02
    start_timeout_s: float = 300
    reconcile_grace_s: float = 60
    unknown_min_free: Resources | float = 0.10
    # Rule P for jobs without a trusted estimate only (default), or for every job.
    probation: Literal["untrusted_only", "all"] = "untrusted_only"
    # Rule V: vacate when managed load exceeds C * (1 - headroom). From the simulator's
    # noise bound (2 %); calibrate on real hardware.
    vacate_headroom: float = 0.02
    # Warn(JobNotObserved) after this many consecutive device samples without an
    # observation of a running job there (0: never).
    missing_samples_warn: int = 5

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "unknown_min_free":
                _check_unknown_min_free(value)
                continue
            if f.name == "probation":
                if value not in ("untrusted_only", "all"):
                    raise ValueError(f"probation must be 'untrusted_only' or 'all', got {value!r}")
                continue
            if f.name == "stable_requires_epoch":
                if not isinstance(value, bool):
                    raise TypeError("stable_requires_epoch must be a bool")
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{f.name} must be a number, got {type(value).__name__}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and non-negative, got {value!r}")
        for name in (
            "stable_min_steps",
            "max_oom_retries",
            "max_lost_retries",
            "min_trusted_observations",
            "missing_samples_warn",
        ):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an int")
        if self.safety_margin < 1:
            raise ValueError("safety_margin must be >= 1")
        if self.trusted_margin < 1:
            raise ValueError("trusted_margin must be >= 1")
        if self.oom_growth <= 1:
            raise ValueError("oom_growth must be > 1, otherwise an OOM teaches nothing")
        if self.min_warmup_s > self.max_warmup_s:
            raise ValueError("min_warmup_s must be <= max_warmup_s")
        if self.foreign_tolerance >= 1:
            raise ValueError("foreign_tolerance must be < 1")
        if self.vacate_headroom >= 1:
            raise ValueError("vacate_headroom must be < 1")

    def unknown_min_free_for(self, capacity: Resources) -> Resources:
        """``unknown_min_free`` on a device: absolute, or a fraction of its capacity."""
        if isinstance(self.unknown_min_free, Resources):
            return self.unknown_min_free
        return capacity.scale(self.unknown_min_free)


def _check_unknown_min_free(value: object) -> None:
    if isinstance(value, Resources):
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("unknown_min_free must be Resources or a fraction of capacity")
    if not 0 <= value <= 1:
        raise ValueError(f"unknown_min_free as a fraction must be in [0, 1], got {value!r}")
