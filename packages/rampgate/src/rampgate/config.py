"""`Config`: every tunable threshold of the algorithm (SPEC §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from rampgate.resources import Resources

__all__ = ["Config"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Algorithm thresholds. Times are in seconds of the caller's clock."""

    safety_margin: float = 1.15
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
    unknown_min_free: Resources = field(default_factory=Resources)

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "unknown_min_free":
                if not isinstance(value, Resources):
                    raise TypeError("unknown_min_free must be a Resources")
                continue
            if f.name == "stable_requires_epoch":
                if not isinstance(value, bool):
                    raise TypeError("stable_requires_epoch must be a bool")
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{f.name} must be a number, got {type(value).__name__}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and non-negative, got {value!r}")
        for name in ("stable_min_steps", "max_oom_retries", "max_lost_retries"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an int")
        if self.safety_margin < 1:
            raise ValueError("safety_margin must be >= 1")
        if self.oom_growth <= 1:
            raise ValueError("oom_growth must be > 1, otherwise an OOM teaches nothing")
        if self.min_warmup_s > self.max_warmup_s:
            raise ValueError("min_warmup_s must be <= max_warmup_s")
        if self.foreign_tolerance >= 1:
            raise ValueError("foreign_tolerance must be < 1")
