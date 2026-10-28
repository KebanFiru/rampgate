"""Synthetic workloads: memory over time since the process started (SPEC §8)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from rampgate.resources import Resources

__all__ = ["EpochEval", "RampPlateau", "Spiky", "StepProfile", "Workload"]


class Workload(Protocol):
    @property
    def duration_s(self) -> float: ...

    @property
    def steps_per_s(self) -> float: ...

    @property
    def checkpoint_every_s(self) -> float: ...

    @property
    def noise(self) -> float: ...

    def usage(self, t: float) -> Resources:
        """Usage ``t`` seconds after the process started (no noise)."""
        ...

    def epoch(self, t: float) -> int | None:
        """Completed epochs ``t`` seconds after the process started, or None if not reported."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _Common:
    duration_s: float = 600.0
    steps_per_s: float = 1.0
    checkpoint_every_s: float = 60.0
    noise: float = 0.0
    # False: the process reports no step counter (Rule S judges it on time alone).
    reports_steps: bool = True

    def epoch(self, t: float) -> int | None:
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RampPlateau(_Common):
    """Model load (up to 30 % of the plateau), a linear ramp to the plateau, then flat."""

    load_s: float
    ramp_s: float
    plateau: Resources

    def usage(self, t: float) -> Resources:
        if t < self.load_s:
            return self.plateau.scale(0.3 * t / self.load_s)
        if t < self.load_s + self.ramp_s:
            return self.plateau.scale(0.3 + 0.7 * (t - self.load_s) / self.ramp_s)
        return self.plateau


@dataclass(frozen=True, slots=True, kw_only=True)
class StepProfile(_Common):
    """Piecewise constant: the last ``(t, usage)`` step with ``t <= now``."""

    steps: tuple[tuple[float, Resources], ...]

    def usage(self, t: float) -> Resources:
        current = Resources()
        for start, value in sorted(self.steps, key=lambda s: s[0]):
            if start <= t:
                current = value
        return current


@dataclass(frozen=True, slots=True, kw_only=True)
class Spiky(_Common):
    """``base`` plus ``spike`` for ``spike_s`` seconds every ``every_s`` seconds."""

    base: Resources
    spike: Resources
    every_s: float
    spike_s: float = 1.0

    def usage(self, t: float) -> Resources:
        if t >= self.every_s and (t % self.every_s) < self.spike_s:
            return self.base + self.spike
        return self.base


@dataclass(frozen=True, slots=True, kw_only=True)
class EpochEval(_Common):
    """Training at ``base``, then an end-of-epoch evaluation that adds ``eval_spike``.

    After ``load_s`` of loading (linear ramp to ``base``), each epoch is
    ``epoch_s`` of training followed by ``eval_s`` of evaluation. The reported
    ``epoch`` counter is incremented *after* the evaluation, as SPEC §3.5 assumes.
    """

    base: Resources
    eval_spike: Resources
    epoch_s: float
    eval_s: float
    load_s: float = 0.0
    reports_epoch: bool = True

    def usage(self, t: float) -> Resources:
        if t < self.load_s:
            return self.base.scale(t / self.load_s)
        pos = (t - self.load_s) % (self.epoch_s + self.eval_s)
        return self.base + self.eval_spike if pos >= self.epoch_s else self.base

    def epoch(self, t: float) -> int | None:
        if not self.reports_epoch:
            return None
        if t < self.load_s:
            return 0
        return math.floor((t - self.load_s) / (self.epoch_s + self.eval_s))
