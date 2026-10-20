"""Rule S (stability) and peak-growth detection (SPEC §5). Pure functions."""

from __future__ import annotations

from rampgate.config import Config
from rampgate.model import JobRuntime, JobState
from rampgate.resources import Resources

__all__ = ["is_peak_growth", "is_stable"]


def is_peak_growth(old: Resources, new: Resources, tol: float) -> bool:
    """True if ``new`` exceeds ``old`` by more than ``tol`` (relative) in any dimension.

    Any increase from 0 counts as growth.
    """
    return any(v > old[d] * (1 + tol) for d, v in new.items())


def is_stable(rt: JobRuntime, now: float, config: Config) -> bool:
    """Rule S: may this WARMING job be marked STABLE at ``now``?

    ``now - warm_start >= min_warmup_s`` and either
    - the peak has been quiet for ``stable_window_s`` and, if the job reports
      steps, it made ``stable_min_steps`` steps in this attempt; or
    - ``now - warm_start >= max_warmup_s``.

    Jobs that never report steps are judged on the time conditions alone.
    """
    if rt.state is not JobState.WARMING or rt.warm_start is None:
        return False
    warmed = now - rt.warm_start
    if warmed < config.min_warmup_s:
        return False
    if warmed >= config.max_warmup_s:
        return True
    quiet_since = rt.warm_start
    if rt.last_peak_growth_t is not None:
        quiet_since = max(quiet_since, rt.last_peak_growth_t)
    if now - quiet_since < config.stable_window_s:
        return False
    steps = rt.steps_in_attempt
    return steps is None or steps >= config.stable_min_steps
