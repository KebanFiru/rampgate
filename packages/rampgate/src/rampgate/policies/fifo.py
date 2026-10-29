"""`FifoPolicy`: submit order (docs/policies.md)."""

from __future__ import annotations

from collections.abc import Iterable

from rampgate.model import JobId, JobState
from rampgate.policies.base import BasePolicy, StateView

__all__ = ["FifoPolicy"]


class FifoPolicy(BasePolicy):
    """Submit order. ``strict=True`` holds each device for the earliest job that needs it."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self.head_of_line_blocking = strict

    def candidates(self, view: StateView, now: float) -> Iterable[JobId]:
        pending = (rt for rt in view.jobs.values() if rt.state is JobState.PENDING)
        return [rt.id for rt in sorted(pending, key=lambda rt: rt.submit_seq)]
