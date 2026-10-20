"""`PriorityPolicy`: higher priority first, with optional aging (SPEC §6)."""

from __future__ import annotations

from collections.abc import Iterable

from rampgate.model import JobId, JobRuntime, JobState
from rampgate.policies.base import BasePolicy, StateView

__all__ = ["PriorityPolicy"]


class PriorityPolicy(BasePolicy):
    """Order by ``priority + aging_per_s * (now - submit_t)``, then submit order.

    With ``head_of_line_blocking`` (the default) the top job holds its devices
    until it fits, which is what lets aging prevent starvation.
    """

    def __init__(self, aging_per_s: float = 0.0, *, head_of_line_blocking: bool = True) -> None:
        if aging_per_s < 0:
            raise ValueError("aging_per_s must be non-negative")
        self.aging_per_s = aging_per_s
        self.head_of_line_blocking = head_of_line_blocking

    def effective_priority(self, rt: JobRuntime, now: float) -> float:
        if rt.submit_t is None:
            return rt.job.priority
        return rt.job.priority + self.aging_per_s * max(0.0, now - rt.submit_t)

    def candidates(self, view: StateView, now: float) -> Iterable[JobId]:
        pending = [rt for rt in view.jobs.values() if rt.state is JobState.PENDING]
        pending.sort(key=lambda rt: (-self.effective_priority(rt, now), rt.submit_seq))
        return [rt.id for rt in pending]
