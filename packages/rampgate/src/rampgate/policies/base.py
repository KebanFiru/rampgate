"""The ordering policy protocol (SPEC §6)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeAlias

from rampgate.actions import GroupOutcome
from rampgate.explain import Explanation
from rampgate.model import GroupId, JobId, State

__all__ = ["BasePolicy", "OrderingPolicy", "StateView"]

StateView: TypeAlias = State


class OrderingPolicy(Protocol):
    """Decides *which* jobs are eligible and in what order; `decide` checks resources."""

    @property
    def head_of_line_blocking(self) -> bool:
        """If true, a candidate that is not admitted on a device holds it for this tick."""
        ...

    def candidates(self, view: StateView, now: float) -> Iterable[JobId]:
        """PENDING jobs eligible by structure, in admission order."""
        ...

    def explain_blocked(self, view: StateView, job_id: JobId) -> Explanation | None:
        """Why a PENDING job is not a candidate, if the policy is the reason."""
        ...

    def completed_groups(self, view: StateView) -> Iterable[tuple[GroupId, GroupOutcome]]:
        """Groups whose jobs are all terminal, with their outcome."""
        ...

    def queue_stops(self, view: StateView) -> Iterable[tuple[GroupId, JobId]]:
        """Failed jobs that should stop the queue (StagePolicy's STOP_QUEUE)."""
        ...


class BasePolicy:
    """Convenience base for custom policies: no blocking, no groups, no queue stops."""

    head_of_line_blocking: bool = False

    def candidates(self, view: StateView, now: float) -> Iterable[JobId]:
        raise NotImplementedError

    def explain_blocked(self, view: StateView, job_id: JobId) -> Explanation | None:
        return None

    def completed_groups(self, view: StateView) -> Iterable[tuple[GroupId, GroupOutcome]]:
        return ()

    def queue_stops(self, view: StateView) -> Iterable[tuple[GroupId, JobId]]:
        return ()
