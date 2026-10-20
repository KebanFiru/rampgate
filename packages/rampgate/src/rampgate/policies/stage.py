"""`StagePolicy`: groups run one after another (SPEC §6)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

from rampgate.actions import GroupOutcome
from rampgate.explain import Explanation, GroupBarrier
from rampgate.model import TERMINAL_STATES, DeviceId, GroupId, JobId, JobRuntime, JobState
from rampgate.policies.base import BasePolicy, StateView

__all__ = ["FailurePolicy", "StagePolicy"]


class FailurePolicy(StrEnum):
    CONTINUE = "CONTINUE"
    STOP_QUEUE = "STOP_QUEUE"


def _by_position(rt: JobRuntime) -> tuple[int, int]:
    return rt.job.position, rt.submit_seq


class StagePolicy(BasePolicy):
    """Barrier groups in ``groups_order``; side jobs (``group=None``) always run.

    Never blocks: jobs inside the active group run in parallel, and a group is
    finite, so skipping a job that does not fit cannot starve it.
    """

    def __init__(
        self,
        groups_order: Sequence[GroupId] = (),
        *,
        backfill: bool = False,
        failure_policy: Mapping[GroupId, FailurePolicy] | None = None,
    ) -> None:
        self.groups_order = tuple(groups_order)
        self.backfill = backfill
        self.failure_policy = dict(failure_policy or {})
        self.head_of_line_blocking = False

    # -- structure ---------------------------------------------------------------

    def groups(self, view: StateView) -> list[GroupId]:
        """Listed groups first, then the others in order of first submission."""
        seen = list(self.groups_order)
        for rt in sorted(view.jobs.values(), key=lambda rt: rt.submit_seq):
            g = rt.job.group
            if g is not None and g not in seen:
                seen.append(g)
        return seen

    @staticmethod
    def _members(view: StateView, group: GroupId) -> list[JobRuntime]:
        return [rt for rt in view.jobs.values() if rt.job.group == group]

    def _complete(self, view: StateView, group: GroupId) -> bool:
        return all(rt.state in TERMINAL_STATES for rt in self._members(view, group))

    def active_group(self, view: StateView) -> GroupId | None:
        return next((g for g in self.groups(view) if not self._complete(view, g)), None)

    def _backfill_devices(self, view: StateView, active: GroupId) -> set[DeviceId]:
        busy = {
            d
            for rt in self._members(view, active)
            if rt.state not in TERMINAL_STATES
            for d in rt.job.placement
        }
        return set(view.devices) - busy

    def _backfill_jobs(self, view: StateView, active: GroupId) -> list[JobRuntime]:
        if not self.backfill:
            return []
        free = self._backfill_devices(view, active)
        groups = self.groups(view)
        later = groups[groups.index(active) + 1 :]
        out: list[JobRuntime] = []
        for g in later:
            out.extend(
                sorted(
                    (
                        rt
                        for rt in self._members(view, g)
                        if rt.state is JobState.PENDING and set(rt.job.placement) <= free
                    ),
                    key=_by_position,
                )
            )
        return out

    # -- OrderingPolicy ------------------------------------------------------------

    def candidates(self, view: StateView, now: float) -> Iterable[JobId]:
        pending = [rt for rt in view.jobs.values() if rt.state is JobState.PENDING]
        active = self.active_group(view)
        stage = sorted(
            (rt for rt in pending if active is not None and rt.job.group == active),
            key=_by_position,
        )
        side = sorted((rt for rt in pending if rt.job.group is None), key=lambda r: r.submit_seq)
        backfill = self._backfill_jobs(view, active) if active is not None else []
        return [rt.id for rt in (*stage, *side, *backfill)]

    def explain_blocked(self, view: StateView, job_id: JobId) -> Explanation | None:
        rt = view.jobs[job_id]
        active = self.active_group(view)
        if rt.job.group is None or active is None or rt.job.group == active:
            return None
        if any(b.id == job_id for b in self._backfill_jobs(view, active)):
            return None
        return GroupBarrier(active)

    def completed_groups(self, view: StateView) -> Iterable[tuple[GroupId, GroupOutcome]]:
        out = []
        for g in self.groups(view):
            members = self._members(view, g)
            if members and self._complete(view, g):
                failed = any(rt.state is JobState.FAILED for rt in members)
                out.append((g, GroupOutcome.HAD_FAILURES if failed else GroupOutcome.OK))
        return out

    def queue_stops(self, view: StateView) -> Iterable[tuple[GroupId, JobId]]:
        return [
            (g, rt.id)
            for g, policy in sorted(self.failure_policy.items())
            if policy is FailurePolicy.STOP_QUEUE
            for rt in sorted(self._members(view, g), key=lambda r: r.submit_seq)
            if rt.state is JobState.FAILED
        ]
