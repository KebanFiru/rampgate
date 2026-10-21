"""`Engine`: the stateful, single-owner wrapper around ``reduce`` and ``decide`` (SPEC §4)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from rampgate.actions import Action, Unreconciled, Warn, WarnCleared, key_id, warning_key
from rampgate.config import Config
from rampgate.decide import PROBATION, Admission, Decision, decide
from rampgate.estimators import Estimator, NullEstimator, apply_history, resolve_estimates
from rampgate.events import (
    CancelRequested,
    DeviceAdded,
    DeviceObserved,
    DeviceRemoved,
    Event,
    JobExited,
    JobObserved,
    JobStarted,
    JobSubmitted,
    JobUpdated,
    Paused,
    PreemptRequested,
    ResumeRequested,
    Unpaused,
)
from rampgate.explain import Explanation
from rampgate.model import (
    ACTIVE_STATES,
    Device,
    DeviceId,
    ExitKind,
    Job,
    JobId,
    Snapshot,
    State,
    frozen_map,
)
from rampgate.policies.base import OrderingPolicy
from rampgate.reducer import InvalidTransition, drain, reduce
from rampgate.serialization import UnsupportedSchema, dump_state, load_state

__all__ = ["ENGINE_FORMAT", "Engine"]

ENGINE_FORMAT = "rampgate.engine"
_ENGINE_SCHEMA = 1


class Engine:
    """Owns the state; the caller serializes access (single-threaded).

    Every method except `tick` is a thin ``reduce`` call; `InvalidTransition`
    propagates to the caller. `tick` is the only place that talks to the
    estimator (history records out, estimates in) and runs ``decide``.
    """

    def __init__(
        self,
        devices: Iterable[Device],
        policy: OrderingPolicy,
        estimator: Estimator | None = None,
        config: Config | None = None,
        admission: Admission = PROBATION,
    ) -> None:
        self.policy = policy
        self.admission = admission
        self.estimator: Estimator = estimator or NullEstimator()
        self.config = config or Config()
        self._state = State.initial(devices)
        self._now: float | None = None
        self._unreconciled: set[JobId] = set()
        self._explanations: Mapping[JobId, Explanation] = frozen_map()
        self._last_snapshot: Snapshot | None = None
        self._reconcile_since: float | None = None

    # -- read-only views ---------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def now(self) -> float | None:
        """``now`` of the last tick, or None before the first one."""
        return self._now

    @property
    def explanations(self) -> Mapping[JobId, Explanation]:
        """Why each waiting job was not started, as of the last tick."""
        return self._explanations

    @property
    def last_snapshot(self) -> Snapshot | None:
        """The snapshot the last tick decided on (state plus resolved estimates)."""
        return self._last_snapshot

    def unreconciled(self) -> list[JobId]:
        """Jobs that were running at `load` and that nobody has reported on since.

        Reported by a start or exit, an observation naming the current attempt,
        or `confirm`. After ``reconcile_grace_s`` (counted from the first tick
        after `load`) each one raises ``Warn(Unreconciled)``; rampgate never
        declares a job LOST by itself.
        """
        return sorted(j for j in self._unreconciled if self._state.jobs[j].state in ACTIVE_STATES)

    # -- events ------------------------------------------------------------------

    def _apply(self, event: Event) -> None:
        self._state = reduce(self._state, event, self.config)

    def _fact(self, event: JobStarted | JobExited | JobObserved) -> None:
        """Apply a fact and update reconciliation.

        A start or exit that is not ignored as stale or duplicate reconciles its
        job; so does an observation that names the job's current attempt.
        """
        ignored_before = len(self._state.ignored_events)
        self._apply(event)
        if isinstance(event, JobObserved):
            rt = self._state.jobs[event.job_id]
            if event.attempt is not None and event.attempt == rt.attempt:
                self._unreconciled.discard(event.job_id)
        elif len(self._state.ignored_events) == ignored_before:
            self._unreconciled.discard(event.job_id)

    def confirm(self, job_id: JobId, attempt: int) -> None:
        """The caller has checked that ``attempt`` of ``job_id`` is still running as recorded."""
        rt = self._state.jobs.get(job_id)
        if rt is None:
            raise InvalidTransition(f"unknown job {job_id!r}")
        if attempt != rt.attempt:
            raise InvalidTransition(
                f"job {job_id!r}: confirm for attempt {attempt}, current attempt is {rt.attempt}"
            )
        self._unreconciled.discard(job_id)

    def apply(self, event: Event) -> None:
        """Apply any input event; facts go through the same path as the dedicated methods."""
        if isinstance(event, JobStarted | JobExited | JobObserved):
            self._fact(event)
        else:
            self._apply(event)

    def submit(self, job: Job, *, t: float | None = None) -> None:
        """Submit a job; ``t`` (for priority aging) defaults to the last tick's ``now``."""
        self._apply(JobSubmitted(job, self._now if t is None else t))

    def update(self, job_id: JobId, **fields: Any) -> None:
        self._apply(JobUpdated(job_id, **fields))

    def observe(self, event: JobObserved | DeviceObserved) -> None:
        if isinstance(event, JobObserved):
            self._fact(event)
        else:
            self._apply(event)

    def on_started(self, job_id: JobId, attempt: int, t: float) -> None:
        self._fact(JobStarted(job_id, attempt, t))

    def on_exited(
        self,
        job_id: JobId,
        attempt: int,
        kind: ExitKind,
        t: float,
        oom_dims: tuple[str, ...] | None = None,
    ) -> None:
        self._fact(JobExited(job_id, attempt, kind, t, oom_dims))

    def preempt(self, job_id: JobId) -> None:
        self._apply(PreemptRequested(job_id))

    def resume(self, job_id: JobId) -> None:
        self._apply(ResumeRequested(job_id))

    def cancel(self, job_id: JobId) -> None:
        self._apply(CancelRequested(job_id))

    def pause(self) -> None:
        self._apply(Paused())

    def unpause(self) -> None:
        self._apply(Unpaused())

    def add_device(self, device: Device) -> None:
        self._apply(DeviceAdded(device))

    def remove_device(self, device_id: DeviceId) -> None:
        self._apply(DeviceRemoved(device_id))

    # -- tick ----------------------------------------------------------------------

    def tick(self, now: float) -> Decision:
        """Hand history to the estimator, resolve estimates, decide, and apply the actions.

        The returned decision carries the events ignored since the previous tick.
        """
        if self._now is not None and now < self._now:
            raise ValueError(f"time went backwards: tick({now}) after tick({self._now})")
        state, ignored, history = drain(self._state)
        apply_history(self.estimator, state, history)
        snapshot = resolve_estimates(state, self.estimator)
        decision = decide(snapshot, now, self.config, self.policy, self.admission)
        for action in decision.actions:
            state = reduce(state, action, self.config)
        self._state = state
        self._now = now
        self._explanations = decision.explanations
        self._last_snapshot = snapshot
        extra = self._unreconciled_warnings(now)
        return replace(decision, actions=(*decision.actions, *extra), ignored_events=ignored)

    def _unreconciled_warnings(self, now: float) -> list[Action]:
        """Edge-triggered ``Warn(Unreconciled)`` after ``reconcile_grace_s`` without news."""
        if self._unreconciled and self._reconcile_since is None:
            self._reconcile_since = now
        overdue = (
            self._reconcile_since is not None
            and now - self._reconcile_since >= self.config.reconcile_grace_s
        )
        current: dict[str, Unreconciled] = {}
        if overdue:
            for job_id in self.unreconciled():
                w = Unreconciled(job_id, self._state.jobs[job_id].attempt)
                current[key_id(*warning_key(w))] = w
        active = {
            k: w for k, w in self._state.active_warnings.items() if isinstance(w, Unreconciled)
        }
        actions: list[Action] = [Warn(current[k]) for k in sorted(current.keys() - active.keys())]
        actions += [
            WarnCleared(*warning_key(active[k])) for k in sorted(active.keys() - current.keys())
        ]
        for action in actions:
            self._state = reduce(self._state, action, self.config)
        return actions

    # -- persistence ------------------------------------------------------------------

    def dump(self) -> dict[str, Any]:
        """Versioned, JSON-serializable engine state.

        The policy, estimator, config and admission are not part of it.
        """
        return {
            "format": ENGINE_FORMAT,
            "schema_version": _ENGINE_SCHEMA,
            "now": self._now,
            "state": dump_state(self._state),
        }

    @classmethod
    def load(
        cls,
        data: Mapping[str, Any],
        *,
        policy: OrderingPolicy,
        estimator: Estimator | None = None,
        config: Config | None = None,
        admission: Admission = PROBATION,
    ) -> Engine:
        """Restore a dumped engine. Every job that was running is then *unreconciled*.

        Report reality for those jobs (`on_started`, `on_exited` with ``ExitKind.LOST``,
        or observations) — see ``docs/reconciliation.md``.
        """
        if not isinstance(data, Mapping) or data.get("format") != ENGINE_FORMAT:
            raise UnsupportedSchema("not a rampgate engine dump")
        if data.get("schema_version") != _ENGINE_SCHEMA:
            raise UnsupportedSchema(f"unsupported engine schema {data.get('schema_version')!r}")
        engine = cls((), policy, estimator, config, admission)
        engine._state = load_state(data["state"])
        now = data.get("now")
        engine._now = None if now is None else float(now)
        engine._unreconciled = {
            j for j, rt in engine._state.jobs.items() if rt.state in ACTIVE_STATES
        }
        return engine
