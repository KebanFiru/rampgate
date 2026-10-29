"""`SimExecutor`: applies actions to simulated processes and reports what happens
(docs/simulator.md)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from rampgate.actions import Action, Requeue, RequeueReason, Start, Stop
from rampgate.events import DeviceObserved, Event, JobExited, JobObserved, JobStarted
from rampgate.model import Device, DeviceId, ExitKind, Job, JobId, StopReason
from rampgate.resources import Resources
from rampgate.sim.workloads import Workload

__all__ = ["ForeignLoad", "SimExecutor", "TimelineEntry", "WorldStats"]

# Constant foreign load, or a step schedule: the last ``(t, usage)`` with ``t <= now``.
ForeignLoad = Resources | Sequence[tuple[float, Resources]]


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """A process began (``kind="start"``) or exited (``kind`` = the ``ExitKind`` value).

    ``device`` is the device whose overcommit caused an OOM kill (None otherwise).
    """

    t: float
    job_id: JobId
    attempt: int
    kind: str
    device: DeviceId | None = None


@dataclass(frozen=True, slots=True)
class WorldStats:
    """Ground truth the executor measured; the source of `rampgate.sim.Report`.

    - ``oom_count``: OOM kills, including those nobody heard of during downtime.
    - ``lost_work_step_s``: steps thrown away (OOM rollback to the last
      checkpoint, a running process killed on a start timeout or replaced).
    - ``usage_s``: per device, job usage integrated over time (foreign excluded).
    - ``max_overcommit``: per device, the largest ``(F + sum(usage) - C) / C``
      in any dimension, measured before OOM kills resolve it (0 if never over).
    - ``observed_until``: the last ``advance`` time.
    """

    oom_count: int
    lost_work_step_s: float
    usage_s: Mapping[DeviceId, Resources]
    max_overcommit: Mapping[DeviceId, float]
    timeline: tuple[TimelineEntry, ...]
    observed_until: float | None


@dataclass(slots=True)
class _Proc:
    job: Job
    attempt: int
    issued_at: float
    began: float | None = None
    stop: tuple[StopReason, float] | None = None
    usage: Resources = field(default_factory=Resources)
    grew_at: float = -math.inf


class SimExecutor:
    """Deterministic stand-in for real processes.

    - ``Start``: the process reports ``JobStarted`` after ``start_delay_s``.
    - Every ``advance`` reports usage per device, the job's cumulative ``steps``
      and ``epoch`` counters, and device totals (including constant foreign load).
    - OOM model: while a device's total usage exceeds its capacity, the job whose
      usage grew most recently (ties: job id) is killed with ``OOM``.
    - A job whose progress reaches its workload duration exits ``SUCCESS``.
    - ``Stop(PREEMPT)``: the process saves a clean checkpoint and exits PREEMPTED
      after ``ckpt_save_s``, keeping all its progress. ``Stop(CANCEL)``: it exits
      CANCELLED after ``stop_delay_s``.
    - After an OOM, progress rolls back to the last checkpoint; the rest is lost work.
    - ``foreign``: per device, constant load or a step schedule (`ForeignLoad`).
    - ``reset_counters_on_resume`` makes resumed attempts count steps/epochs from 0.
    """

    def __init__(
        self,
        devices: Iterable[Device],
        workloads: Mapping[JobId, Workload],
        *,
        foreign: Mapping[DeviceId, ForeignLoad] | None = None,
        start_delay_s: float = 1.0,
        stop_delay_s: float = 1.0,
        ckpt_save_s: float = 5.0,
        reset_counters_on_resume: bool = False,
        seed: int = 0,
    ) -> None:
        self.devices = {d.id: d for d in devices}
        self.workloads = dict(workloads)
        self.foreign = dict(foreign or {})
        self.start_delay_s = start_delay_s
        self.stop_delay_s = stop_delay_s
        self.ckpt_save_s = ckpt_save_s
        self.reset_counters_on_resume = reset_counters_on_resume
        self.rng = random.Random(seed)
        self.procs: dict[JobId, _Proc] = {}
        self.progress: dict[JobId, float] = {}
        self.jobs: dict[JobId, Job] = {}
        self.exits: dict[tuple[JobId, int], JobExited] = {}
        self.oom_count = 0
        self.lost_work_step_s = 0.0
        self.usage_s: dict[DeviceId, Resources] = {d: Resources() for d in self.devices}
        self.max_overcommit: dict[DeviceId, float] = dict.fromkeys(self.devices, 0.0)
        self.timeline: list[TimelineEntry] = []
        self._last_t: float | None = None

    def foreign_at(self, device_id: DeviceId, now: float) -> Resources:
        load = self.foreign.get(device_id)
        if load is None:
            return Resources()
        if isinstance(load, Resources):
            return load
        current = Resources()
        for start, value in sorted(load, key=lambda s: s[0]):
            if start <= now:
                current = value
        return current

    def stats(self) -> WorldStats:
        return WorldStats(
            self.oom_count,
            self.lost_work_step_s,
            dict(self.usage_s),
            dict(self.max_overcommit),
            tuple(self.timeline),
            self._last_t,
        )

    def _discard(self, proc: _Proc, now: float) -> None:
        """A process is killed without an exit report: everything since it began is lost."""
        if proc.began is not None:
            rate = self.workloads[proc.job.id].steps_per_s
            self.lost_work_step_s += rate * (now - proc.began)

    def apply(self, actions: Iterable[Action], now: float, jobs: Mapping[JobId, Job]) -> None:
        for action in actions:
            if isinstance(action, Start):
                old = self.procs.get(action.job_id)
                if old is not None:
                    self._discard(old, now)
                self.jobs[action.job_id] = jobs[action.job_id]
                self.procs[action.job_id] = _Proc(jobs[action.job_id], action.attempt, now)
            elif isinstance(action, Requeue) and action.reason is RequeueReason.START_TIMEOUT:
                proc = self.procs.get(action.job_id)
                if proc is not None and proc.attempt < action.next_attempt:
                    self._discard(proc, now)
                    del self.procs[action.job_id]
            elif isinstance(action, Stop):
                proc = self.procs.get(action.job_id)
                if proc is not None and proc.attempt == action.attempt and proc.stop is None:
                    delay = (
                        self.ckpt_save_s
                        if action.reason is StopReason.PREEMPT
                        else self.stop_delay_s
                    )
                    proc.stop = (action.reason, now + delay)

    def _counters(self, job_id: JobId, elapsed: float) -> tuple[int, int | None]:
        workload = self.workloads[job_id]
        before = 0.0 if self.reset_counters_on_resume else self.progress.get(job_id, 0.0)
        steps = math.floor(workload.steps_per_s * before) + math.floor(
            workload.steps_per_s * elapsed
        )
        epoch = workload.epoch(elapsed)
        if epoch is not None and not self.reset_counters_on_resume:
            epoch += _epochs(workload, self.progress.get(job_id, 0.0))
        return steps, epoch

    def _end(
        self,
        proc: _Proc,
        kept: float,
        kind: ExitKind,
        now: float,
        *,
        oom_dims: tuple[str, ...] | None = None,
        device: DeviceId | None = None,
    ) -> Event:
        job_id = proc.job.id
        self.progress[job_id] = self.progress.get(job_id, 0.0) + kept
        if proc.began is not None:
            elapsed = now - proc.began
            self.lost_work_step_s += self.workloads[job_id].steps_per_s * max(0.0, elapsed - kept)
        del self.procs[job_id]
        exit_event = JobExited(job_id, proc.attempt, kind, now, oom_dims)
        self.exits[(job_id, proc.attempt)] = exit_event
        self.timeline.append(TimelineEntry(now, job_id, proc.attempt, kind.value, device))
        if kind is ExitKind.OOM:
            self.oom_count += 1
        return exit_event

    def truth(self, job_id: JobId, attempt: int) -> tuple[str, float | None, JobExited | None]:
        """Ground truth for reconciliation: ``("running", began, None)``,
        ``("starting", None, None)``, ``("exited", None, exit)`` or ``("unknown", None, None)``.
        """
        proc = self.procs.get(job_id)
        if proc is not None and proc.attempt == attempt:
            if proc.began is None:
                return "starting", None, None
            return "running", proc.began, None
        exit_event = self.exits.get((job_id, attempt))
        if exit_event is not None:
            return "exited", None, exit_event
        return "unknown", None, None

    def advance(self, now: float) -> list[Event]:
        """Everything that happened up to ``now``, in a deterministic order."""
        events: list[Event] = []
        for job_id in sorted(self.procs):
            proc = self.procs[job_id]
            if proc.began is None and now >= proc.issued_at + self.start_delay_s:
                proc.began = now
                events.append(JobStarted(job_id, proc.attempt, now))
                self.timeline.append(TimelineEntry(now, job_id, proc.attempt, "start"))

        for job_id in sorted(self.procs):
            proc = self.procs[job_id]
            elapsed = 0.0 if proc.began is None else now - proc.began
            if proc.stop is not None and now >= proc.stop[1]:
                kind = (
                    ExitKind.PREEMPTED if proc.stop[0] is StopReason.PREEMPT else ExitKind.CANCELLED
                )
                events.append(self._end(proc, elapsed, kind, now))
                continue
            workload = self.workloads[job_id]
            if proc.began is not None:
                done = self.progress.get(job_id, 0.0) + elapsed >= workload.duration_s
                if done:
                    events.append(self._end(proc, elapsed, ExitKind.SUCCESS, now))
                    continue
                usage = workload.usage(elapsed)
                if workload.noise:
                    usage = usage.scale(max(0.0, 1 + workload.noise * self.rng.uniform(-1, 1)))
                if usage.exceeds(proc.usage):
                    proc.grew_at = now
                proc.usage = usage

        events.extend(self._oom(now))

        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        totals: dict[DeviceId, Resources] = {d: self.foreign_at(d, now) for d in self.devices}
        for job_id in sorted(self.procs):
            proc = self.procs[job_id]
            if proc.began is None:
                continue
            steps: int | None
            steps, epoch = self._counters(job_id, now - proc.began)
            if not getattr(self.workloads[job_id], "reports_steps", True):
                steps = None
            for d in proc.job.devices:
                totals[d] = totals[d] + proc.usage
                self.usage_s[d] = self.usage_s[d] + proc.usage.scale(dt)
                events.append(
                    JobObserved(
                        job_id, d, now, proc.usage, steps=steps, epoch=epoch, attempt=proc.attempt
                    )
                )
        events.extend(DeviceObserved(d, now, totals[d]) for d in sorted(totals))
        return events

    def _oom(self, now: float) -> list[Event]:
        events: list[Event] = []
        for d in sorted(self.devices):
            while True:
                running = [
                    p for p in self.procs.values() if p.began is not None and d in p.job.placement
                ]
                total = self.foreign_at(d, now)
                for p in running:
                    total = total + p.usage
                cap = self.devices[d].capacity
                over = total.exceeds(cap)
                for dim in over:
                    if cap[dim] > 0:
                        ratio = (total[dim] - cap[dim]) / cap[dim]
                        self.max_overcommit[d] = max(self.max_overcommit[d], ratio)
                if not over or not running:
                    break
                victim = max(running, key=lambda p: (p.grew_at, p.job.id))
                assert victim.began is not None
                elapsed = now - victim.began
                ckpt = self.workloads[victim.job.id].checkpoint_every_s
                kept = math.floor(elapsed / ckpt) * ckpt if ckpt > 0 else 0.0
                events.append(
                    self._end(victim, kept, ExitKind.OOM, now, oom_dims=tuple(over), device=d)
                )
        return events


def _epochs(workload: Workload, progress: float) -> int:
    epoch = workload.epoch(progress)
    return 0 if epoch is None else epoch
