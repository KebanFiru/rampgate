"""`SimExecutor`: applies actions to simulated processes and reports what happens (SPEC §8)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from rampgate.actions import Action, Requeue, RequeueReason, Start, Stop
from rampgate.events import DeviceObserved, Event, JobExited, JobObserved, JobStarted
from rampgate.model import Device, DeviceId, ExitKind, Job, JobId, StopReason
from rampgate.resources import Resources
from rampgate.sim.workloads import Workload

__all__ = ["SimExecutor"]


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
    - ``Stop``: the process exits PREEMPTED/CANCELLED after ``stop_delay_s`` and
      keeps all its progress (checkpoint timing comes with L4).
    - After an OOM, progress rolls back to the last checkpoint.
    - ``reset_counters_on_resume`` makes resumed attempts count steps/epochs from 0.
    """

    def __init__(
        self,
        devices: Iterable[Device],
        workloads: Mapping[JobId, Workload],
        *,
        foreign: Mapping[DeviceId, Resources] | None = None,
        start_delay_s: float = 1.0,
        stop_delay_s: float = 1.0,
        reset_counters_on_resume: bool = False,
        seed: int = 0,
    ) -> None:
        self.devices = {d.id: d for d in devices}
        self.workloads = dict(workloads)
        self.foreign = dict(foreign or {})
        self.start_delay_s = start_delay_s
        self.stop_delay_s = stop_delay_s
        self.reset_counters_on_resume = reset_counters_on_resume
        self.rng = random.Random(seed)
        self.procs: dict[JobId, _Proc] = {}
        self.progress: dict[JobId, float] = {}
        self.jobs: dict[JobId, Job] = {}

    def apply(self, actions: Iterable[Action], now: float, jobs: Mapping[JobId, Job]) -> None:
        for action in actions:
            if isinstance(action, Start):
                self.jobs[action.job_id] = jobs[action.job_id]
                self.procs[action.job_id] = _Proc(jobs[action.job_id], action.attempt, now)
            elif isinstance(action, Requeue) and action.reason is RequeueReason.START_TIMEOUT:
                proc = self.procs.get(action.job_id)
                if proc is not None and proc.attempt < action.next_attempt:
                    del self.procs[action.job_id]
            elif isinstance(action, Stop):
                proc = self.procs.get(action.job_id)
                if proc is not None and proc.attempt == action.attempt and proc.stop is None:
                    proc.stop = (action.reason, now + self.stop_delay_s)

    def _counters(self, job_id: JobId, elapsed: float) -> tuple[int, int | None]:
        workload = self.workloads[job_id]
        before = 0.0 if self.reset_counters_on_resume else self.progress.get(job_id, 0.0)
        steps = math.floor(workload.steps_per_s * (before + elapsed))
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
        oom_dims: tuple[str, ...] | None = None,
    ) -> Event:
        job_id = proc.job.id
        self.progress[job_id] = self.progress.get(job_id, 0.0) + kept
        del self.procs[job_id]
        return JobExited(job_id, proc.attempt, kind, now, oom_dims)

    def advance(self, now: float) -> list[Event]:
        """Everything that happened up to ``now``, in a deterministic order."""
        events: list[Event] = []
        for job_id in sorted(self.procs):
            proc = self.procs[job_id]
            if proc.began is None and now >= proc.issued_at + self.start_delay_s:
                proc.began = now
                events.append(JobStarted(job_id, proc.attempt, now))

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

        totals: dict[DeviceId, Resources] = {
            d: self.foreign.get(d, Resources()) for d in self.devices
        }
        for job_id in sorted(self.procs):
            proc = self.procs[job_id]
            if proc.began is None:
                continue
            steps, epoch = self._counters(job_id, now - proc.began)
            for d in proc.job.devices:
                totals[d] = totals[d] + proc.usage
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
                total = self.foreign.get(d, Resources())
                for p in running:
                    total = total + p.usage
                over = total.exceeds(self.devices[d].capacity)
                if not over or not running:
                    break
                victim = max(running, key=lambda p: (p.grew_at, p.job.id))
                assert victim.began is not None
                elapsed = now - victim.began
                ckpt = self.workloads[victim.job.id].checkpoint_every_s
                kept = math.floor(elapsed / ckpt) * ckpt if ckpt > 0 else 0.0
                events.append(self._end(victim, kept, ExitKind.OOM, now, oom_dims=tuple(over)))
        return events


def _epochs(workload: Workload, progress: float) -> int:
    epoch = workload.epoch(progress)
    return 0 if epoch is None else epoch
