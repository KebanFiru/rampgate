"""Small driver that feeds events/actions through `reduce` for tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rampgate import (
    CancelRequested,
    Config,
    Device,
    ExitKind,
    IgnoreReason,
    Job,
    JobExited,
    JobObserved,
    JobRuntime,
    JobStarted,
    JobState,
    JobSubmitted,
    MarkStable,
    Message,
    PreemptRequested,
    Resources,
    Start,
    State,
    Stop,
    StopReason,
    reduce,
)

GPU = "gpu:0"
GPU1 = "gpu:1"
CAP = Resources(vram=100)


def gpus(*ids: str, cap: Resources = CAP) -> list[Device]:
    return [Device(i, cap) for i in (ids or (GPU,))]


class Driver:
    def __init__(self, devices: Iterable[Device] | None = None, config: Config | None = None):
        self.config = config or Config()
        self.state = State.initial(gpus() if devices is None else devices)

    def __call__(self, *msgs: Message) -> Driver:
        for m in msgs:
            self.state = reduce(self.state, m, self.config)
        return self

    def rt(self, job_id: str = "j") -> JobRuntime:
        return self.state.jobs[job_id]

    @property
    def ignored(self) -> list[IgnoreReason]:
        return [i.reason for i in self.state.ignored_events]

    def submit(
        self,
        job_id: str = "j",
        placement: Mapping[str, Resources | None] | None = None,
        **kw: object,
    ) -> Driver:
        job = Job(job_id, placement if placement is not None else {GPU: None}, **kw)  # type: ignore[arg-type]
        return self(JobSubmitted(job))

    def start_action(self, job_id: str = "j", free: Mapping[str, Resources] | None = None) -> Start:
        rt = self.rt(job_id)
        devices = rt.job.devices
        admitted = free if free is not None else {d: Resources() for d in devices}
        return Start(job_id, rt.attempt, devices, rt.has_run, admitted)

    def start(self, job_id: str = "j", free: Mapping[str, Resources] | None = None) -> Driver:
        return self(self.start_action(job_id, free))

    def started(self, job_id: str = "j", t: float = 0.0) -> Driver:
        return self(JobStarted(job_id, self.rt(job_id).attempt, t))

    def observe(
        self,
        t: float,
        job_id: str = "j",
        device: str = GPU,
        steps: int | None = None,
        attempt: int | None = None,
        **usage: float,
    ) -> Driver:
        return self(JobObserved(job_id, device, t, Resources(usage), steps, attempt))

    def exited(
        self,
        kind: ExitKind,
        job_id: str = "j",
        t: float = 100.0,
        oom_dims: tuple[str, ...] | None = None,
    ) -> Driver:
        return self(JobExited(job_id, self.rt(job_id).attempt, kind, t, oom_dims))

    def mark_stable_action(self, job_id: str = "j") -> MarkStable:
        rt = self.rt(job_id)
        reservation = {
            d: rt.peak.get(d, Resources()).scale(self.config.safety_margin) for d in rt.job.devices
        }
        return MarkStable(job_id, rt.attempt, reservation)

    def stable(self, job_id: str = "j") -> Driver:
        return self(self.mark_stable_action(job_id))

    def stop(self, reason: StopReason, job_id: str = "j") -> Driver:
        event = (
            PreemptRequested(job_id) if reason is StopReason.PREEMPT else CancelRequested(job_id)
        )
        self(event)
        return self(Stop(job_id, self.rt(job_id).attempt, reason, self.config.preempt_grace_s))

    def to(self, target: JobState, job_id: str = "j", **submit_kw: object) -> Driver:
        """Drive a fresh job to ``target`` along the main path."""
        self.submit(job_id, **submit_kw)  # type: ignore[arg-type]
        if target is JobState.PENDING:
            return self
        if target is JobState.CANCELLED:
            return self(CancelRequested(job_id))
        self.start(job_id)
        if target is JobState.STARTING:
            return self
        self.started(job_id, t=0.0)
        if target is JobState.WARMING:
            return self
        if target is JobState.STOPPING:
            return self.stop(StopReason.PREEMPT, job_id)
        if target is JobState.PREEMPTED:
            return self.stop(StopReason.PREEMPT, job_id).exited(ExitKind.PREEMPTED, job_id)
        self.observe(1.0, job_id, vram=10).stable(job_id)
        if target is JobState.STABLE:
            return self
        if target is JobState.SUCCEEDED:
            return self.exited(ExitKind.SUCCESS, job_id)
        if target is JobState.FAILED:
            return self.exited(ExitKind.FAILURE, job_id)
        raise AssertionError(target)
