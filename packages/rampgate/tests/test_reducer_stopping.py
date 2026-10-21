"""The 'exits while STOPPING' table of SPEC §3.4. A stopped job is never re-queued nor FAILED."""

import pytest

from rampgate import (
    CancelRequested,
    Config,
    ExitKind,
    HistoryRecord,
    JobState,
    PreemptRequested,
    Resources,
    ResumeRequested,
    StopReason,
    UncleanStop,
    Warn,
)
from rampgate_testkit import GPU, Driver

P, C = StopReason.PREEMPT, StopReason.CANCEL

TABLE = [
    (ExitKind.SUCCESS, P, JobState.SUCCEEDED),
    (ExitKind.SUCCESS, C, JobState.SUCCEEDED),
    (ExitKind.OOM, P, JobState.PREEMPTED),
    (ExitKind.OOM, C, JobState.CANCELLED),
    (ExitKind.FAILURE, P, JobState.PREEMPTED),
    (ExitKind.FAILURE, C, JobState.CANCELLED),
    (ExitKind.PREEMPTED, P, JobState.PREEMPTED),
    (ExitKind.PREEMPTED, C, JobState.PREEMPTED),
    (ExitKind.CANCELLED, P, JobState.CANCELLED),
    (ExitKind.CANCELLED, C, JobState.CANCELLED),
    (ExitKind.LOST, P, JobState.PREEMPTED),
    (ExitKind.LOST, C, JobState.CANCELLED),
]


def stopping(reason: StopReason, config: Config | None = None) -> Driver:
    d = Driver(config=config).to(JobState.WARMING).observe(1.0, vram=40).stable()
    return d.stop(reason)


@pytest.mark.parametrize(("kind", "reason", "end"), TABLE)
def test_stopping_exit_table(kind: ExitKind, reason: StopReason, end: JobState) -> None:
    d = stopping(reason).exited(kind)
    rt = d.rt()
    assert rt.state is end
    assert rt.attempt == 1
    assert rt.stop_intent is None
    assert rt.reservation == {}


@pytest.mark.parametrize(("kind", "reason", "end"), TABLE)
def test_exit_with_requested_but_unsent_stop_follows_table(
    kind: ExitKind, reason: StopReason, end: JobState
) -> None:
    d = Driver().to(JobState.STABLE)
    d(PreemptRequested("j") if reason is P else CancelRequested("j"))
    assert d.rt().state is JobState.STABLE
    d.exited(kind)
    assert d.rt().state is end


def test_success_while_stopping_records_history() -> None:
    d = stopping(P).exited(ExitKind.SUCCESS)
    assert d.state.history_records == (
        HistoryRecord("j", GPU, Resources(vram=40), True, 1, running=True),
        HistoryRecord("j", GPU, Resources(vram=40), False, 1),
    )


def test_oom_while_preempting_learns_but_does_not_fail() -> None:
    cfg = Config(max_oom_retries=0)
    d = stopping(P, cfg).room(vram=40).exited(ExitKind.OOM)
    rt = d.rt()
    assert rt.state is JobState.PREEMPTED
    assert rt.oom_count == 1
    assert rt.learned[GPU]["vram"] == pytest.approx(40 * 1.25)
    assert d.state.history_records == (
        HistoryRecord("j", GPU, Resources(vram=40), True, 1, running=True),
        HistoryRecord("j", GPU, Resources(vram=40), True, 1),
    )
    assert d.state.outbox == ()


def test_oom_while_preempting_at_capacity_alone_does_not_fail() -> None:
    d = Driver().submit("j", {GPU: Resources(vram=100)}).start().started()
    d.stop(P).exited(ExitKind.OOM)
    assert d.rt().state is JobState.PREEMPTED


def test_oom_while_cancelling_records_peak_as_lower_bound() -> None:
    d = stopping(C).exited(ExitKind.OOM)
    assert d.rt().oom_count == 0
    assert d.state.history_records == (
        HistoryRecord("j", GPU, Resources(vram=40), True, 1, running=True),
        HistoryRecord("j", GPU, Resources(vram=40), True, 1),
    )


def test_failure_while_preempting_warns_unclean_stop() -> None:
    d = stopping(P).exited(ExitKind.FAILURE)
    assert d.state.outbox == (Warn(UncleanStop("j", 1)),)
    d2 = stopping(C).exited(ExitKind.FAILURE)
    assert d2.state.outbox == ()


def test_stopped_job_resumes_with_next_attempt() -> None:
    d = stopping(P).exited(ExitKind.OOM)
    d(ResumeRequested("j"))
    assert (d.rt().state, d.rt().attempt) == (JobState.PENDING, 2)
    start = d.start_action()
    assert start.resume
