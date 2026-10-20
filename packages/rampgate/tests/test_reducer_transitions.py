"""Every lifecycle transition of SPEC §3.4, valid and invalid."""

import pytest

from rampgate import (
    UNSET,
    CancelRequested,
    Config,
    Device,
    DeviceAdded,
    DeviceRemoved,
    ExitKind,
    Fail,
    FailReason,
    GroupCompleted,
    GroupOutcome,
    HistoryRecord,
    IgnoreReason,
    InvalidTransition,
    Job,
    JobExited,
    JobStarted,
    JobState,
    JobSubmitted,
    JobUpdated,
    MarkStable,
    Paused,
    PreemptRequested,
    Requeue,
    RequeueReason,
    Resources,
    ResumeRequested,
    Start,
    State,
    Stop,
    StopReason,
    Unpaused,
    drain,
    reduce,
)
from rampgate_testkit import GPU, GPU1, Driver, gpus

RUNNING = [JobState.STARTING, JobState.WARMING, JobState.STABLE]
TERMINAL = [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED]


# -- submit ----------------------------------------------------------------------------


def test_submit_creates_pending_attempt_1() -> None:
    d = Driver().submit("a").submit("b")
    assert d.rt("a").state is JobState.PENDING
    assert d.rt("a").attempt == 1
    assert (d.rt("a").submit_seq, d.rt("b").submit_seq) == (0, 1)


def test_submit_duplicate_identical_is_ignored() -> None:
    d = Driver().submit("a").submit("a")
    assert d.ignored == [IgnoreReason.DUPLICATE]
    assert d.state.next_seq == 1


def test_submit_same_id_different_content_raises() -> None:
    d = Driver().submit("a")
    with pytest.raises(InvalidTransition, match="different content"):
        d.submit("a", priority=5)


def test_submit_unknown_device_raises() -> None:
    with pytest.raises(InvalidTransition, match="unknown devices"):
        Driver().submit("a", {"nope": None})


def test_unknown_job_raises() -> None:
    with pytest.raises(InvalidTransition, match="unknown job"):
        Driver()(JobStarted("ghost", 1, 0.0))


# -- PENDING -> STARTING ---------------------------------------------------------------


def test_start_moves_to_starting_and_records_admission() -> None:
    d = Driver().submit().start(free={GPU: Resources(vram=60)})
    rt = d.rt()
    assert rt.state is JobState.STARTING
    assert rt.admitted_free == {GPU: Resources(vram=60)}
    assert rt.admitted_alone == {GPU: True}


def test_start_not_alone_with_other_active_job() -> None:
    d = Driver().submit("a").start("a").submit("b").start("b")
    assert d.rt("b").admitted_alone == {GPU: False}
    # A finished neighbour does not count.
    d2 = Driver().to(JobState.SUCCEEDED, "a").submit("b").start("b")
    assert d2.rt("b").admitted_alone == {GPU: True}


def test_start_twice_is_a_noop() -> None:
    d = Driver().submit()
    action = d.start_action()
    d(action)
    before = d.state
    d(action)
    assert d.state == before


def test_start_from_preempted_raises() -> None:
    d = Driver().submit()(PreemptRequested("j"))
    with pytest.raises(InvalidTransition, match="only a PENDING"):
        d.start()


def test_start_with_wrong_placement_raises() -> None:
    d = Driver(gpus(GPU, GPU1)).submit()
    with pytest.raises(InvalidTransition, match="placement"):
        d(Start("j", 1, (GPU1,), False, {}))


def test_start_future_attempt_raises_and_stale_is_noop() -> None:
    d = Driver().submit()
    with pytest.raises(InvalidTransition, match="future"):
        d(Start("j", 2, (GPU,), False, {}))
    d = Driver().to(JobState.WARMING).exited(ExitKind.OOM)
    before = d.state
    d(Start("j", 1, (GPU,), False, {}))
    assert d.state == before


# -- STARTING -> WARMING ---------------------------------------------------------------


def test_started_moves_to_warming() -> None:
    d = Driver().submit().start().started(t=5.0)
    assert d.rt().state is JobState.WARMING
    assert d.rt().warm_start == 5.0
    assert d.rt().has_run


def test_started_without_start_raises() -> None:
    with pytest.raises(InvalidTransition, match="never told to start"):
        Driver().submit().started()


def test_started_future_attempt_raises() -> None:
    d = Driver().submit().start()
    with pytest.raises(InvalidTransition, match="future"):
        d(JobStarted("j", 2, 0.0))


def test_started_after_stop_sent_while_starting_stays_stopping() -> None:
    d = Driver().submit().start().stop(StopReason.PREEMPT)
    assert d.rt().state is JobState.STOPPING
    d.started(t=3.0)
    assert d.rt().state is JobState.STOPPING
    assert d.rt().warm_start == 3.0


# -- WARMING -> STABLE -----------------------------------------------------------------


def test_mark_stable() -> None:
    d = Driver().to(JobState.WARMING).observe(1.0, vram=40).stable()
    rt = d.rt()
    assert rt.state is JobState.STABLE
    assert rt.reservation == {GPU: Resources(vram=40).scale(1.15)}


def test_mark_stable_twice_is_noop() -> None:
    d = Driver().to(JobState.WARMING)
    action = d.mark_stable_action()
    d(action)
    before = d.state
    d(action)
    assert d.state == before


@pytest.mark.parametrize("state", [JobState.PENDING, JobState.STARTING])
def test_mark_stable_requires_warming(state: JobState) -> None:
    d = Driver().to(state)
    with pytest.raises(InvalidTransition, match="WARMING"):
        d(MarkStable("j", 1, {GPU: Resources()}))


def test_mark_stable_reservation_must_cover_placement() -> None:
    d = Driver().to(JobState.WARMING)
    with pytest.raises(InvalidTransition, match="cover"):
        d(MarkStable("j", 1, {}))
    with pytest.raises(InvalidTransition, match="future"):
        d(MarkStable("j", 2, {GPU: Resources()}))


# -- exits from running states ---------------------------------------------------------


@pytest.mark.parametrize("state", RUNNING)
def test_exit_success(state: JobState) -> None:
    d = Driver().to(state)
    d.exited(ExitKind.SUCCESS)
    assert d.rt().state is JobState.SUCCEEDED
    assert d.rt().reservation == {}


def test_exit_success_records_history_only_for_observed_devices() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("j", {GPU: None, GPU1: None}).start().started()
    d.observe(1.0, device=GPU, vram=30).observe(2.0, device=GPU, vram=20)
    d.exited(ExitKind.SUCCESS)
    assert d.state.history_records == (HistoryRecord("j", GPU, Resources(vram=30), False),)


@pytest.mark.parametrize("state", RUNNING)
def test_exit_failure(state: JobState) -> None:
    d = Driver().to(state).exited(ExitKind.FAILURE)
    assert d.rt().state is JobState.FAILED
    assert d.state.outbox == (Fail("j", 1, FailReason.EXITED_FAILURE),)


@pytest.mark.parametrize("state", RUNNING)
def test_exit_oom_requeues(state: JobState) -> None:
    d = Driver().to(state).exited(ExitKind.OOM)
    rt = d.rt()
    assert rt.state is JobState.PENDING
    assert rt.attempt == 2
    assert rt.oom_count == 1
    (requeue,) = d.state.outbox
    assert isinstance(requeue, Requeue)
    assert (requeue.next_attempt, requeue.reason) == (2, RequeueReason.OOM)


@pytest.mark.parametrize("state", RUNNING)
def test_exit_preempted_without_stop(state: JobState) -> None:
    d = Driver().to(state).exited(ExitKind.PREEMPTED)
    assert d.rt().state is JobState.PREEMPTED
    d(ResumeRequested("j"))
    assert (d.rt().state, d.rt().attempt) == (JobState.PENDING, 2)
    assert d.start_action().resume  # any exit means the attempt ran


@pytest.mark.parametrize("state", RUNNING)
def test_exit_cancelled_without_stop(state: JobState) -> None:
    d = Driver().to(state).exited(ExitKind.CANCELLED)
    assert d.rt().state is JobState.CANCELLED


@pytest.mark.parametrize("state", [JobState.PENDING, *TERMINAL])
def test_exit_without_running_raises(state: JobState) -> None:
    d = Driver().to(state)
    if state in (JobState.SUCCEEDED, JobState.FAILED):
        d.exited(ExitKind.SUCCESS)  # replay of the applied exit: ignored
        assert d.ignored == [IgnoreReason.DUPLICATE]
        return
    with pytest.raises(InvalidTransition, match="never started"):
        d.exited(ExitKind.SUCCESS)


def test_exit_future_attempt_raises() -> None:
    d = Driver().to(JobState.WARMING)
    with pytest.raises(InvalidTransition, match="future"):
        d(JobExited("j", 2, ExitKind.SUCCESS, 1.0))


# -- LOST ------------------------------------------------------------------------------


@pytest.mark.parametrize("state", RUNNING)
def test_lost_requeues_without_oom_accounting(state: JobState) -> None:
    d = Driver().to(state)
    before = d.rt()
    d.exited(ExitKind.LOST)
    rt = d.rt()
    assert (rt.state, rt.attempt, rt.lost_count) == (JobState.PENDING, 2, 1)
    assert rt.oom_count == before.oom_count
    assert rt.learned == before.learned
    assert rt.reservation == {}
    assert d.state.history_records == ()
    (requeue,) = d.state.outbox
    assert requeue == Requeue("j", 2, {GPU: None}, RequeueReason.LOST)
    assert d.start_action().resume


@pytest.mark.parametrize(
    ("reason", "end"),
    [(StopReason.PREEMPT, JobState.PREEMPTED), (StopReason.CANCEL, JobState.CANCELLED)],
)
def test_lost_while_stopping_follows_stop_intent(reason: StopReason, end: JobState) -> None:
    d = Driver().to(JobState.STABLE).stop(reason).exited(ExitKind.LOST)
    assert d.rt().state is end
    assert d.rt().lost_count == 0
    assert d.state.outbox == ()


def test_lost_retries_are_bounded() -> None:
    d = Driver(config=Config(max_lost_retries=3)).submit()
    for attempt in range(1, 3):
        d.start().started().exited(ExitKind.LOST)
        assert (d.rt().state, d.rt().attempt) == (JobState.PENDING, attempt + 1)
    d.start().started().exited(ExitKind.LOST)  # the 3rd consecutive LOST
    assert d.rt().state is JobState.FAILED
    assert d.state.outbox[-1] == Fail("j", 3, FailReason.LOST_RETRIES_EXHAUSTED)


def test_lost_counter_resets_after_stable() -> None:
    d = Driver(config=Config(max_lost_retries=2)).submit()
    d.start().started().exited(ExitKind.LOST)
    d.start().started().observe(1.0, vram=1).stable().exited(ExitKind.LOST)
    assert d.rt().state is JobState.PENDING
    assert d.rt().lost_count == 1


# -- preempt / resume / cancel ---------------------------------------------------------


def test_preempt_pending_holds_job() -> None:
    d = Driver().submit()(PreemptRequested("j"))
    assert (d.rt().state, d.rt().attempt) == (JobState.PREEMPTED, 1)
    d(ResumeRequested("j"))
    assert (d.rt().state, d.rt().attempt) == (JobState.PENDING, 1)
    assert not d.start_action().resume


@pytest.mark.parametrize("state", RUNNING)
def test_preempt_running_sets_intent_then_stop(state: JobState) -> None:
    d = Driver().to(state)(PreemptRequested("j"))
    assert d.rt().state is state
    assert d.rt().stop_intent is StopReason.PREEMPT
    d(Stop("j", 1, StopReason.PREEMPT, 120))
    assert d.rt().state is JobState.STOPPING
    d.exited(ExitKind.PREEMPTED)
    assert d.rt().state is JobState.PREEMPTED
    assert d.rt().stop_intent is None
    d(ResumeRequested("j"))
    assert (d.rt().state, d.rt().attempt) == (JobState.PENDING, 2)
    assert d.start_action().resume


def test_preempt_duplicates_are_ignored() -> None:
    d = Driver().to(JobState.WARMING)(PreemptRequested("j"), PreemptRequested("j"))
    d(Stop("j", 1, StopReason.PREEMPT, 1), PreemptRequested("j"))
    d.exited(ExitKind.PREEMPTED)(PreemptRequested("j"))
    assert d.ignored == [IgnoreReason.DUPLICATE] * 3


@pytest.mark.parametrize("state", TERMINAL)
def test_preempt_terminal_raises(state: JobState) -> None:
    with pytest.raises(InvalidTransition, match="finished"):
        Driver().to(state)(PreemptRequested("j"))


@pytest.mark.parametrize("state", [JobState.PENDING, JobState.PREEMPTED])
def test_cancel_idle_job(state: JobState) -> None:
    d = Driver().to(state)(CancelRequested("j"))
    assert d.rt().state is JobState.CANCELLED
    d(CancelRequested("j"))
    assert d.ignored == [IgnoreReason.DUPLICATE]


@pytest.mark.parametrize("state", RUNNING)
def test_cancel_running_via_stop(state: JobState) -> None:
    d = Driver().to(state).stop(StopReason.CANCEL)
    assert d.rt().state is JobState.STOPPING
    d(CancelRequested("j"))
    assert d.ignored == [IgnoreReason.DUPLICATE]
    d.exited(ExitKind.CANCELLED)
    assert d.rt().state is JobState.CANCELLED


def test_cancel_upgrades_preempt_intent() -> None:
    d = Driver().to(JobState.STABLE)(PreemptRequested("j"), CancelRequested("j"))
    assert d.rt().stop_intent is StopReason.CANCEL
    # And while already stopping for a preempt:
    d2 = Driver().to(JobState.STABLE).stop(StopReason.PREEMPT)(CancelRequested("j"))
    d2.exited(ExitKind.PREEMPTED)
    assert d2.rt().state is JobState.PREEMPTED  # "as named" (SPEC §3.4 table)
    d3 = Driver().to(JobState.STABLE).stop(StopReason.PREEMPT)(CancelRequested("j"))
    d3.exited(ExitKind.FAILURE)
    assert d3.rt().state is JobState.CANCELLED


@pytest.mark.parametrize("state", [JobState.SUCCEEDED, JobState.FAILED])
def test_cancel_finished_raises(state: JobState) -> None:
    with pytest.raises(InvalidTransition, match="finished"):
        Driver().to(state)(CancelRequested("j"))


def test_resume_pending_is_duplicate() -> None:
    d = Driver().submit()(ResumeRequested("j"))
    assert d.ignored == [IgnoreReason.DUPLICATE]


@pytest.mark.parametrize("state", [*RUNNING, JobState.STOPPING, *TERMINAL])
def test_resume_requires_preempted(state: JobState) -> None:
    with pytest.raises(InvalidTransition, match="PREEMPTED"):
        Driver().to(state)(ResumeRequested("j"))


# -- Stop action -----------------------------------------------------------------------


def test_stop_without_request_raises() -> None:
    d = Driver().to(JobState.WARMING)
    with pytest.raises(InvalidTransition, match="matching request"):
        d(Stop("j", 1, StopReason.PREEMPT, 1))
    d(PreemptRequested("j"))
    with pytest.raises(InvalidTransition, match="matching request"):
        d(Stop("j", 1, StopReason.CANCEL, 1))
    with pytest.raises(InvalidTransition, match="future"):
        d(Stop("j", 2, StopReason.PREEMPT, 1))


def test_stop_is_idempotent_and_noop_after_exit() -> None:
    d = Driver().to(JobState.WARMING)(PreemptRequested("j"))
    stop = Stop("j", 1, StopReason.PREEMPT, 1)
    d(stop)
    before = d.state
    d(stop)
    assert d.state == before
    # The job exits on its own before the Stop is reduced.
    d = Driver().to(JobState.WARMING)(PreemptRequested("j")).exited(ExitKind.SUCCESS)
    before = d.state
    d(stop)
    assert d.state == before


# -- Requeue(START_TIMEOUT) ------------------------------------------------------------


def test_start_timeout_requeues() -> None:
    d = Driver().to(JobState.STARTING)
    timeout = Requeue("j", 2, {GPU: None}, RequeueReason.START_TIMEOUT)
    d(timeout)
    assert (d.rt().state, d.rt().attempt) == (JobState.PENDING, 2)
    before = d.state
    d(timeout)
    assert d.state == before
    # A late JobStarted for the timed-out attempt is stale.
    d(JobStarted("j", 1, 9.0))
    assert d.ignored == [IgnoreReason.STALE_ATTEMPT]


@pytest.mark.parametrize(
    ("reason", "end"),
    [(StopReason.PREEMPT, JobState.PREEMPTED), (StopReason.CANCEL, JobState.CANCELLED)],
)
def test_start_timeout_honours_stop_intent(reason: StopReason, end: JobState) -> None:
    d = Driver().to(JobState.STARTING)
    d(PreemptRequested("j") if reason is StopReason.PREEMPT else CancelRequested("j"))
    d(Requeue("j", 2, {GPU: None}, RequeueReason.START_TIMEOUT))
    assert d.rt().state is end
    if end is JobState.PREEMPTED:
        d(ResumeRequested("j"))
        assert d.rt().attempt == 2


def test_start_timeout_requires_starting() -> None:
    d = Driver().to(JobState.WARMING)
    with pytest.raises(InvalidTransition, match="START_TIMEOUT"):
        d(Requeue("j", 2, {GPU: None}, RequeueReason.START_TIMEOUT))


# -- JobUpdated ------------------------------------------------------------------------


def test_update_pending_any_field() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("j", group="g1")
    d(JobUpdated("j", group=None, position=3, priority=7, placement={GPU1: None}, force=True))
    job = d.rt().job
    assert (job.group, job.position, job.priority, job.force) == (None, 3, 7, True)
    assert job.devices == (GPU1,)


def test_update_without_changes_is_noop() -> None:
    d = Driver().submit()
    before = d.state
    d(JobUpdated("j"))
    assert d.state == before


def test_update_drops_learned_estimates_for_removed_devices() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.WARMING).exited(ExitKind.OOM)
    assert GPU in d.rt().learned
    d(JobUpdated("j", placement={GPU1: None}))
    assert d.rt().learned == {}


@pytest.mark.parametrize("state", [*RUNNING, JobState.STOPPING])
def test_update_running_allows_ordering_fields_only(state: JobState) -> None:
    d = Driver().to(state)
    d(JobUpdated("j", group="g2", position=1, priority=2))
    assert d.rt().job.group == "g2"
    with pytest.raises(InvalidTransition, match="placement"):
        d(JobUpdated("j", placement={GPU: None}))
    with pytest.raises(InvalidTransition, match="force"):
        d(JobUpdated("j", force=True))


@pytest.mark.parametrize("state", TERMINAL)
def test_update_terminal_raises(state: JobState) -> None:
    with pytest.raises(InvalidTransition, match="finished"):
        Driver().to(state)(JobUpdated("j", priority=1))


def test_update_unknown_device_raises() -> None:
    with pytest.raises(InvalidTransition, match="unknown devices"):
        Driver().submit()(JobUpdated("j", placement={"nope": None}))


def test_job_updated_unset_sentinel() -> None:
    assert JobUpdated("j").group is UNSET


# -- terminal states reject everything else --------------------------------------------


@pytest.mark.parametrize("state", [JobState.SUCCEEDED, JobState.FAILED])
def test_replayed_start_of_finished_attempt_is_noop(state: JobState) -> None:
    d = Driver().to(state)
    before = d.state
    d(Start("j", 1, (GPU,), False, {}))
    assert d.state == before


def test_cancelled_pending_rejects_start() -> None:
    d = Driver().to(JobState.CANCELLED)
    with pytest.raises(InvalidTransition, match="only a PENDING"):
        d(Start("j", 1, (GPU,), False, {}))


def test_cancelled_pending_rejects_started() -> None:
    d = Driver().to(JobState.CANCELLED)
    with pytest.raises(InvalidTransition, match="never told"):
        d.started()


# -- pause, devices --------------------------------------------------------------------


def test_pause_unpause() -> None:
    d = Driver()(Paused())
    assert d.state.paused
    d(Paused(), Unpaused())
    assert not d.state.paused
    d(Unpaused())
    assert d.ignored == [IgnoreReason.DUPLICATE, IgnoreReason.DUPLICATE]


def test_device_added_and_removed() -> None:
    dev = Device(GPU1, Resources(vram=8), {"kind": "a100"})
    d = Driver()(DeviceAdded(dev), DeviceAdded(dev))
    assert d.state.devices[GPU1].device == dev
    assert d.ignored == [IgnoreReason.DUPLICATE]
    with pytest.raises(InvalidTransition, match="different content"):
        d(DeviceAdded(Device(GPU1, Resources(vram=9))))
    d(DeviceRemoved(GPU1))
    assert GPU1 not in d.state.devices
    with pytest.raises(InvalidTransition, match="unknown device"):
        d(DeviceRemoved(GPU1))


def test_device_removed_in_use_raises() -> None:
    d = Driver().submit()
    with pytest.raises(InvalidTransition, match="still used"):
        d(DeviceRemoved(GPU))
    d(CancelRequested("j"), DeviceRemoved(GPU))
    assert d.state.devices == {}


def test_state_initial_rejects_duplicate_devices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        State.initial([Device(GPU, Resources()), Device(GPU, Resources())])


# -- outbox / drain / dispatch ---------------------------------------------------------


def test_clear_only_actions_drain_the_outbox() -> None:
    d = Driver().to(JobState.WARMING).exited(ExitKind.FAILURE)
    (fail,) = d.state.outbox
    d(fail)
    assert d.state.outbox == ()
    before = d.state
    d(fail, GroupCompleted("g", GroupOutcome.OK))  # unknown / already cleared: no-op
    assert d.state == before


def test_drain_takes_ignored_and_history() -> None:
    d = Driver().to(JobState.SUCCEEDED).submit("j")
    state, ignored, history = drain(d.state)
    assert [i.reason for i in ignored] == [IgnoreReason.DUPLICATE]
    assert len(history) == 1
    assert state.ignored_events == ()
    assert state.history_records == ()
    assert state.jobs == d.state.jobs


def test_reduce_rejects_unknown_messages() -> None:
    with pytest.raises(TypeError):
        reduce(State.initial(), object(), Config())  # type: ignore[arg-type]


def test_reduce_is_pure() -> None:
    state = State.initial(gpus())
    job = Job("j", {GPU: None})
    reduce(state, JobSubmitted(job), Config())
    assert state.jobs == {}
