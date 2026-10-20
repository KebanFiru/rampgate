"""Ordering policies (SPEC §6)."""

import pytest

from rampgate import (
    BasePolicy,
    CancelRequested,
    FailurePolicy,
    FifoPolicy,
    GroupBarrier,
    GroupOutcome,
    JobState,
    PreemptRequested,
    PriorityPolicy,
    StagePolicy,
)
from rampgate_testkit import GPU, GPU1, Driver, gpus


def test_stage_groups_listed_first_then_by_first_submission() -> None:
    d = Driver().submit("x", group="late").submit("y", group="b").submit("z", group="a")
    assert StagePolicy(["a"]).groups(d.state) == ["a", "late", "b"]


def test_stage_candidates_active_group_by_position_then_side_jobs() -> None:
    d = Driver().submit("s1", group=None).submit("p2", group="g", position=2)
    d.submit("p1", group="g", position=1).submit("other", group="h")
    policy = StagePolicy(["g", "h"])
    assert list(policy.candidates(d.state, 0.0)) == ["p1", "p2", "s1"]
    assert policy.explain_blocked(d.state, "other") == GroupBarrier("g")
    assert policy.explain_blocked(d.state, "s1") is None
    assert policy.explain_blocked(d.state, "p1") is None
    assert not policy.head_of_line_blocking


def test_stage_empty_listed_group_is_skipped_and_all_complete_leaves_side_jobs() -> None:
    d = Driver().submit("a", group="g")
    policy = StagePolicy(["empty", "g"])
    assert policy.active_group(d.state) == "g"
    d(CancelRequested("a")).submit("side")
    assert policy.active_group(d.state) is None
    assert list(policy.candidates(d.state, 0.0)) == ["side"]
    assert policy.explain_blocked(d.state, "side") is None


def test_stage_preempted_job_keeps_group_open() -> None:
    d = Driver().submit("a", group="g").submit("b", group="h")(PreemptRequested("a"))
    policy = StagePolicy(["g", "h"])
    assert policy.active_group(d.state) == "g"
    assert list(policy.completed_groups(d.state)) == []


def test_stage_backfill_uses_only_devices_free_of_the_active_group() -> None:
    d = Driver(gpus(GPU, GPU1)).submit("a", {GPU: None}, group="g")
    d.submit("b1", {GPU1: None}, group="h").submit("b2", {GPU: None, GPU1: None}, group="h")
    off = StagePolicy(["g", "h"])
    on = StagePolicy(["g", "h"], backfill=True)
    assert list(off.candidates(d.state, 0.0)) == ["a"]
    assert list(on.candidates(d.state, 0.0)) == ["a", "b1"]
    assert on.explain_blocked(d.state, "b1") is None
    assert on.explain_blocked(d.state, "b2") == GroupBarrier("g")


def test_stage_completed_groups_and_outcomes() -> None:
    d = Driver().to(JobState.SUCCEEDED, "a", group="ok")
    d.to(JobState.FAILED, "b", group="bad").to(JobState.CANCELLED, "c", group="cancelled")
    policy = StagePolicy()
    assert sorted(policy.completed_groups(d.state)) == [
        ("bad", GroupOutcome.HAD_FAILURES),
        ("cancelled", GroupOutcome.OK),
        ("ok", GroupOutcome.OK),
    ]


def test_stage_queue_stops() -> None:
    d = Driver().to(JobState.FAILED, "a", group="g").to(JobState.FAILED, "b", group="h")
    policy = StagePolicy(
        failure_policy={"g": FailurePolicy.STOP_QUEUE, "h": FailurePolicy.CONTINUE}
    )
    assert list(policy.queue_stops(d.state)) == [("g", "a")]


def test_fifo_order_and_blocking_flag() -> None:
    d = Driver().submit("b").submit("a")
    assert list(FifoPolicy().candidates(d.state, 0.0)) == ["b", "a"]
    assert FifoPolicy(strict=True).head_of_line_blocking
    assert not FifoPolicy(strict=False).head_of_line_blocking
    assert list(FifoPolicy().completed_groups(d.state)) == []
    assert FifoPolicy().explain_blocked(d.state, "a") is None


def test_priority_order() -> None:
    d = Driver().submit("low", priority=1).submit("high", priority=9).submit("mid", priority=5)
    d.submit("high2", priority=9)
    assert list(PriorityPolicy().candidates(d.state, 0.0)) == ["high", "high2", "mid", "low"]
    assert PriorityPolicy().head_of_line_blocking


def test_priority_aging_needs_submit_time() -> None:
    d = Driver().submit("aged", t=0.0, priority=0).submit("untimed", priority=1)
    policy = PriorityPolicy(aging_per_s=0.01)
    assert list(policy.candidates(d.state, 50.0)) == ["untimed", "aged"]
    assert list(policy.candidates(d.state, 200.0)) == ["aged", "untimed"]
    with pytest.raises(ValueError, match="aging"):
        PriorityPolicy(aging_per_s=-1)


def test_base_policy_defaults() -> None:
    d = Driver().submit("a")
    policy = BasePolicy()
    assert not policy.head_of_line_blocking
    assert policy.explain_blocked(d.state, "a") is None
    assert list(policy.completed_groups(d.state)) == []
    assert list(policy.queue_stops(d.state)) == []
    with pytest.raises(NotImplementedError):
        policy.candidates(d.state, 0.0)
