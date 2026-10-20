"""Serialization (I8), the Engine API, restarts and reconciliation (SPEC §4, L3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import rampgate.serialization as ser
from rampgate import (
    Config,
    Device,
    Engine,
    ExitKind,
    FifoPolicy,
    HistoryEstimator,
    InMemoryHistoryStore,
    InvalidTransition,
    Job,
    JobObserved,
    JobStarted,
    JobState,
    JobSubmitted,
    JobUpdated,
    OrderingPolicy,
    Paused,
    Requeue,
    RequeueReason,
    Resources,
    Start,
    StopReason,
    UnsupportedSchema,
    WaitingForWarmup,
    dump_state,
    load_state,
)
from rampgate.serialization import MIGRATIONS, SCHEMA_VERSION, from_data, to_data
from rampgate.sim import RampPlateau, Scenario, SimJob, simulate
from rampgate_testkit import GPU, GPU1, Driver, examples, gpus
from test_invariants import CFG, scenarios

# -- I8: dump / load round trip ---------------------------------------------------------------


def roundtrip(state: Any) -> Any:
    back = load_state(json.loads(json.dumps(dump_state(state))))
    # Equal *and* identically typed (a StrEnum decoded as str would still compare equal).
    assert to_data(back) == to_data(state)
    return back


def test_roundtrip_of_a_driven_state() -> None:
    d = Driver(gpus(GPU, GPU1)).to(JobState.STABLE, "a").to(JobState.PREEMPTED, "p")
    d.submit("g", {GPU: None, GPU1: Resources(vram=3)}, group="g", t=1.5)
    d(JobUpdated("g", priority=2), JobUpdated("g"))
    d.exited(ExitKind.OOM, "a")
    assert roundtrip(d.state) == d.state


@settings(max_examples=examples(0.3))
@given(scenarios(), st.data())
def test_i8_dump_load_roundtrips_to_an_equal_state(
    case: tuple[Scenario, OrderingPolicy], data: st.DataObject
) -> None:
    scenario, policy = case
    trace = simulate(scenario, policy, config=CFG, max_time=400.0)
    tick = data.draw(st.sampled_from(trace.ticks))
    for state in (tick.snapshot.state, tick.after):
        assert roundtrip(state) == state


def test_schema_errors() -> None:
    good = dump_state(Driver().submit().state)
    with pytest.raises(UnsupportedSchema, match="not a rampgate"):
        load_state({"format": "other"})
    with pytest.raises(UnsupportedSchema, match="reads up to"):
        load_state({**good, "schema_version": SCHEMA_VERSION + 1})
    with pytest.raises(UnsupportedSchema, match="invalid"):
        load_state({**good, "schema_version": "1"})
    with pytest.raises(UnsupportedSchema, match="State"):
        load_state({**good, "state": to_data(Resources(vram=1))})


def test_migration_path(monkeypatch: pytest.MonkeyPatch) -> None:
    state = Driver().submit().state
    current = dump_state(state)
    monkeypatch.setattr(ser, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    with pytest.raises(UnsupportedSchema, match="no migration"):
        ser.load_state(current)
    monkeypatch.setitem(MIGRATIONS, SCHEMA_VERSION, lambda payload: payload)
    assert ser.load_state(current) == state


def test_decoder_rejects_unknown_input() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        from_data({"$t": "os.system", "f": {}})
    with pytest.raises(ValueError, match="unexpected JSON object"):
        from_data({"x": 1})
    with pytest.raises(ValueError, match="unexpected JSON value"):
        from_data([1])
    with pytest.raises(TypeError):
        to_data(object())
    with pytest.raises(TypeError):
        to_data([1])


def test_in_memory_store_json_roundtrip() -> None:
    store = InMemoryHistoryStore()
    store.put("k", "a100", Resources(vram=3), lower_bound=False)
    back = InMemoryHistoryStore.from_json(json.loads(json.dumps(store.to_json())))
    assert back.get("k", "a100") == Resources(vram=3)


# -- Engine API ------------------------------------------------------------------------------------


def engine(*devices: Device, **kw: Any) -> Engine:
    return Engine(devices or (Device(GPU, Resources(vram=100)),), FifoPolicy(), **kw)


def test_engine_lifecycle_through_the_api() -> None:
    e = engine(config=Config(min_warmup_s=0, stable_window_s=0))
    e.submit(Job("a", {GPU: None}))
    e.submit(Job("b", {GPU: None}))
    first = e.tick(0.0)
    assert [a.job_id for a in first.actions if isinstance(a, Start)] == ["a"]
    assert e.explanations["b"] == WaitingForWarmup(GPU, "a")
    assert e.now == 0.0
    e.on_started("a", 1, 1.0)
    e.observe(JobObserved("a", GPU, 2.0, Resources(vram=40), attempt=1))
    second = e.tick(3.0)  # a becomes stable, b starts
    assert [type(a).__name__ for a in second.actions] == ["MarkStable", "Start"]
    e.on_exited("a", 1, ExitKind.SUCCESS, 4.0)
    e.update("b", priority=3)
    e.preempt("b")
    assert [type(a).__name__ for a in e.tick(5.0).actions] == ["Stop"]
    e.on_exited("b", 1, ExitKind.PREEMPTED, 6.0)
    e.resume("b")
    e.pause()
    assert e.tick(7.0).actions == ()
    e.unpause()
    assert [type(a).__name__ for a in e.tick(8.0).actions] == ["Start"]
    e.cancel("b")
    e.add_device(Device(GPU1, Resources(vram=10)))
    e.remove_device(GPU1)
    assert e.state.jobs["b"].stop_intent is StopReason.CANCEL


def test_engine_submit_stamps_last_tick_time() -> None:
    e = engine()
    e.submit(Job("early", {GPU: None}))
    e.tick(10.0)
    e.submit(Job("late", {GPU: None}))
    e.submit(Job("explicit", {GPU: None}), t=99.0)
    jobs = e.state.jobs
    assert (jobs["early"].submit_t, jobs["late"].submit_t, jobs["explicit"].submit_t) == (
        None,
        10.0,
        99.0,
    )


def test_engine_rejects_time_going_backwards_and_invalid_calls() -> None:
    e = engine()
    e.tick(5.0)
    e.tick(5.0)
    with pytest.raises(ValueError, match="backwards"):
        e.tick(4.0)
    with pytest.raises(InvalidTransition):
        e.on_started("ghost", 1, 0.0)


def test_engine_tick_hands_history_to_the_estimator_and_reports_ignored_events() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    e = engine(estimator=est, config=Config(min_warmup_s=0, stable_window_s=0))
    job = Job("a", {GPU: None}, fingerprint=("m",))
    e.submit(job)
    e.tick(0.0)
    e.on_started("a", 1, 1.0)
    e.observe(JobObserved("a", GPU, 2.0, Resources(vram=40), attempt=1))
    e.on_exited("a", 1, ExitKind.SUCCESS, 3.0)
    e.on_exited("a", 1, ExitKind.SUCCESS, 3.0)
    decision = e.tick(4.0)
    assert len(decision.ignored_events) == 1
    assert est.estimate(job, e.state.devices[GPU].device) == Resources(vram=40)
    assert e.tick(5.0).ignored_events == ()


# -- dump / load / reconcile ---------------------------------------------------------------------


def test_engine_dump_load_and_reconciliation() -> None:
    e = engine(Device(GPU, Resources(vram=100)), Device(GPU1, Resources(vram=100)))
    e.submit(Job("a", {GPU: None}))
    e.submit(Job("b", {GPU1: None}))
    e.tick(0.0)
    e.on_started("a", 1, 1.0)
    e.on_started("b", 1, 1.0)
    e.tick(2.0)
    data = json.loads(json.dumps(e.dump()))
    back = Engine.load(data, policy=FifoPolicy())
    assert back.state == e.state
    assert back.now == 2.0
    assert back.unreconciled() == ["a", "b"]
    back.observe(JobObserved("a", GPU, 3.0, Resources(vram=5), attempt=1))
    back.on_exited("b", 1, ExitKind.LOST, 3.0)
    assert back.unreconciled() == []
    decision = back.tick(4.0)
    assert Requeue("b", 2, {GPU1: None}, RequeueReason.LOST) in decision.actions
    restart = next(a for a in decision.actions if isinstance(a, Start))
    assert (restart.job_id, restart.attempt, restart.resume) == ("b", 2, True)


def test_stale_reports_do_not_reconcile() -> None:
    e = engine()
    e.submit(Job("a", {GPU: None}))
    e.tick(0.0)
    e.on_started("a", 1, 1.0)
    back = Engine.load(e.dump(), policy=FifoPolicy())
    back.on_started("a", 1, 1.0)  # duplicate of an already applied fact
    assert back.unreconciled() == ["a"]


def test_engine_load_rejects_other_formats() -> None:
    with pytest.raises(UnsupportedSchema, match="engine"):
        Engine.load({"format": "rampgate.state"}, policy=FifoPolicy())
    with pytest.raises(UnsupportedSchema, match="schema"):
        Engine.load({"format": "rampgate.engine", "schema_version": 2}, policy=FifoPolicy())


# -- restart in the middle of a run ---------------------------------------------------------


@settings(max_examples=examples(0.3))
@given(scenarios(), st.data())
def test_restart_mid_run_gives_the_same_trace(
    case: tuple[Scenario, OrderingPolicy], data: st.DataObject
) -> None:
    scenario, policy = case
    whole = simulate(scenario, policy, config=CFG, max_time=400.0)
    at = data.draw(st.sampled_from([t.now for t in whole.ticks]))
    restarted = simulate(scenario, policy, config=CFG, max_time=400.0, restart_at=at)
    assert [t.decision for t in restarted.ticks] == [t.decision for t in whole.ticks]
    assert restarted.final == whole.final


def test_restart_keeps_learning_with_a_durable_store() -> None:
    store = InMemoryHistoryStore()
    job = SimJob(
        Job("a", {GPU: None}, fingerprint=("m",)),
        RampPlateau(load_s=2, ramp_s=2, plateau=Resources(vram=30), duration_s=30),
    )
    scenario = Scenario((Device(GPU, Resources(vram=100)),), (job,))
    simulate(scenario, FifoPolicy(), estimator=HistoryEstimator(store), restart_at=10.0)
    assert store.get("m", GPU) == Resources(vram=30)


def test_encoder_rejects_foreign_dataclasses() -> None:
    @dataclass
    class Job:  # same name as a rampgate class, but not one
        id: str

    with pytest.raises(TypeError, match="not a serializable"):
        to_data(Job("x"))


def test_engine_apply_routes_commands_and_facts() -> None:
    e = engine()
    e.apply(JobSubmitted(Job("a", {GPU: None})))
    e.apply(Paused())
    assert e.state.paused
    e.unpause()
    e.tick(0.0)
    e.apply(JobStarted("a", 1, 1.0))
    assert e.state.jobs["a"].state is JobState.WARMING
