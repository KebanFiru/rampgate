"""Estimators and history stores (SPEC §7)."""

import json

import pytest

from rampgate import (
    Device,
    EstimateResult,
    ExitKind,
    HistoryEntry,
    HistoryEstimator,
    HistoryRecord,
    InMemoryHistoryStore,
    Job,
    JobState,
    NullEstimator,
    Resources,
    StaticEstimator,
    apply_history,
    drain,
    resolve_estimates,
)
from rampgate_testkit import GPU, Driver, stored

A100 = Device("node/gpu:0", Resources(vram=80), {"kind": "a100"})
A100b = Device("node/gpu:1", Resources(vram=80), {"kind": "a100"})
UNLABELED = Device("gpu:x", Resources(vram=80))


def job(*fingerprint: str) -> Job:
    return Job("j", {A100.id: None}, fingerprint=fingerprint)


def test_in_memory_store_keeps_max() -> None:
    store = InMemoryHistoryStore()
    assert store.get("k", "a100") is None
    store.put("k", "a100", Resources(vram=10, ram=5), lower_bound=False)
    store.put("k", "a100", Resources(vram=8, ram=9), lower_bound=True)
    assert stored(store, "k", "a100") == Resources(vram=10, ram=9)
    assert store.get("k", "h100") is None


def test_history_estimator_walks_fingerprint_most_specific_first() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("model:gpt2", "family:gpt"), A100, Resources(vram=20))
    assert est.estimate(job("model:gpt2-xl", "family:gpt"), A100) == EstimateResult(
        Resources(vram=20), 1
    )
    est.record(job("model:gpt2-xl", "family:gpt"), A100, Resources(vram=60))
    assert est.estimate(job("model:gpt2", "family:gpt"), A100) == EstimateResult(
        Resources(vram=20), 0
    )
    assert est.estimate(job("model:gpt2-xl", "family:gpt"), A100) == EstimateResult(
        Resources(vram=60), 0
    )
    assert est.estimate(job("family:gpt"), A100) == EstimateResult(Resources(vram=60), 0)
    assert est.estimate(job(), A100) is None


def test_history_estimator_device_kind_from_labels() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("k"), A100, Resources(vram=20))
    assert est.estimate(job("k"), A100b) == EstimateResult(Resources(vram=20), 0)
    assert est.estimate(job("k"), UNLABELED) is None
    custom = HistoryEstimator(InMemoryHistoryStore(), device_kind=lambda d: "all")
    custom.record(job("k"), A100, Resources(vram=1))
    assert custom.estimate(job("k"), UNLABELED) == EstimateResult(Resources(vram=1), 0)


def test_history_estimator_lower_bounds_raise_the_estimate() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("k"), A100, Resources(vram=20))
    est.record(job("k"), A100, Resources(vram=30), lower_bound=True)
    # Newer than the success and above it: counted, but "more than 30" is not trusted.
    assert est.estimate(job("k"), A100) == EstimateResult(Resources(vram=30), 0, complete=False)


def test_lower_bounds_alone_are_not_trusted() -> None:
    # An OOM'd job's lower bound must not let a same-template sibling skip probation.
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("k", "family"), A100, Resources(vram=15), lower_bound=True)
    result = est.estimate(job("k", "family"), A100)
    assert result == EstimateResult(Resources(vram=15), 0, complete=False)
    assert result is not None
    assert not result.trusted
    est.record(job("family"), A100, Resources(vram=36))  # complete, at a fallback level only
    assert est.estimate(job("k", "family"), A100) == EstimateResult(
        Resources(vram=15), 0, complete=False
    )
    est.record(job("k"), A100, Resources(vram=10))  # a complete record at level 0, newer
    result = est.estimate(job("k", "family"), A100)
    assert result == EstimateResult(Resources(vram=10), 0)  # the older lower bound is dropped
    assert result is not None
    assert result.trusted


def test_a_running_record_makes_siblings_trusted() -> None:
    d = Driver().submit("j", fingerprint=("k",)).start().started().observe(1.0, vram=40).stable()
    est = HistoryEstimator(InMemoryHistoryStore())
    apply_history(est, d.state, d.state.history_records)
    d.submit("sibling", fingerprint=("k",))
    snap = resolve_estimates(d.state, est)
    assert snap.estimates["sibling"][GPU] == Resources(vram=40)
    assert snap.is_trusted(d.state.jobs["sibling"], GPU)


def test_an_older_lower_bound_does_not_override_a_later_success() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("k"), A100, Resources(vram=8), lower_bound=True, record_id="x:1:d:lower_bound")
    est.record(job("k"), A100, Resources(vram=6.4), record_id="y:1:d:peak")
    assert est.estimate(job("k"), A100) == EstimateResult(Resources(vram=6.4), 0)


def test_a_newer_lower_bound_above_the_complete_value_counts_and_is_untrusted() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("k"), A100, Resources(vram=36), record_id="x:1:d:peak")
    est.record(job("k"), A100, Resources(vram=40), lower_bound=True, record_id="y:1:d:lower_bound")
    result = est.estimate(job("k"), A100)
    assert result == EstimateResult(Resources(vram=40), 0, complete=False)
    est.record(job("k"), A100, Resources(vram=30), record_id="z:1:d:peak")  # a later success
    assert est.estimate(job("k"), A100) == EstimateResult(Resources(vram=36), 0)


def test_a_newer_lower_bound_below_the_complete_value_keeps_it_trusted() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    est.record(job("k"), A100, Resources(vram=36), record_id="x:1:d:peak")
    est.record(job("k"), A100, Resources(vram=20), lower_bound=True, record_id="y:1:d:lower_bound")
    assert est.estimate(job("k"), A100) == EstimateResult(Resources(vram=36), 0)


def test_replaying_a_record_does_not_make_it_newer() -> None:
    store = InMemoryHistoryStore()
    store.put("k", "a100", Resources(vram=8), lower_bound=True, record_id="old:lower_bound")
    store.put("k", "a100", Resources(vram=6), lower_bound=False, record_id="new:peak")
    store.put("k", "a100", Resources(vram=8), lower_bound=True, record_id="old:lower_bound")
    assert store.get("k", "a100") == HistoryEntry(Resources(vram=6), None)
    assert store.puts_applied == 2


def test_a_grown_running_record_is_newer() -> None:
    store = InMemoryHistoryStore()
    store.put("k", "a100", Resources(vram=40), lower_bound=False, record_id="r:running")
    store.put("k", "a100", Resources(vram=45), lower_bound=True, record_id="o:lower_bound")
    assert store.get("k", "a100") == HistoryEntry(Resources(vram=40), Resources(vram=45))
    store.put("k", "a100", Resources(vram=42), lower_bound=False, record_id="r:running")
    assert store.get("k", "a100") == HistoryEntry(Resources(vram=42), None)


def test_store_json_round_trip_keeps_the_order() -> None:
    store = InMemoryHistoryStore()
    store.put("k", "a100", Resources(vram=8), lower_bound=True, record_id="a:lower_bound")
    store.put("k", "a100", Resources(vram=6), lower_bound=False, record_id="b:peak")
    back = InMemoryHistoryStore.from_json(json.loads(json.dumps(store.to_json())))
    assert back.get("k", "a100") == store.get("k", "a100")
    back.put("k", "a100", Resources(vram=9), lower_bound=True, record_id="c:lower_bound")
    assert back.get("k", "a100") == HistoryEntry(Resources(vram=6), Resources(vram=9))


def test_legacy_store_json_loads() -> None:
    legacy = {
        "peaks": [["k", "a100", {"vram": 8.0}], ["m", "a100", {"vram": 3.0}]],
        "applied": [["k", "a100", "j:1:d:lower_bound", {"vram": 8.0}]],
    }
    store = InMemoryHistoryStore.from_json(legacy)
    assert store.get("k", "a100") == HistoryEntry(None, Resources(vram=8))
    assert store.get("m", "a100") == HistoryEntry(Resources(vram=3), None)


def test_only_max_aggregate() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        HistoryEstimator(InMemoryHistoryStore(), aggregate="p95")  # type: ignore[arg-type]


def test_null_and_static_estimators() -> None:
    j = job("k")
    assert NullEstimator().estimate(j, A100) is None
    NullEstimator().record(j, A100, Resources(vram=1))
    static = StaticEstimator({"j": {A100.id: Resources(vram=7)}})
    assert static.estimate(j, A100) == EstimateResult(Resources(vram=7), 0)
    assert static.estimate(j, A100b) is None
    static.record(j, A100, Resources(vram=1))
    assert static.estimate(j, A100) == EstimateResult(Resources(vram=7), 0)


def test_resolve_estimates_covers_unfinished_jobs() -> None:
    d = Driver().submit("p").to(JobState.SUCCEEDED, "done")
    snap = resolve_estimates(d.state, StaticEstimator({"p": {GPU: Resources(vram=3)}}))
    assert snap.estimates == {"p": {GPU: Resources(vram=3)}}
    assert snap.state is d.state


def test_apply_history_feeds_success_and_oom_records() -> None:
    est = HistoryEstimator(InMemoryHistoryStore())
    d = Driver().submit("j", fingerprint=("k",)).start(free={GPU: Resources(vram=40)})
    d.room(vram=40).exited(ExitKind.OOM)
    state, _, history = drain(d.state)
    assert history == (HistoryRecord("j", GPU, Resources(vram=40), True, 1),)
    apply_history(est, state, history)
    gpu = state.devices[GPU].device
    lower = EstimateResult(Resources(vram=40), 0, complete=False)
    assert est.estimate(state.jobs["j"].job, gpu) == lower
    apply_history(est, state, [HistoryRecord("ghost", GPU, Resources(vram=1), False, 1)])


def test_rule_h_running_record_at_stable_and_rewritten_on_reservation_growth() -> None:
    d = Driver().to(JobState.WARMING).observe(1.0, vram=40).stable()
    running = HistoryRecord("j", GPU, Resources(vram=40), False, 1, running=True)
    assert d.state.history_records == (running,)
    assert running.record_id == f"j:1:{GPU}:running"
    d.observe(2.0, vram=45)  # within the 1.15 margin: no UpdateReservation, no record
    assert len(d.state.history_records) == 1
    d.observe(3.0, vram=60)  # exceeds the reservation
    grown = d.state.history_records[-1]
    assert grown == HistoryRecord("j", GPU, Resources(vram=60), False, 1, running=True)
    assert grown.record_id == running.record_id
    store = InMemoryHistoryStore()
    est = HistoryEstimator(store)
    apply_history(est, d.state, reversed(d.state.history_records))  # any order
    assert store.get("k", GPU) is None  # the job has no fingerprint
    d2 = Driver().submit("j", fingerprint=("k",)).start().started()
    d2.observe(1.0, vram=40).stable().observe(3.0, vram=60)
    apply_history(est, d2.state, reversed(d2.state.history_records))
    assert stored(store, "k", GPU) == Resources(vram=60)
