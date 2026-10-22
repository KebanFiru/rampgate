"""Smoke test for ``benchmarks/run.py`` (the full run is in the nightly workflow)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[3] / "benchmarks"


@pytest.fixture(scope="module")
def bench() -> Any:
    sys.path.insert(0, str(BENCH))
    try:
        import run as bench_run  # type: ignore[import-not-found]  # noqa: PLC0415

        yield bench_run
    finally:
        sys.path.remove(str(BENCH))


def test_one_seed_run_writes_results_and_passes_the_check(bench: Any, tmp_path: Path) -> None:
    argv = ["--seeds", "1", "--groups", "main", "--scenarios", "a_small", "--no-plots"]
    argv += ["--check", "--min-seeds", "1", "--jobs", "1", "--out", str(tmp_path)]
    assert bench.main(argv) == 0
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["meta"]["seeds"] == [0]
    assert data["meta"]["warm_seed_offset"] == 10_000
    assert {r["strategy"] for r in data["runs"]} == {
        "Probation",
        "Probation (all)",
        "GreedyFree",
        "DeclaredReservation",
        "Serial",
    }
    assert data["strategies"]["Probation"]["config"]["probation"] == "untrusted_only"
    assert "## Main" in (tmp_path / "results.md").read_text()


def _row(scenario: str, strategy: str, seed: int, **metrics: Any) -> dict[str, Any]:
    base = {"oom_count": 0, "first_contact_ooms": 0, "repeat_device_ooms": 0, "makespan": 100.0}
    keys = {"group": "main", "scenario": scenario, "strategy": strategy, "seed": seed}
    return keys | base | metrics


def test_check_flags_non_first_contact_ooms_and_weak_baselines(bench: Any) -> None:
    runs = [
        _row("b_mixed", "Probation", 0, oom_count=3, first_contact_ooms=2),
        _row("b_mixed", "GreedyFree", 0, oom_count=2),
        _row("b_mixed", "DeclaredReservation", 0, oom_count=50),
    ]
    failures, _ = bench.check(runs, min_seeds=1)
    assert any("1 OOM(s) not first-contact" in f for f in failures)
    assert any("not below GreedyFree" in f for f in failures)
    assert not any("DeclaredReservation" in f for f in failures)
    failures, _ = bench.check(runs, min_seeds=10)
    assert any("seeds per row" in f for f in failures)


def test_check_gates_every_rampgate_row_but_not_ablations(bench: Any) -> None:
    runs = [
        _row("d_under_declared", "Probation", 0, oom_count=2, first_contact_ooms=1),
        _row("e_trainq", "Probation (warm)", 0, repeat_device_ooms=1) | {"group": "warm"},
        _row("a_small", "rampgate - Rule P", 0, oom_count=9) | {"group": "ablations"},
    ]
    failures, _ = bench.check(runs, min_seeds=1)
    assert failures == [
        "d_under_declared Probation seed 0: 1 OOM(s) not first-contact",
        "e_trainq Probation (warm) seed 0: 1 job(s) OOMed twice on a device",
    ]
