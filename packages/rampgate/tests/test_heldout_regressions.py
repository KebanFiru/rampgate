"""The bench-freeze-1 held-out failures as regression tests (SPEC §12 L4).

Each (scenario, seed) had cold `Probation` OOM outside first contact on tag
``bench-freeze-1``; the traces are in ``docs/benchmark/freeze-1/heldout-failures.md``.
(b) 1002 and 1008: a sibling admitted on a lower bound alone (lower-bound
estimates now grow by ``oom_growth``). (c) 1005, 1009, 1013, 1016: trusted jobs
started next to an unknown warming job (I17). These seeds are spent: they are
replayed here, never re-reported as held-out results.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rampgate.sim import STRATEGIES, run

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"


@pytest.mark.parametrize(
    ("scenario", "seed"),
    [
        ("b_mixed", 1002),
        ("b_mixed", 1008),
        ("c_late_spike", 1005),
        ("c_late_spike", 1009),
        ("c_late_spike", 1013),
        ("c_late_spike", 1016),
    ],
)
def test_freeze_1_held_out_failure_has_no_oom_outside_first_contact(
    scenario: str, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    scenarios = importlib.import_module("scenarios")  # benchmarks/ is not a package

    case = scenarios.make(scenario, seed)
    rep = run(
        case.scenario,
        STRATEGIES["Probation"],
        policy=case.policy,
        max_time=case.max_time,
        **scenarios.SIM_KWARGS,
    )
    assert rep.oom_count == rep.first_contact_ooms
    assert rep.repeat_device_ooms == 0
    assert not rep.censored
    sys.modules.pop("scenarios", None)
