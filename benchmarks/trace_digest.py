"""Per-run decision digests: prove that a change leaves benchmark runs identical.

Usage::

    python benchmarks/trace_digest.py --out digests.json [--seeds 50] [--groups main warm ablations]
    python benchmarks/trace_digest.py --compare a.json b.json

For every task of ``run.py`` (scenario x strategy x seed in the groups), it runs
the task exactly as ``run.py`` does and records the ``results.json`` row and a
SHA-256 of the per-tick sequence ``[(now, [action_id, ...]), ...]`` of the
measured run (the warm-up run of a warm row is not part of it). ``--compare``
lists every task whose digest or row differs; exit code 1 if any does.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as bench

# `rampgate.sim.report` is also the name of a function in `rampgate.sim`; take the module.
report_module = importlib.import_module("rampgate.sim.report")

_traces: list[Any] = []
_simulate = report_module.simulate


def _recording_simulate(*args: Any, **kwargs: Any) -> Any:
    trace = _simulate(*args, **kwargs)
    _traces.append(trace)
    return trace


report_module.simulate = _recording_simulate  # type: ignore[assignment]


def digest(task: bench.Task) -> tuple[str, dict[str, Any]]:
    _traces.clear()
    row = bench.execute(task)
    trace = _traces[-1]  # the measured run (a warm row's warm-up run comes first)
    ticks = [[t.now, [a.action_id for a in t.decision.actions]] for t in trace.ticks]
    sha = hashlib.sha256(json.dumps(ticks).encode()).hexdigest()
    key = f"{task.group}/{task.scenario}/{task.strategy}/{task.seed}"
    return key, {"digest": sha, "ticks": len(ticks), "row": row}


def compare(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text())["runs"]
    b = json.loads(b_path.read_text())["runs"]
    differ = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            differ.append(f"{key}: only in {'b' if key not in a else 'a'}")
            continue
        if a[key]["digest"] != b[key]["digest"]:
            differ.append(
                f"{key}: action_id sequence differs ({a[key]['ticks']} vs {b[key]['ticks']} ticks)"
            )
        rows = {k for k in a[key]["row"] if a[key]["row"].get(k) != b[key]["row"].get(k)}
        if rows:
            differ.append(f"{key}: row differs in {sorted(rows)}")
    print(f"{len(a)} vs {len(b)} runs; {len(differ)} difference(s)")
    for line in differ:
        print("  " + line)
    return 1 if differ else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--groups", nargs="+", default=["main", "warm", "ablations"])
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    ns = argparse.Namespace(
        groups=args.groups,
        strategies=[],
        scenarios=list(bench.NAMES),
        seeds=args.seeds,
        seed_start=args.seed_start,
    )
    tasks = bench.tasks(ns)
    with ProcessPoolExecutor(args.jobs) as pool:
        runs = dict(pool.map(digest, tasks, chunksize=4))
    args.out.write_text(json.dumps({"runs": runs}, sort_keys=True) + "\n")
    print(f"{len(runs)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
