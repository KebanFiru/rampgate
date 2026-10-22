"""Run the benchmark (SPEC §8) and write ``results.json`` and ``results.md``.

    uv run python benchmarks/run.py                      # full run, 10 seeds
    uv run python benchmarks/run.py --check --no-plots   # also enforce SPEC §12 L4
    uv run --group bench python benchmarks/run.py        # with plots (matplotlib)

Groups of runs:

- ``main``: `MAIN` rows (cold) on every scenario, primary simulator settings;
- ``ablations``: `ABLATIONS`;
- ``warm``: `WARM` rows, each filled first by an independent run at ``seed + 10_000``;
- ``sensitivity``: `MAIN` rows over ``start_delay_s`` x ``checkpoint_s``.

The output depends only on the code and the arguments (no timestamps), so a
rerun reproduces ``results.json`` byte for byte.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios import NAMES, SIM_KWARGS, make

import rampgate
from rampgate import Config, InMemoryHistoryStore, Resources
from rampgate.sim import ABLATIONS, MAIN, STRATEGIES, WARM, Strategy, run, summarize

WARM_SEED_OFFSET = 10_000
START_DELAYS = (15.0, 60.0, 120.0)
CHECKPOINTS = (60.0, 120.0, 300.0)
METRICS = (
    "makespan",
    "completed",
    "oom_count",
    "first_contact_ooms",
    "first_contact_unknown",
    "first_contact_hint",
    "lost_work_step_s",
    "utilization",
    "wait_p50",
    "wait_p95",
)
GROUPS: dict[str, tuple[Strategy, ...]] = {
    "main": MAIN,
    "ablations": ABLATIONS,
    "warm": WARM,
    "sensitivity": MAIN,
}


@dataclasses.dataclass(frozen=True)
class Task:
    group: str
    scenario: str
    strategy: str
    seed: int
    start_delay_s: float
    checkpoint_s: float | None


def execute(task: Task) -> dict[str, Any]:
    strategy = STRATEGIES[task.strategy]
    case = make(task.scenario, task.seed, checkpoint_s=task.checkpoint_s)
    sim = {**SIM_KWARGS, "start_delay_s": task.start_delay_s, "max_time": case.max_time}
    store: InMemoryHistoryStore | None = None
    if strategy.warm:
        warmup = make(task.scenario, task.seed + WARM_SEED_OFFSET, checkpoint_s=task.checkpoint_s)
        filled = InMemoryHistoryStore()
        run(warmup.scenario, strategy, policy=warmup.policy, history_store=filled, **sim)
        store = InMemoryHistoryStore.from_json(json.loads(json.dumps(filled.to_json())))
    rep = run(case.scenario, strategy, policy=case.policy, history_store=store, **sim)
    return {**dataclasses.asdict(task), "total": rep.total, **rep.metrics()}


def tasks(args: argparse.Namespace) -> list[Task]:
    out: list[Task] = []
    primary = SIM_KWARGS["start_delay_s"]
    for group in args.groups:
        rows = [s for s in GROUPS[group] if not args.strategies or s.name in args.strategies]
        settings: Iterable[tuple[float, float | None]] = [(primary, None)]
        if group == "sensitivity":
            settings = [(d, c) for d in START_DELAYS for c in CHECKPOINTS]
        for scenario in args.scenarios:
            for delay, ckpt in settings:
                for strategy in rows:
                    for seed in range(args.seeds):
                        out.append(Task(group, scenario, strategy.name, seed, delay, ckpt))
    return out


# -- statistics -------------------------------------------------------------------------------


def values(runs: Sequence[dict[str, Any]], metric: str) -> list[float]:
    """Metric values with censoring as +inf (makespan, waits)."""
    return [math.inf if r[metric] is None else float(r[metric]) for r in runs]


def fmt(x: float, digits: int = 0) -> str:
    if math.isinf(x):
        return "> max_time"
    return f"{x:,.{digits}f}"


def cell(runs: Sequence[dict[str, Any]], metric: str, digits: int = 0) -> str:
    q1, med, q3 = summarize(values(runs, metric))
    return f"{fmt(med, digits)} [{fmt(q1, digits)}-{fmt(q3, digits)}]"


def by(runs: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def median(runs: Sequence[dict[str, Any]], metric: str) -> float:
    return summarize(values(runs, metric))[1]


# -- acceptance (SPEC §12 L4) ------------------------------------------------------------------


def rampgate_row(strategy: str) -> bool:
    """Full rampgate (Rules P, A, O, H; cold or warm), not an ablation or a baseline."""
    return strategy.startswith("Probation")


def check(runs: list[dict[str, Any]], min_seeds: int) -> tuple[list[str], list[str]]:
    """Returns (failures, notes). Failures break the build; notes report the target."""
    failures: list[str] = []
    notes: list[str] = []
    main = by((r for r in runs if r["group"] == "main"), "scenario", "strategy")
    seeds = {len(v) for v in main.values()}
    if seeds and min(seeds) < min_seeds:
        failures.append(f"only {min(seeds)} seeds per row, need >= {min_seeds}")
    # Every OOM of a rampgate row that is not first-contact is a bug.
    for r in runs:
        if r["group"] not in ("main", "warm") or not rampgate_row(r["strategy"]):
            continue
        where = f"{r['scenario']} {r['strategy']} seed {r['seed']}"
        other = r["oom_count"] - r["first_contact_ooms"]
        if other:
            failures.append(f"{where}: {other} OOM(s) not first-contact")
    for scenario in ("a_small", "b_mixed", "c_late_spike"):
        rows = main.get((scenario, "Probation"), [])
        greedy = {r["seed"]: r["oom_count"] for r in main.get((scenario, "GreedyFree"), [])}
        declared = {
            r["seed"]: r["oom_count"] for r in main.get((scenario, "DeclaredReservation"), [])
        }
        for r in rows:
            fc = r["first_contact_ooms"]
            for name, baseline in (("GreedyFree", greedy), ("DeclaredReservation", declared)):
                if fc and not fc < baseline.get(r["seed"], 0):
                    failures.append(
                        f"{scenario} seed {r['seed']}: {fc} first-contact OOM(s), "
                        f"not below {name}'s {baseline.get(r['seed'])}"
                    )
    d = "d_under_declared"
    prob, serial = main.get((d, "Probation"), []), main.get((d, "Serial"), [])
    if prob and serial and not median(prob, "makespan") < median(serial, "makespan"):
        failures.append(f"{d}: Probation's median makespan is not below Serial's")
    for name in ("GreedyFree", "DeclaredReservation"):
        base = main.get((d, name), [])
        if prob and base and not median(prob, "oom_count") < median(base, "oom_count"):
            failures.append(f"{d}: Probation's median OOMs are not below {name}'s")
    # Target, not a gate: warm Probation makespan <= 1.2 x GreedyFree (medians).
    warm = by((r for r in runs if r["group"] == "warm"), "scenario", "strategy")
    for scenario in sorted({s for s, _ in warm}):
        w = warm.get((scenario, "Probation (warm)"), [])
        g = main.get((scenario, "GreedyFree"), [])
        if w and g:
            ratio = median(w, "makespan") / median(g, "makespan")
            verdict = "met" if ratio <= 1.2 else "missed"
            notes.append(
                f"{scenario}: warm Probation / GreedyFree makespan = {ratio:.2f} ({verdict})"
            )
    return failures, notes


# -- output ----------------------------------------------------------------------------------


def config_data(config: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        out[f.name] = value.to_json() if isinstance(value, Resources) else value
    return out


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def results_json(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    used = sorted({r["strategy"] for r in runs})
    return {
        "meta": {
            "rampgate_version": rampgate.__version__,
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "seeds": list(range(args.seeds)),
            "warm_seed_offset": WARM_SEED_OFFSET,
            "sim": SIM_KWARGS,
            "max_time": {name: make(name, 0).max_time for name in args.scenarios},
            "sensitivity": {"start_delay_s": START_DELAYS, "checkpoint_s": CHECKPOINTS},
            "groups": args.groups,
        },
        "strategies": {
            name: {
                "admission": STRATEGIES[name].admission.name,
                "probation": STRATEGIES[name].admission.probation,
                "estimator": STRATEGIES[name].estimator,
                "warm": STRATEGIES[name].warm,
                "config": config_data(STRATEGIES[name].config),
            }
            for name in used
        },
        "runs": sorted(
            runs,
            key=lambda r: (
                r["group"],
                r["scenario"],
                r["start_delay_s"],
                r["checkpoint_s"] or 0,
                r["strategy"],
                r["seed"],
            ),
        ),
    }


def table(runs: list[dict[str, Any]], group: str, order: Sequence[str]) -> list[str]:
    lines: list[str] = []
    rows = by((r for r in runs if r["group"] == group), "scenario", "strategy")
    for scenario in sorted({s for s, _ in rows}):
        lines += [
            f"#### {scenario}",
            "",
            "| Strategy | Makespan (s) | Completed | OOMs | First-contact: unknown"
            " | First-contact: under-declared hint"
            " | Lost work (step-s) | Utilization | Wait p50 (s) | Wait p95 (s) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in order:
            rs = rows.get((scenario, name))
            if not rs:
                continue
            total = rs[0]["total"]
            lines.append(
                f"| {name} | {cell(rs, 'makespan')} | {cell(rs, 'completed')} / {total}"
                f" | {cell(rs, 'oom_count')} | {cell(rs, 'first_contact_unknown')}"
                f" | {cell(rs, 'first_contact_hint')}"
                f" | {cell(rs, 'lost_work_step_s')} | {cell(rs, 'utilization', 2)}"
                f" | {cell(rs, 'wait_p50')} | {cell(rs, 'wait_p95')} |"
            )
        lines.append("")
    return lines


def sensitivity_table(runs: list[dict[str, Any]]) -> list[str]:
    rows = by(
        (r for r in runs if r["group"] == "sensitivity"),
        "scenario",
        "start_delay_s",
        "checkpoint_s",
        "strategy",
    )
    names = [s.name for s in MAIN]
    lines: list[str] = []
    for scenario in sorted({k[0] for k in rows}):
        lines += [
            f"#### {scenario}",
            "",
            "Median makespan in seconds (median OOMs).",
            "",
            "| start_delay_s | checkpoint_s | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names),
        ]
        for delay in START_DELAYS:
            for ckpt in CHECKPOINTS:
                cells = []
                for name in names:
                    rs = rows.get((scenario, delay, ckpt, name), [])
                    if not rs:
                        cells.append("-")
                        continue
                    cells.append(f"{fmt(median(rs, 'makespan'))} ({fmt(median(rs, 'oom_count'))})")
                lines.append(f"| {delay:.0f} | {ckpt:.0f} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def results_md(data: dict[str, Any], failures: list[str], notes: list[str]) -> str:
    runs = data["runs"]
    meta = data["meta"]
    lines = [
        "# rampgate benchmark results",
        "",
        f"rampgate {meta['rampgate_version']}, commit `{meta['git_commit']}`,"
        f" Python {meta['python']}, seeds 0-{len(meta['seeds']) - 1}"
        f" (warm runs pre-filled from seed + {meta['warm_seed_offset']}).",
        f"Simulator: {', '.join(f'{k}={v}' for k, v in meta['sim'].items())}.",
        "Cells: median [Q1-Q3] over the seeds; `> max_time` marks censored runs.",
        "Generated by `benchmarks/run.py` from `results.json`; see SPEC §8 for definitions.",
        "",
    ]
    if "main" in meta["groups"]:
        lines += ["## Main", "", *table(runs, "main", [s.name for s in MAIN])]
    if "ablations" in meta["groups"]:
        order = [MAIN[0].name, *(s.name for s in ABLATIONS)]
        merged = runs + [{**r, "group": "ablations"} for r in runs if r["group"] == "main"]
        lines += ["## Ablations", "", *table(merged, "ablations", order)]
    if "warm" in meta["groups"]:
        order = [MAIN[0].name, WARM[0].name, MAIN[1].name, WARM[1].name]
        merged = runs + [{**r, "group": "warm"} for r in runs if r["group"] == "main"]
        lines += ["## Cold vs warm history", "", *table(merged, "warm", order)]
    if "sensitivity" in meta["groups"]:
        lines += ["## Sensitivity", "", *sensitivity_table(runs)]
    lines += ["## Acceptance (SPEC §12 L4)", ""]
    lines += [f"- FAIL: {f}" for f in failures] or ["- All criteria met."]
    lines += ["", "Target (not a gate):", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", default=list(NAMES), choices=NAMES)
    parser.add_argument("--strategies", nargs="+", default=[], choices=sorted(STRATEGIES))
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--check", action="store_true", help="enforce SPEC §12 L4")
    parser.add_argument("--min-seeds", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    todo = tasks(args)
    print(f"{len(todo)} runs on {args.jobs} processes", file=sys.stderr)
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            runs = list(pool.map(execute, todo, chunksize=4))
    else:
        runs = [execute(t) for t in todo]

    data = results_json(args, runs)
    failures, notes = check(data["runs"], args.min_seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    (args.out / "results.md").write_text(results_md(data, failures, notes))
    if not args.no_plots:
        from plots import plot_all  # noqa: PLC0415 - matplotlib: the `bench` group

        plot_all(data, args.out)
    for note in notes:
        print(note, file=sys.stderr)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if args.check and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
