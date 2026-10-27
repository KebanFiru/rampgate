"""Plots for the benchmark (``uv run --group bench``): every seed plus median and IQR.

One figure per scenario (``<scenario>.png``, four metrics), an overview
(``overview.png``: makespan and OOMs for every scenario), and the README's main
figures, Pareto plots of makespan against OOMs (``pareto_ooms.png``) and lost
work (``pareto_lost.png``): one point per strategy (median, IQR bars), with the
baselines, ablations, cold and warm rampgate and the ``trusted_margin`` sweep.
In the per-scenario figures strategies are on the x axis, so color only
separates rampgate's rows (blue) from the baselines (gray). Censored runs are
drawn as a triangle at the top of the panel, labelled ``> max_time``.

Every figure carries a footer with the tag, commit and seed set of the run.
Plots can be redrawn from a saved ``results.json`` without rerunning:

    uv run python benchmarks/plots.py docs/benchmark/freeze-1/heldout/results.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from rampgate.sim import ABLATIONS, BASELINES, MAIN, MARGIN_SWEEP, WARM, quantile

RAMPGATE = "#2a78d6"  # categorical slot 1 (references/palette.md)
WARM_COLOR = "#eb6834"  # slot 2
ABLATION = "#1baf7a"  # slot 3 (below 3:1 on the surface: legend, marker shape, results.md)
BASELINE = "#8a8984"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

LABELS = {
    "Probation (all)": "Probation\n(all)",
    "GreedyFree": "Greedy\nFree",
    "DeclaredReservation": "Declared\nReservation",
}

PANELS = (
    ("makespan", "Makespan (s)", False),
    ("oom_count", "OOM kills", True),
    ("lost_work_step_s", "Lost work (step-s)", True),
    ("wait_p95", "Queue wait p95 (s)", False),
)


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _panel(
    ax: Axes, runs: dict[str, list[dict[str, Any]]], metric: str, title: str, log: bool
) -> None:
    names = [s.name for s in MAIN if s.name in runs]
    finite = [float(r[metric]) for n in names for r in runs[n] if r[metric] is not None]
    top = max(finite, default=1.0) * 1.15 or 1.0
    for i, name in enumerate(names):
        color = RAMPGATE if name.startswith("Probation") else BASELINE
        vals = [math.inf if r[metric] is None else float(r[metric]) for r in runs[name]]
        n = len(vals)
        xs = [i + (j - (n - 1) / 2) * (0.5 / max(n, 1)) for j in range(n)]
        shown = [v if not math.isinf(v) else top for v in vals]
        if log:
            shown = [v + 1 for v in shown]  # log scale with zeros: plot value + 1
        ax.scatter(
            [x for x, v in zip(xs, vals, strict=True) if not math.isinf(v)],
            [s for s, v in zip(shown, vals, strict=True) if not math.isinf(v)],
            s=14,
            color=color,
            alpha=0.75,
            edgecolors=SURFACE,
            linewidths=0.6,
            zorder=3,
        )
        censored = [x for x, v in zip(xs, vals, strict=True) if math.isinf(v)]
        if censored:
            y = (top + 1) if log else top
            ax.scatter(censored, [y] * len(censored), marker="^", s=30, color=color, zorder=3)
            ax.annotate(
                "> max_time",
                (i, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=MUTED,
            )
        q1, med, q3 = (quantile(vals, q) for q in (0.25, 0.5, 0.75))
        if not math.isinf(med):
            lo = q1 + (1 if log else 0)
            hi = (top if math.isinf(q3) else q3) + (1 if log else 0)
            ax.vlines(i + 0.36, lo, hi, color=INK, linewidth=2)
            ax.hlines(med + (1 if log else 0), i + 0.29, i + 0.43, color=INK, linewidth=2)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS.get(n, n) for n in names], fontsize=7, color=INK)
    ax.set_title(title + (" (+1, log)" if log else ""), fontsize=9, color=INK, loc="left")
    _style(ax)


def _grouped(data: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in data["runs"]:
        if r["group"] == "main":
            out[r["scenario"]][r["strategy"]].append(r)
    return out


def provenance(data: dict[str, Any]) -> str:
    """``tag · commit · seeds``: which frozen code and which seed set a figure shows."""
    meta = data["meta"]
    seeds = meta["seeds"]
    commit = meta.get("git_commit") or "unknown"
    span = f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else seeds
    return f"{meta.get('tag') or 'untagged'} · commit {commit[:12]} · seeds {span} ({len(seeds)})"


def _save(fig: Any, data: dict[str, Any], path: Path, dpi: int) -> None:
    fig.text(0.995, 0.005, provenance(data), fontsize=7, color=MUTED, ha="right", va="bottom")
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_all(data: dict[str, Any], out: Path) -> None:
    grouped = _grouped(data)
    for scenario, runs in sorted(grouped.items()):
        fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 3.6), facecolor=SURFACE)
        for ax, (metric, title, log) in zip(axes, PANELS, strict=True):
            _panel(ax, runs, metric, title, log)
        fig.suptitle(
            f"{scenario}: every seed, median and IQR (black)",
            fontsize=10,
            color=INK,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, data, out / f"{scenario}.png", 130)

    scenarios = sorted(grouped)
    fig, axes = plt.subplots(
        2, len(scenarios), figsize=(3.4 * len(scenarios), 6.4), facecolor=SURFACE, squeeze=False
    )
    for col, scenario in enumerate(scenarios):
        _panel(axes[0][col], grouped[scenario], "makespan", f"{scenario}\nmakespan (s)", False)
        _panel(axes[1][col], grouped[scenario], "oom_count", "OOM kills", True)
    fig.suptitle(
        "rampgate (blue) vs baselines (gray): every seed, median and IQR (black)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, data, out / "overview.png", 110)
    plot_pareto(data, out)


# -- Pareto: makespan against OOMs / lost work ----------------------------------------------

# family -> (color, marker, legend label); identity is never color alone.
FAMILIES = {
    "cold": (RAMPGATE, "o", "rampgate, cold history"),
    "warm": (WARM_COLOR, "s", "rampgate, warm history"),
    "ablation": (ABLATION, "^", "ablations"),
    "baseline": (BASELINE, "D", "baselines"),
}
SHORT = {
    "Probation": "Probation",
    "Probation (all)": "all",
    "Probation (warm)": "warm",
    "Probation (all, warm)": "all, warm",
    "GreedyFree": "GreedyFree",
    "DeclaredReservation": "Declared",
    "Serial": "Serial",
    "GreedyFree + Rule O": "Greedy+O",
    "rampgate - Rule P": "-P",
    "rampgate - history": "-history",
}


def _family(name: str) -> str:
    if name in {s.name for s in BASELINES}:
        return "baseline"
    if name in {s.name for s in ABLATIONS}:
        return "ablation"
    return "warm" if "warm" in name else "cold"


def _stats(rs: list[dict[str, Any]], metric: str) -> tuple[float, float, float]:
    vals = [math.inf if r[metric] is None else float(r[metric]) for r in rs]
    return quantile(vals, 0.25), quantile(vals, 0.5), quantile(vals, 0.75)


def _pareto_panel(ax: Axes, runs: dict[str, list[dict[str, Any]]], metric: str, title: str) -> None:
    named = [*MAIN, *ABLATIONS, *WARM]
    xmax = max((_stats(rs, "makespan")[2] for rs in runs.values() if rs), default=1.0)
    finite_x = [
        _stats(rs, "makespan")[2]
        for rs in runs.values()
        if not math.isinf(_stats(rs, "makespan")[2])
    ]
    edge = max(finite_x, default=1.0) * 1.08 if math.isinf(xmax) else xmax * 1.05
    labels: list[tuple[float, float, str, str, tuple[int, int]]] = []
    # The trusted_margin sweep: a thin line per family through its margins, ends labelled.
    for warm in (False, True):
        sweep = [s.name for s in MARGIN_SWEEP if ("(warm)" in s.name) == warm and s.name in runs]
        pts = [(_stats(runs[n], "makespan")[1], _stats(runs[n], metric)[1], n) for n in sweep]
        pts = [p for p in pts if not math.isinf(p[0])]
        if not pts:
            continue
        color = WARM_COLOR if warm else RAMPGATE
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=1, alpha=0.6)
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=12,
            color=color,
            edgecolors=SURFACE,
            linewidths=1,
            zorder=3,
        )
    for strategy in named:
        rs = runs.get(strategy.name)
        if not rs:
            continue
        color, marker, _ = FAMILIES[_family(strategy.name)]
        xq1, xm, xq3 = _stats(rs, "makespan")
        yq1, ym, yq3 = _stats(rs, metric)
        x = edge if math.isinf(xm) else xm
        ax.hlines(ym, xq1, edge if math.isinf(xq3) else xq3, color=color, linewidth=1.2, alpha=0.8)
        ax.vlines(x, yq1, yq3, color=color, linewidth=1.2, alpha=0.8)
        ax.scatter(
            [x], [ym], s=46, marker=marker, color=color, edgecolors=SURFACE, linewidths=2, zorder=4
        )
        label = SHORT.get(strategy.name, strategy.name) + (" >max" if math.isinf(xm) else "")
        labels.append((x, ym, label, INK, (5, 4)))
    _place_labels(ax, labels, edge)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.4, max(2.0, ax.get_ylim()[1] * 1.6))
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)


def _place_labels(
    ax: Axes, labels: list[tuple[float, float, str, str, tuple[int, int]]], edge: float
) -> None:
    """Direct labels. Points that (nearly) coincide share one label; labels on the same
    row alternate above and below their point so neighbours do not overprint."""
    merged: list[tuple[float, float, list[str], str]] = []
    for x, y, text, color, _ in sorted(labels, key=lambda t: (t[0], t[1])):
        for m in merged:
            if abs(m[0] - x) <= 0.05 * edge and abs(math.asinh(m[1]) - math.asinh(y)) <= 0.35:
                m[2].append(text)
                break
        else:
            merged.append((x, y, [text], color))
    row_turn: dict[int, int] = defaultdict(int)
    for x, y, texts, color in merged:
        row = round(math.asinh(y) / 0.35)
        above = row_turn[row] % 2 == 0
        row_turn[row] += 1
        ax.annotate(
            ", ".join(texts),
            (x, y),
            textcoords="offset points",
            xytext=(4, 5 if above else -12),
            fontsize=6.5,
            color=INK if color == INK else MUTED,
        )


def plot_pareto(data: dict[str, Any], out: Path) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in data["runs"]:
        if r["group"] != "sensitivity":
            grouped[r["scenario"]][r["strategy"]].append(r)
    scenarios = sorted(grouped)
    for metric, ylabel, name in (
        ("oom_count", "OOM kills per run", "pareto_ooms.png"),
        ("lost_work_step_s", "Lost work per run (step-s)", "pareto_lost.png"),
    ):
        cols = 3
        rows = math.ceil(len(scenarios) / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(5.2 * cols, 4.0 * rows), facecolor=SURFACE, squeeze=False
        )
        for i, scenario in enumerate(scenarios):
            ax = axes[i // cols][i % cols]
            _pareto_panel(ax, grouped[scenario], metric, scenario)
            ax.set_xlabel("Makespan (s), median and IQR", fontsize=8, color=MUTED)
            ax.set_ylabel(ylabel + " (symlog)", fontsize=8, color=MUTED)
        for j in range(len(scenarios), rows * cols):
            axes[j // cols][j % cols].set_visible(False)
        handles = [
            Line2D([], [], marker=m, color=c, linestyle="", markersize=7, label=label)
            for c, m, label in FAMILIES.values()
        ]
        handles.append(Line2D([], [], color=MUTED, linewidth=1, label="trusted_margin sweep (tm)"))
        seeds = data["meta"]["seeds"]
        fig.suptitle(
            f"Makespan vs {ylabel.lower()}: lower-left is better\n"
            f"Seeds {seeds[0]}-{seeds[-1]}; points are medians, bars the IQR;"
            " '>max' = censored (drawn at the right edge); tm = trusted_margin sweep",
            fontsize=10,
            color=INK,
            x=0.01,
            y=0.995,
            ha="left",
            va="top",
        )
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.925),
            ncol=len(handles),
            fontsize=8,
            frameon=False,
            labelcolor=INK,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.88))
        _save(fig, data, out / name, 120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw the plots from a saved results.json.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, help="default: next to results.json")
    parser.add_argument("--tag", help="tag to show when results.json does not record one")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    if args.tag and not data["meta"].get("tag"):
        data["meta"]["tag"] = args.tag
    out = args.out or args.results.parent
    out.mkdir(parents=True, exist_ok=True)
    plot_all(data, out)


if __name__ == "__main__":
    main()
