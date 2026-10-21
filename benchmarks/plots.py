"""Plots for the benchmark (``uv run --group bench``): every seed plus median and IQR.

One figure per scenario (``<scenario>.png``, four metrics) and an overview for
the README (``overview.png``: makespan and OOMs for every scenario). Strategies
are on the x axis, so color only separates rampgate's rows (blue) from the
baselines (gray). Censored runs are drawn as a triangle at the top of the
panel, labelled ``> max_time``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from rampgate.sim import MAIN, quantile

RAMPGATE = "#2a78d6"  # categorical slot 1 (references/palette.md)
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
        fig.tight_layout()
        fig.savefig(out / f"{scenario}.png", dpi=130, facecolor=SURFACE)
        plt.close(fig)

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
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
