"""Charts written to ``results/``."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
# Fixed categorical order: each model keeps its colour wherever it appears.
_SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_recall_by_segment(recall: Mapping[str, Mapping[str, float]],
                           segment_sizes: Mapping[str, int], path: Path, k: int = 10) -> Path:
    """Grouped column chart of recall@k per user segment and model.

    Args:
        recall: ``model name -> {segment name -> recall}``; insertion order sets the
            order of models (at most three) and segments.
        segment_sizes: ``segment name -> number of evaluated users`` (shown on the axis).
        path: Output PNG path.
        k: Cut-off, used in labels only.
    """
    models = list(recall)
    segments = list(next(iter(recall.values())))
    if len(models) > len(_SERIES_COLOURS):
        raise ValueError("The chart supports at most three models")

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    width, gap = 0.2, 0.03
    peak = max(max(values.values()) for values in recall.values())
    for m, model in enumerate(models):
        offset = (m - (len(models) - 1) / 2) * (width + gap)
        values = [recall[model][segment] for segment in segments]
        positions = [s + offset for s in range(len(segments))]
        ax.bar(positions, values, width=width, color=_SERIES_COLOURS[m], linewidth=0,
               label=model)
        # Values are printed on every column: the chart is a static image with no hover.
        for position, value in zip(positions, values):
            ax.text(position, value + peak * 0.015, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=_TEXT)

    ax.set_xlim(-0.5, len(segments) - 0.5)
    ax.set_ylim(0, peak * 1.14)
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels([f"{segment}\nn = {segment_sizes[segment]:,}" for segment in segments],
                       fontsize=9.5, color=_TEXT_SECONDARY)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=_TEXT_SECONDARY)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.set_ylabel(f"recall@{k}", fontsize=10, color=_TEXT_SECONDARY)

    fig.suptitle(f"Recall@{k} on the held-out test window, by model and user segment",
                 x=0.065, y=0.975, ha="left", fontsize=12.5, color=_TEXT, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.11), ncols=len(models), frameon=False,
              fontsize=9.5, labelcolor=_TEXT_SECONDARY, handlelength=1.0, handleheight=1.0,
              columnspacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.83, bottom=0.14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path
