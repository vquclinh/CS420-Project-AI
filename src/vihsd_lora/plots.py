"""PNG charts for the slides. Each run keeps a fixed color across all charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .config import LABELS, Settings  # noqa: E402
from .report import collect_runs, load_history, load_published, results_dir  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RUN_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}
LORA_COLOR = RUN_COLORS["A"]
FULL_COLOR = RUN_COLORS["B"]
BLUE_RAMP = ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 12,
    "text.color": INK, "axes.labelcolor": SECONDARY, "axes.titlecolor": INK,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 11, "legend.labelcolor": SECONDARY,
}


def _run_name(row) -> str:
    if row["mode"] == "full":
        return f"Full fine-tune, {int(row['epochs'])} epochs"
    return f"LoRA r={int(row['rank'])}, {int(row['epochs'])} epochs"


def _clean_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis == "y":
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["bottom"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(figure, directory: Path, name: str) -> Path:
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)
    return path


def _primary_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Keep the smallest seed of each run so the charts have no duplicate lines."""
    return runs.sort_values(["run_key", "seed"]).drop_duplicates("run_key")


def _horizontal_bar(ax, value, center, thickness, color, x_max) -> None:
    """Bar with a rounded end. The end at the axis is cut flat by drawing it past the axis."""
    box = ax.get_window_extent()
    y_low, y_high = ax.get_ylim()
    x_per_px = x_max / box.width
    y_per_px = abs(y_high - y_low) / box.height
    radius = 8 * x_per_px
    patch = FancyBboxPatch((-radius, center - thickness / 2), value + radius, thickness,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           mutation_aspect=y_per_px / x_per_px, linewidth=0, facecolor=color, clip_on=True)
    ax.add_patch(patch)


def plot_line_history(runs: pd.DataFrame, settings: Settings, column: str, title: str,
                      ylabel: str, directory: Path, name: str) -> Optional[Path]:
    with plt.rc_context(STYLE):
        figure, ax = plt.subplots(figsize=(9.6, 5.2))
        for _, row in _primary_runs(runs).iterrows():
            history = load_history(row["slug"], settings).dropna(subset=[column])
            if history.empty:
                continue
            ax.plot(history["epoch"], history[column], color=RUN_COLORS[row["run_key"]], linewidth=1.8,
                    solid_capstyle="round", solid_joinstyle="round", label=_run_name(row))
            ax.plot(history["epoch"].iloc[-1], history[column].iloc[-1], "o", markersize=7,
                    color=RUN_COLORS[row["run_key"]], markeredgecolor=SURFACE, markeredgewidth=1.4)
        ax.set_title(title, pad=44)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        _clean_axes(ax)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2,
                  handlelength=1.4, columnspacing=1.4)
        return _save(figure, directory, name)


def plot_rank_effect(runs: pd.DataFrame, directory: Path) -> Optional[Path]:
    data = _primary_runs(runs)
    lora = data[(data["mode"] == "lora") & (data["run_key"] != "E")].sort_values("rank")
    full = data[data["mode"] == "full"]
    if len(lora) < 2:
        return None
    with plt.rc_context(STYLE):
        figure, ax = plt.subplots(figsize=(8.4, 5.0))
        positions = np.arange(len(lora))
        ax.plot(positions, lora["dev_macro_f1"], color=LORA_COLOR, linewidth=1.8, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.4, label="LoRA")
        for position, value in zip(positions, lora["dev_macro_f1"]):
            ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 12), textcoords="offset points",
                        ha="center", color=INK, fontsize=11)
        if len(full):
            ax.axhline(full["dev_macro_f1"].iloc[0], color=FULL_COLOR, linewidth=1.8, label="Full fine-tune")
        ax.set_xticks(positions, [f"r={int(rank)}\n{int(params) / 1e6:.2f}M params"
                                  for rank, params in zip(lora["rank"], lora["trainable_params"])])
        ax.set_xlim(-0.4, len(lora) - 0.6)
        ax.set_title("Does a higher rank still improve the score?", pad=40)
        ax.set_ylabel("Dev macro F1")
        _clean_axes(ax)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, handlelength=1.4)
        return _save(figure, directory, "rank_effect")


def plot_f1_vs_params(runs: pd.DataFrame, directory: Path) -> Optional[Path]:
    if "test_macro_f1" not in runs:
        return None
    tested = runs.dropna(subset=["test_macro_f1"])
    if tested.empty:
        return None
    published = load_published()
    published = published[(published["table"] == "main")].dropna(subset=["trainable_params"])
    with plt.rc_context(STYLE):
        figure, ax = plt.subplots(figsize=(9.6, 5.4))
        ax.scatter(published["trainable_params"], published["macro_f1"], s=62, color=MUTED,
                   edgecolors=SURFACE, linewidths=1.4, label="Published full fine-tune", zorder=3)
        for mode, color, label in (("lora", LORA_COLOR, "Our LoRA"), ("full", FULL_COLOR, "Our full fine-tune")):
            group = tested[tested["mode"] == mode]
            if len(group):
                ax.scatter(group["trainable_params"], group["test_macro_f1"], s=86, color=color,
                           edgecolors=SURFACE, linewidths=1.4, label=label, zorder=4)
        best = published.sort_values("macro_f1", ascending=False).iloc[0]
        ax.annotate(f"{best['model']} (paper)", (best["trainable_params"], best["macro_f1"]), xytext=(-10, 4),
                    textcoords="offset points", ha="right", color=SECONDARY, fontsize=11)
        top_lora = tested[tested["mode"] == "lora"].sort_values("test_macro_f1", ascending=False).head(1)
        for _, row in top_lora.iterrows():
            ax.annotate(f"LoRA r={int(row['rank'])}: {row['test_macro_f1']:.2f}",
                        (row["trainable_params"], row["test_macro_f1"]), xytext=(8, -16),
                        textcoords="offset points", color=INK, fontsize=11)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}M"))
        ax.minorticks_off()
        ax.margins(y=0.1)
        ax.set_title("Test macro F1 vs. trainable parameters", pad=40)
        ax.set_xlabel("Trainable parameters (log scale)")
        ax.set_ylabel("Macro F1")
        _clean_axes(ax)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, handletextpad=0.3)
        return _save(figure, directory, "f1_vs_params")


def plot_per_class(runs: pd.DataFrame, directory: Path) -> Optional[Path]:
    if "test_macro_f1" not in runs:
        return None
    tested = runs.dropna(subset=["test_macro_f1"])
    lora = tested[tested["mode"] == "lora"].sort_values("test_macro_f1", ascending=False).head(1)
    full = tested[tested["mode"] == "full"].head(1)
    if lora.empty or full.empty:
        return None
    with plt.rc_context(STYLE):
        figure, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.set_ylim(len(LABELS) - 0.45, -0.55)
        ax.set_xlim(0, 110)
        figure.canvas.draw()
        thickness = 0.26
        for index, label in enumerate(LABELS):
            values = [(lora, LORA_COLOR, -0.15), (full, FULL_COLOR, 0.15)]
            for frame, color, offset in values:
                value = float(frame[f"test_f1_{label}"].iloc[0])
                _horizontal_bar(ax, value, index + offset, thickness, color, 110)
                ax.text(value + 1.2, index + offset, f"{value:.1f}", va="center", color=INK, fontsize=11)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Test F1")
        ax.set_title("Per-label F1 on test", pad=40)
        _clean_axes(ax, "x")
        handles = [plt.Rectangle((0, 0), 1, 1, color=LORA_COLOR), plt.Rectangle((0, 0), 1, 1, color=FULL_COLOR)]
        ax.legend(handles, [f"LoRA r={int(lora['rank'].iloc[0])}", "Full fine-tune"], loc="lower left",
                  bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, handlelength=1.0)
        return _save(figure, directory, "per_class_f1")


def plot_cost(runs: pd.DataFrame, directory: Path) -> Optional[Path]:
    data = _primary_runs(runs)
    lora = data[data["run_key"] == "A"]
    full = data[data["mode"] == "full"]
    if lora.empty or full.empty:
        return None
    measures = [("trainable_params", "Trainable params (millions)", 1e6), ("mean_epoch_seconds", "Seconds per epoch", 1),
                ("gpu_peak_mb", "Peak VRAM (GB)", 1024), ("saved_mb", "Saved size (MB)", 1)]
    measures = [item for item in measures if float(full[item[0]].iloc[0]) > 0]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(measures), figsize=(3.0 * len(measures) + 1.0, 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (column, title, scale) in zip(axes, measures):
            values = [float(lora[column].iloc[0]) / scale, float(full[column].iloc[0]) / scale]
            x_max = max(values) * 1.32
            ax.set_xlim(0, x_max)
            ax.set_ylim(1.6, -0.6)
            figure.canvas.draw()
            for index, (value, color) in enumerate(zip(values, (LORA_COLOR, FULL_COLOR))):
                _horizontal_bar(ax, value, index, 0.34, color, x_max)
                text = f"{value:.2f}" if value < 10 else f"{value:.1f}" if value < 100 else f"{value:,.0f}"
                ax.text(value + x_max * 0.02, index, text, va="center", color=INK, fontsize=11)
            ax.set_title(title, fontsize=12, pad=8, fontweight="normal", color=SECONDARY)
            ax.set_yticks([0, 1], [f"LoRA r={int(lora['rank'].iloc[0])}", "Full"])
            ax.set_xticks([])
            for side in ("top", "right", "bottom"):
                ax.spines[side].set_visible(False)
        figure.suptitle("Cost of LoRA vs. full fine-tune", x=0.02, ha="left",
                        fontsize=15, fontweight="bold", color=INK, y=1.0)
        figure.subplots_adjust(top=0.72, wspace=0.12, left=0.09, right=0.98, bottom=0.04)
        return _save(figure, directory, "cost_comparison")


def plot_confusion(slug: str, settings: Settings, directory: Path, label: str) -> Optional[Path]:
    path = Path(settings.output_root) / "runs" / slug / "test_metrics.json"
    if not path.exists():
        return None
    matrix = np.array(json.loads(path.read_text(encoding="utf-8"))["confusion_matrix"], dtype=float)
    rates = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    colormap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    with plt.rc_context(STYLE):
        figure, ax = plt.subplots(figsize=(5.8, 5.2))
        image = ax.imshow(rates, cmap=colormap, vmin=0, vmax=1)
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
                color = "#ffffff" if rates[row, column] > 0.55 else INK
                ax.text(column, row, f"{rates[row, column] * 100:.1f}%\n{int(matrix[row, column])}",
                        ha="center", va="center", color=color, fontsize=11)
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Test confusion matrix: {label}", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(LABELS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LABELS)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Rate per true label", color=SECONDARY)
        colorbar.outline.set_visible(False)
        return _save(figure, directory, f"confusion_{slug}")


def make_all(settings: Optional[Settings] = None) -> list:
    """Create every chart that has enough data and return the list of files."""
    settings = settings or Settings.from_env()
    runs = collect_runs(settings)
    if runs.empty:
        raise FileNotFoundError("No finished runs yet.")
    directory = results_dir(settings) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_line_history(runs, settings, "dev_macro_f1", "Dev macro F1 by epoch", "Macro F1",
                          directory, "dev_macro_f1"),
        plot_line_history(runs, settings, "train_loss", "Train loss by epoch", "Loss", directory, "train_loss"),
        plot_rank_effect(runs, directory),
        plot_f1_vs_params(runs, directory),
        plot_per_class(runs, directory),
        plot_cost(runs, directory),
    ]
    if "test_macro_f1" in runs:
        for _, row in runs.dropna(subset=["test_macro_f1"]).iterrows():
            paths.append(plot_confusion(row["slug"], settings, directory, _run_name(row)))
    return [path for path in paths if path is not None]
