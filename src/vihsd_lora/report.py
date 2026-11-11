"""Combine the run results with the published numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Settings
from .data import EXPECTED_COUNTS
from .metrics import majority_baseline, paired_bootstrap_difference

PUBLISHED_PATH = Path(__file__).resolve().parents[2] / "data" / "published_results.csv"


def results_dir(settings: Settings) -> Path:
    path = Path(settings.output_root) / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_published() -> pd.DataFrame:
    return pd.read_csv(PUBLISHED_PATH)


def collect_runs(settings: Optional[Settings] = None) -> pd.DataFrame:
    """One row per run, with training info and test scores if already scored."""
    settings = settings or Settings.from_env()
    rows = []
    for summary_path in sorted((Path(settings.output_root) / "runs").glob("*/run_summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        test_path = summary_path.with_name("test_metrics.json")
        if test_path.exists():
            test = json.loads(test_path.read_text(encoding="utf-8"))
            for key, value in test.items():
                if key != "confusion_matrix":
                    row[f"test_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_history(slug: str, settings: Settings) -> pd.DataFrame:
    path = Path(settings.output_root) / "runs" / slug / "history.csv"
    return pd.read_csv(path).drop_duplicates("epoch", keep="last")


def majority_row() -> dict:
    counts = EXPECTED_COUNTS["test"]
    y_true = np.repeat([0, 1, 2], [counts["CLEAN"], counts["OFFENSIVE"], counts["HATE"]])
    scores = majority_baseline(y_true)
    return {"System": "Always predict CLEAN", "Acc": round(scores["accuracy"], 2),
            "Macro F1": round(scores["macro_f1"], 2), "Trainable params": 0,
            "Source": "Computed from the test label distribution"}


def _run_label(row) -> str:
    model = str(row["model_name"])
    name = "ViSoBERT" if "visobert" in model.lower() else Path(model).name
    if row["mode"] == "lora":
        return f"{name} + LoRA r={int(row['rank'])}, {int(row['epochs'])} epochs (ours)"
    return f"{name} full fine-tune, {int(row['epochs'])} epochs (ours)"


def main_table(settings: Optional[Settings] = None) -> pd.DataFrame:
    """Main table: 13 published rows, the always-CLEAN baseline, and the runs scored on test."""
    settings = settings or Settings.from_env()
    published = load_published()
    published = published[published["table"] == "main"]
    rows = [{"System": row.model, "Acc": row.accuracy, "Macro F1": row.macro_f1,
             "Trainable params": row.trainable_params, "Source": row.source}
            for row in published.itertuples()]
    rows.append(majority_row())
    runs = collect_runs(settings)
    if len(runs) and "test_macro_f1" in runs:
        for _, row in runs.dropna(subset=["test_macro_f1"]).iterrows():
            rows.append({"System": _run_label(row), "Acc": round(row["test_accuracy"], 2),
                         "Macro F1": round(row["test_macro_f1"], 2),
                         "Trainable params": row["trainable_params"], "Source": "Our run"})
    table = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    return table


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "---|" * len(table.columns)
    lines = [header, divider]
    for _, row in table.iterrows():
        cells = []
        for column in table.columns:
            value = row[column]
            if column == "Trainable params":
                cells.append("Not counted" if pd.isna(value) else "0" if value == 0 else f"{value / 1e6:.2f}M")
            elif isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def export_tables(settings: Optional[Settings] = None) -> dict:
    settings = settings or Settings.from_env()
    directory = results_dir(settings)
    table = main_table(settings)
    table.to_csv(directory / "main_results.csv", index=False, encoding="utf-8-sig")
    (directory / "main_results.md").write_text(_markdown(table), encoding="utf-8")
    runs = collect_runs(settings)
    if len(runs):
        runs.to_csv(directory / "all_runs.csv", index=False, encoding="utf-8-sig")
    return {"csv": directory / "main_results.csv", "markdown": directory / "main_results.md"}


def compare_on_test(slug_a: str, slug_b: str, settings: Optional[Settings] = None) -> dict:
    """Compare two runs on the same test samples with a paired bootstrap."""
    settings = settings or Settings.from_env()
    root = Path(settings.output_root) / "runs"
    first = pd.read_csv(root / slug_a / "predictions_test.csv")
    second = pd.read_csv(root / slug_b / "predictions_test.csv")
    if not (first["label"].values == second["label"].values).all():
        raise ValueError("The two runs were not scored on the same test set.")
    return paired_bootstrap_difference(first["label"], first["pred"], second["pred"])


def log_figures_to_wandb(paths, settings: Optional[Settings] = None) -> None:
    """Log the charts to W&B as a separate run."""
    settings = settings or Settings.from_env()
    if not settings.use_wandb:
        return
    import wandb

    session = wandb.init(project=settings.wandb_project, entity=settings.wandb_entity,
                         name="slide-figures", group=settings.wandb_group, job_type="report")
    wandb.log({Path(path).stem: wandb.Image(str(path)) for path in paths})
    session.finish()
