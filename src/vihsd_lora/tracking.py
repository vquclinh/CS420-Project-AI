"""Kết nối W&B và ghi lịch sử train ra CSV."""

from __future__ import annotations

import csv
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainerCallback

from .config import LABELS, Run, Settings

HISTORY_FIELDS = [
    "epoch", "step", "train_loss", "dev_loss", "dev_accuracy", "dev_macro_f1", "dev_weighted_f1",
    *[f"dev_f1_{label}" for label in LABELS], "epoch_seconds", "gpu_peak_mb",
]


def _run_id(run_dir: Path) -> str:
    """Giữ id cố định để train tiếp vẫn nối vào cùng một run trên W&B."""
    path = run_dir / "wandb_id.txt"
    if path.exists():
        return path.read_text().strip()
    identifier = uuid.uuid4().hex[:10]
    path.write_text(identifier)
    return identifier


def start_wandb(run: Run, seed: int, slug: str, run_dir: Path, settings: Settings,
                parameters: Optional[dict] = None):
    if not settings.use_wandb:
        return None
    import wandb

    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")
    config = {
        "run_key": run.key, "mode": run.mode, "rank": run.rank, "lora_alpha": run.lora_alpha,
        "learning_rate": run.learning_rate, "epochs": settings.epochs_for(run), "seed": seed,
        "model_name": settings.model_name, "batch_size": settings.batch_size,
        "max_length": settings.max_length, "warmup_ratio": settings.warmup_ratio,
        "lora_dropout": settings.lora_dropout, **(parameters or {}),
    }
    session = wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        name=slug,
        group=settings.wandb_group,
        job_type=run.mode,
        tags=[run.mode, f"run-{run.key}", f"seed-{seed}"] + ([f"r{run.rank}"] if run.rank else []),
        id=_run_id(run_dir),
        resume="allow",
        config=config,
        dir=str(run_dir),
    )
    wandb.define_metric("by_epoch/epoch")
    wandb.define_metric("by_epoch/*", step_metric="by_epoch/epoch")
    return session


def resume_wandb(run_dir: Path, settings: Settings):
    """Mở lại run đã có để ghi điểm test."""
    if not settings.use_wandb or not (run_dir / "wandb_id.txt").exists():
        return None
    import wandb

    return wandb.init(project=settings.wandb_project, entity=settings.wandb_entity,
                      id=_run_id(run_dir), resume="allow", dir=str(run_dir))


class HistoryCallback(TrainerCallback):
    """Ghi số liệu theo epoch ra history.csv và lên W&B."""

    def __init__(self, path: Path, use_wandb: bool):
        self.path = Path(path)
        self.use_wandb = use_wandb
        self.epoch_start = None
        self.epoch_seconds = None
        self.losses = []
        self.active = True

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.perf_counter()
        self.losses = []

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start is not None:
            self.epoch_seconds = time.perf_counter() - self.epoch_start

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.active or not metrics:
            return
        peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        row = {
            "epoch": round(state.epoch or 0, 2),
            "step": state.global_step,
            "train_loss": sum(self.losses) / len(self.losses) if self.losses else "",
            "dev_loss": metrics.get("eval_loss", ""),
            "dev_accuracy": metrics.get("eval_accuracy", ""),
            "dev_macro_f1": metrics.get("eval_macro_f1", ""),
            "dev_weighted_f1": metrics.get("eval_weighted_f1", ""),
            "epoch_seconds": self.epoch_seconds or "",
            "gpu_peak_mb": round(peak, 1),
        }
        for label in LABELS:
            row[f"dev_f1_{label}"] = metrics.get(f"eval_f1_{label}", "")
        new_file = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        if self.use_wandb:
            import wandb

            if wandb.run is not None:
                payload = {"by_epoch/epoch": row["epoch"]}
                for key, value in row.items():
                    if key not in ("epoch", "step") and value != "":
                        payload[f"by_epoch/{key}"] = value
                wandb.log(payload)
