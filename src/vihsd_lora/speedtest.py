"""Run a few hundred steps to measure speed and VRAM, and estimate training time."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainerCallback, set_seed

from .config import RUNS, Settings, get_run
from .modeling import build_model, count_parameters
from .train import _prepare, training_arguments


class _StepTimer(TrainerCallback):
    """Skip the first steps, which are slower because of warm-up."""

    def __init__(self, skip: int):
        self.skip = skip
        self.start = None
        self.steps = 0
        self.seconds = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.skip:
            self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start is not None:
            self.steps = state.global_step - self.skip
            self.seconds = time.perf_counter() - self.start


def measure(run_key: str = "A", steps: int = 200, settings: Optional[Settings] = None) -> dict:
    settings = settings or Settings.from_env()
    run = get_run(run_key)
    set_seed(42)
    tokenizer, _, tokenized = _prepare(settings)
    model = build_model(run, settings)
    parameters = count_parameters(model)
    arguments = training_arguments(run, 42, settings, Path(settings.output_root) / "speedtest",
                                   f"speedtest_{run_key}", len(tokenized["train"]),
                                   max_steps=steps, evaluate=False)
    skip = min(20, steps // 4)
    timer = _StepTimer(skip)
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized["train"],
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                      callbacks=[timer])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    seconds_per_step = timer.seconds / max(timer.steps, 1)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"run": run_key, "mode": run.mode, "seconds_per_step": seconds_per_step,
            "gpu_peak_mb": round(peak, 1), **parameters}


def estimate_hours(speed_by_mode: dict, settings: Optional[Settings] = None, train_size: int = 24048) -> pd.DataFrame:
    """Estimate the training time of each run from the measured speed. Dev evaluation is not included."""
    settings = settings or Settings.from_env()
    rows = []
    for run in RUNS.values():
        steps = math.ceil(train_size / settings.batch_size) * settings.epochs_for(run)
        seconds = speed_by_mode[run.mode] * steps
        rows.append({"run": run.key, "description": run.describe(), "steps": steps,
                     "hours": round(seconds / 3600, 2)})
    return pd.DataFrame(rows)
