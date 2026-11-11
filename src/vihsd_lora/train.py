"""Train one run, resuming from the last checkpoint if it was interrupted.

Command line: python -m vihsd_lora.train --run A --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from . import data as data_module
from .config import LABELS, Run, Settings, get_run
from .inference import predict_logits
from .metrics import compute_metrics, confusion, score_predictions
from .modeling import (assert_trainable_scope, build_model, count_parameters, load_from_dir, load_tokenizer,
                       weights_size_mb)
from .tracking import HistoryCallback, start_wandb


def training_arguments(run: Run, seed: int, settings: Settings, checkpoint_dir: Path, slug: str,
                       train_size: int, max_steps: Optional[int] = None, evaluate: bool = True,
                       report_to=None) -> TrainingArguments:
    epochs = settings.epochs_for(run)
    steps_per_epoch = math.ceil(train_size / settings.batch_size)
    total_steps = max_steps or steps_per_epoch * epochs
    strategy = "epoch" if evaluate else "no"
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        run_name=slug,
        num_train_epochs=epochs,
        max_steps=max_steps or -1,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        learning_rate=run.learning_rate,
        weight_decay=settings.weight_decay,
        adam_epsilon=settings.adam_epsilon,
        lr_scheduler_type="linear",
        warmup_steps=int(total_steps * settings.warmup_ratio),
        eval_strategy=strategy,
        save_strategy=strategy,
        save_total_limit=1,
        logging_steps=settings.logging_steps,
        fp16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=settings.num_workers,
        label_names=["labels"],
        report_to=report_to or [],
    )


class BestModelCallback(TrainerCallback):
    """Save the model with the highest dev macro F1 to best/ right after each dev evaluation.

    The best model is kept apart from the checkpoints, so checkpoints can be saved less often
    without losing the best epoch. Its state is written to best_state.json so resumed training remembers it.
    """

    def __init__(self, run_dir: Path, tokenizer):
        self.best_dir = run_dir / "best"
        self.state_path = run_dir / "best_state.json"
        self.tokenizer = tokenizer
        self.best = None
        if self.state_path.exists():
            self.best = json.loads(self.state_path.read_text(encoding="utf-8"))["macro_f1"]

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        score = (metrics or {}).get("eval_macro_f1")
        if score is None or model is None or (self.best is not None and score <= self.best):
            return
        temporary = self.best_dir.with_name("best_tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        model.save_pretrained(str(temporary))
        self.tokenizer.save_pretrained(str(temporary))
        shutil.rmtree(self.best_dir, ignore_errors=True)
        temporary.rename(self.best_dir)
        self.best = score
        self.state_path.write_text(json.dumps({"macro_f1": score, "epoch": round(state.epoch or 0, 2)}),
                                   encoding="utf-8")


class SaveIntervalCallback(TrainerCallback):
    """Only save a checkpoint every `every` epochs and at the last epoch."""

    def __init__(self, every: int, total_epochs: int):
        self.every = every
        self.total_epochs = total_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch or 0))
        if epoch % self.every != 0 and epoch < self.total_epochs:
            control.should_save = False
        return control


def _prepare(settings: Settings):
    tokenizer = load_tokenizer(settings.model_name)
    dataset = data_module.load_vihsd(settings)
    tokenized = data_module.tokenize_dataset(dataset, tokenizer, settings.max_length)
    return tokenizer, dataset, tokenized


def _summarize_history(path: Path) -> dict:
    history = pd.read_csv(path).drop_duplicates("epoch", keep="last")
    best = history.loc[history["dev_macro_f1"].idxmax()]
    return {
        "best_epoch": float(best["epoch"]),
        "train_seconds": float(history["epoch_seconds"].sum()),
        "mean_epoch_seconds": float(history["epoch_seconds"].mean()),
        "gpu_peak_mb": float(history["gpu_peak_mb"].max()),
    }


def train_run(run_key: str, seed: int = 42, settings: Optional[Settings] = None, overwrite: bool = False) -> dict:
    """Train one run and return its run summary."""
    settings = settings or Settings.from_env()
    run = get_run(run_key)
    slug = settings.slug(run, seed)
    run_dir = settings.run_dir(run, seed)
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists() and not overwrite:
        print(f"{slug} is already trained, skipping. Use overwrite=True to train it again.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if overwrite and run_dir.exists():
        if (run_dir / "test_metrics.json").exists():
            print("Warning: this run has already been scored on test. Retraining after seeing test scores "
                  "makes those scores meaningless.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    tokenizer, dataset, tokenized = _prepare(settings)
    model = build_model(run, settings)
    if run.mode == "lora":
        assert_trainable_scope(model)
    parameters = count_parameters(model)
    print(f"{slug}: train {parameters['trainable_params']:,} / {parameters['model_params']:,} parameters "
          f"({parameters['trainable_pct']}%)")

    session = start_wandb(run, seed, slug, run_dir, settings, parameters)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    arguments = training_arguments(run, seed, settings, checkpoint_dir, slug, len(tokenized["train"]),
                                   report_to=["wandb"] if session else [])
    history = HistoryCallback(run_dir / "history.csv", use_wandb=session is not None)
    every = settings.save_interval(run)
    print(f"Saving a checkpoint every {every} epoch(s) to {checkpoint_dir}")
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[history, BestModelCallback(run_dir, tokenizer),
                   SaveIntervalCallback(every, settings.epochs_for(run))],
    )

    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    wall_seconds = time.perf_counter() - started

    history.active = False
    best_dir = run_dir / "best"
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_model = load_from_dir(best_dir, run.mode, settings.model_name)
    dev_labels = np.array(list(dataset["validation"]["labels"]))
    dev_logits = predict_logits(best_model, tokenizer, list(dataset["validation"]["text"]),
                                batch_size=settings.eval_batch_size, max_length=settings.max_length)
    dev_predictions = dev_logits.argmax(axis=-1)
    dev_metrics = score_predictions(dev_labels, dev_predictions)
    dev_metrics["confusion_matrix"] = confusion(dev_labels, dev_predictions)
    del best_model
    (run_dir / "dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2), encoding="utf-8")

    summary = {
        "slug": slug, "run_key": run.key, "mode": run.mode, "rank": run.rank, "lora_alpha": run.lora_alpha,
        "learning_rate": run.learning_rate, "epochs": settings.epochs_for(run), "seed": seed,
        "model_name": settings.model_name, "batch_size": settings.batch_size, "max_length": settings.max_length,
        **parameters,
        **_summarize_history(run_dir / "history.csv"),
        "wall_seconds": wall_seconds,
        "saved_mb": weights_size_mb(best_dir),
        "dev_accuracy": dev_metrics["accuracy"],
        "dev_macro_f1": dev_metrics["macro_f1"],
        "dev_weighted_f1": dev_metrics["weighted_f1"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if session is not None:
        import wandb

        session.summary.update({key: value for key, value in summary.items() if isinstance(value, (int, float))})
        wandb.log({"dev/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=dev_labels.tolist(), preds=dev_predictions.tolist(), class_names=list(LABELS))})
        session.finish()

    # Checkpoints are only needed for unfinished training. The best model is already in best/.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(f"Finished {slug}: dev macro F1 {summary['dev_macro_f1']:.2f}, Acc {summary['dev_accuracy']:.2f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ViHSD experiment run")
    parser.add_argument("--run", required=True, help="A, B, C, D or E")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    train_run(arguments.run, arguments.seed, overwrite=arguments.overwrite)


if __name__ == "__main__":
    main()
