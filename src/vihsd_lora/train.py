"""Train một lượt, tự train tiếp từ checkpoint nếu bị ngắt.

Dòng lệnh: python -m vihsd_lora.train --run A --seed 42
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
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from . import data as data_module
from .config import LABELS, Run, Settings, get_run
from .metrics import compute_metrics, confusion
from .modeling import assert_trainable_scope, build_model, count_parameters, load_tokenizer, weights_size_mb
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
        load_best_model_at_end=evaluate,
        metric_for_best_model="macro_f1" if evaluate else None,
        greater_is_better=True if evaluate else None,
        logging_steps=settings.logging_steps,
        fp16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=settings.num_workers,
        label_names=["labels"],
        report_to=report_to or [],
    )


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
    """Train một lượt và trả về run_summary."""
    settings = settings or Settings.from_env()
    run = get_run(run_key)
    slug = settings.slug(run, seed)
    run_dir = settings.run_dir(run, seed)
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists() and not overwrite:
        print(f"{slug} đã train xong, bỏ qua. Dùng overwrite=True nếu muốn chạy lại.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if overwrite and run_dir.exists():
        if (run_dir / "test_metrics.json").exists():
            print("Cảnh báo: lượt này đã chấm test. Train lại sau khi xem điểm test làm điểm mất ý nghĩa.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    tokenizer, _, tokenized = _prepare(settings)
    model = build_model(run, settings)
    if run.mode == "lora":
        assert_trainable_scope(model)
    parameters = count_parameters(model)
    print(f"{slug}: train {parameters['trainable_params']:,} / {parameters['model_params']:,} tham số "
          f"({parameters['trainable_pct']}%)")

    session = start_wandb(run, seed, slug, run_dir, settings, parameters)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    arguments = training_arguments(run, seed, settings, checkpoint_dir, slug, len(tokenized["train"]),
                                   report_to=["wandb"] if session else [])
    history = HistoryCallback(run_dir / "history.csv", use_wandb=session is not None)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[history],
    )

    last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
    if last_checkpoint:
        print(f"Train tiếp từ {last_checkpoint}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    wall_seconds = time.perf_counter() - started

    history.active = False
    best_dir = run_dir / "best"
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    prediction = trainer.predict(tokenized["validation"], metric_key_prefix="dev")
    dev_predictions = np.argmax(prediction.predictions, axis=-1)
    dev_metrics = {key.replace("dev_", ""): value for key, value in prediction.metrics.items()
                   if isinstance(value, (int, float))}
    dev_metrics["confusion_matrix"] = confusion(prediction.label_ids, dev_predictions)
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
            y_true=prediction.label_ids.tolist(), preds=dev_predictions.tolist(), class_names=list(LABELS))})
        session.finish()

    # Checkpoint chỉ cần khi train dở, model tốt nhất đã nằm trong best/.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print(f"Xong {slug}: dev macro F1 {summary['dev_macro_f1']:.2f}, Acc {summary['dev_accuracy']:.2f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train một lượt thí nghiệm ViHSD")
    parser.add_argument("--run", required=True, help="A, B, C, D hoặc E")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    train_run(arguments.run, arguments.seed, overwrite=arguments.overwrite)


if __name__ == "__main__":
    main()
