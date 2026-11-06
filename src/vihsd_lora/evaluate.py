"""So các lượt trên dev và chấm test.

Dòng lệnh: python -m vihsd_lora.evaluate --slug A_lora_r8_e10_s42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data as data_module
from .config import LABELS, Settings
from .inference import predict_logits
from .metrics import bootstrap_interval, confusion, score_predictions
from .modeling import load_trained
from .tracking import resume_wandb


def runs_root(settings: Settings) -> Path:
    return Path(settings.output_root) / "runs"


def compare_on_dev(settings: Optional[Settings] = None) -> pd.DataFrame:
    """Bảng các lượt đã train, xếp theo macro F1 trên dev."""
    settings = settings or Settings.from_env()
    rows = []
    for path in sorted(runs_root(settings).glob("*/run_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"Chưa có lượt nào train xong trong {runs_root(settings)}")
    table = pd.DataFrame(rows)
    columns = ["slug", "mode", "rank", "epochs", "seed", "trainable_params", "dev_accuracy",
               "dev_macro_f1", "best_epoch", "mean_epoch_seconds", "gpu_peak_mb", "saved_mb"]
    return table[columns].sort_values("dev_macro_f1", ascending=False).reset_index(drop=True)


def select_for_test(settings: Optional[Settings] = None) -> dict:
    """Chọn LoRA tốt nhất và full fine-tune chỉ dựa trên dev."""
    table = compare_on_dev(settings)
    lora = table[table["mode"] == "lora"]
    full = table[table["mode"] == "full"]
    choice = {}
    if len(lora):
        best = lora.sort_values(["dev_macro_f1", "trainable_params"], ascending=[False, True]).iloc[0]
        choice["lora"] = best["slug"]
    if len(full):
        choice["full"] = full.iloc[0]["slug"]
    return choice


def evaluate_on_test(slug: str, settings: Optional[Settings] = None, force: bool = False) -> dict:
    """Chấm test một lần. Lần sau trả lại kết quả cũ, trừ khi force=True."""
    settings = settings or Settings.from_env()
    run_dir = runs_root(settings) / slug
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists() and not force:
        print(f"{slug} đã chấm test, trả lại kết quả đã lưu.")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    model, tokenizer, summary = load_trained(run_dir)
    dataset = data_module.load_vihsd(settings)["test"]
    texts = list(dataset["text"])
    y_true = np.array(list(dataset["labels"]))

    started = time.perf_counter()
    logits = predict_logits(model, tokenizer, texts, batch_size=settings.eval_batch_size,
                            max_length=summary["max_length"])
    seconds = time.perf_counter() - started

    y_pred = logits.argmax(axis=-1)
    result = score_predictions(y_true, y_pred)
    result.update(bootstrap_interval(y_true, y_pred))
    result["confusion_matrix"] = confusion(y_true, y_pred)
    result["inference_seconds"] = seconds
    result["samples_per_second"] = len(texts) / seconds
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Không lưu nội dung bình luận để tránh phát tán dữ liệu xúc phạm theo kết quả.
    pd.DataFrame({"index": np.arange(len(y_true)), "label": y_true, "pred": y_pred}).to_csv(
        run_dir / "predictions_test.csv", index=False)

    session = resume_wandb(run_dir, settings)
    if session is not None:
        import wandb

        session.summary.update({f"test/{key}": value for key, value in result.items()
                                if isinstance(value, (int, float))})
        wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=list(LABELS))})
        session.finish()

    low, high = result["macro_f1_ci95"]
    print(f"{slug}: test Acc {result['accuracy']:.2f}, macro F1 {result['macro_f1']:.2f} "
          f"(95% CI {low:.2f}–{high:.2f}), {result['samples_per_second']:.0f} mẫu/giây")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chấm test một lượt đã train")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    evaluate_on_test(arguments.slug, force=arguments.force)


if __name__ == "__main__":
    main()
