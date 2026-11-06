"""Nạp model, gắn LoRA, đếm tham số và nạp lại model đã train."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ID2LABEL, LABEL2ID, NUM_LABELS, Run, Settings


def _token():
    return os.environ.get("HF_TOKEN") or None


def load_tokenizer(model_name: str):
    """ViSoBERT chỉ có file sentencepiece, nên cần thư viện sentencepiece."""
    try:
        return AutoTokenizer.from_pretrained(model_name, token=_token())
    except Exception as first_error:
        print(f"Tokenizer mặc định lỗi ({type(first_error).__name__}), thử bản slow.")
        return AutoTokenizer.from_pretrained(model_name, token=_token(), use_fast=False)


def load_base_model(model_name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        token=_token(),
    )


def attach_lora(model, run: Run, settings: Settings):
    """Đóng băng trọng số gốc, gắn LoRA vào query và value của self-attention."""
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=run.rank,
        lora_alpha=run.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(settings.target_modules),
        bias="none",
    )
    return get_peft_model(model, config)


def build_model(run: Run, settings: Settings):
    model = load_base_model(settings.model_name)
    if run.mode == "lora":
        model = attach_lora(model, run, settings)
    return model


def count_parameters(model) -> dict:
    """Đếm tham số. model_params không tính phần LoRA và bản sao đầu phân loại của PEFT."""
    trainable = lora = model_params = 0
    for name, parameter in model.named_parameters():
        size = parameter.numel()
        is_lora = "lora_" in name
        if parameter.requires_grad:
            trainable += size
            if is_lora:
                lora += size
        if not is_lora and ".modules_to_save." not in name:
            model_params += size
    return {
        "trainable_params": trainable,
        "lora_params": lora,
        "head_params": trainable - lora if lora else 0,
        "model_params": model_params,
        "trainable_pct": round(trainable / model_params * 100, 3),
    }


def assert_trainable_scope(model) -> None:
    """Dừng lại nếu có tham số ngoài LoRA và đầu phân loại được train."""
    unexpected = [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "lora_" not in name and ".modules_to_save." not in name]
    if unexpected:
        raise RuntimeError(f"Có tham số ngoài LoRA đang được train: {unexpected[:5]}")


def weights_size_mb(directory) -> float:
    total = 0
    for path in Path(directory).iterdir():
        if path.suffix in (".safetensors", ".bin") and path.name != "training_args.bin":
            total += path.stat().st_size
    return round(total / 1024**2, 2)


def load_trained(run_dir, device=None):
    """Nạp model tốt nhất của một lượt. Trả về (model, tokenizer, summary)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    best_dir = run_dir / "best"
    if summary["mode"] == "lora":
        base = load_base_model(summary["model_name"])
        model = PeftModel.from_pretrained(base, str(best_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tokenizer = load_tokenizer(str(best_dir))
    return model, tokenizer, summary
