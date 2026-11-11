"""Load the model, attach LoRA, count parameters and reload trained models."""

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
    """ViSoBERT only ships a sentencepiece model file, so the sentencepiece library is required."""
    try:
        return AutoTokenizer.from_pretrained(model_name, token=_token())
    except Exception as first_error:
        print(f"Default tokenizer failed ({type(first_error).__name__}), trying the slow one.")
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
    """Freeze the base weights and attach LoRA to the self-attention query and value."""
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
    """Count parameters. model_params excludes the LoRA weights and PEFT's copy of the classifier head."""
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
    """Stop if any parameter outside LoRA and the classifier head is trainable."""
    unexpected = [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "lora_" not in name and ".modules_to_save." not in name]
    if unexpected:
        raise RuntimeError(f"Parameters outside LoRA are trainable: {unexpected[:5]}")


def weights_size_mb(directory) -> float:
    total = 0
    for path in Path(directory).iterdir():
        if path.suffix in (".safetensors", ".bin") and path.name != "training_args.bin":
            total += path.stat().st_size
    return round(total / 1024**2, 2)


def load_from_dir(best_dir, mode: str, model_name: str, device=None):
    """Load a LoRA adapter onto the base model, or load a full fine-tuned model directly."""
    if mode == "lora":
        model = PeftModel.from_pretrained(load_base_model(model_name), str(best_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def load_trained(run_dir, device=None):
    """Load the best model of a run. Returns (model, tokenizer, summary)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    best_dir = run_dir / "best"
    model = load_from_dir(best_dir, summary["mode"], summary["model_name"], device)
    tokenizer = load_tokenizer(str(best_dir))
    return model, tokenizer, summary
