"""Predict new text and measure latency."""

from __future__ import annotations

import time

import numpy as np
import torch

from .config import LABELS
from .modeling import load_trained


@torch.no_grad()
def predict_logits(model, tokenizer, texts, batch_size: int = 80, max_length: int = 128) -> np.ndarray:
    device = next(model.parameters()).device
    autocast = torch.cuda.is_available() and device.type == "cuda"
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[start:start + batch_size]), truncation=True,
                          max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=autocast):
            logits = model(**batch).logits
        outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


class Predictor:
    """Predictor that uses the best model of a run."""

    def __init__(self, run_dir, device=None, max_length: int = 128):
        self.model, self.tokenizer, self.summary = load_trained(run_dir, device)
        self.max_length = self.summary.get("max_length", max_length)
        self.merged = False

    def predict(self, texts) -> list:
        logits = predict_logits(self.model, self.tokenizer, texts, max_length=self.max_length)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        rows = []
        for text, scores in zip(texts, probabilities):
            rows.append({
                "text": text,
                "label": LABELS[int(scores.argmax())],
                **{f"p_{label}": round(float(scores[index]), 4) for index, label in enumerate(LABELS)},
            })
        return rows

    def merge_lora(self) -> None:
        """Merge BA into the base weights. Afterwards the model has no separate LoRA layers."""
        if self.summary["mode"] != "lora":
            raise ValueError("This run does not use LoRA.")
        if not self.merged:
            self.model = self.model.merge_and_unload()
            self.model.eval()
            self.merged = True

    def latency_ms(self, texts, batch_size: int = 1, repeats: int = 3) -> float:
        """Average time per model call, in milliseconds (best of `repeats`)."""
        best = float("inf")
        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            predict_logits(self.model, self.tokenizer, texts, batch_size=batch_size, max_length=self.max_length)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            calls = int(np.ceil(len(texts) / batch_size))
            best = min(best, elapsed / calls * 1000)
        return round(best, 2)
