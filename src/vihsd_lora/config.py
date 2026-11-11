"""Experiment configuration: training runs and shared settings."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "uitnlp/visobert"
FALLBACK_MODEL = "xlm-roberta-large"
LABELS = ("CLEAN", "OFFENSIVE", "HATE")
NUM_LABELS = len(LABELS)
LABEL2ID = {name: index for index, name in enumerate(LABELS)}
ID2LABEL = {index: name for index, name in enumerate(LABELS)}


@dataclass(frozen=True)
class Run:
    """One training run in the plan."""

    key: str
    mode: str
    learning_rate: float
    epochs: int
    rank: Optional[int] = None
    lora_alpha: Optional[int] = None

    def describe(self) -> str:
        if self.mode == "lora":
            return f"LoRA r={self.rank}, {self.epochs} epochs"
        return f"Full fine-tune, {self.epochs} epochs"


RUNS = {
    "A": Run("A", "lora", 5e-4, 10, rank=8, lora_alpha=8),
    "B": Run("B", "full", 2e-5, 10),
    "C": Run("C", "lora", 5e-4, 10, rank=2, lora_alpha=8),
    "D": Run("D", "lora", 5e-4, 10, rank=16, lora_alpha=8),
    "E": Run("E", "lora", 5e-4, 30, rank=8, lora_alpha=8),
}


def get_run(key: str) -> Run:
    normalized = key.strip().upper()
    if normalized not in RUNS:
        raise ValueError(f"Unknown run '{key}'. Valid runs: {', '.join(RUNS)}")
    return RUNS[normalized]


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    return int(value) if value else None


@dataclass
class Settings:
    """Settings shared by all runs."""

    model_name: str = DEFAULT_MODEL
    dataset_ids: tuple = ("uitnlp/vihsd", "phucdev/ViHSD")
    data_dir: Optional[str] = None
    output_root: str = "outputs"
    batch_size: int = 40
    eval_batch_size: int = 80
    max_length: int = 128
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    lora_dropout: float = 0.1
    target_modules: tuple = ("query", "value")
    logging_steps: int = 50
    num_workers: int = 2
    check_counts: bool = True
    epochs_override: Optional[int] = None
    save_every_epochs: Optional[int] = None
    use_wandb: bool = True
    wandb_project: str = "vihsd-lora-seminar"
    wandb_entity: Optional[str] = None
    wandb_group: str = field(default="vihsd")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Read environment variables, then apply values passed in directly."""
        values = {}
        env_strings = {
            "model_name": "VIHSD_MODEL_NAME",
            "data_dir": "VIHSD_DATA_DIR",
            "output_root": "VIHSD_OUTPUT_ROOT",
            "wandb_project": "WANDB_PROJECT",
            "wandb_entity": "WANDB_ENTITY",
        }
        for name, env_name in env_strings.items():
            if os.environ.get(env_name):
                values[name] = os.environ[env_name]
        env_ints = {
            "batch_size": "VIHSD_BATCH_SIZE",
            "eval_batch_size": "VIHSD_EVAL_BATCH_SIZE",
            "max_length": "VIHSD_MAX_LENGTH",
            "logging_steps": "VIHSD_LOGGING_STEPS",
            "num_workers": "VIHSD_NUM_WORKERS",
            "epochs_override": "VIHSD_EPOCHS",
            "save_every_epochs": "VIHSD_SAVE_EVERY_EPOCHS",
        }
        for name, env_name in env_ints.items():
            value = _env_int(env_name)
            if value is not None:
                values[name] = value
        if os.environ.get("VIHSD_SKIP_COUNT_CHECK") == "1":
            values["check_counts"] = False
        if os.environ.get("WANDB_MODE") == "disabled":
            values["use_wandb"] = False
        values.update(overrides)
        return cls(**values)

    def epochs_for(self, run: Run) -> int:
        return self.epochs_override or run.epochs

    def save_interval(self, run: Run) -> int:
        """Number of epochs between two resumable checkpoint saves.

        A LoRA checkpoint is only a few MB, so it is saved every epoch. A full fine-tune checkpoint is
        about 1.2GB (model and optimizer state), so by default it is saved every 2 epochs to save Drive space.
        """
        if self.save_every_epochs:
            return max(1, self.save_every_epochs)
        return 2 if run.mode == "full" else 1

    def slug(self, run: Run, seed: int) -> str:
        """Directory name and W&B run name."""
        epochs = self.epochs_for(run)
        method = f"lora_r{run.rank}" if run.mode == "lora" else "full"
        slug = f"{run.key}_{method}_e{epochs}_s{seed}"
        if self.model_name != DEFAULT_MODEL:
            slug += "_" + Path(self.model_name).name.replace("/", "-")
        return slug

    def run_dir(self, run: Run, seed: int) -> Path:
        return Path(self.output_root) / "runs" / self.slug(run, seed)

    def as_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}
