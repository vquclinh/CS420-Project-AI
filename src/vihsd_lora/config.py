"""Cấu hình thí nghiệm: các lượt train và thiết lập dùng chung."""

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
    """Một lượt train trong kế hoạch."""

    key: str
    mode: str
    learning_rate: float
    epochs: int
    rank: Optional[int] = None
    lora_alpha: Optional[int] = None

    def describe(self) -> str:
        if self.mode == "lora":
            return f"LoRA r={self.rank}, {self.epochs} epoch"
        return f"Full fine-tune, {self.epochs} epoch"


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
        raise ValueError(f"Không có lượt '{key}'. Các lượt hợp lệ: {', '.join(RUNS)}")
    return RUNS[normalized]


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    return int(value) if value else None


@dataclass
class Settings:
    """Thiết lập dùng chung cho mọi lượt."""

    model_name: str = DEFAULT_MODEL
    dataset_ids: tuple = ("uitnlp/vihsd", "sonlam1102/vihsd")
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
    use_wandb: bool = True
    wandb_project: str = "vihsd-lora-seminar"
    wandb_entity: Optional[str] = None
    wandb_group: str = field(default="vihsd")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Đọc biến môi trường, sau đó áp dụng giá trị truyền trực tiếp."""
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

    def slug(self, run: Run, seed: int) -> str:
        """Tên thư mục và tên run trên W&B."""
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
