"""Nhận biết Colab/Kaggle, nạp secret và chọn nơi lưu kết quả."""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Optional


def detect_platform() -> str:
    if os.environ.get("COLAB_RELEASE_TAG") or os.path.exists("/content"):
        try:
            import google.colab  # noqa: F401

            return "colab"
        except ImportError:
            pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"):
        return "kaggle"
    return "local"


def _read_secret(name: str, platform: str) -> Optional[str]:
    if os.environ.get(name):
        return os.environ[name]
    if platform == "colab":
        try:
            from google.colab import userdata

            return userdata.get(name)
        except Exception:
            return None
    if platform == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        except Exception:
            return None
    return None


def load_secrets(ask_if_missing: bool = True) -> dict:
    """Đưa HF_TOKEN và WANDB_API_KEY vào biến môi trường, không in giá trị."""
    platform = detect_platform()
    status = {}
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        value = _read_secret(name, platform)
        wandb_off = name == "WANDB_API_KEY" and os.environ.get("WANDB_MODE") in ("offline", "disabled")
        if not value and ask_if_missing and not wandb_off:
            value = getpass.getpass(f"Nhập {name} (bỏ trống nếu không có): ").strip()
        if value:
            os.environ[name] = value
        status[name] = bool(value)
    return status


def default_output_root() -> str:
    """Chọn thư mục kết quả bền vững theo nền tảng."""
    if os.environ.get("VIHSD_OUTPUT_ROOT"):
        return os.environ["VIHSD_OUTPUT_ROOT"]
    platform = detect_platform()
    if platform == "colab":
        drive_root = Path("/content/drive")
        if not (drive_root / "MyDrive").exists():
            from google.colab import drive

            drive.mount(str(drive_root))
        os.environ.setdefault("WANDB_DIR", "/content/wandb_local")
        return str(drive_root / "MyDrive" / "vihsd-lora" / "outputs")
    if platform == "kaggle":
        return "/kaggle/working/outputs"
    return "outputs"


def free_space_gb(path: str) -> float:
    """Dung lượng còn trống tại thư mục lưu kết quả, với Drive là dung lượng còn lại của tài khoản."""
    import shutil

    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return round(shutil.disk_usage(target).free / 1024**3, 1)


def describe_gpu() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "Không có GPU", "memory_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {"device": "cuda", "name": props.name, "memory_gb": round(props.total_memory / 1024**3, 1)}
