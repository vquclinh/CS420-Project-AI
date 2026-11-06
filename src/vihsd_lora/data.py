"""Tải, chuẩn hóa và kiểm tra dữ liệu ViHSD."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from .config import LABEL2ID, LABELS, Settings

# Số mẫu theo paper gốc ViHSD và các paper dùng lại split này.
EXPECTED_COUNTS = {
    "train": {"CLEAN": 19886, "OFFENSIVE": 1606, "HATE": 2556},
    "validation": {"CLEAN": 2190, "OFFENSIVE": 212, "HATE": 270},
    "test": {"CLEAN": 5548, "OFFENSIVE": 444, "HATE": 688},
}
# Một vài dòng rỗng có thể bị loại khi đọc CSV, ví dụ chuỗi bị hiểu là giá trị thiếu.
MISSING_ROWS_TOLERANCE = 5

TEXT_COLUMNS = ("free_text", "text", "comment", "sentence", "content")
LABEL_COLUMNS = ("label_id", "label", "labels", "class")
SPLIT_ALIASES = {"train": "train", "validation": "validation", "valid": "validation",
                 "dev": "validation", "test": "test"}


def _pick_column(columns, candidates, kind: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"Không tìm thấy cột {kind}. Các cột hiện có: {sorted(columns)}")


def _load_local(data_dir: str) -> DatasetDict:
    root = Path(data_dir)
    files = {}
    for path in sorted(root.iterdir()):
        split = SPLIT_ALIASES.get(path.stem.split("-")[0].lower())
        if split and path.suffix in (".csv", ".parquet"):
            files[split] = str(path)
    missing = {"train", "validation", "test"} - set(files)
    if missing:
        raise FileNotFoundError(f"Thiếu file cho split {sorted(missing)} trong {root}")
    builder = "csv" if next(iter(files.values())).endswith(".csv") else "parquet"
    return load_dataset(builder, data_files=files)


def _load_hub(settings: Settings) -> DatasetDict:
    token = os.environ.get("HF_TOKEN") or None
    errors = []
    for dataset_id in settings.dataset_ids:
        try:
            dataset = load_dataset(dataset_id, token=token)
            print(f"Đã tải dữ liệu từ {dataset_id}")
            return dataset
        except Exception as error:  # Thử nguồn tiếp theo trước khi báo lỗi.
            errors.append(f"{dataset_id}: {type(error).__name__}: {error}")
    raise RuntimeError(
        "Không tải được ViHSD. Hãy mở https://huggingface.co/datasets/uitnlp/vihsd, "
        "đăng nhập, chấp nhận điều kiện, rồi kiểm tra HF_TOKEN.\n" + "\n".join(errors)
    )


def infer_label_mapping(values) -> dict:
    """Ánh xạ giá trị nhãn trong file sang tên nhãn.

    Nhãn dạng chữ được đối chiếu theo tên. Nhãn dạng số được xác định theo số mẫu
    train, vì ba nhãn có số lượng khác xa nhau: CLEAN nhiều nhất, HATE ở giữa,
    OFFENSIVE ít nhất.
    """
    counts = Counter(values)
    if len(counts) != len(LABELS):
        raise ValueError(f"Cần đúng 3 nhãn, nhưng tìm thấy: {dict(counts)}")
    if all(isinstance(value, str) for value in counts):
        mapping = {value: value.strip().upper() for value in counts}
        unknown = set(mapping.values()) - set(LABELS)
        if unknown:
            raise ValueError(f"Nhãn không hợp lệ: {sorted(unknown)}")
        return mapping
    ranked = [value for value, _ in counts.most_common()]
    return {ranked[0]: "CLEAN", ranked[1]: "HATE", ranked[2]: "OFFENSIVE"}


def normalize(dataset: DatasetDict) -> DatasetDict:
    """Đưa dữ liệu về hai cột text và labels, với id theo LABELS."""
    renamed = {}
    for name, split in dataset.items():
        target = SPLIT_ALIASES.get(name)
        if target is None:
            continue
        renamed[target] = split
    missing = {"train", "validation", "test"} - set(renamed)
    if missing:
        raise ValueError(f"Thiếu split {sorted(missing)}. Bản dữ liệu này không phù hợp.")

    columns = renamed["train"].column_names
    text_column = _pick_column(columns, TEXT_COLUMNS, "văn bản")
    label_column = _pick_column(columns, LABEL_COLUMNS, "nhãn")
    mapping = infer_label_mapping(list(renamed["train"][label_column]))

    output = {}
    for name, split in renamed.items():
        before = len(split)
        split = split.filter(lambda row: row[text_column] is not None and str(row[text_column]).strip() != "")
        removed = before - len(split)
        if removed:
            print(f"{name}: loại {removed} dòng văn bản rỗng")
        split = split.map(
            lambda row: {"text": str(row[text_column]), "labels": LABEL2ID[mapping[row[label_column]]]},
            remove_columns=split.column_names,
        )
        output[name] = split
    return DatasetDict(output)


def label_counts(dataset: DatasetDict) -> dict:
    result = {}
    for name, split in dataset.items():
        counts = Counter(list(split["labels"]))
        result[name] = {label: counts.get(index, 0) for index, label in enumerate(LABELS)}
    return result


def check_counts(dataset: DatasetDict, tolerance: int = MISSING_ROWS_TOLERANCE) -> dict:
    """So số mẫu từng nhãn với bản chuẩn. Lệch quá ngưỡng thì dừng lại."""
    actual = label_counts(dataset)
    problems = []
    for split, expected in EXPECTED_COUNTS.items():
        for label, count in expected.items():
            difference = count - actual[split][label]
            if difference < 0 or difference > tolerance:
                problems.append(f"{split}/{label}: mong đợi {count}, thực tế {actual[split][label]}")
    if problems:
        raise ValueError(
            "Dữ liệu không khớp bản ViHSD chuẩn, nên kết quả sẽ không so được với bảng số liệu:\n"
            + "\n".join(problems)
        )
    return actual


def load_vihsd(settings: Optional[Settings] = None) -> DatasetDict:
    settings = settings or Settings.from_env()
    raw = _load_local(settings.data_dir) if settings.data_dir else _load_hub(settings)
    dataset = normalize(raw)
    if settings.check_counts:
        check_counts(dataset)
        print("Số mẫu từng nhãn khớp bản ViHSD chuẩn.")
    else:
        print("Đã bỏ qua kiểm tra số mẫu. Chỉ dùng khi chạy thử.")
    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(encode, batched=True, remove_columns=["text"], desc="Tokenize")


def length_report(split: Dataset, tokenizer, max_length: int) -> dict:
    """Thống kê độ dài token để biết max_length cắt mất bao nhiêu bình luận."""
    import numpy as np

    lengths = np.array([len(ids) for ids in tokenizer(list(split["text"]), truncation=False)["input_ids"]])
    return {
        "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "truncated_pct": round(float((lengths > max_length).mean() * 100), 2),
    }
