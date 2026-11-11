"""Load, normalize and validate the ViHSD data."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from .config import LABEL2ID, LABELS, Settings

# Sample counts from the original ViHSD paper and the papers that reuse this split.
EXPECTED_COUNTS = {
    "train": {"CLEAN": 19886, "OFFENSIVE": 1606, "HATE": 2556},
    "validation": {"CLEAN": 2190, "OFFENSIVE": 212, "HATE": 270},
    "test": {"CLEAN": 5548, "OFFENSIVE": 444, "HATE": 688},
}
# A few empty rows may be dropped when reading the CSV, e.g. strings parsed as missing values.
MISSING_ROWS_TOLERANCE = 5

TEXT_COLUMNS = ("free_text", "text", "comment", "sentence", "content")
LABEL_COLUMNS = ("label_id", "label", "labels", "class")
SPLIT_ALIASES = {"train": "train", "validation": "validation", "valid": "validation",
                 "dev": "validation", "test": "test"}


def _pick_column(columns, candidates, kind: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"No {kind} column found. Available columns: {sorted(columns)}")


def _load_local(data_dir: str) -> DatasetDict:
    root = Path(data_dir)
    files = {}
    for path in sorted(root.iterdir()):
        split = SPLIT_ALIASES.get(path.stem.split("-")[0].lower())
        if split and path.suffix in (".csv", ".parquet"):
            files[split] = str(path)
    missing = {"train", "validation", "test"} - set(files)
    if missing:
        raise FileNotFoundError(f"Missing files for split(s) {sorted(missing)} in {root}")
    builder = "csv" if next(iter(files.values())).endswith(".csv") else "parquet"
    return load_dataset(builder, data_files=files)


def _load_hub(settings: Settings) -> DatasetDict:
    token = os.environ.get("HF_TOKEN") or None
    errors = []
    for dataset_id in settings.dataset_ids:
        try:
            dataset = load_dataset(dataset_id, token=token)
            print(f"Loaded data from {dataset_id}")
            return dataset
        except Exception as error:  # Try the next source before raising.
            errors.append(f"{dataset_id}: {type(error).__name__}: {error}")
    raise RuntimeError(
        "Could not load ViHSD. Open https://huggingface.co/datasets/uitnlp/vihsd, "
        "log in, accept the conditions, then check HF_TOKEN.\n" + "\n".join(errors)
    )


def infer_label_mapping(values) -> dict:
    """Map the label values in the file to label names.

    String labels are matched by name. Numeric labels are identified by their train
    counts, since the three labels differ greatly in size: CLEAN is the largest, HATE is
    in the middle, and OFFENSIVE is the smallest.
    """
    counts = Counter(values)
    if len(counts) != len(LABELS):
        raise ValueError(f"Expected exactly 3 labels, but found: {dict(counts)}")
    if all(isinstance(value, str) for value in counts):
        mapping = {value: value.strip().upper() for value in counts}
        unknown = set(mapping.values()) - set(LABELS)
        if unknown:
            raise ValueError(f"Invalid labels: {sorted(unknown)}")
        return mapping
    ranked = [value for value, _ in counts.most_common()]
    return {ranked[0]: "CLEAN", ranked[1]: "HATE", ranked[2]: "OFFENSIVE"}


def normalize(dataset: DatasetDict) -> DatasetDict:
    """Reduce the data to two columns, text and labels, with ids following LABELS."""
    renamed = {}
    for name, split in dataset.items():
        target = SPLIT_ALIASES.get(name)
        if target is None:
            continue
        renamed[target] = split
    missing = {"train", "validation", "test"} - set(renamed)
    if missing:
        raise ValueError(f"Missing split(s) {sorted(missing)}. This version of the data is not suitable.")

    columns = renamed["train"].column_names
    text_column = _pick_column(columns, TEXT_COLUMNS, "text")
    label_column = _pick_column(columns, LABEL_COLUMNS, "label")
    mapping = infer_label_mapping(list(renamed["train"][label_column]))

    output = {}
    for name, split in renamed.items():
        before = len(split)
        split = split.filter(lambda row: row[text_column] is not None and str(row[text_column]).strip() != "")
        removed = before - len(split)
        if removed:
            print(f"{name}: dropped {removed} rows with empty text")
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
    """Compare per-label counts with the reference version. Stop if the gap exceeds the tolerance."""
    actual = label_counts(dataset)
    problems = []
    for split, expected in EXPECTED_COUNTS.items():
        for label, count in expected.items():
            difference = count - actual[split][label]
            if difference < 0 or difference > tolerance:
                problems.append(f"{split}/{label}: expected {count}, got {actual[split][label]}")
    if problems:
        raise ValueError(
            "The data does not match the reference ViHSD version, so results would not be comparable "
            "with the published tables:\n"
            + "\n".join(problems)
        )
    return actual


def load_vihsd(settings: Optional[Settings] = None) -> DatasetDict:
    settings = settings or Settings.from_env()
    raw = _load_local(settings.data_dir) if settings.data_dir else _load_hub(settings)
    dataset = normalize(raw)
    if settings.check_counts:
        check_counts(dataset)
        print("Per-label counts match the reference ViHSD version.")
    else:
        print("Skipped the sample count check. Only do this for test runs.")
    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def encode(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(encode, batched=True, remove_columns=["text"], desc="Tokenize")


def length_report(split: Dataset, tokenizer, max_length: int) -> dict:
    """Token length statistics, to see how many comments max_length truncates."""
    import numpy as np

    lengths = np.array([len(ids) for ids in tokenizer(list(split["text"]), truncation=False)["input_ids"]])
    return {
        "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "truncated_pct": round(float((lengths > max_length).mean() * 100), 2),
    }
