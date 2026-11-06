"""Kiểm tra dữ liệu, LoRA, train, resume, đánh giá và inference trên CPU."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict

from vihsd_lora import data as data_module
from vihsd_lora import train as train_module
from vihsd_lora.config import LABELS, RUNS, Settings, get_run
from vihsd_lora.evaluate import compare_on_dev, evaluate_on_test, select_for_test
from vihsd_lora.inference import Predictor
from vihsd_lora.metrics import bootstrap_interval, majority_baseline, paired_bootstrap_difference, score_predictions
from vihsd_lora.modeling import assert_trainable_scope, build_model, count_parameters, load_tokenizer
from vihsd_lora.report import export_tables, majority_row, main_table


def _real_counts_dataset(missing=0):
    splits = {}
    for split, counts in data_module.EXPECTED_COUNTS.items():
        labels = [0] * (counts["CLEAN"] - missing) + [1] * counts["OFFENSIVE"] + [2] * counts["HATE"]
        splits[split] = Dataset.from_dict({"labels": labels})
    return DatasetDict(splits)


def test_expected_counts_match_paper_totals():
    totals = {split: sum(counts.values()) for split, counts in data_module.EXPECTED_COUNTS.items()}
    assert totals == {"train": 24048, "validation": 2672, "test": 6680}


def test_label_mapping_from_counts():
    values = [7] * 19886 + [3] * 1606 + [5] * 2556
    assert data_module.infer_label_mapping(values) == {7: "CLEAN", 5: "HATE", 3: "OFFENSIVE"}


def test_label_mapping_from_names():
    mapping = data_module.infer_label_mapping(["CLEAN", "hate", "Offensive"])
    assert sorted(mapping.values()) == ["CLEAN", "HATE", "OFFENSIVE"]


def test_count_check_accepts_small_gap_and_rejects_wrong_split():
    data_module.check_counts(_real_counts_dataset(missing=2))
    with pytest.raises(ValueError):
        data_module.check_counts(_real_counts_dataset(missing=50))


def test_normalize_drops_empty_text_and_renames_dev():
    rows = {"free_text": ["a", "", None, "b", "c", "d", "e"], "label_id": [0, 0, 0, 0, 1, 2, 2]}
    raw = DatasetDict({name: Dataset.from_dict(rows) for name in ("train", "dev", "test")})
    dataset = data_module.normalize(raw)
    assert set(dataset) == {"train", "validation", "test"}
    assert list(dataset["train"]["text"]) == ["a", "b", "c", "d", "e"]
    assert list(dataset["train"]["labels"]) == [0, 0, 1, 2, 2]


def test_metrics_match_hand_calculation():
    scores = score_predictions([0, 0, 1, 2], [0, 1, 1, 2])
    assert scores["accuracy"] == pytest.approx(75.0)
    assert scores["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3 * 100)


def test_majority_row_matches_plan():
    row = majority_row()
    assert (row["Acc"], row["Macro F1"]) == (83.05, 30.25)
    y_true = np.repeat([0, 1, 2], [5548, 444, 688])
    assert majority_baseline(y_true)["weighted_f1"] == pytest.approx(75.37, abs=0.01)


def test_bootstrap_outputs_ordered_interval():
    y_true = np.array([0, 1, 2] * 40)
    interval = bootstrap_interval(y_true, y_true, n_resamples=20)
    assert interval["macro_f1_ci95"] == [100.0, 100.0]
    same = paired_bootstrap_difference(y_true, y_true, y_true, n_resamples=20)
    assert same["difference"] == 0 and not same["significant"]


def test_real_visobert_parameter_counts():
    """Kiến trúc thật: khớp số trong kế hoạch."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    expected = {2: 666627, 8: 887811, 16: 1182723}
    config = XLMRobertaConfig(vocab_size=15004, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                              intermediate_size=3072, max_position_embeddings=514, type_vocab_size=2, num_labels=3)
    for rank, trainable in expected.items():
        model = XLMRobertaForSequenceClassification(config)
        full = count_parameters(model)
        assert full["trainable_params"] == 97568259
        lora = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=8,
                                                target_modules=["query", "value"], bias="none"))
        counts = count_parameters(lora)
        assert counts["trainable_params"] == trainable
        assert counts["model_params"] == 97568259
        assert counts["head_params"] == 592899
        assert_trainable_scope(lora)


def test_tokenizer_loads_from_sentencepiece_only_folder(tiny_model):
    assert not (tiny_model / "tokenizer.json").exists()
    tokenizer = load_tokenizer(str(tiny_model))
    encoded = tokenizer("hom nay troi dep", truncation=True, max_length=16)
    assert len(encoded["input_ids"]) > 2


def test_lora_trains_only_adapter_and_head(settings):
    from safetensors.torch import load_file

    run = get_run("A")
    assert_trainable_scope(build_model(run, settings))
    summary = train_module.train_run("A", settings=settings)
    assert summary["mode"] == "lora" and summary["epochs"] == 2

    run_dir = settings.run_dir(run, 42)
    assert not (run_dir / "checkpoints").exists()
    history = pd.read_csv(run_dir / "history.csv")
    assert list(history["epoch"]) == [1.0, 2.0]
    assert history["epoch_seconds"].notna().all()

    saved = load_file(run_dir / "best" / "adapter_model.safetensors")
    assert all("lora_" in key or "classifier" in key for key in saved)
    learned = [value for key, value in saved.items() if "lora_B" in key]
    assert learned and any(value.abs().sum() > 0 for value in learned)


def test_merge_gives_same_predictions(settings):
    train_module.train_run("C", settings=settings)
    predictor = Predictor(settings.run_dir(get_run("C"), 42), device="cpu")
    texts = ["hom nay troi dep qua", "ghet bon do cut di"]
    before = predictor.predict(texts)
    predictor.merge_lora()
    after = predictor.predict(texts)
    for first, second in zip(before, after):
        assert first["label"] == second["label"]
        for label in LABELS:
            assert first[f"p_{label}"] == pytest.approx(second[f"p_{label}"], abs=1e-3)


class _Interrupt(train_module.HistoryCallback):
    def on_save(self, args, state, control, **kwargs):
        raise KeyboardInterrupt("Giả lập Colab ngắt phiên")


def test_resume_after_interruption(settings, monkeypatch):
    monkeypatch.setattr(train_module, "HistoryCallback", _Interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_module.train_run("D", settings=settings)
    run_dir = settings.run_dir(get_run("D"), 42)
    assert not (run_dir / "run_summary.json").exists()
    assert list((run_dir / "checkpoints").glob("checkpoint-*"))

    monkeypatch.undo()
    summary = train_module.train_run("D", settings=settings)
    history = pd.read_csv(run_dir / "history.csv").drop_duplicates("epoch", keep="last")
    assert summary["epochs"] == 2
    assert list(history["epoch"]) == [1.0, 2.0]


def test_full_run_saves_whole_model_and_is_larger(settings):
    lora = train_module.train_run("A", settings=settings)
    full = train_module.train_run("B", settings=settings)
    assert full["trainable_pct"] == 100
    assert full["saved_mb"] > lora["saved_mb"]


def test_second_train_call_is_skipped(settings, capsys):
    train_module.train_run("A", settings=settings)
    train_module.train_run("A", settings=settings)
    assert "đã train xong" in capsys.readouterr().out


def test_test_set_is_scored_once_and_keeps_no_text(settings):
    train_module.train_run("A", settings=settings)
    train_module.train_run("B", settings=settings)
    assert len(compare_on_dev(settings)) == 2
    choice = select_for_test(settings)
    assert set(choice) == {"lora", "full"}

    first = evaluate_on_test(choice["lora"], settings)
    path = Path(settings.output_root) / "runs" / choice["lora"] / "test_metrics.json"
    path.write_text(json.dumps({**first, "marker": True}), encoding="utf-8")
    assert evaluate_on_test(choice["lora"], settings)["marker"] is True

    columns = pd.read_csv(path.with_name("predictions_test.csv")).columns
    assert list(columns) == ["index", "label", "pred"]


def test_main_table_has_published_rows_and_groups_new_runs(settings):
    train_module.train_run("A", settings=settings)
    evaluate_on_test(settings.slug(get_run("A"), 42), settings)
    table = main_table(settings)
    assert len(table) == 13 + 1 + 1
    assert table.iloc[0]["Hệ thống"] == "ViSoBERT"
    paths = export_tables(settings)
    assert paths["markdown"].read_text(encoding="utf-8").count("\n") == len(table) + 2


def test_wandb_offline_logging(settings, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    settings.use_wandb = True
    summary = train_module.train_run("A", settings=settings)
    run_dir = settings.run_dir(get_run("A"), 42)
    assert (run_dir / "wandb_id.txt").exists()
    assert list((run_dir / "wandb").glob("offline-run-*"))
    evaluate_on_test(summary["slug"], settings)


def test_settings_slug_marks_other_models():
    run = RUNS["A"]
    assert Settings().slug(run, 42) == "A_lora_r8_e10_s42"
    assert Settings(model_name="xlm-roberta-large").slug(run, 42).endswith("xlm-roberta-large")
