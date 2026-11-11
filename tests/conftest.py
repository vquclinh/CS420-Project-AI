"""Build a tiny model with the same file layout as ViSoBERT, and fake data in the ViHSD format."""

import csv
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Fake Vietnamese words (without diacritics) used to build distinguishable sentences per label.
CLEAN_WORDS = ["hom", "nay", "troi", "dep", "qua", "ban", "that", "vui", "cam", "on"]
OFFENSIVE_WORDS = ["chan", "that", "te", "qua", "buc", "minh", "lam", "nan", "gat"]
HATE_WORDS = ["ghet", "bon", "do", "cut", "di", "tham", "hoa", "khong", "chiu"]


def _sentences(words, count, generator):
    return [" ".join(generator.choices(words, k=generator.randint(4, 10))) for _ in range(count)]


def write_vihsd_csv(directory: Path, sizes=None, seed: int = 0) -> Path:
    """Create train.csv, dev.csv and test.csv with the columns free_text and label_id."""
    generator = random.Random(seed)
    sizes = sizes or {"train": (120, 20, 40), "dev": (40, 8, 12), "test": (40, 8, 12)}
    directory.mkdir(parents=True, exist_ok=True)
    for split, (clean, offensive, hate) in sizes.items():
        rows = ([(text, 0) for text in _sentences(CLEAN_WORDS, clean, generator)]
                + [(text, 1) for text in _sentences(OFFENSIVE_WORDS, offensive, generator)]
                + [(text, 2) for text in _sentences(HATE_WORDS, hate, generator)])
        generator.shuffle(rows)
        with (directory / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["free_text", "label_id"])
            writer.writerows(rows)
    return directory


def build_tiny_visobert(directory: Path) -> Path:
    """Directory with only config.json, pytorch_model.bin and sentencepiece.bpe.model, like the real repo."""
    import sentencepiece as spm
    import torch
    from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForMaskedLM

    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory.parent / "corpus.txt"
    words = CLEAN_WORDS + OFFENSIVE_WORDS + HATE_WORDS
    generator = random.Random(1)
    corpus.write_text("\n".join(" ".join(generator.choices(words, k=8)) for _ in range(800)), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(directory / "sentencepiece.bpe"), vocab_size=60,
        model_type="bpe", pad_id=-1, bos_id=0, eos_id=1, unk_id=2, hard_vocab_limit=False,
        minloglevel=2,
    )
    (directory / "sentencepiece.bpe.vocab").unlink()

    config = XLMRobertaConfig(vocab_size=70, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=64, max_position_embeddings=64, type_vocab_size=2)
    config.save_pretrained(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    config.vocab_size = len(tokenizer) + 2
    config.save_pretrained(directory)
    model = XLMRobertaForMaskedLM(config)
    torch.save(model.state_dict(), directory / "pytorch_model.bin")
    return directory


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return build_tiny_visobert(tmp_path_factory.mktemp("visobert") / "model")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return write_vihsd_csv(tmp_path_factory.mktemp("vihsd"))


@pytest.fixture()
def settings(tiny_model, data_dir, tmp_path):
    from vihsd_lora.config import Settings

    return Settings(model_name=str(tiny_model), data_dir=str(data_dir), output_root=str(tmp_path / "outputs"),
                    batch_size=8, eval_batch_size=16, max_length=32, logging_steps=5, num_workers=0,
                    check_counts=False, epochs_override=2, use_wandb=False)
