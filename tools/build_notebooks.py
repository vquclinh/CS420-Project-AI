"""Generate the notebooks in notebooks/. Re-run this whenever the notebook content needs to change."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path(__file__).resolve().parents[1] / "notebooks"

BOOTSTRAP = '''import os, subprocess, sys
from pathlib import Path

# Change this to your repo URL. For a private repo, add a GITHUB_TOKEN secret on Colab.
REPO_URL = "https://github.com/YOUR-USERNAME/vihsd-lora.git"

def github_token():
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None

def find_repo():
    places = [Path.cwd(), *Path.cwd().parents, Path("/content/vihsd-lora"), Path("/kaggle/working/vihsd-lora")]
    places += list(Path("/kaggle/input").glob("*")) + list(Path("/kaggle/input").glob("*/vihsd-lora"))
    return next((place for place in places if (place / "src" / "vihsd_lora").is_dir()), None)

repo = find_repo()
if repo is None:
    if "YOUR-USERNAME" in REPO_URL:
        raise ValueError("Set REPO_URL to your GitHub repo URL, then run this cell again.")
    repo = Path("/content" if Path("/content").exists() else "/kaggle/working") / "vihsd-lora"
    token = github_token()
    url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
    subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
elif (repo / ".git").exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only", "-q"], check=False)
if not os.environ.get("VIHSD_SKIP_INSTALL"):
    # Colab ships an old torchao that makes peft fail when attaching LoRA. This repo does not use torchao, so remove it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
sys.path.insert(0, str(repo / "src"))
print("Repo:", repo)'''

SESSION = '''from vihsd_lora.config import Settings
from vihsd_lora.env import default_output_root, describe_gpu, detect_platform, free_space_gb, load_secrets

print("Platform:", detect_platform())
print("GPU:", describe_gpu())
print("Secrets found:", load_secrets())

settings = Settings.from_env(output_root=default_output_root())
free_gb = free_space_gb(settings.output_root)
print("Results are saved to:", settings.output_root)
print(f"Free space: {free_gb} GB")
if free_gb < 6:
    print("Warning: less than 6GB left. A full fine-tune run needs about 3GB, so clean up Drive and empty the trash.")'''


def notebook(cells):
    built = []
    for kind, source in cells:
        built.append(new_markdown_cell(source) if kind == "md" else new_code_cell(source))
    result = new_notebook(cells=built)
    result.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    result.metadata["accelerator"] = "GPU"
    return result


NOTEBOOKS = {
    "00_setup_and_check.ipynb": [
        ("md", "# 00. Setup and checks\n\nRun this notebook once before training. It checks the GPU, secrets, "
               "data and model, then measures the real speed to replace the estimates in the plan.\n\n"
               "**Requirements:** enable a GPU and add the `HF_TOKEN` and `WANDB_API_KEY` secrets as described "
               "in the README."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Check the data\n\nThe cell below stops if the per-label counts do not match the reference "
               "ViHSD version."),
        ("code", '''import pandas as pd
from vihsd_lora.data import label_counts, length_report, load_vihsd
from vihsd_lora.metrics import majority_baseline

dataset = load_vihsd(settings)
display(pd.DataFrame(label_counts(dataset)).T)

baseline = majority_baseline(dataset["test"]["labels"])
print(f"Always predicting CLEAN on test: Acc {baseline['accuracy']:.2f}, macro F1 {baseline['macro_f1']:.2f}")'''),
        ("md", "## Check the tokenizer and comment lengths\n\n"
               "`truncated_pct` is the share of comments longer than `max_length` that get truncated."),
        ("code", '''from vihsd_lora.modeling import load_tokenizer

tokenizer = load_tokenizer(settings.model_name)
print(length_report(dataset["train"], tokenizer, settings.max_length))'''),
        ("md", "## Check the parameter counts\n\nExpected with ViSoBERT: LoRA r=8 trains 887,811 parameters, "
               "full fine-tune trains 97,568,259 parameters."),
        ("code", '''from vihsd_lora.config import RUNS
from vihsd_lora.modeling import build_model, count_parameters

rows = []
for key, run in RUNS.items():
    if key == "E":
        continue
    rows.append({"run": key, "description": run.describe(), **count_parameters(build_model(run, settings))})
display(pd.DataFrame(rows))'''),
        ("md", "## Measure the real speed\n\nRuns 200 steps for LoRA and for full fine-tune. The final table is the "
               "estimated training time based on the measured speed, excluding dev evaluation."),
        ("code", '''from vihsd_lora.speedtest import estimate_hours, measure

SPEED_STEPS = 200
speed_lora = measure("A", steps=SPEED_STEPS, settings=settings)
speed_full = measure("B", steps=SPEED_STEPS, settings=settings)
print(speed_lora)
print(speed_full)
display(estimate_hours({"lora": speed_lora["seconds_per_step"], "full": speed_full["seconds_per_step"]}, settings))'''),
    ],
    "01_train.ipynb": [
        ("md", "# 01. Train\n\nEach run is saved on Drive in `MyDrive/vihsd-lora/outputs/runs/<run name>` "
               "and logged to W&B.\n\n- **Checkpoints:** LoRA saves every epoch, full fine-tune every 2 epochs. "
               "Drive keeps only the latest checkpoint.\n- **Best model:** saved to `best/` as soon as the dev "
               "macro F1 improves.\n"
               "- **Session disconnected:** re-run the whole notebook with the same run. Training resumes from "
               "the latest checkpoint and continues the same W&B run."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Choose runs\n\n| Run | Method | r | Epochs |\n|---|---|---|---|\n"
               "| A | LoRA | 8 | 10 |\n| B | Full fine-tune | | 10 |\n| C | LoRA | 2 | 10 |\n"
               "| D | LoRA | 16 | 10 |\n| E | LoRA | 8 | 30 |\n\n"
               "You can train several runs back to back, or give each team member one run if you share a "
               "W&B project."),
        ("code", '''RUNS_TO_TRAIN = ["A"]
SEED = 42
WANDB_TEAM = None  # Set your W&B team name if the whole team shares one workspace
SAVE_EVERY_EPOCHS = None  # Leave empty for the defaults: LoRA every epoch, full fine-tune every 2 epochs

settings.wandb_entity = WANDB_TEAM or settings.wandb_entity
settings.save_every_epochs = SAVE_EVERY_EPOCHS'''),
        ("code", '''import gc
import pandas as pd
import torch
from vihsd_lora.train import train_run

summaries = []
for key in RUNS_TO_TRAIN:
    summaries.append(train_run(key, seed=SEED, settings=settings))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

columns = ["slug", "trainable_params", "trainable_pct", "dev_accuracy", "dev_macro_f1", "best_epoch",
           "mean_epoch_seconds", "gpu_peak_mb", "saved_mb"]
display(pd.DataFrame(summaries)[columns])'''),
        ("md", "## View the charts\n\nOpen the W&B link printed during training. Per-epoch charts are in the "
               "`by_epoch` group: `dev_macro_f1`, `dev_accuracy`, `train_loss`, `epoch_seconds` and `gpu_peak_mb`. "
               "Select several runs in the Runs table to compare them on the same chart."),
    ],
    "02_evaluate.ipynb": [
        ("md", "# 02. Compare on dev and score on test\n\nThe configuration is chosen on dev only, then scored on "
               "test once. Test scores are saved, so later runs reuse them instead of scoring again."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Compare runs on dev"),
        ("code", '''from vihsd_lora.evaluate import compare_on_dev, evaluate_on_test, select_for_test

display(compare_on_dev(settings))
choice = select_for_test(settings)
print("Will score on test:", choice)'''),
        ("md", "## Score on test\n\nOnly run this after all runs to compare have finished training."),
        ("code", '''results = {name: evaluate_on_test(slug, settings) for name, slug in choice.items()}'''),
        ("md", "## Compare LoRA with full fine-tune on the same test samples\n\n"
               "If the confidence interval of the difference contains 0, there is not enough evidence to say "
               "which one is better."),
        ("code", '''from vihsd_lora.report import compare_on_test

if {"lora", "full"} <= set(choice):
    print(compare_on_test(choice["lora"], choice["full"], settings))'''),
    ],
    "03_inference.ipynb": [
        ("md", "# 03. Inference\n\nPredict new text with a trained model, and check that merging LoRA into the "
               "base weights does not change the results."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("code", '''from pathlib import Path
import pandas as pd
from vihsd_lora.evaluate import select_for_test
from vihsd_lora.inference import Predictor

SLUG = select_for_test(settings)["lora"]
predictor = Predictor(Path(settings.output_root) / "runs" / SLUG)
print("Using run:", SLUG)'''),
        ("md", "## Predict\n\nReplace the list below with the Vietnamese sentences you want to try."),
        ("code", '''TEXTS = [
    "Hôm nay thời tiết đẹp quá, cả nhà đi chơi vui nhé.",  # "The weather is so nice today, have fun everyone."
    "Bài viết rất hữu ích, cảm ơn bạn đã chia sẻ.",  # "Very useful post, thanks for sharing."
]
before = pd.DataFrame(predictor.predict(TEXTS))
display(before)'''),
        ("md", "## Merge LoRA into the base weights\n\nAfter merging, the results should stay almost the same. "
               "This is why LoRA adds no latency at deployment."),
        ("code", '''latency_adapter = predictor.latency_ms(TEXTS, batch_size=1)
predictor.merge_lora()
latency_merged = predictor.latency_ms(TEXTS, batch_size=1)
after = pd.DataFrame(predictor.predict(TEXTS))

print(f"Latency per sentence: unmerged {latency_adapter} ms, merged {latency_merged} ms")
print("Labels unchanged:", (before["label"] == after["label"]).all())
display(after)'''),
    ],
    "04_results_for_slides.ipynb": [
        ("md", "# 04. Tables and charts for the slides\n\nCreates the results table and PNG files in "
               "`outputs/results`. Charts without enough data are skipped, so you can re-run this after each run."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Results table\n\nIncludes the 13 published rows with the same protocol, the always-CLEAN "
               "baseline, and the runs scored on test."),
        ("code", '''from vihsd_lora.report import export_tables, main_table

display(main_table(settings))
print(export_tables(settings))'''),
        ("md", "## Charts"),
        ("code", '''from IPython.display import Image, display as show
from vihsd_lora.plots import make_all

figures = make_all(settings)
for figure in figures:
    print(figure.name)
    show(Image(str(figure), width=760))'''),
        ("md", "## Log the charts to W&B\n\nCreates a `slide-figures` run that the whole team can view."),
        ("code", '''from vihsd_lora.report import log_figures_to_wandb

log_figures_to_wandb(figures, settings)'''),
    ],
}


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nbformat.write(notebook(cells), OUTPUT / name)
        print("Created", name)


if __name__ == "__main__":
    main()
