# LoRA on ViHSD

Code for the experimental part of a LoRA seminar: comparing LoRA with full fine-tuning of ViSoBERT on Vietnamese hate speech detection (ViHSD). The full plan is in [docs/seminar-plan-lora-vihsd.md](docs/seminar-plan-lora-vihsd.md).

## Layout

```text
notebooks/   Notebooks to run, numbered in order
src/         Training, evaluation, inference, tables and charts
data/        Published results (published_results.csv); contains no ViHSD data
docs/        Seminar plan
tests/       CPU tests with a tiny model and fake data
tools/       Script that regenerates the notebooks
```

## Setup

1. **Hugging Face:** log in, open [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd) and accept the conditions. Then create a token with Read access under Settings > Access Tokens.
2. **W&B:** create an account at [wandb.ai](https://wandb.ai) and get an API key from the [authorize](https://wandb.ai/authorize) page.
3. **Secrets:** add `HF_TOKEN` and `WANDB_API_KEY`.
   - Colab: open the key icon in the left sidebar, add both secrets and enable notebook access.
   - Kaggle: open Add-ons > Secrets, add both secrets and enable them for the notebook.
4. **GPU:** on Colab, choose Runtime > Change runtime type, then pick T4 or L4. On Kaggle, choose GPU T4 and turn on Internet.
5. **Repo:** set `REPO_URL` in the first cell of each notebook to your GitHub repo URL. If the repo is private, create a fine-grained token with read-only access to this repo and add it to Colab as `GITHUB_TOKEN`.

## Run order

| Notebook | What it does | Estimated time |
|---|---|---|
| `00_setup_and_check` | Checks GPU, data, parameter counts, measures real speed | About 10–15 minutes |
| `01_train` | Trains runs A, B, C, D, E | 30–70 minutes per 10-epoch run on T4 |
| `02_evaluate` | Compares on dev, picks configurations, scores on test | A few minutes |
| `03_inference` | Predicts new sentences, checks LoRA merging | A few minutes |
| `04_results_for_slides` | Exports tables and PNG charts | Under 2 minutes |

These times are estimates. Notebook `00` reports the real times for your GPU.

If each team member trains one run, copy all `outputs/runs/<run name>` folders into one shared Drive before running notebooks `02` and `04`.

## Training runs

| Run | Method | r | alpha | Learning rate | Epochs |
|---|---|---|---|---|---|
| A | LoRA | 8 | 8 | 5e-4 | 10 |
| B | Full fine-tune | | | 2e-5 | 10 |
| C | LoRA | 2 | 8 | 5e-4 | 10 |
| D | LoRA | 16 | 8 | 5e-4 | 10 |
| E | LoRA | 8 | 8 | 5e-4 | 30 |

All runs use batch size 40, max length 128, 6% warmup, FP16, and keep the epoch with the best dev macro F1.

## Saving to Drive and resuming

On Colab, the notebooks mount Drive automatically and save everything important there, not in `/content`.

- **Resumable checkpoints:** LoRA saves every epoch (a few MB); full fine-tune saves every 2 epochs (about 1.2GB each time, model plus optimizer). Drive keeps only the latest checkpoint. Change the interval with `SAVE_EVERY_EPOCHS` in notebook `01`.
- **Best model:** saved to `best/` as soon as the dev macro F1 improves, independently of the checkpoints, so the best epoch is never lost even with sparse checkpoints.
- **Session disconnected:** reopen notebook `01` and run it from the top with the same run. Training resumes from the latest checkpoint and continues the same W&B run. Finished runs are skipped.
- **Drive trash:** deleted checkpoints may stay in the trash and still use quota. Empty the Drive trash after a full fine-tune run.
- **W&B local files:** kept in `/content/wandb_local`, since the data is already synced to W&B. Drive only keeps `wandb_id.txt` to resume the right run.

## Where the results are

On Colab, results are saved to Drive at `MyDrive/vihsd-lora/outputs`. On Kaggle, they are at `/kaggle/working/outputs`.

```text
outputs/
  runs/<run name>/
    best/                 LoRA adapter or full model with the highest dev macro F1
    best_state.json       Score and epoch of the model in best/
    checkpoints/          Resumable checkpoints, deleted automatically when training finishes
    history.csv           Per-epoch metrics
    run_summary.json      Hyperparameters, time, VRAM, saved size, dev scores
    test_metrics.json     Test scores, confidence intervals, confusion matrix
    predictions_test.csv  True and predicted labels, without the text
  results/
    main_results.csv      Results table
    main_results.md       Markdown table to paste into the report
    figures/              PNG charts for the slides
```

## Charts on W&B

The default project is `vihsd-lora-seminar`. If the whole team shares a W&B team, set its name in `WANDB_TEAM` in notebook `01_train`.

- The `by_epoch` group has per-epoch metrics: `dev_macro_f1`, `dev_accuracy`, `train_loss`, `dev_loss`, `epoch_seconds`, `gpu_peak_mb`.
- The `eval` and `train` groups have per-step metrics logged by the Trainer.
- In the Runs table, select several runs to compare them on the same chart.
- Open the menu in the corner of a chart to download a PNG.

Notebook `04` also creates color-consistent charts for the slides, independent of W&B: macro F1 curves, train loss, rank effect, macro F1 vs. parameter count, per-label F1, cost, and confusion matrices.

## Command line

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m vihsd_lora.train --run A --seed 42
python -m vihsd_lora.evaluate --slug A_lora_r8_e10_s42
```

To use local data, set `VIHSD_DATA_DIR` to a folder containing `train.csv`, `dev.csv` and `test.csv`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Error 401 or 403 when loading the data | Check that you accepted the conditions on `uitnlp/vihsd` and that `HF_TOKEN` is correct |
| Sample count mismatch | Stop and check the data source. Do not skip the check, or the results will not be comparable with the published tables |
| Cannot load `uitnlp/visobert` | Set `VIHSD_MODEL_NAME=xlm-roberta-large`. Training will take longer |
| Out of VRAM | Set a smaller `VIHSD_BATCH_SIZE` and state the change in the report |
| Colab session disconnected | Re-run `01_train` with the same run to resume |
| Import error right after installing packages | Choose Runtime > Restart session, then run from the top |
| `Found an incompatible version of torchao` | The first cell already removes torchao. If it still happens, run `!pip uninstall -y torchao`, choose Runtime > Restart session and run again |
| You do not want to use W&B | Set `WANDB_MODE=disabled`; `history.csv` and the PNG charts are still created |

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests -q
```

## Test status

All 22 tests and 5 notebooks have run successfully on CPU with a tiny model and fake data, with W&B in offline mode. The ViSoBERT parameter counts were verified on the real architecture built from its config. Not yet tested: a real GPU, the real ViHSD data, the real ViSoBERT weights, and online W&B logging. So run notebook `00` first.

## Data notice

ViHSD is for research use only and contains real offensive comments. This repo does not store the data, and `predictions_test.csv` contains no text. Do not commit the data to git, and avoid putting raw comments on slides.
