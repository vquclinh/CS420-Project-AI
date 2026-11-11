# LoRA seminar plan: Vietnamese hate speech detection on ViHSD

Updated: 2026-11-04

## 1. Summary of decisions

| Item | Decision |
|---|---|
| Task | Classify Vietnamese social media comments as CLEAN, OFFENSIVE or HATE |
| Dataset | ViHSD: 24,048 train, 2,672 dev, 6,680 test |
| Model | `uitnlp/visobert`, 97.6M parameters |
| Method | LoRA on `query` and `value`, compared with full fine-tuning of the same model |
| Metric | Macro F1 as the main metric, plus Accuracy, on test |
| Hardware | T4 or L4 is enough |
| Training time | 5 runs: 3.5–8 hours on T4, 1.2–3 hours on L4 (estimated, not measured) |
| Inference time | About 1 minute per test evaluation (estimated, not measured) |
| Goal | LoRA within 1 Acc point and 2 macro F1 points of full fine-tuning, while training only about 0.91% of the parameters |

## 2. Why ViHSD

- **No published LoRA results on ViHSD found.** As of 2026-11-04, the team has not found any paper or public model that does this. This means "not found", not "certainly nobody has done it", and it must be presented that way.
- **The closest work does not overlap.** "ViHateT5 with LoRA" applies LoRA to a different news dataset with 2,962 samples. "Crossing Linguistic Horizons" applies LoRA to general data and only uses ViHSD for evaluation.
- **Plenty of published numbers.** The main table has 13 models under the same protocol, plus 23 reference rows from 3 other papers.
- **The data is accessible.** ViHSD is on Hugging Face and does not depend on the lab's website.
- **Lightweight.** The model has only 97.6M parameters and the data consists of short comments, so many runs fit within a 10-hour budget.

**Why the earlier directions were dropped:**

- **ViNLI:** the UIT NLP Group data page is currently unreachable.
- **DreamBooth + LoRA:** DreamBench++ and Diffu-LoRA have already published results, and a single run over all 30 subjects is estimated to take 11–16 hours on a T4.

## 3. Dataset

ViHSD has 33,400 comments from Facebook and YouTube about entertainment, celebrities, society and politics. Personal names were removed for anonymity.

| Label | Meaning | Train | Dev | Test |
|---|---|---|---|---|
| CLEAN | No harassing content | 19,886 | 2,190 | 5,548 |
| OFFENSIVE | Harassing content, but not aimed at a specific target | 1,606 | 212 | 444 |
| HATE | Directly insults an individual or a group | 2,556 | 270 | 688 |
| **Total** | | **24,048** | **2,672** | **6,680** |

The labels are highly imbalanced: CLEAN makes up 82.71% of the dataset. Inter-annotator agreement is κ = 0.52.

**How to get the data:** use `uitnlp/vihsd` on Hugging Face, which requires logging in and accepting the conditions. The authors' README states that the data is for research use only.

**Requirements before training:**

- Count the samples per label in each split; they must match the table above. These counts also identify which label each `label_id` corresponds to.
- Keep the original train/dev/test split; do not re-split.
- Do not preprocess the text, to match the protocol of the main table.
- Do not use `visolex/ViHSD`, because it uses a different split (about 26K/3.3K/3.3K).
- `phucdev/ViHSD` has only 24,046 train samples. Use it only as a fallback and state the 2-sample difference.

## 4. Metrics

- **Macro F1:** the main metric. It computes F1 per label and averages them, so it reflects performance on the two rare labels.
- **Accuracy:** the secondary metric. It is not used alone, because always predicting CLEAN already scores 83.05.

```python
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
```

## 5. Results tables

### 5.1. Main table: directly comparable

Numbers from Table 4 of the ViSoBERT paper: Accuracy and macro F1, all run by the same authors with 10 epochs, batch size 40, max length 128, learning rate 2e-5 and no preprocessing. The paper does not name the split, but reports the dataset split as 24,048/2,672/6,680.

| System | Acc | Macro F1 | Trainable params |
|---|---|---|---|
| ViSoBERT | 88.51 | 68.77 | 97.6M |
| PhoBERT Large | 87.32 | 65.14 | 369.2M |
| TwHIN-BERT Large | 87.23 | 65.23 | Not counted |
| XLM-R Large | 87.15 | 65.13 | 559.9M |
| PhoBERT Base | 87.12 | 65.01 | 135.0M |
| vELECTRA | 86.96 | 63.95 | Not counted |
| TwHIN-BERT Base | 86.63 | 63.67 | Not counted |
| XLM-R Base | 86.36 | 63.39 | 278.0M |
| XLM-T | 86.22 | 63.48 | Not counted |
| Bernice | 86.12 | 64.32 | Not counted |
| viBERT | 85.34 | 62.07 | 115.4M |
| mBERT cased | 83.55 | 60.62 | 177.9M |
| mBERT uncased | 83.38 | 58.92 | 167.4M |
| Always predict CLEAN (computed by us) | 83.05 | 30.25 | 0 |
| **ViSoBERT full fine-tune (ours)** | ? | ? | 97.6M |
| **ViSoBERT + LoRA r=16 (ours)** | ? | ? | 1.18M |
| **ViSoBERT + LoRA r=8 (ours)** | ? | ? | 0.89M |
| **ViSoBERT + LoRA r=2 (ours)** | ? | ? | 0.67M |

The parameter column was computed by us from `config.json`, including a 3-label classifier head, except for the "Not counted" cells. The paper does not give checkpoint IDs, so this assumes the most common checkpoints. The remaining cells can be counted with `model.num_parameters()`, without training.

### 5.2. Secondary tables: same data, different protocol

For reference only; not ranked together with the main table.

**ViHateT5 (4 epochs, batch size 16, max length 256, learning rate 2e-5, no preprocessing):**

| System | Acc | Macro F1 |
|---|---|---|
| ViHateT5 | 88.76 | 68.67 |
| ViSoBERT | 88.17 | 67.71 |
| PhoBERT_v2 | 87.42 | 66.60 |
| BERT cased | 87.36 | 64.44 |
| XLM-RoBERTa | 87.29 | 65.08 |
| PhoBERT | 86.75 | 64.76 |
| BERT uncased | 86.66 | 62.92 |
| viBERT | 86.33 | 62.85 |
| DistilBERT | 86.30 | 63.34 |

**Original ViHSD paper (4 epochs, batch size 16, max length 100, with preprocessing):**

| System | Acc | Macro F1 |
|---|---|---|
| BERT cased | 86.88 | 62.69 |
| Text CNN + fastText | 86.69 | 61.11 |
| BERT uncased | 86.60 | 62.38 |
| DistilBERT | 86.22 | 62.42 |
| XLM-R Base | 86.12 | 61.28 |
| GRU + fastText | 85.41 | 60.47 |

**PhoBERT-CNN (batch size 64, learning rate 2e-5, with preprocessing, no data augmentation):**

| System | Acc | Macro F1 |
|---|---|---|
| PhoBERT-CNN | 87.17 | 64.43 |
| PhoBERT | 87.13 | 63.51 |
| Text-CNN + PhoW2V word | 86.11 | 63.01 |
| Bi-LSTM + PhoW2V word | 85.99 | 62.66 |
| XLMR-CNN | 85.48 | 63.34 |
| BERT | 84.52 | 60.29 |
| XLM-R | 83.62 | 62.38 |
| RoBERTa | 83.04 | 61.49 |

The PhoBERT-CNN paper has 10 more rows in Table 6. The SMTCE paper is not used because it reports only macro F1.

### 5.3. Source papers

| Paper | Used for | Rows |
|---|---|---|
| ViSoBERT, arXiv 2310.11166 | Main table | 13 |
| ViHateT5, Findings of ACL 2024 | Secondary table | 9 |
| Luu et al. 2021 (original ViHSD paper) | Secondary table | 6 |
| PhoBERT-CNN, arXiv 2206.00524 | Secondary table | 8 |

## 6. Model and training configuration

We use `uitnlp/visobert` because it is the best model in the main table, has a public training protocol, and is pre-trained on Vietnamese social media text. It has 12 layers, 768 dimensions and the XLM-R architecture. If it cannot be loaded, fall back to `xlm-roberta-large`, the best XLM-R row in the main table, and recompute the parameter counts and timings.

| Run | Priority | Method | r | Learning rate | Epochs | Trainable params | Saved size |
|---|---|---|---|---|---|---|---|
| A | 1 | LoRA | 8 | 5e-4 | 10 | 887,811 | About 3.6MB |
| B | 2 | Full fine-tune | N/A | 2e-5 | 10 | 97,568,259 | About 390MB |
| C | 3 | LoRA | 2 | 5e-4 | 10 | 666,627 | About 2.7MB |
| D | 4 | LoRA | 16 | 5e-4 | 10 | 1,182,723 | About 4.7MB |
| E | 5 | LoRA | 8 | 5e-4 | 30 | 887,811 | About 3.6MB |

**Each run** is a complete, independent training run: reload the pre-trained model, train for all epochs, evaluate on dev after each epoch and keep the best epoch.

Run E exists because 10 epochs may not be enough for LoRA: the LoRA paper trains RoBERTa Base for 25–80 epochs depending on the task.

**Shared configuration:**

- Batch size 40, max length 128, dynamic padding per batch. If VRAM runs out, reduce the batch size and record it.
- AdamW, 6% warmup, linear learning rate decay, FP16.
- No preprocessing, no class weights.
- Select the best epoch by dev macro F1, seed 42.
- `lora_alpha=8` is kept fixed for all three values of r.
- With `TaskType.SEQ_CLS`, PEFT automatically makes the classifier head trainable.

**Sources of the configuration:** r, alpha, the LoRA learning rate and warmup follow the RoBERTa Base configuration in the LoRA paper. Batch size, max length, number of epochs and the 2e-5 learning rate follow the ViSoBERT paper. Dropout, seed and epoch selection were chosen by the team.

```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

**Steps:**

1. Get the data and count the labels as in section 3.
2. Run 200 test steps to measure speed and VRAM, then recompute the total time.
3. Train the runs in priority order, saving a checkpoint to Drive after each epoch with `save_total_limit=1`.
4. Compare the LoRA runs by dev macro F1 to pick the best configuration.
5. Score on test exactly once for the chosen LoRA run and for full fine-tuning.
6. If time allows, run 2 more seeds for the best LoRA configuration and for full fine-tuning, and report the mean.

**What to record for each run:** train loss, dev Acc and macro F1 per epoch, time per epoch, `torch.cuda.max_memory_allocated()`, the output of `print_trainable_parameters()`, saved file size, per-label F1 and the confusion matrix.

## 7. Time

| Task | T4 | L4 |
|---|---|---|
| One 10-epoch run, including dev evaluation | 30–70 minutes | 10–25 minutes |
| One 30-epoch run, including dev evaluation | 1.5–3.5 hours | 30–75 minutes |
| All 5 runs | 3.5–8 hours | 1.2–3 hours |
| Scoring one model on test, including loading | About 1 minute | Under 1 minute |
| 2 extra seeds for 2 configurations (10 epochs each) | 2–4.7 hours | 40–100 minutes |

All of these are estimates, not measurements. They start from a BERT benchmark on T4 (257 seconds per epoch, batch size 16, length 128), assume ViSoBERT costs about the same as BERT-base, then assume FP16 and short sentences make it 1.5–3.5 times faster. L4 is taken as 2.8 times faster than T4, following McCormick's benchmark. Test scoring is assumed to be 3 times faster than training.

Colab does not guarantee that sessions stay alive, so checkpoints must be saved and `resume_from_checkpoint` used after a disconnect. With several team members, each can train one run on their own account.

## 8. Chances of beating the baselines

This is the team's qualitative assessment, without measurements. Comparisons use the macro F1 of the main table.

| Baseline | Macro F1 | Chance of beating | Reason |
|---|---|---|---|
| Always predict CLEAN | 30.25 | Almost certain | Minimum bar |
| mBERT, viBERT, XLM-R Base, XLM-T, TwHIN-BERT Base, vELECTRA, Bernice | 58.92–64.32 | High | At least 4.45 points below ViSoBERT |
| PhoBERT Base, PhoBERT Large, XLM-R Large, TwHIN-BERT Large | 65.01–65.23 | Fair | LoRA can be up to 3.5 points below ViSoBERT and still beat them |
| ViSoBERT full fine-tune | 68.77 | Close, uncertain | In the LoRA paper, LoRA wins on some tasks and loses on others against full fine-tuning |

On RoBERTa Base, the LoRA paper reports a GLUE average of 87.2 versus 86.4 for full fine-tuning. The results are uneven, though: SST-2 is 95.1 vs. 94.8, MNLI is 87.5 vs. 87.6, and QQP is 90.8 vs. 91.9.

Topping the table is not a goal. Success means LoRA comes close to full fine-tuning while training about 110 times fewer parameters.

## 9. Discussion of the results table

**For each possible outcome:**

- **LoRA is close to full fine-tuning (meets the goal in section 1):** conclude that the trade-off is favorable, since trainable parameters and saved size drop about 110 times with almost no loss in performance.
- **LoRA beats full fine-tuning:** do not conclude that LoRA is better if the gap is small and there is only one seed, because macro F1 on imbalanced data fluctuates a lot. Claiming it requires more seeds.
- **LoRA is clearly worse:** first check run E to see whether it is due to too few epochs. Then discuss the limits of the configuration: the embeddings and feed-forward layers are frozen, and only `query` and `value` are adapted.

**Analyses to include:**

- **Performance vs. parameters:** plot macro F1 against trainable parameters on a log scale, including the main table rows and our runs.
- **Effect of rank:** compare r=2, 8, 16 to see whether more parameters still improve the score.
- **Effect of epochs:** compare run A with run E to see whether LoRA needs longer training than full fine-tuning.
- **Classifier head:** at r=8, the classifier head accounts for 592,899 of 887,811 trainable parameters, i.e. 66.8%. Counting only the LoRA weights gives 294,912 parameters, about 0.30% of the model. State how parameters are counted when comparing with papers.
- **Label imbalance:** compare per-label F1 and confusion matrices between LoRA and full fine-tuning, focusing on OFFENSIVE and HATE.
- **Data limits:** an agreement of κ = 0.52 shows the label boundaries are inherently fuzzy, so macro F1 cannot get close to perfect.
- **Real cost:** compare time per epoch, VRAM and file size between runs A and B. Do not expect LoRA to be much faster, since gradients still flow through the whole network. The adapter is small but still needs the base model at inference time.
- **Reproducibility:** compare our full fine-tune with 88.51 / 68.77. If it is far off, investigate before drawing conclusions about LoRA.
- **Protocol affects scores:** for the same mBERT cased, macro F1 is 60.62, 62.69 and 64.44 in three papers. This is why ranking is done only within the main table.

**Limitations to state:** a single seed for most runs, the paper does not give checkpoint IDs, the paper does not name the split, the secondary table rows use different protocols, and "no publication found" does not mean "nobody has done it".

## 10. Class presentation

| Slide | Content | Key message |
|---|---|---|
| 1 | Problem | Full fine-tuning trains and stores all parameters for every task |
| 2 | LoRA idea | Freeze the base weights and learn only a low-rank update |
| 3 | Formula | `h = W₀x + (α/r)·BAx`, with `A` initialized randomly and `B` at zero |
| 4 | Applying it to a Transformer | Attach to `query` and `value`; everything else is frozen except the classifier head |
| 5 | Why ViHSD | Published tables available, imbalanced labels, no LoRA publication found |
| 6 | Experiment design | 5 runs, same data, same scoring, only the training method differs |
| 7 | Training | Loss curves, dev macro F1, `print_trainable_parameters()` output, VRAM, time |
| 8 | Results table | Main table, with our rows next to ViSoBERT |
| 9 | Discussion | Performance vs. parameter trade-off, rank, epochs, rare labels |
| 10 | Conclusion | Results, limitations and future work |

**How to explain the concepts:**

- **Why low rank is enough:** the paper hypothesizes that the weight change when adapting to a new task has a low "intrinsic rank", so a product of two small matrices can represent most of that change.
- **Why it saves resources:** instead of learning a `d × d` matrix, we learn `A` of size `r × d` and `B` of size `d × r`, so the parameter count drops from `d²` to `2dr`.
- **Role of `r` and `α`:** `r` sets the expressiveness of the update, while `α/r` scales its magnitude.
- **Why `B` starts at zero:** the product `BA` is zero at the start, so the model begins exactly as the pre-trained model.
- **Why the classifier head is still trained:** the 3-label classifier head is new and has no pre-trained weights.
- **Why the learning rates differ:** full fine-tuning only nudges weights that are already good, while the LoRA matrices are learned from scratch and need larger steps.
- **Why there is no added latency:** at deployment, `BA` can be merged into `W₀`, so the network size and inference time do not grow.
- **Why macro F1 is the main metric:** the labels are imbalanced, so high Accuracy can still miss the two labels we need to detect.

**Questions to prepare for:**

- Why not compare directly with ViHateT5: it belongs to a table with a different protocol. In its own table ViHateT5 beats ViSoBERT, but numbers should not be compared across the two tables.
- Why run full fine-tuning ourselves: to get time and VRAM figures and to check the data with the same scoring.
- Can we claim nobody has done this: no, only that no publication was found as of the search date.

**Content warning:** the dataset contains real offensive comments, so avoid putting raw text on slides. When an example is needed, mask the offensive words, and do not republish the data.

## 11. Open items

- [ ] Download `uitnlp/vihsd` and check the counts against section 3.
- [ ] Confirm that `uitnlp/visobert` can be loaded.
- [ ] Confirm with the instructor that ViHSD may be used for the experiments.
- [ ] Run 200 test steps to replace the estimates in section 7 with real numbers.
- [ ] Open the ViSoBERT and ViHateT5 PDFs and check the tables before putting them on slides.
- [ ] Assign who trains each run and who handles the slides and report.

## Sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685v2)
- [ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing](https://arxiv.org/html/2310.11166v1)
- [ViHateT5 (Findings of ACL 2024)](https://aclanthology.org/2024.findings-acl.355.pdf)
- [A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts](https://arxiv.org/pdf/2103.11528)
- [Vietnamese Hate and Offensive Detection using PhoBERT-CNN](https://arxiv.org/pdf/2206.00524)
- [SMTCE benchmark](https://arxiv.org/pdf/2209.10482)
- [ViHateT5 with LoRA](https://zenodo.org/records/20265021)
- [Crossing Linguistic Horizons](https://arxiv.org/html/2403.02715v2)
- [DreamBench++](https://arxiv.org/html/2406.16855v2)
- [Diffu-LoRA](https://arxiv.org/pdf/2610.10550v1)
- [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd)
- [sonlam1102/vihsd on GitHub](https://github.com/sonlam1102/vihsd)
- [uitnlp/visobert](https://huggingface.co/uitnlp/visobert)
- [GPU Benchmarks for Fine-Tuning BERT](https://mccormickml.com/2020/07/21/gpu-benchmarks-for-fine-tuning-bert/)
- [Colab GPUs Features & Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
