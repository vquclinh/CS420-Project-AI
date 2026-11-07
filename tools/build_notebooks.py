"""Sinh các notebook trong thư mục notebooks/. Chạy lại khi cần sửa nội dung notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path(__file__).resolve().parents[1] / "notebooks"

BOOTSTRAP = '''import os, subprocess, sys
from pathlib import Path

# Đổi thành địa chỉ repo của bạn. Repo private thì thêm secret GITHUB_TOKEN trên Colab.
REPO_URL = "https://github.com/TEN-TAI-KHOAN/vihsd-lora.git"

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
    if "TEN-TAI-KHOAN" in REPO_URL:
        raise ValueError("Hãy sửa REPO_URL thành địa chỉ repo GitHub của bạn rồi chạy lại ô này.")
    repo = Path("/content" if Path("/content").exists() else "/kaggle/working") / "vihsd-lora"
    token = github_token()
    url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
    subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
elif (repo / ".git").exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only", "-q"], check=False)
if not os.environ.get("VIHSD_SKIP_INSTALL"):
    # Colab cài sẵn torchao bản cũ, peft sẽ báo lỗi khi gắn LoRA. Repo không dùng torchao nên gỡ đi.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
sys.path.insert(0, str(repo / "src"))
print("Repo:", repo)'''

SESSION = '''from vihsd_lora.config import Settings
from vihsd_lora.env import default_output_root, describe_gpu, detect_platform, free_space_gb, load_secrets

print("Nền tảng:", detect_platform())
print("GPU:", describe_gpu())
print("Secret đã có:", load_secrets())

settings = Settings.from_env(output_root=default_output_root())
free_gb = free_space_gb(settings.output_root)
print("Kết quả lưu tại:", settings.output_root)
print(f"Dung lượng còn trống: {free_gb} GB")
if free_gb < 6:
    print("Cảnh báo: còn dưới 6GB. Lượt full fine-tune cần khoảng 3GB, hãy dọn Drive và làm trống thùng rác.")'''


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
        ("md", "# 00. Chuẩn bị và kiểm tra\n\nChạy notebook này một lần trước khi train. Nó xác nhận GPU, secret, "
               "dữ liệu và model, rồi đo tốc độ thật để thay các con số ước tính trong kế hoạch.\n\n"
               "**Yêu cầu:** bật GPU và thêm secret `HF_TOKEN`, `WANDB_API_KEY` theo hướng dẫn trong README."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Kiểm tra dữ liệu\n\nÔ dưới dừng lại nếu số mẫu từng nhãn không khớp bản ViHSD chuẩn."),
        ("code", '''import pandas as pd
from vihsd_lora.data import label_counts, length_report, load_vihsd
from vihsd_lora.metrics import majority_baseline

dataset = load_vihsd(settings)
display(pd.DataFrame(label_counts(dataset)).T)

baseline = majority_baseline(dataset["test"]["labels"])
print(f"Luôn đoán CLEAN trên test: Acc {baseline['accuracy']:.2f}, macro F1 {baseline['macro_f1']:.2f}")'''),
        ("md", "## Kiểm tra tokenizer và độ dài bình luận\n\n"
               "`truncated_pct` là tỉ lệ bình luận dài hơn `max_length` và bị cắt."),
        ("code", '''from vihsd_lora.modeling import load_tokenizer

tokenizer = load_tokenizer(settings.model_name)
print(length_report(dataset["train"], tokenizer, settings.max_length))'''),
        ("md", "## Kiểm tra số tham số\n\nSố mong đợi với ViSoBERT: LoRA r=8 train 887,811 tham số, "
               "full fine-tune train 97,568,259 tham số."),
        ("code", '''from vihsd_lora.config import RUNS
from vihsd_lora.modeling import build_model, count_parameters

rows = []
for key, run in RUNS.items():
    if key == "E":
        continue
    rows.append({"lượt": key, "mô tả": run.describe(), **count_parameters(build_model(run, settings))})
display(pd.DataFrame(rows))'''),
        ("md", "## Đo tốc độ thật\n\nChạy 200 step cho LoRA và cho full fine-tune. Bảng cuối là thời gian train "
               "ước tính theo tốc độ đo được, chưa tính thời gian chấm dev."),
        ("code", '''from vihsd_lora.speedtest import estimate_hours, measure

SPEED_STEPS = 200
speed_lora = measure("A", steps=SPEED_STEPS, settings=settings)
speed_full = measure("B", steps=SPEED_STEPS, settings=settings)
print(speed_lora)
print(speed_full)
display(estimate_hours({"lora": speed_lora["seconds_per_step"], "full": speed_full["seconds_per_step"]}, settings))'''),
    ],
    "01_train.ipynb": [
        ("md", "# 01. Train\n\nMỗi lượt được lưu trên Drive trong `MyDrive/vihsd-lora/outputs/runs/<tên lượt>` "
               "và ghi lên W&B.\n\n- **Checkpoint:** LoRA lưu mỗi epoch, full fine-tune lưu cách 2 epoch. "
               "Drive chỉ giữ checkpoint mới nhất.\n- **Model tốt nhất:** lưu vào `best/` ngay khi macro F1 trên dev tăng.\n"
               "- **Bị ngắt phiên:** chạy lại toàn bộ notebook với cùng lượt, quá trình tiếp tục từ checkpoint gần nhất "
               "và nối vào cùng run trên W&B."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Chọn lượt\n\n| Lượt | Cách train | r | Epoch |\n|---|---|---|---|\n"
               "| A | LoRA | 8 | 10 |\n| B | Full fine-tune | | 10 |\n| C | LoRA | 2 | 10 |\n"
               "| D | LoRA | 16 | 10 |\n| E | LoRA | 8 | 30 |\n\n"
               "Có thể chạy nhiều lượt liền nhau, hoặc chia mỗi thành viên một lượt nếu dùng chung một W&B project."),
        ("code", '''RUNS_TO_TRAIN = ["A"]
SEED = 42
WANDB_TEAM = None  # Điền tên team W&B nếu cả nhóm dùng chung một không gian
SAVE_EVERY_EPOCHS = None  # Bỏ trống để dùng mặc định: LoRA 1 epoch, full fine-tune 2 epoch

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
        ("md", "## Xem biểu đồ\n\nMở link W&B in ra trong lúc train. Biểu đồ theo epoch nằm ở nhóm `by_epoch`, "
               "gồm `dev_macro_f1`, `dev_accuracy`, `train_loss`, `epoch_seconds` và `gpu_peak_mb`. "
               "Chọn nhiều lượt trong bảng Runs để so trên cùng một biểu đồ."),
    ],
    "02_evaluate.ipynb": [
        ("md", "# 02. So trên dev và chấm test\n\nChọn cấu hình chỉ dựa trên dev, rồi chấm test một lần. "
               "Điểm test được lưu lại, lần chạy sau sẽ dùng kết quả cũ thay vì chấm lại."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## So các lượt trên dev"),
        ("code", '''from vihsd_lora.evaluate import compare_on_dev, evaluate_on_test, select_for_test

display(compare_on_dev(settings))
choice = select_for_test(settings)
print("Sẽ chấm test:", choice)'''),
        ("md", "## Chấm test\n\nChỉ chạy sau khi đã train xong các lượt cần so."),
        ("code", '''results = {name: evaluate_on_test(slug, settings) for name, slug in choice.items()}'''),
        ("md", "## So LoRA với full fine-tune trên cùng các mẫu test\n\n"
               "Nếu khoảng tin cậy của chênh lệch chứa 0 thì chưa đủ cơ sở để nói bên nào tốt hơn."),
        ("code", '''from vihsd_lora.report import compare_on_test

if {"lora", "full"} <= set(choice):
    print(compare_on_test(choice["lora"], choice["full"], settings))'''),
    ],
    "03_inference.ipynb": [
        ("md", "# 03. Inference\n\nDự đoán văn bản mới bằng model đã train, và kiểm tra rằng gộp LoRA vào trọng số "
               "gốc không làm đổi kết quả."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("code", '''from pathlib import Path
import pandas as pd
from vihsd_lora.evaluate import select_for_test
from vihsd_lora.inference import Predictor

SLUG = select_for_test(settings)["lora"]
predictor = Predictor(Path(settings.output_root) / "runs" / SLUG)
print("Dùng lượt:", SLUG)'''),
        ("md", "## Dự đoán\n\nThay danh sách dưới bằng câu cần thử."),
        ("code", '''TEXTS = [
    "Hôm nay thời tiết đẹp quá, cả nhà đi chơi vui nhé.",
    "Bài viết rất hữu ích, cảm ơn bạn đã chia sẻ.",
]
before = pd.DataFrame(predictor.predict(TEXTS))
display(before)'''),
        ("md", "## Gộp LoRA vào trọng số gốc\n\nSau khi gộp, kết quả phải gần như giữ nguyên. "
               "Đây là lý do LoRA không làm tăng độ trễ khi triển khai."),
        ("code", '''latency_adapter = predictor.latency_ms(TEXTS, batch_size=1)
predictor.merge_lora()
latency_merged = predictor.latency_ms(TEXTS, batch_size=1)
after = pd.DataFrame(predictor.predict(TEXTS))

print(f"Độ trễ mỗi câu: chưa gộp {latency_adapter} ms, đã gộp {latency_merged} ms")
print("Nhãn giữ nguyên:", (before["label"] == after["label"]).all())
display(after)'''),
    ],
    "04_results_for_slides.ipynb": [
        ("md", "# 04. Bảng và biểu đồ cho slide\n\nTạo bảng kết quả và các file PNG trong `outputs/results`. "
               "Biểu đồ nào thiếu dữ liệu sẽ được bỏ qua, nên có thể chạy lại sau mỗi lượt."),
        ("code", BOOTSTRAP),
        ("code", SESSION),
        ("md", "## Bảng kết quả\n\nGồm 13 dòng đã công bố cùng quy trình, mốc luôn đoán CLEAN và các lượt đã chấm test."),
        ("code", '''from vihsd_lora.report import export_tables, main_table

display(main_table(settings))
print(export_tables(settings))'''),
        ("md", "## Biểu đồ"),
        ("code", '''from IPython.display import Image, display as show
from vihsd_lora.plots import make_all

figures = make_all(settings)
for figure in figures:
    print(figure.name)
    show(Image(str(figure), width=760))'''),
        ("md", "## Đưa biểu đồ lên W&B\n\nTạo run `slide-figures` để cả nhóm xem chung."),
        ("code", '''from vihsd_lora.report import log_figures_to_wandb

log_figures_to_wandb(figures, settings)'''),
    ],
}


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nbformat.write(notebook(cells), OUTPUT / name)
        print("Đã tạo", name)


if __name__ == "__main__":
    main()
