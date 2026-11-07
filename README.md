# LoRA trên ViHSD

Repo cho phần thực nghiệm của seminar LoRA: so sánh LoRA với full fine-tune trên ViSoBERT cho bài toán phát hiện ngôn từ thù ghét tiếng Việt (ViHSD). Kế hoạch đầy đủ nằm ở [docs/ke-hoach-seminar-lora-vihsd.md](docs/ke-hoach-seminar-lora-vihsd.md).

## Cấu trúc

```text
notebooks/   Các notebook cần chạy, đánh số theo thứ tự
src/         Code train, đánh giá, inference, bảng và biểu đồ
data/        Số liệu đã công bố (published_results.csv), không chứa dữ liệu ViHSD
docs/        Kế hoạch seminar
tests/       Kiểm thử chạy trên CPU với model tí hon và dữ liệu giả
tools/       Script sinh lại notebook
```

## Chuẩn bị

1. **Hugging Face:** đăng nhập, mở [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd) và chấp nhận điều kiện. Sau đó tạo một token quyền Read trong phần Settings > Access Tokens.
2. **W&B:** tạo tài khoản tại [wandb.ai](https://wandb.ai) và lấy API key ở trang [authorize](https://wandb.ai/authorize).
3. **Secret:** thêm `HF_TOKEN` và `WANDB_API_KEY`.
   - Colab: mở biểu tượng chìa khóa ở thanh bên trái, thêm hai secret và bật quyền truy cập cho notebook.
   - Kaggle: mở Add-ons > Secrets, thêm hai secret và bật cho notebook.
4. **GPU:** trên Colab chọn Runtime > Change runtime type, rồi chọn T4 hoặc L4. Trên Kaggle chọn GPU T4 và bật Internet.
5. **Repo:** sửa `REPO_URL` ở ô đầu tiên của mỗi notebook thành địa chỉ repo GitHub. Nếu repo private, tạo một fine-grained token chỉ có quyền đọc repo này và thêm vào Colab với tên `GITHUB_TOKEN`.

## Thứ tự chạy

| Notebook | Việc thực hiện | Thời gian ước tính |
|---|---|---|
| `00_setup_and_check` | Kiểm tra GPU, dữ liệu, số tham số, đo tốc độ thật | Khoảng 10–15 phút |
| `01_train` | Train từng lượt A, B, C, D, E | 30–70 phút mỗi lượt 10 epoch trên T4 |
| `02_evaluate` | So trên dev, chọn cấu hình, chấm test | Vài phút |
| `03_inference` | Dự đoán câu mới, kiểm tra gộp LoRA | Vài phút |
| `04_results_for_slides` | Xuất bảng và biểu đồ PNG | Dưới 2 phút |

Các con số trên là ước tính. Notebook `00` sẽ cho thời gian thật theo GPU của bạn.

Nếu mỗi thành viên chạy một lượt, hãy chép các thư mục `outputs/runs/<tên lượt>` về chung một Drive trước khi chạy notebook `02` và `04`.

## Các lượt train

| Lượt | Cách train | r | alpha | Learning rate | Epoch |
|---|---|---|---|---|---|
| A | LoRA | 8 | 8 | 5e-4 | 10 |
| B | Full fine-tune | | | 2e-5 | 10 |
| C | LoRA | 2 | 8 | 5e-4 | 10 |
| D | LoRA | 16 | 8 | 5e-4 | 10 |
| E | LoRA | 8 | 8 | 5e-4 | 30 |

Mọi lượt dùng batch 40, max length 128, warmup 6%, FP16 và chọn epoch tốt nhất theo macro F1 trên dev.

## Lưu trên Drive và train tiếp

Trên Colab, notebook tự mount Drive và lưu mọi thứ quan trọng lên Drive, không để trong `/content`.

- **Checkpoint để train tiếp:** LoRA lưu mỗi epoch (vài MB), full fine-tune lưu cách 2 epoch (khoảng 1.2GB mỗi lần, gồm model và optimizer). Drive chỉ giữ checkpoint mới nhất. Đổi khoảng cách bằng `SAVE_EVERY_EPOCHS` trong notebook `01`.
- **Model tốt nhất:** lưu vào `best/` ngay khi macro F1 trên dev tăng, độc lập với checkpoint, nên không mất epoch tốt nhất dù checkpoint lưu thưa.
- **Bị ngắt phiên:** mở lại notebook `01`, chạy lại từ đầu với cùng lượt. Quá trình tiếp tục từ checkpoint gần nhất và nối vào cùng run trên W&B. Lượt đã train xong sẽ được bỏ qua.
- **Thùng rác Drive:** checkpoint cũ bị xóa có thể nằm trong thùng rác và vẫn chiếm dung lượng. Sau lượt full fine-tune, hãy làm trống thùng rác Drive.
- **File tạm của W&B:** để ở `/content/wandb_local`, vì dữ liệu đã được đồng bộ lên W&B. Trên Drive chỉ giữ `wandb_id.txt` để nối lại đúng run.

## Kết quả nằm ở đâu

Trên Colab, kết quả lưu trong Drive tại `MyDrive/vihsd-lora/outputs`. Trên Kaggle, kết quả nằm ở `/kaggle/working/outputs`.

```text
outputs/
  runs/<tên lượt>/
    best/                 Adapter LoRA hoặc model đầy đủ có macro F1 dev cao nhất
    best_state.json       Điểm và epoch của model trong best/
    checkpoints/          Checkpoint để train tiếp, tự xóa khi train xong
    history.csv           Số liệu theo epoch
    run_summary.json      Tham số, thời gian, VRAM, dung lượng, điểm dev
    test_metrics.json     Điểm test, khoảng tin cậy, confusion matrix
    predictions_test.csv  Nhãn thật và nhãn dự đoán, không chứa văn bản
  results/
    main_results.csv      Bảng kết quả
    main_results.md       Bảng dạng markdown để dán vào report
    figures/              Biểu đồ PNG cho slide
```

## Biểu đồ trên W&B

Project mặc định là `vihsd-lora-seminar`. Nếu cả nhóm dùng chung một team W&B, điền tên team vào `WANDB_TEAM` trong notebook `01_train`.

- Nhóm `by_epoch` chứa số liệu theo epoch: `dev_macro_f1`, `dev_accuracy`, `train_loss`, `dev_loss`, `epoch_seconds`, `gpu_peak_mb`.
- Nhóm `eval` và `train` chứa số liệu theo step do Trainer ghi.
- Trong bảng Runs, chọn nhiều lượt để so trên cùng một biểu đồ.
- Mở menu ở góc biểu đồ để tải PNG.

Notebook `04` cũng tạo sẵn các biểu đồ cùng màu cho slide, không phụ thuộc W&B: đường macro F1, train loss, ảnh hưởng của rank, macro F1 theo số tham số, F1 từng nhãn, chi phí, và confusion matrix.

## Chạy bằng dòng lệnh

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m vihsd_lora.train --run A --seed 42
python -m vihsd_lora.evaluate --slug A_lora_r8_e10_s42
```

Có thể dùng dữ liệu cục bộ bằng cách đặt `VIHSD_DATA_DIR` tới thư mục chứa `train.csv`, `dev.csv`, `test.csv`.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Lỗi 401 hoặc 403 khi tải dữ liệu | Kiểm tra đã chấp nhận điều kiện ở `uitnlp/vihsd` và `HF_TOKEN` đúng |
| Báo số mẫu không khớp | Dừng lại và kiểm tra nguồn dữ liệu. Không nên bỏ qua vì kết quả sẽ không so được với bảng số liệu |
| Không tải được `uitnlp/visobert` | Đặt `VIHSD_MODEL_NAME=xlm-roberta-large`. Thời gian sẽ dài hơn |
| Hết VRAM | Đặt `VIHSD_BATCH_SIZE` nhỏ hơn và ghi rõ thay đổi trong report |
| Colab ngắt phiên | Chạy lại `01_train` với cùng lượt để train tiếp |
| Lỗi import ngay sau khi cài thư viện | Chọn Runtime > Restart session rồi chạy lại từ đầu |
| `Found an incompatible version of torchao` | Ô đầu tiên đã tự gỡ torchao. Nếu vẫn gặp, chạy `!pip uninstall -y torchao`, chọn Runtime > Restart session rồi chạy lại |
| Không muốn dùng W&B | Đặt `WANDB_MODE=disabled`, `history.csv` và biểu đồ PNG vẫn được tạo |

## Kiểm thử

```bash
pip install -r requirements.txt pytest
python -m pytest tests -q
```

## Trạng thái kiểm thử

Toàn bộ 22 test và 5 notebook đã chạy thành công trên CPU với model tí hon và dữ liệu giả, W&B ở chế độ offline. Số tham số của ViSoBERT đã được kiểm tra trên kiến trúc thật dựng từ config. Chưa kiểm thử trên GPU thật, với ViHSD thật, với trọng số ViSoBERT thật, và chưa đẩy lên W&B online. Vì vậy hãy chạy notebook `00` trước.

## Lưu ý về dữ liệu

ViHSD chỉ dùng cho nghiên cứu và chứa bình luận xúc phạm thật. Repo không lưu dữ liệu, và `predictions_test.csv` không chứa văn bản. Không commit dữ liệu lên git, và hạn chế đưa nguyên văn bình luận lên slide.
