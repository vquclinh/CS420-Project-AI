# Kế hoạch seminar LoRA: phát hiện ngôn từ thù ghét tiếng Việt trên ViHSD

Cập nhật: 04/11/2026

## 1. Tóm tắt quyết định

| Hạng mục | Quyết định |
|---|---|
| Task | Phân loại bình luận mạng xã hội tiếng Việt thành CLEAN, OFFENSIVE, HATE |
| Dataset | ViHSD: 24,048 train, 2,672 dev, 6,680 test |
| Model | `uitnlp/visobert`, 97.6M tham số |
| Phương pháp | LoRA trên `query` và `value`, so với full fine-tune cùng model |
| Metric | Macro F1 là metric chính, kèm Accuracy, trên test |
| Phần cứng | T4 hoặc L4 đều đủ |
| Thời gian train | 5 lượt: 3.5–8 giờ trên T4, 1.2–3 giờ trên L4 (ước tính, chưa đo) |
| Thời gian inference | Khoảng 1 phút cho mỗi lần chấm test (ước tính, chưa đo) |
| Mục tiêu | LoRA cách full fine-tune không quá 1 điểm Acc và 2 điểm macro F1, chỉ train khoảng 0.91% tham số |

## 2. Vì sao chọn ViHSD

- **Chưa thấy công bố LoRA trên ViHSD.** Tính đến 04/11/2026, nhóm chưa tìm thấy paper hay model công khai nào làm việc này. Đây là "chưa tìm thấy", không phải "chắc chắn chưa ai làm", nên khi trình bày cần nói đúng như vậy.
- **Trường hợp gần nhất không trùng.** "ViHateT5 with LoRA" dùng LoRA trên một dataset tin tức khác có 2,962 mẫu. "Crossing Linguistic Horizons" dùng LoRA trên dữ liệu tổng quát, ViHSD chỉ dùng để đánh giá.
- **Có sẵn nhiều số liệu.** Bảng chính có 13 model cùng một quy trình, ngoài ra còn 23 dòng tham khảo từ 3 paper khác.
- **Dữ liệu lấy được.** ViHSD có trên Hugging Face, không phụ thuộc vào trang web của lab.
- **Nhẹ.** Model chỉ 97.6M tham số và dữ liệu là bình luận, nên train được nhiều lượt trong giới hạn 10 giờ.

**Lý do bỏ các hướng trước:**

- **ViNLI:** trang dữ liệu của UIT NLP Group đang không truy cập được.
- **DreamBooth + LoRA:** DreamBench++ và Diffu-LoRA đã công bố kết quả, và ước tính cần 11–16 giờ trên T4 cho một lượt đủ 30 chủ thể.

## 3. Dataset

ViHSD có 33,400 bình luận từ Facebook và YouTube, về giải trí, người nổi tiếng, xã hội và chính trị. Tên riêng đã được loại bỏ để ẩn danh.

| Nhãn | Ý nghĩa | Train | Dev | Test |
|---|---|---|---|---|
| CLEAN | Không có nội dung quấy rối | 19,886 | 2,190 | 5,548 |
| OFFENSIVE | Có nội dung quấy rối, nhưng không nhắm vào đối tượng cụ thể | 1,606 | 212 | 444 |
| HATE | Xúc phạm trực tiếp một cá nhân hoặc nhóm người | 2,556 | 270 | 688 |
| **Tổng** | | **24,048** | **2,672** | **6,680** |

Dữ liệu lệch nhãn mạnh: CLEAN chiếm 82.71% toàn bộ dataset. Độ đồng thuận giữa người gán nhãn ở mức κ = 0.52.

**Cách lấy dữ liệu:** dùng `uitnlp/vihsd` trên Hugging Face, cần đăng nhập và chấp nhận điều kiện. README của tác giả ghi dữ liệu chỉ dùng cho nghiên cứu.

**Yêu cầu trước khi train:**

- Đếm số mẫu từng nhãn ở từng split, phải khớp bảng trên. Số lượng này cũng giúp xác định từng `label_id` ứng với nhãn nào.
- Giữ nguyên train/dev/test, không chia lại.
- Không tiền xử lý văn bản, để khớp quy trình của bảng chính.
- Không dùng `visolex/ViHSD`, vì bản này chia split khác (khoảng 26K/3.3K/3.3K).
- `phucdev/ViHSD` chỉ có 24,046 mẫu train, chỉ dùng làm dự phòng và phải ghi rõ chênh lệch 2 mẫu.

## 4. Metric

- **Macro F1:** metric chính, tính F1 riêng từng nhãn rồi lấy trung bình. Metric này phản ánh năng lực trên hai nhãn hiếm.
- **Accuracy:** metric thứ hai. Không dùng một mình, vì luôn đoán CLEAN đã đạt 83.05.

```python
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
```

## 5. Bảng số liệu

### 5.1. Bảng chính: so sánh trực tiếp được

Số lấy từ Table 4 của paper ViSoBERT: Accuracy và macro F1, cùng tác giả chạy với 10 epoch, batch 40, max length 128, learning rate 2e-5, không tiền xử lý. Paper không gọi tên split, nhưng ghi split dataset là 24,048/2,672/6,680.

| Hệ thống | Acc | Macro F1 | Tham số train |
|---|---|---|---|
| ViSoBERT | 88.51 | 68.77 | 97.6M |
| PhoBERT Large | 87.32 | 65.14 | 369.2M |
| TwHIN-BERT Large | 87.23 | 65.23 | Cần đếm |
| XLM-R Large | 87.15 | 65.13 | 559.9M |
| PhoBERT Base | 87.12 | 65.01 | 135.0M |
| vELECTRA | 86.96 | 63.95 | Cần đếm |
| TwHIN-BERT Base | 86.63 | 63.67 | Cần đếm |
| XLM-R Base | 86.36 | 63.39 | 278.0M |
| XLM-T | 86.22 | 63.48 | Cần đếm |
| Bernice | 86.12 | 64.32 | Cần đếm |
| viBERT | 85.34 | 62.07 | 115.4M |
| mBERT cased | 83.55 | 60.62 | 177.9M |
| mBERT uncased | 83.38 | 58.92 | 167.4M |
| Luôn đoán CLEAN (nhóm tự tính) | 83.05 | 30.25 | 0 |
| **ViSoBERT full fine-tune (nhóm)** | ? | ? | 97.6M |
| **ViSoBERT + LoRA r=16 (nhóm)** | ? | ? | 1.18M |
| **ViSoBERT + LoRA r=8 (nhóm)** | ? | ? | 0.89M |
| **ViSoBERT + LoRA r=2 (nhóm)** | ? | ? | 0.67M |

Cột tham số do nhóm tính từ `config.json`, gồm đầu phân loại 3 nhãn, trừ các ô "Cần đếm". Paper không ghi mã checkpoint, nên đây là giả định theo checkpoint phổ biến. Các ô còn lại đếm bằng `model.num_parameters()`, không cần train.

### 5.2. Bảng phụ: cùng dữ liệu, khác quy trình

Chỉ dùng để tham khảo, không xếp hạng chung với bảng chính.

**ViHateT5 (4 epoch, batch 16, max length 256, learning rate 2e-5, không tiền xử lý):**

| Hệ thống | Acc | Macro F1 |
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

**Paper gốc ViHSD (4 epoch, batch 16, max length 100, có tiền xử lý):**

| Hệ thống | Acc | Macro F1 |
|---|---|---|
| BERT cased | 86.88 | 62.69 |
| Text CNN + fastText | 86.69 | 61.11 |
| BERT uncased | 86.60 | 62.38 |
| DistilBERT | 86.22 | 62.42 |
| XLM-R Base | 86.12 | 61.28 |
| GRU + fastText | 85.41 | 60.47 |

**PhoBERT-CNN (batch 64, learning rate 2e-5, có tiền xử lý, không tăng cường dữ liệu):**

| Hệ thống | Acc | Macro F1 |
|---|---|---|
| PhoBERT-CNN | 87.17 | 64.43 |
| PhoBERT | 87.13 | 63.51 |
| Text-CNN + PhoW2V word | 86.11 | 63.01 |
| Bi-LSTM + PhoW2V word | 85.99 | 62.66 |
| XLMR-CNN | 85.48 | 63.34 |
| BERT | 84.52 | 60.29 |
| XLM-R | 83.62 | 62.38 |
| RoBERTa | 83.04 | 61.49 |

Paper PhoBERT-CNN còn 10 dòng khác ở Table 6. Paper SMTCE không dùng vì chỉ có macro F1.

### 5.3. Paper chứa số liệu

| Paper | Dùng cho | Số dòng |
|---|---|---|
| ViSoBERT, arXiv 2310.11166 | Bảng chính | 13 |
| ViHateT5, Findings of ACL 2024 | Bảng phụ | 9 |
| Luu et al. 2021 (paper gốc ViHSD) | Bảng phụ | 6 |
| PhoBERT-CNN, arXiv 2206.00524 | Bảng phụ | 8 |

## 6. Model và cấu hình train

Dùng `uitnlp/visobert` vì đây là model tốt nhất của bảng chính, có quy trình train công khai và được pre-train cho văn bản mạng xã hội tiếng Việt. Model có 12 lớp, 768 chiều, kiến trúc XLM-R. Nếu không tải được thì chuyển sang `xlm-roberta-large`, là dòng XLM-R tốt nhất của bảng chính, và tính lại tham số cùng thời gian.

| Lượt | Ưu tiên | Cách train | r | Learning rate | Epoch | Tham số train | Dung lượng lưu |
|---|---|---|---|---|---|---|---|
| A | 1 | LoRA | 8 | 5e-4 | 10 | 887,811 | Khoảng 3.6MB |
| B | 2 | Full fine-tune | Không có | 2e-5 | 10 | 97,568,259 | Khoảng 390MB |
| C | 3 | LoRA | 2 | 5e-4 | 10 | 666,627 | Khoảng 2.7MB |
| D | 4 | LoRA | 16 | 5e-4 | 10 | 1,182,723 | Khoảng 4.7MB |
| E | 5 | LoRA | 8 | 5e-4 | 30 | 887,811 | Khoảng 3.6MB |

**Mỗi lượt** là một lần train trọn vẹn và độc lập: nạp lại model pre-train, train đủ số epoch, chấm dev sau mỗi epoch và giữ epoch tốt nhất.

Lượt E tồn tại vì 10 epoch có thể chưa đủ với LoRA: paper LoRA train RoBERTa Base 25–80 epoch tùy tác vụ.

**Cấu hình chung:**

- Batch 40, max length 128, padding động theo batch. Nếu hết VRAM thì giảm batch và ghi lại.
- AdamW, warmup 6%, learning rate giảm tuyến tính, FP16.
- Không tiền xử lý, không dùng trọng số lớp.
- Chọn epoch tốt nhất bằng macro F1 trên dev, seed 42.
- `lora_alpha=8` giữ nguyên cho cả ba giá trị r.
- Với `TaskType.SEQ_CLS`, PEFT tự đưa đầu phân loại vào phần được train.

**Nguồn cấu hình:** r, alpha, learning rate LoRA và warmup theo cấu hình RoBERTa Base của paper LoRA. Batch, max length, số epoch và learning rate 2e-5 theo paper ViSoBERT. Dropout, seed và cách chọn epoch do nhóm tự chọn.

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

**Các bước thực hiện:**

1. Lấy dữ liệu và đếm nhãn theo mục 3.
2. Chạy thử 200 step để đo tốc độ và VRAM, rồi tính lại tổng thời gian.
3. Chạy các lượt theo thứ tự ưu tiên, lưu checkpoint lên Drive sau mỗi epoch, đặt `save_total_limit=1`.
4. So các lượt LoRA bằng macro F1 trên dev để chọn cấu hình tốt nhất.
5. Chấm test đúng một lần cho LoRA đã chọn và cho full fine-tune.
6. Nếu còn giờ, chạy thêm 2 seed cho cấu hình LoRA tốt nhất và full fine-tune để báo cáo trung bình.

**Dữ liệu cần ghi lại ở mỗi lượt:** train loss, Acc và macro F1 trên dev theo epoch, thời gian mỗi epoch, `torch.cuda.max_memory_allocated()`, kết quả `print_trainable_parameters()`, dung lượng file lưu, F1 từng nhãn và confusion matrix.

## 7. Thời gian

| Việc | T4 | L4 |
|---|---|---|
| Một lượt 10 epoch, gồm chấm dev | 30–70 phút | 10–25 phút |
| Một lượt 30 epoch, gồm chấm dev | 1.5–3.5 giờ | 30–75 phút |
| Cả 5 lượt | 3.5–8 giờ | 1.2–3 giờ |
| Chấm test một model, gồm nạp model | Khoảng 1 phút | Dưới 1 phút |
| Thêm 2 seed cho 2 cấu hình (mỗi lượt 10 epoch) | 2–4.7 giờ | 40–100 phút |

Tất cả là ước tính, chưa đo thực tế. Cách tính xuất phát từ benchmark BERT trên T4 (257 giây mỗi epoch, batch 16, độ dài 128), coi ViSoBERT có chi phí tương đương BERT-base, rồi giả định FP16 và câu ngắn giúp nhanh hơn 1.5–3.5 lần. L4 lấy nhanh hơn T4 2.8 lần theo benchmark của McCormick. Chấm test giả định nhanh hơn train 3 lần.

Colab không đảm bảo giữ phiên, nên phải lưu checkpoint và dùng `resume_from_checkpoint` khi bị ngắt. Nếu có nhiều thành viên, mỗi người có thể chạy một lượt trên tài khoản riêng.

## 8. Cơ hội vượt

Đây là đánh giá định tính của nhóm, chưa có số đo. So sánh theo macro F1 của bảng chính.

| Mốc so sánh | Macro F1 | Khả năng vượt | Lý do |
|---|---|---|---|
| Luôn đoán CLEAN | 30.25 | Gần như chắc chắn | Mốc tối thiểu |
| mBERT, viBERT, XLM-R Base, XLM-T, TwHIN-BERT Base, vELECTRA, Bernice | 58.92–64.32 | Cao | Thấp hơn ViSoBERT ít nhất 4.45 điểm |
| PhoBERT Base, PhoBERT Large, XLM-R Large, TwHIN-BERT Large | 65.01–65.23 | Khá | LoRA có thể thấp hơn ViSoBERT tới 3.5 điểm mà vẫn vượt |
| ViSoBERT full fine-tune | 68.77 | Ngang ngửa, không chắc | Paper LoRA có tác vụ thắng và có tác vụ thua full fine-tune |

Trên RoBERTa Base, paper LoRA báo cáo trung bình GLUE 87.2 so với 86.4 của full fine-tune. Tuy nhiên kết quả không đồng đều: SST-2 là 95.1 so với 94.8, MNLI là 87.5 so với 87.6, QQP là 90.8 so với 91.9.

Không đặt mục tiêu đứng đầu bảng. Thành công là LoRA đạt gần full fine-tune với số tham số train nhỏ hơn khoảng 110 lần.

## 9. Discussion trên bảng kết quả

**Theo từng kết quả có thể xảy ra:**

- **LoRA gần bằng full fine-tune (đạt mục tiêu ở mục 1):** kết luận đánh đổi có lợi, vì giảm khoảng 110 lần tham số train và dung lượng lưu mà gần như không mất hiệu suất.
- **LoRA cao hơn full fine-tune:** không kết luận LoRA tốt hơn nếu chênh ít và chỉ có một seed, vì macro F1 trên dữ liệu lệch nhãn dao động mạnh. Muốn khẳng định phải chạy thêm seed.
- **LoRA thấp hơn rõ rệt:** xem lượt E trước để biết có phải do thiếu epoch không. Sau đó bàn về giới hạn của cấu hình: embedding và feed-forward bị đóng băng, chỉ `query` và `value` được điều chỉnh.

**Các góc phân tích nên có:**

- **Hiệu suất theo tham số:** vẽ macro F1 theo số tham số train, thang log, gồm các dòng bảng chính và các lượt của nhóm.
- **Ảnh hưởng của rank:** so r=2, 8, 16 để xem tăng tham số có còn tăng điểm không.
- **Ảnh hưởng của số epoch:** so lượt A với lượt E để xem LoRA có cần train lâu hơn full fine-tune không.
- **Đầu phân loại:** ở r=8, đầu phân loại chiếm 592,899 trên 887,811, tức 66.8% tham số train. Chỉ tính phần LoRA thì 294,912 tham số, khoảng 0.30% model. Khi so với paper phải nói rõ cách tính.
- **Lệch nhãn:** so F1 từng nhãn và confusion matrix giữa LoRA và full fine-tune, tập trung vào OFFENSIVE và HATE.
- **Giới hạn của dữ liệu:** độ đồng thuận κ = 0.52 cho thấy ranh giới giữa các nhãn vốn không rõ, nên macro F1 khó tiến gần tuyệt đối.
- **Chi phí thật:** so thời gian mỗi epoch, VRAM và dung lượng file giữa lượt A và B. Không nên kỳ vọng LoRA nhanh hơn nhiều, vì gradient vẫn phải đi qua toàn bộ mạng. Adapter nhỏ nhưng vẫn cần model gốc khi chạy.
- **Độ tái lập:** so full fine-tune của nhóm với 88.51 / 68.77. Lệch nhiều thì phải kiểm tra lại trước khi kết luận về LoRA.
- **Quy trình ảnh hưởng đến điểm:** cùng mBERT cased, macro F1 là 60.62, 62.69 và 64.44 ở ba paper. Đây là lý do chỉ xếp hạng trong bảng chính.

**Giới hạn phải nêu:** một seed cho hầu hết các lượt, paper không ghi mã checkpoint, paper không gọi tên split, các dòng bảng phụ khác quy trình, và "chưa thấy công bố" không đồng nghĩa với "chưa ai làm".

## 10. Trình bày trước lớp

| Slide | Nội dung | Ý chính cần nói |
|---|---|---|
| 1 | Vấn đề | Full fine-tune phải train và lưu toàn bộ tham số cho mỗi tác vụ |
| 2 | Ý tưởng LoRA | Đóng băng trọng số gốc, chỉ học phần thay đổi có hạng thấp |
| 3 | Công thức | `h = W₀x + (α/r)·BAx`, với `A` khởi tạo ngẫu nhiên, `B` bằng 0 |
| 4 | Gắn vào Transformer | Gắn vào `query` và `value`, phần còn lại đóng băng, trừ đầu phân loại |
| 5 | Vì sao chọn ViHSD | Có sẵn bảng số liệu, dữ liệu lệch nhãn, chưa thấy công bố LoRA |
| 6 | Thiết kế thí nghiệm | 5 lượt, cùng dữ liệu, cùng cách chấm, chỉ khác cách train |
| 7 | Quá trình train | Đường loss, macro F1 dev, kết quả `print_trainable_parameters()`, VRAM, thời gian |
| 8 | Bảng kết quả | Bảng chính, đặt dòng của nhóm cạnh ViSoBERT |
| 9 | Discussion | Đánh đổi hiệu suất và tham số, rank, số epoch, nhãn hiếm |
| 10 | Kết luận | Kết quả đạt được, giới hạn và hướng làm tiếp |

**Cách giải thích các khái niệm:**

- **Vì sao hạng thấp đủ dùng:** paper giả thuyết rằng thay đổi trọng số khi thích nghi với tác vụ mới có "hạng nội tại" thấp, nên một tích hai ma trận nhỏ có thể biểu diễn phần lớn thay đổi đó.
- **Vì sao tiết kiệm:** thay vì học ma trận `d × d`, ta học `A` cỡ `r × d` và `B` cỡ `d × r`, nên số tham số giảm từ `d²` xuống `2dr`.
- **Vai trò của `r` và `α`:** `r` quyết định sức biểu diễn của phần cập nhật, còn `α/r` điều chỉnh độ lớn của nó.
- **Vì sao `B` bằng 0:** tích `BA` bằng 0 lúc đầu, nên model bắt đầu đúng như model pre-train.
- **Vì sao đầu phân loại vẫn train:** đầu phân loại 3 nhãn là mới, chưa có trọng số pre-train.
- **Vì sao learning rate khác nhau:** full fine-tune chỉ chỉnh nhẹ trọng số đã tốt, còn ma trận LoRA học từ đầu nên cần bước lớn hơn.
- **Vì sao không thêm độ trễ:** khi triển khai có thể gộp `BA` vào `W₀`, nên kích thước mạng và thời gian suy luận không tăng.
- **Vì sao macro F1 là metric chính:** dữ liệu lệch nhãn nên Accuracy cao vẫn có thể bỏ sót hai nhãn cần phát hiện.

**Câu hỏi nên chuẩn bị:**

- Vì sao không so trực tiếp với ViHateT5: vì nó thuộc bảng khác quy trình. Trong bảng của nó, ViHateT5 cao hơn ViSoBERT, nhưng không nên so chéo số giữa hai bảng.
- Vì sao vẫn tự chạy full fine-tune: để có thời gian, VRAM và kiểm tra dữ liệu cùng cách chấm.
- Có khẳng định chưa ai làm không: không, chỉ khẳng định chưa tìm thấy công bố tính đến ngày tìm kiếm.

**Lưu ý nội dung:** dataset chứa bình luận xúc phạm thật, nên hạn chế đưa nguyên văn lên slide. Khi cần minh họa thì che từ ngữ xúc phạm, và không đăng lại dữ liệu công khai.

## 11. Việc cần chốt

- [ ] Tải `uitnlp/vihsd` và đếm khớp mục 3.
- [ ] Xác nhận tải được `uitnlp/visobert`.
- [ ] Xác nhận với giảng viên việc dùng ViHSD cho phần thực nghiệm.
- [ ] Chạy thử 200 step để thay các ước tính ở mục 7 bằng số thật.
- [ ] Mở PDF ViSoBERT và ViHateT5 để đối chiếu bảng trước khi đưa lên slide.
- [ ] Chia người chạy từng lượt và người phụ trách slide, report.

## Nguồn

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
- [sonlam1102/vihsd trên GitHub](https://github.com/sonlam1102/vihsd)
- [uitnlp/visobert](https://huggingface.co/uitnlp/visobert)
- [GPU Benchmarks for Fine-Tuning BERT](https://mccormickml.com/2020/07/21/gpu-benchmarks-for-fine-tuning-bert/)
- [Colab GPUs Features & Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
