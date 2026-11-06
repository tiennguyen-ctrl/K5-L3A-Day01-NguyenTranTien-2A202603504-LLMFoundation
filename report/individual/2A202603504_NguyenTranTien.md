# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603504**
- **Họ và tên: Nguyễn Trần Tiến**
- **SEED 6010** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.859 | 0.857 | 0.348 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.943 | 0.974 | 4.654 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.898 | 0.974 | 81.218 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.944 | 1.000 | 66.963 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.943 | 1.000 | 45.831 | 0.000 |
| 4a. LLM nhỏ zero-shot | — | 1.000 | 1299.214 | 0.007 |
| 4b. LLM lớn zero-shot | — | 1.000 | 1170.340 | 0.066 |
| 4c. LLM nhỏ few-shot | — | 1.000 | 2011.747 | 0.026 |

*LLM chỉ đo trên SUB 40 tin, nên không có F1 (test). Độ trễ phụ thuộc máy Colab (CPU/GPU) ở mỗi lần chạy.**

## 3. Dự đoán và thực tế

**1. `ml_lr` — Learning rate**
- Đoán: lr=0.001 loss giảm rất chậm; lr=50 loss dao động, không giảm mượt.
- Kết quả: lr=0.001 đúng (0.69 → 0.627 sau 300 epoch). Nhưng lr=50 chỉ vọt lên ~2.6 ở vài epoch đầu rồi giảm rất nhanh xuống 0.007, thấp hơn cả lr=0.5 (0.086).
- Vì sao lệch: loss của logistic regression là hàm lồi (chỉ một đáy) và tập train gần như tách được bằng từ khoá, nên sau cú vượt đáy ban đầu, bước lớn đẩy trọng số lên rất lớn → mô hình cực tự tin, loss train ≈ 0. Loss train thấp không có nghĩa là tổng quát hoá tốt hơn.

**2. `dl_curve` — MLP so với LogReg**
- Đoán: F1 của MLP xấp xỉ LogReg vì cả hai đều dùng BoW.
- Kết quả: MLP đạt F1 test 0.943 so với 0.859, hơn ~0.08 (recall tăng từ 0.76 lên 0.91) — lớn hơn nhiều so với nhiễu ngẫu nhiên.
- Vì sao lệch: LogReg tự code chưa hội tụ hết (loss cuối vẫn ~0.086), còn MLP dùng Adam nên tối ưu tốt hơn. Thí nghiệm bỏ ReLU vẫn cho F1 0.932, cho thấy phần lớn khoảng cách đến từ việc tối ưu, không phải từ hidden layer.

**3. `hard_eval` — 50 case khó**
- Đoán: LLM đạt ~85% và khó nhất ở nhóm prompt injection; với BoW khó nhất là né từ khoá và "tưởng spam nhưng không phải".
- Kết quả: LLM (cả 4a và 4d) đạt 100% ở mọi nhóm, kể cả prompt injection. BoW đạt 100% ở nhóm "tưởng spam" nhưng chỉ 10–25% ở spam giọng tự nhiên và thao túng.
- Vì sao lệch: model mới đã được huấn luyện chống các câu ra lệnh đơn giản (dù 6 case là quá ít để kết luận an toàn). BoW có xu hướng đoán ham khi không thấy từ khoá spam, nên các nhóm ham dễ, còn spam không có từ khoá thì lọt qua.

Ngoài dự đoán: với `max_completion_tokens=3`, LLM trả về chuỗi rỗng (`finish_reason=length`, `reasoning_tokens=0`) → F1 = 0, invalid = 40/40; phải nâng giới hạn token mới chạy đúng.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất:**  BoW + MLP — F1 từ 0.974 (SUB thường) xuống 0.389 trên 50 case khó (accuracy 0.56). BoW + LogReg cũng chỉ đạt accuracy 0.54. Trong khi đó encoder đạt 0.60–0.74 và LLM (4a, 4d) đạt 1.00.

Các nhóm BoW tụt mạnh nhất: né từ khoá (0–17%), spam giọng tự nhiên** (10–20%) và thao túng / lừa đảo (12–25%). Ngược lại, BoW đạt 100% ở nhóm "tưởng spam nhưng không phải" và "nói về spam" — vì mô hình có xu hướng đoán ham khi không thấy từ khoá spam quen thuộc.

**Case 1 — #25 (thao túng, thật = spam):** "Netflix: your payment was declined. Update your…"
- Cơ chế sai: BoW nhìn văn bản như các token rời rạc. Tin không có từ spam mạnh (free, call, txt, prize); các từ *payment*, *update*, *declined* lại hay gặp trong tin giao dịch bình thường → trọng số kéo về ham. BoW không hiểu ý đồ mạo danh thương hiệu để lừa người nhận bấm link.

**Case 2 — nhóm né từ khoá (thật = spam):** tin kiểu "C0ngr4tulat10ns… F R E E…"
- Cơ chế sai: `tokenize` biến `C0ngr4tulat10ns` thành một token không có trong vocab, còn `F R E E` bị tách thành 4 chữ cái đơn lẻ, không khớp từ "free". Vector BoW gần như trống → mô hình chỉ còn dựa vào bias, nghiêng về ham (lớp chiếm ~87%).

**Kết luận:** F1 trên test thường (~0.94–0.97) quá lạc quan, vì test cùng phân phối với train và còn chứa spam mẫu gần trùng nhau. Bộ 50 case khó cho thấy BoW chỉ học thuộc từ khoá; embedding chịu được né từ khoá tốt hơn (kNN đạt 74%), còn LLM đọc được ý đồ nên đúng toàn bộ.

## 5. Thử thách cá nhân

**Đề #5 (SEED 6010) — Embedding:** Thiết kế 4 tin nhắn kiểm thử độc lập (2 tin spam né từ khoá / l33tspeak, 2 tin ham chứa từ khoá nhạy cảm như "free", "win"). So sánh dự đoán giữa BoW + Logistic Regression và Embedding + Logistic Regression.

### Cách làm

Xây dựng 4 test case mang tính đánh lừa:
- Case 0 & 1 (`obfuscation_spam_1`, `obfuscation_spam_2`): spam cố tình biến đổi ký tự (`W!n`, `r1ght`, `cl1ck1ng`, `Fr33`, `ch3ck`) để qua mặt bộ lọc từ khoá.
- Case 2 & 3 (`ham_with_free`, `ham_with_win`): tin bình thường nhưng chứa từ khoá hay gặp trong spam (*free coupon*, *win first prize*).

Đưa 4 tin qua hai pipeline:
- BoW + LogReg: biểu diễn văn bản theo sự xuất hiện của từng từ trong từ điển cố định.
- Encoder all-MiniLM-L6-v2 + LogReg: biểu diễn văn bản thành vector ngữ nghĩa 384 chiều, nắm ý nghĩa toàn câu trước khi phân loại.

### Kết quả

| STT | Tin nhắn | Nhãn thật | BoW | Embedding + LogReg |
|---|---|---|---|---|
| 0 | W!n 1000$ r1ght n0w by cl1ck1ng h3r3! | spam | ❌ ham | ✅ spam |
| 1 | Fr33 cr3d1t card 0nline, no ch3ck r3qu1r3d. | spam | ❌ ham | ✅ spam |
| 2 | Grab sent me a free coupon code for my morning coffee ride! | ham | ❌ spam | ✅ ham |
| 3 | If we work hard on the embedded firmware, our team will win first prize! | ham | ❌ spam | ✅ ham |

### Giải thích

- Tin 0 & 1 (spam né từ khoá): BoW đoán nhầm ham vì các từ "win", "right", "click", "free" đã bị biến đổi mặt chữ, không khớp token nào trong từ điển nên mô hình không thấy tín hiệu spam. Embedding + LogReg đoán đúng spam nhờ encoder nắm được cấu trúc ngữ nghĩa và ý đồ mời gọi ẩn sau các ký tự cách điệu.
- Tin 2 & 3 (ham chứa từ khoá nhạy cảm): BoW đoán nhầm spam vì "sập bẫy" các từ đơn lẻ như *free*, *win* — vốn có trọng số rất cao về phía spam. Embedding + LogReg đoán đúng ham nhờ nắm ngữ cảnh xung quanh (*Grab*, *coffee ride*, *embedded firmware*), phân biệt được đây là mã giảm giá dịch vụ hoặc ngữ cảnh công việc chứ không phải tin lừa đảo.

## 6. Phương án cho sản phẩm thật

**Đạt p95 < 50 ms:** BoW + LogReg (0.35 ms), BoW + MLP (4.65 ms) và 3c Encoder + MLP head (45.8 ms — sát ngưỡng). Các phương án 3a (81.2 ms), 3b (67.0 ms) và toàn bộ LLM (1.170–2.012 ms) đều vượt SLA.

**Chọn: BoW + MLP.**
- Chất lượng: F1 test 0.943 — bằng 3c (0.943), xấp xỉ 3b (0.944), hơn hẳn BoW + LogReg (0.859).
- Độ trễ: p95 4.65 ms, dư khoảng 10 lần so với SLA, còn 3c chỉ dư ~4 ms nên dễ vượt khi tải tăng.
- Thông lượng: ~6.600 tin/s (cả lô), dư rất nhiều so với ~116 tin/s cần thiết; chạy trên CPU, không tốn phí API.

**Chi phí nếu dùng LLM** (10 triệu tin/ngày ≈ 300 triệu tin/tháng = 300.000 × 1.000 tin):
- LLM nhỏ zero-shot: $0.007 × 300.000 ≈ $2.100/tháng
- LLM nhỏ few-shot: $0.026 × 300.000 ≈ $7.800/tháng
- LLM lớn: $0.066 × 300.000 ≈ $19.800/tháng

Dù chịu chi phí, LLM vẫn không đạt SLA vì độ trễ ~1–2 giây/tin.

**Đánh đổi chấp nhận:** BoW + MLP yếu trước tin "ngụy trang" — chỉ đạt 56% trên 50 case khó (3c đạt 64%, LLM đạt 100%), đặc biệt kém ở nhóm né từ khoá và spam giọng tự nhiên. Để bù, đề xuất router: BoW + MLP xử lý mọi tin; chỉ tin "lưỡng lự" (0.3 ≤ p ≤ 0.7) mới chuyển sang encoder hoặc LLM nhỏ với prompt phòng thủ. Giữ tỉ lệ tin lưỡng lự dưới 5% để p95 vẫn dưới 50 ms và chi phí LLM chỉ còn vài chục USD/tháng.

## 7. AI log và điều đã học

### AI log (Claude)

| Phần | Hỏi gì | Nhận gì | Tôi sửa / tự làm gì |
|---|---|---|---|
| 0 | Cách lưu bản sao Colab, chạy các ô đầu, điền STUDENT_ID | Hướng dẫn thao tác giao diện | Tự thao tác |
| 0–1 | Giải thích data leakage, accuracy khi dữ liệu lệch lớp | Giải thích khái niệm | Tự viết lại câu trả lời ngắn gọn |
| 1–3 (vùng tự code) | Hướng làm TODO 1.1–1.4, 2.1–2.2, 3.1–3.3 | AI đưa các bước bằng lời, ví dụ công cụ (np.zeros, enumerate…) không liên quan bài, và chỉ ra lỗi trong code tôi viết | Tự viết code; sửa các lỗi AI chỉ ra: lồng hàm thừa, `vocab-size`, docstring sai chỗ, thiếu dấu phẩy dict, thụt lề, `otp.step()`, quên xoá `raise`, thiếu `return`, sai tên biến `prediction(s)` |
| 1–3 | Giải thích các câu hỏi (gradient của "free", lr=50, overfitting, đếm tham số MLP, BCE, vì sao hidden layer ít giúp trên embedding, bắt lỗi 1–2, checkpoint A–B) | Giải thích + số liệu đối chiếu với output của tôi | Rút gọn, diễn đạt lại khi điền vào notebook |
| 1–5 | Gợi ý cho các ô `predict(...)` | Gợi ý cách suy luận + câu mẫu | Chỉnh lại theo dự đoán của mình |
| 4 (được dùng AI) | Viết `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` | AI viết code và giải thích| Tự viết bản `parse_label` và prompt tiếng Việt trước; sau đó chuyển sang prompt tiếng Anh của AI. Đã review: key đọc từ biến môi trường, tham số qua `provider_kwargs`, xử lý content rỗng |
| 4 (debug) | Vì sao F1 = 0, invalid = 40/40 | AI hướng dẫn thêm dòng debug tạm thời in `repr(resp)`, `finish_reason`, `usage` | Phát hiện: `reasoning_effort="none"` đã gửi, `reasoning_tokens=0`, nhưng với 3 token model trả về rỗng (`finish_reason=length`). Đổi prompt sang tiếng Anh vẫn rỗng → nâng giới hạn token trong `PROVIDERS` thì chạy đúng (invalid = 0). Đã xoá dòng debug |
| 5 | Câu hỏi 4–6, bắt lỗi 3, Pareto, phản ánh cuối bài, tính tay | AI giải thích dựa trên bảng số liệu của tôi | Đối chiếu số liệu, tự tính lại trên giấy cho `hand_calc.jpg` |
| 5 (bonus) | Viết case riêng | AI không viết sẵn, chỉ gợi ý hướng và góp ý nhãn | Tự viết 5 case |
| Thử thách cá nhân | Hướng viết code | AI chỉ dẫn các bước viết code | Tự làm; AI chỉ nhận xét 4 tin sau khi tôi báo thầy đã nhận xét, và định dạng markdown phần báo cáo (sửa 2 chỗ sai: BoW là 0/1 chứ không phải tần suất; MiniLM là encoder nhỏ, không phải LLM) |
| Báo cáo | Soạn mục 2, 3, 4, 6 | AI soạn nháp từ số liệu của tôi | Đối chiếu với notebook, chỉnh phần dự đoán cho đúng câu tôi đã ghi |

### Điều đã học

- Điểm F1 cao trên tập test thường có thể rất lạc quan: các model nhỏ đạt ~0.94–1.0 nhưng tụt xuống 0.39–0.64 trên 50 case khó, vì chúng học thuộc từ khoá chứ không hiểu ý đồ.
- Model mạnh nhất chưa chắc là lựa chọn đúng: LLM đạt 100% nhưng chậm ~1–2 giây/tin và tốn tiền; với SLA 50 ms, BoW + MLP (hoặc router kết hợp nhiều tầng) mới là phương án thực tế.
- Khi gọi LLM phải đo và debug thay vì tin vào cấu hình: giới hạn 3 token khiến model trả về rỗng, và chỉ khi in `finish_reason` / `usage` tôi mới tìm ra nguyên nhân.