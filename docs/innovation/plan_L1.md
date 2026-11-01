# Kế hoạch L1: gọng kính adversarial

Ngày lập: 2026-11-01. Thuộc [`plan.md`](plan.md) (P4a, các thí nghiệm E1a, E1b, E1c). Các quyết định chốt ngày 2026-11-01 nằm ở [`decisions.md`](decisions.md).

## 1. Mục tiêu

Chứng minh bằng số đo rằng **liveness không trả lời được câu hỏi "đúng người"**. Kẻ tấn công là người thật (nên qua được anti-spoofing), đeo một gọng kính in hoa văn, và được nhận diện thành người khác (impersonation). Sau đó cho thấy kênh giọng (P1) chặn được tấn công này (E1c).

Dùng thêm tấn công né tránh (dodging, đeo kính để **không** bị nhận ra) làm bài kiểm tra dễ hơn, để kiểm chứng pipeline trước.

## 2. Hai giai đoạn

| | Giai đoạn A: kiểm chứng digital (không cần người hỗ trợ) | Giai đoạn B: vật lý với người hỗ trợ |
|---|---|---|
| Kẻ tấn công | Owner (ảnh webcam của owner) | Người hỗ trợ |
| Nạn nhân | Danh tính ảo: người trong dataset public, enroll vào DB bằng `enroll_public` | Owner (đã enroll mặt và giọng) |
| Mục đích | Biết chắc phương pháp, pipeline và loss đúng trước khi tốn công in | Kết quả chính cho report: tấn công thật và chặn bằng giọng |
| Tầng đo | D, D+EOT, transfer; dodging trên owner | D, D+EOT, **P (vật lý)**, control; liveness; giọng (E1c) |
| Cổng chuyển sang B | E1a ≥ 80% ở tầng D (impersonation) trên ảnh giữ lại | |

Nếu giai đoạn A không đạt 80% sau khi debug, áp dụng dự phòng của E1a ở `plan.md` §6 và ghi lại trong `experiment_log.md`.

## 3. Pipeline phải giống hệt backend

Tấn công chỉ có nghĩa khi tối ưu đúng trên đường đi mà frame thật phải qua:

1. Webcam 640×480.
2. Frontend nén JPEG chất lượng 0.95.
3. MediaPipe Face Detection (bbox).
4. `crop_face_from_bbox` (nới 15%).
5. Resize về 112×112, chuẩn hoá về [−1, 1].
6. FaceCNN_ArcFace (method3) cho embedding 512 chiều.
7. `find_match`: lấy max cosine theo từng góc đã enroll, chọn người có điểm cao nhất, khớp nếu ≥ 0.5.

Nguyên tắc cài đặt:
- **Import trực tiếp** `backend.face_recognizer` và `backend.processing.crop_face_from_bbox`, không viết lại, để không bị lệch pipeline.
- **JPEG không lấy đạo hàm được**, nên trong vòng tối ưu dùng JPEG ngẫu nhiên kiểu BPDA hoặc straight-through. Khi đánh giá thì dùng JPEG thật.
- **Bbox được phát hiện lại trên ảnh đã đeo kính** lúc đánh giá, vì chính gọng kính có thể làm bbox xê dịch.
- **Thành công = `find_match` trả về đúng nạn nhân**, với điểm ≥ 0.5 và top-1 trong cả danh sách người đã đăng ký (1:N), không chỉ vượt ngưỡng.

Backend ghi điểm danh ngay khi **một frame** khớp. Vì vậy ngoài tỉ lệ thành công theo frame, còn báo cáo **"có ít nhất một frame thành công trong cửa sổ 3 s"**, vì đó mới là điều kiện để tấn công điểm danh hộ được.

## 4. Gọng kính

- **Template:** một ảnh gọng kính 2D chuẩn (khoảng 400×150 px), gồm hai vành mắt và cầu mũi. Vùng được tối ưu là chính phần gọng (mask), không phải tròng kính.
- **Đặt lên mặt:** lấy landmark FaceMesh (khoé mắt, sống mũi, thái dương) trên từng ảnh, rồi warp template theo phép phối cảnh. Đầu đọc khi xoay đầu, gọng xoay theo.
- **Kích thước:** gọng chiếm **15–30% diện tích ảnh crop 112×112**. Có thể nới bản lề và độ dày gọng; >30% vẫn chấp nhận nếu cần (owner đồng ý).
- **Ảnh ra để in:** PNG theo kích thước thật của gọng (mm, 300 dpi), kèm hướng dẫn cắt và dán lên gọng nhựa rẻ.
- **Owner có đeo kính thật.** Ở giai đoạn A, gọng adversarial phủ lên gọng thật. Ở giai đoạn B, ghi lại người hỗ trợ có đeo kính hay không.

## 5. Tối ưu

Tham số là texture `T` của gọng (qua sigmoid để giữ trong [0, 1]). Dùng Adam, khoảng 1–3 nghìn bước, mỗi bước một batch ảnh của kẻ tấn công nhân với các mẫu EOT.

- **Hàm loss (impersonation):**
  - `1 − cos(f(x ⊕ T), e_B)`, với `e_B` là trung bình các embedding đã enroll của nạn nhân.
  - Cộng thêm hạng margin: `max(0, cos(f(x ⊕ T), e_k) − cos(f(x ⊕ T), e_B) + m)` với mọi người khác `k` trong danh sách. Hạng này để thắng ở top-1 (1:N).
  - Cộng `λ_tv · TV(T) + λ_nps · NPS(T)`.
- **Hàm loss (dodging):** giảm `cos(f(x ⊕ T), e_A)` (embedding của chính kẻ tấn công), cùng TV và NPS.
- **EOT** (lấy mẫu ngẫu nhiên mỗi bước):
  - Hình học: dịch ±3 px, xoay ±5°, scale ±5% so với vị trí landmark, phối cảnh nhẹ.
  - Màu và ảnh: sáng/tương phản ±20%, lệch màu nhỏ, blur Gauss σ ≤ 1, JPEG q 0.6–0.95.
  - Khoảng cách: thu nhỏ còn 80–200 px rồi phóng lại 112.
- **Màu in được (NPS):** in một bảng màu 10×10 ô, chụp lại bằng chính webcam đó. Từ ảnh chụp lấy ra:
  - tập màu in được (dùng cho NPS);
  - ánh xạ "màu in" sang "màu camera thấy", áp vào texture trước khi in. Đây là "mã màu" cuối cùng.
- **Tách tập:** ảnh kẻ tấn công chia thành tập tối ưu (khoảng 70%) và tập đánh giá (khoảng 30%, khác buổi hoặc khác ánh sáng). Mọi số báo cáo lấy trên tập đánh giá.

## 6. Đo lường

| Tầng | Cách làm | Số báo cáo |
|---|---|---|
| D | Dán texture lên ảnh giữ lại, qua đúng pipeline | Tỉ lệ thành công theo frame (E1a, ngưỡng ≥ 80%), điểm cosine với nạn nhân |
| D+EOT | Như D, thêm biến đổi chưa dùng lúc tối ưu (góc lớn hơn, ánh sáng khác) | Độ rơi so với D |
| Transfer | Texture tối ưu trên method3, thử trên method1/2/4/5 | Tấn công có khai thác điểm chung giữa các model không |
| P (giai đoạn B) | Người hỗ trợ đeo kính in, trước webcam, qua backend thật | IAPMR theo frame và theo cửa sổ 3 s, ở 2 khoảng cách (E1b) |
| Control | Gọng in hoa văn ngẫu nhiên (cùng kích thước, cùng dải màu) và gọng trơn | Tách hiệu ứng "đeo kính" khỏi hiệu ứng adversarial |
| Liveness | Chạy anti-spoofing trên các frame đeo kính | Giấy in trên mặt có làm model nghi là giả không |
| Giọng (E1c) | Người hỗ trợ đọc challenge, giọng so với giọng owner đã enroll | IAPMR sau fusion R3 ≤ 5%, BPCER tăng ≤ 5 điểm % |

## 7. Code sẽ viết (`innovation/attacks/adv_glasses/`)

| File | Việc |
|---|---|
| `pipeline.py` | Bản lấy đạo hàm được của crop → resize → chuẩn hoá → FaceCNN, gọi lại code backend |
| `glasses.py` | Template gọng, mask, đặt lên mặt theo landmark FaceMesh, đo % diện tích |
| `eot.py` | Các biến đổi EOT, JPEG kiểu BPDA |
| `printability.py` | Tạo bảng màu để in, đọc ảnh chụp bảng màu, tính NPS và ánh xạ màu |
| `optimize.py` | Vòng tối ưu impersonation và dodging; lưu texture, log loss, config |
| `evaluate.py` | Đo các tầng D, D+EOT, transfer, control, liveness; ghi `results/innovation/<run>/` |
| `export.py` | Ảnh in đúng kích thước (300 dpi) kèm thước đo để kiểm tra tỉ lệ khi in |

Thêm ở frontend: **chế độ chụp frame trong `/lab`**. Chụp N frame qua đúng đường webcam → JPEG 0.95, lưu vào `data/innovation/frames/<người>/<buổi>/`, để ảnh tấn công cùng phân bố với ảnh thật lúc điểm danh.

Chạy được trên CPU để thử nhanh. Lần tối ưu chính chạy trên Kaggle hoặc vast.ai (method3 nhỏ, dự kiến dưới 1 giờ trên một GPU).

## 8. Việc cần chuẩn bị

**Giai đoạn A:**
- [ ] Checkpoint thật `method3_best.pth` (và method1, 2, 4, 5 cho transfer), đặt trong `checkpoints/`.
- [ ] Owner chụp khoảng 60 frame qua chế độ chụp của Lab: 2 buổi hoặc 2 mức sáng, nhiều góc đầu.
- [ ] Nạn nhân ảo: vài ảnh mặt của 1–3 người từ VoxCeleb1 test (owner tải; trang tải bị chặn từ môi trường cloud của Claude).
  - Kiểm tra trước: những người này **không được nằm trong tập train của method3**. Nếu dataset môn học lấy từ VGGFace2 thì có thể trùng người với VoxCeleb.
  - Dự phòng: lấy danh tính từ phần `verification_data` của dataset môn học (người mà model chưa thấy lúc train).
- [ ] Lớp mô phỏng: khoảng 50 danh tính public enroll bằng `enroll_public`, để điều kiện top-1 là 1:N thật.

**Giai đoạn B:**
- [ ] Người hỗ trợ ký đồng ý (dữ liệu lưu cục bộ, xoá sau môn học).
- [ ] **Buổi 1:** khoảng 60 frame qua Lab, cộng khoảng 20 clip giọng đọc chuỗi số **ghi qua trình duyệt trên cùng laptop** (vừa làm impostor cùng kênh để chốt ngưỡng giọng, vừa dùng cho E1c).
- [ ] Máy in màu, giấy mờ; in bảng màu và chụp lại bằng webcam đó.
- [ ] 2–3 gọng kính nhựa rẻ (một cho bản adversarial, một cho bản control).
- [ ] **Buổi 2:** thử vật lý ở 2 khoảng cách, 2 mức sáng; mỗi điều kiện khoảng 30 lần thử, cả bản adversarial và control.

## 9. Đạo đức
- Chỉ owner và người hỗ trợ có đồng ý.
- Texture adversarial không công bố, không đưa lên git (`data/` và `results/` của texture nằm ngoài repo hoặc bị ignore).
- Nạn nhân ảo chỉ dùng embedding, không sinh ảnh hay giọng mới của họ.
