# Kế hoạch L1: gọng kính adversarial

Ngày lập: 2026-11-01 (sửa lần 1 cùng ngày sau review: lighting enhancer, ánh xạ màu, điều kiện 1:N, loss theo max góc, detector, IAPMR, control giai đoạn A, transfer). Thuộc [`plan.md`](plan.md) (P4a, các thí nghiệm E1a, E1b, E1c). Các quyết định chốt ngày 2026-11-01 nằm ở [`decisions.md`](decisions.md).

## 1. Mục tiêu

Chứng minh bằng số đo rằng **liveness không trả lời được câu hỏi "đúng người"**. Kẻ tấn công là người thật (nên qua được anti-spoofing), đeo một gọng kính in hoa văn, và được nhận diện thành người khác (impersonation). Sau đó cho thấy kênh giọng (P1) chặn được tấn công này (E1c).

Dùng thêm tấn công né tránh (dodging, đeo kính để **không** bị nhận ra) làm bài kiểm tra dễ hơn, để kiểm chứng pipeline trước.

## 2. Hai giai đoạn

| | Giai đoạn A: kiểm chứng digital (không cần người hỗ trợ) | Giai đoạn B: vật lý với người hỗ trợ |
|---|---|---|
| Kẻ tấn công | Owner (ảnh webcam của owner) | Người hỗ trợ |
| Nạn nhân | Danh tính ảo: người trong dataset public, enroll vào DB bằng `enroll_public` | Owner (đã enroll mặt và giọng) |
| Kẻ tấn công có trong DB không | **Có.** Owner được enroll (5 góc, như người dùng thật), vì thực tế kẻ tấn công là bạn cùng lớp đã đăng ký | **Có.** Người hỗ trợ được enroll mặt (và giọng cho E1c) |
| Mục đích | Biết chắc phương pháp, pipeline và loss đúng trước khi tốn công in | Kết quả chính cho report: tấn công thật và chặn bằng giọng |
| Tầng đo | D, D+EOT, transfer, **control digital**; dodging trên owner | D, D+EOT, **P (vật lý)**, control; liveness; giọng (E1c) |
| Cổng chuyển sang B | E1a ≥ 80% ở tầng D (impersonation, 1:N có cả owner trong DB) trên ảnh giữ lại, **và** control digital thấp hơn rõ rệt (§6) | |

Nếu giai đoạn A không đạt 80% sau khi debug, áp dụng dự phòng của E1a ở `plan.md` §6 và ghi lại trong `experiment_log.md`.

Ghi chú: vì kẻ tấn công có trong DB, đối thủ mạnh nhất trong hạng margin (§5) chính là embedding của kẻ tấn công. Đây là điều kiện 1:N thật. Nếu bỏ kẻ tấn công khỏi DB thì con số sẽ lạc quan và không được dùng cho E1a.

## 3. Pipeline phải giống hệt backend

Tấn công chỉ có nghĩa khi tối ưu đúng trên đường đi mà frame thật phải qua (`backend/processing.py::process_attendance_frame`):

1. Webcam 640×480.
2. Frontend nén JPEG chất lượng 0.95.
3. Phát hiện mặt bằng `LivenessDetector` (MediaPipe Face Detection `model_selection=0`, chọn mặt có confidence cao nhất), cho ra bbox.
4. Liveness đã smooth và distance gate. Recognition chỉ chạy khi kết luận là real.
5. `crop_face_from_bbox` (nới 15%).
6. **`_enhance_inference_crop` (lighting enhancer).** Bước này bỏ qua khi ánh sáng bình thường. Khi tối hoặc chói, crop bị color scaling hoặc Zero-DCE biến đổi, theo một quality gate.
7. Resize về 112×112, chuẩn hoá về [−1, 1].
8. FaceCNN_ArcFace (method3) cho embedding 512 chiều.
9. `find_match`: lấy max cosine theo từng góc đã enroll, chọn người có điểm cao nhất, khớp nếu ≥ 0.5.

Nguyên tắc cài đặt:
- **Import trực tiếp** `backend.face_recognizer`, `backend.processing.crop_face_from_bbox`, `backend.processing._enhance_inference_crop` và `LivenessDetector`. Không viết lại, để không bị lệch pipeline.
- **Đánh giá** (mọi tầng) chạy qua chính `process_attendance_frame`, hoặc qua đúng chuỗi bước 3→9 ở trên, bằng code backend. Mỗi frame log thêm `lighting_enhancement.applied` và lý do, nhãn và điểm liveness, bbox.
- **Lighting enhancer lúc tối ưu:**
  - Mặc định: tối ưu và đánh giá D/D+EOT trên ảnh ánh sáng bình thường. Kiểm tra trên tập ảnh rằng enhancer báo `normal lighting; bypassed`; frame nào enhancer có tác động thì loại khỏi tập tối ưu và ghi số lượng.
  - Với điều kiện tối hoặc chói ở giai đoạn B: thêm color scaling (`enhance_color_scaling`, có thể viết lại bản lấy đạo hàm được hoặc dùng straight-through) vào EOT. Báo cáo riêng IAPMR theo `applied = true/false`.
- **JPEG không lấy đạo hàm được**, nên trong vòng tối ưu dùng JPEG ngẫu nhiên kiểu BPDA hoặc straight-through. Khi đánh giá thì dùng JPEG thật (`cv2.imencode`, chất lượng 95).
- **Resize trong bản lấy đạo hàm được** dùng `F.interpolate(mode="bilinear", align_corners=False, antialias=False)` cho giống `cv2.resize` INTER_LINEAR. Kiểm tra một lần: embedding từ pipeline torch và từ pipeline backend trên cùng 20 ảnh sạch phải có cosine ≥ 0.999; ghi kết quả vào `experiment_log.md`.
- **Bbox được phát hiện lại trên ảnh đã đeo kính** lúc đánh giá, vì chính gọng kính có thể làm bbox xê dịch hoặc làm mất mặt (đếm riêng số frame "No Face").
- **Thành công (impersonation) theo frame** = liveness đã smooth kết luận real **và** `find_match` trả về đúng nạn nhân, với điểm ≥ 0.5 và top-1 trong cả danh sách người đã đăng ký (1:N, có kẻ tấn công). Đây là IAPMR theo nghĩa ISO/IEC 30107-3. Ngoài ra báo cáo riêng hai thành phần: tỉ lệ qua liveness, và tỉ lệ match nạn nhân khi bỏ qua liveness.
- **Detector khi enroll:** `enroll_public.py` hiện dùng `model_selection=1` (khác backend). Với nạn nhân ảo và lớp mô phỏng, enroll lại bằng detector `model_selection=0` nếu phát hiện được mặt. Ảnh nào chỉ model 1 phát hiện được thì giữ, nhưng ghi số lượng. Không đổi tên tham số CLI có sẵn của `enroll_public.py`; nếu cần chọn detector thì thêm cờ mới (ví dụ `--detector-model 0`), mặc định giữ hành vi cũ.

Backend ghi điểm danh ngay khi **một frame** khớp (Quick mode). Vì vậy ngoài tỉ lệ thành công theo frame, còn báo cáo **"có ít nhất một frame thành công trong cửa sổ 3 s"**, vì đó là điều kiện để điểm danh hộ được ở Quick mode. Ở Secure mode, auto-log đã bị chặn (P2), nên chỉ số này không áp dụng; ở đó L1 được đo qua E1c.

## 4. Gọng kính

- **Template:** một ảnh gọng kính 2D chuẩn (khoảng 400×150 px), gồm hai vành mắt và cầu mũi. Vùng được tối ưu là chính phần gọng (mask), không phải tròng kính.
- **Đặt lên mặt:** lấy landmark FaceMesh (khoé mắt, sống mũi, thái dương) trên từng ảnh, rồi warp template theo phép phối cảnh. Đầu xoay thì gọng xoay theo.
- **Kích thước:** (sửa 2026-11-01 sau khi đo: gọng kính chỉ phủ tối đa khoảng 12% ảnh crop, xem `decisions.md`; ablation khoảng 4/8/12%.) Ban đầu đặt mục tiêu gọng chiếm **15–30% diện tích ảnh crop 112×112**. Có thể nới bản lề và độ dày gọng; >30% vẫn chấp nhận nếu cần (owner đồng ý). Lưu ý crop đã nới 15%, nên tỉ lệ so với khuôn mặt còn cao hơn con số này.
  - Luôn báo cáo % diện tích của từng texture.
  - **Ablation kích thước (nếu kịp):** 3 độ dày gọng (ví dụ 10%, 20%, 30% diện tích), mỗi mức tối ưu riêng. Vẽ đồ thị độ thành công tầng D theo diện tích.
- **Ảnh ra để in:** PNG theo kích thước thật của gọng (mm, 300 dpi), kèm hướng dẫn cắt và dán lên gọng nhựa rẻ.
- **Owner có đeo kính thật.** Ở giai đoạn A, gọng adversarial phủ lên gọng thật. Ở giai đoạn B, ghi lại người hỗ trợ có đeo kính hay không. Ghi luôn ảnh enroll của nạn nhân có kính hay không, vì việc đeo kính tự nó có thể làm tăng hoặc giảm similarity; control (§6) dùng để tách hiệu ứng này.

## 5. Tối ưu

Tham số là texture `T` của gọng, ở **không gian màu in** (qua sigmoid để giữ trong [0, 1]). Dùng Adam, khoảng 1–3 nghìn bước, mỗi bước một batch ảnh của kẻ tấn công nhân với các mẫu EOT.

- **Chuỗi biến đổi trong mỗi bước:** `T` (màu in) → `M(T)` (ánh xạ in → màu camera thấy, lấy đạo hàm được) → warp theo landmark → dán lên ảnh → EOT → pipeline torch (§3) → embedding.
- **Ký hiệu:** `E_B` là tập embedding đã enroll của nạn nhân (từng góc, như trong DB); `E_k` là tập của mỗi người khác `k`, gồm cả kẻ tấn công. `S(x, E) = τ · logsumexp(cos(f(x), e) / τ)` trên các `e ∈ E` (xấp xỉ mềm của max theo góc mà `find_match` dùng; τ khoảng 0.05–0.1).
- **Hàm loss (impersonation):**
  - `1 − S(x ⊕ T, E_B)`.
  - Cộng hạng margin: `max(0, S(x ⊕ T, E_k) − S(x ⊕ T, E_B) + m)` với mọi người khác `k` trong danh sách. Hạng này để thắng ở top-1 (1:N). Theo dõi riêng hạng margin với kẻ tấn công, vì đó thường là đối thủ khó nhất.
  - Cộng `λ_tv · TV(T) + λ_nps · NPS(T)`.
- **Hàm loss (dodging):** giảm `S(x ⊕ T, E_A)` (các embedding enroll của chính kẻ tấn công), cùng TV và NPS. Thành công dodging = không được match là kẻ tấn công (điểm < 0.5 hoặc top-1 là người khác).
- **EOT** (lấy mẫu ngẫu nhiên mỗi bước):
  - Hình học: dịch ±3 px, xoay ±5°, scale ±5% so với vị trí landmark, phối cảnh nhẹ.
  - Màu và ảnh: sáng/tương phản ±20%, lệch màu nhỏ, blur Gauss σ ≤ 1, JPEG q 0.6–0.95.
  - Khoảng cách: thu nhỏ còn 80–200 px rồi phóng lại 112.
  - (Chỉ cho điều kiện tối hoặc chói ở giai đoạn B) color scaling của lighting enhancer, xem §3.
- **Màu in được (NPS và ánh xạ màu):** in một bảng màu 10×10 ô, chụp lại bằng chính webcam đó, ở ánh sáng sẽ dùng khi thử. Từ ảnh chụp lấy ra:
  - tập màu in được (dùng cho NPS, tính trên `T`);
  - ánh xạ `M`: "màu in" sang "màu camera thấy" (ví dụ ma trận 3×3 cộng offset, hoặc LUT nội suy; phải lấy đạo hàm được). `M` được áp **trong vòng tối ưu**, trước khi dán lên mặt.
  - Vì vậy `T` sau tối ưu **chính là file in**, không cần đảo ngược ánh xạ. Không áp `M` lên `T` trước khi in.
  - Kiểm tra: in một texture thử, chụp lại, so màu chụp được với `M(T)`; ghi sai số trung bình vào `experiment_log.md`.
- **Tách tập:** ảnh kẻ tấn công chia thành tập tối ưu (khoảng 70%) và tập đánh giá (khoảng 30%, **khác buổi hoặc khác ánh sáng**). Mọi số báo cáo lấy trên tập đánh giá. Tổng cộng **khoảng 120–150 frame** cho mỗi kẻ tấn công (để tập đánh giá có ít nhất khoảng 40 frame). Báo cáo tỉ lệ thành công kèm khoảng tin cậy 95% (Wilson).

## 6. Đo lường

| Tầng | Cách làm | Số báo cáo |
|---|---|---|
| Baseline | Kẻ tấn công không đeo gì, qua đúng pipeline | Tỉ lệ bị match nhầm thành nạn nhân (zero-effort) và cosine với nạn nhân |
| D | Dán texture lên ảnh giữ lại, qua đúng pipeline | Tỉ lệ thành công theo frame (E1a, ngưỡng ≥ 80%), điểm cosine với nạn nhân, kèm khoảng tin cậy |
| **Control digital (giai đoạn A)** | Cùng mask, cùng vị trí, dán (1) texture ngẫu nhiên cùng dải màu và (2) gọng trơn (màu đều) | Tỉ lệ thành công của control. **Cổng sang B:** tầng D ≥ 80% và control ≤ 20% |
| D+EOT | Như D, thêm biến đổi chưa dùng lúc tối ưu (góc lớn hơn, ánh sáng khác) | Độ rơi so với D |
| Transfer | Texture tối ưu trên method3, thử trên method1/2/4/5 | Gọi đúng tên là **transfer cùng kiến trúc, khác cách train** (cả năm dùng chung backbone `FaceCNN`). Cần kiểm tra xem các method có chung dữ liệu train không, rồi ghi vào báo cáo |
| Transfer khác kiến trúc (tùy chọn) | Một model public khác hẳn, ví dụ facenet-pytorch InceptionResnetV1 (kiểm chứng license trước, ghi vào `verification_log.md`) | Chỉ khi có model này mới được kết luận "khai thác điểm chung giữa các model" |
| P (giai đoạn B) | Người hỗ trợ đeo kính in, trước webcam, qua backend thật | IAPMR (liveness real và match nạn nhân) theo frame và theo cửa sổ 3 s, ở 2 khoảng cách và 2 mức sáng (E1b). Tách theo `lighting_enhancement.applied` |
| Control vật lý | Gọng in hoa văn ngẫu nhiên (cùng kích thước, cùng dải màu) và gọng trơn | Tách hiệu ứng "đeo kính" khỏi hiệu ứng adversarial |
| Liveness | Anti-spoofing trên các frame đeo kính (adversarial, control, không kính) | Giấy in trên mặt có làm model nghi là giả không. Đã nằm trong IAPMR, nhưng vẫn báo cáo riêng |
| Giọng (E1c) | Người hỗ trợ đọc challenge, giọng so với giọng owner đã enroll | IAPMR sau fusion R3 ≤ 5%, BPCER tăng ≤ 5 điểm % |

## 7. Code sẽ viết (`innovation/attacks/adv_glasses/`)

| File | Việc |
|---|---|
| `pipeline.py` | Bản lấy đạo hàm được của crop → resize → chuẩn hoá → FaceCNN, gọi lại code backend. Có hàm tự kiểm tra cosine ≥ 0.999 so với pipeline backend (§3) |
| `glasses.py` | Template gọng, mask, đặt lên mặt theo landmark FaceMesh, đo % diện tích |
| `eot.py` | Các biến đổi EOT, JPEG kiểu BPDA, color scaling của enhancer (tùy chọn) |
| `printability.py` | Tạo bảng màu để in, đọc ảnh chụp bảng màu, tính NPS, fit ánh xạ `M` (in → camera) lấy đạo hàm được |
| `optimize.py` | Vòng tối ưu impersonation và dodging, loss theo logsumexp trên các góc; lưu texture (không gian màu in), log loss (tách theo từng hạng, cả margin với kẻ tấn công), config |
| `evaluate.py` | Đo các tầng baseline, D, control digital, D+EOT, transfer, P, control vật lý, liveness, **qua đúng code backend** (gồm lighting enhancer và `LivenessDetector`); log từng frame; ghi `results/innovation/<run>/` |
| `export.py` | Xuất trực tiếp `T` ra ảnh in đúng kích thước (300 dpi) kèm thước đo để kiểm tra tỉ lệ khi in |

Thêm ở frontend: **chế độ chụp frame trong `/lab`**. Chụp N frame qua đúng đường webcam → JPEG 0.95, lưu vào `data/innovation/frames/<người>/<buổi>/`, để ảnh tấn công cùng phân bố với ảnh thật lúc điểm danh.

Chạy được trên CPU để thử nhanh. Lần tối ưu chính chạy trên Kaggle hoặc vast.ai (method3 nhỏ, dự kiến dưới 1 giờ trên một GPU).

## 8. Việc cần chuẩn bị

**Giai đoạn A:**
- [ ] Checkpoint thật `method3_best.pth` (và method1, 2, 4, 5 cho transfer), đặt trong `checkpoints/`.
- [ ] Owner enroll mặt vào DB (5 góc, qua trang Register như người dùng thật).
- [ ] Owner chụp **khoảng 120–150 frame** qua chế độ chụp của Lab: 2 buổi hoặc 2 mức sáng, nhiều góc đầu. Kiểm tra lighting enhancer bypass trên tập tối ưu (§3).
- [ ] Nạn nhân ảo: vài ảnh mặt của 1–3 người từ VoxCeleb1 test (owner tải; trang tải bị chặn từ môi trường cloud của Claude).
  - Kiểm tra trước: những người này **không được nằm trong tập train của method3**. Nếu dataset môn học lấy từ VGGFace2 thì có thể trùng người với VoxCeleb.
  - Dự phòng: lấy danh tính từ phần `verification_data` của dataset môn học (người mà model chưa thấy lúc train).
- [ ] Lớp mô phỏng: khoảng 50 danh tính public enroll bằng `enroll_public` (detector `model_selection=0`, §3), để điều kiện top-1 là 1:N thật.
- [ ] In bảng màu và chụp lại bằng webcam (cần cho `M` ngay ở giai đoạn A, để texture của A in được).

**Giai đoạn B:**
- [ ] Người hỗ trợ ký đồng ý (dữ liệu lưu cục bộ, xoá sau môn học).
- [ ] **Buổi 1:** enroll mặt người hỗ trợ; **khoảng 120–150 frame** qua Lab; cộng khoảng 20 clip giọng đọc chuỗi số **ghi qua trình duyệt trên cùng laptop** (vừa làm impostor cùng kênh để chốt ngưỡng giọng, vừa dùng cho E1c).
- [ ] Máy in màu, giấy mờ; in lại bảng màu nếu đổi máy in hoặc giấy.
- [ ] 2–3 gọng kính nhựa rẻ (một cho bản adversarial, một cho bản control).
- [ ] **Buổi 2:** thử vật lý ở 2 khoảng cách, 2 mức sáng; mỗi điều kiện khoảng 30 lần thử, cả bản adversarial và control.

## 9. Đạo đức
- Chỉ owner và người hỗ trợ có đồng ý.
- Texture adversarial không công bố, không đưa lên git (`data/` và `results/` của texture nằm ngoài repo hoặc bị ignore).
- Nạn nhân ảo chỉ dùng embedding, không sinh ảnh hay giọng mới của họ.
