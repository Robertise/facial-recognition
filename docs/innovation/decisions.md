# Quyết định (innovation)

Mỗi mục: ngày, quyết định, lý do, hệ quả. Hướng đã loại thì ghi ở cuối, để không đề xuất lại.

## 2026-10-31. Trả lời các câu hỏi §11 của spec

| # | Quyết định | Hệ quả |
|---|---|---|
| Q1 | Repo đã tách khỏi team; owner có toàn quyền sửa, nâng cấp mọi phần | Bỏ ràng buộc "không sửa file của người khác". Claude commit lên nhánh riêng, owner review và merge |
| Q2 | Challenge đọc chữ số bằng tiếng Anh | ASR dùng model `.en`; dataset đọc số tiếng Anh (Speech Commands, AudioMNIST) |
| Q3 | Chỉ owner tham gia, cộng người ngoài khoảng một buổi mỗi tuần, cộng dữ liệu public | Thiết kế lại vai trò và cách chia tập (plan §2) |
| Q4 | GPU: vast.ai (5070 Ti, 3090, 4090, 5090), vài giờ mỗi job; hoặc Kaggle chạy qua đêm | Job P4 phải chạy xong trong vài giờ trên một GPU |
| Q5 | Không cần báo trước cho giảng viên (dự án cá nhân) | Đạo đức vẫn giữ: chỉ sinh deepfake của owner, không công bố texture adversarial |
| Q6 | Hạn: vài tuần; làm càng nhanh càng tốt, owner luôn sẵn sàng | Lịch khoảng 3–4 tuần (plan §5) |
| Q7 | Chốt các ngưỡng ở spec §6. Trượt thì chuyển sang phương án dự phòng; mọi lần trượt ghi vào `experiment_log.md` để đưa vào report | Bảng dự phòng ở plan §6 |

## 2026-10-31. Giao diện tiếng Anh
Thay cho spec §8.7 (tiếng Việt). Theme Console, đã có từ commit d930530.

## 2026-10-31. Nạn nhân L1 là danh tính ảo từ dataset public
Owner là kẻ tấn công nên không thể đồng thời là nạn nhân. Nạn nhân B lấy mặt và giọng của cùng một người trong VoxCeleb1, enroll vào DB bằng script. Không sinh media mới của B; kính chỉ tối ưu theo embedding của B.

## 2026-10-31. Train trên dữ liệu public, test trên owner
Thay cho việc chia tập theo người trong team: train và dev trên danh tính public, test trên owner (sau khi có prereg) và người ngoài.

## 2026-10-31. Ngưỡng giọng tạm: 0.40 (thay cho 0.25)
Đo trên AudioMNIST (12 người, giọng thật): ngưỡng 0.25 cho FAR 21%. Điểm EER ≈ 0.44; impostor p99 = 0.40; genuine p05 = 0.49. Chọn 0.40 làm giá trị tạm, ưu tiên ít từ chối nhầm người thật. **Ngưỡng chính thức chốt trong `prereg.md`**, dựa trên dữ liệu dev của owner (khác buổi, khác micro), vì AudioMNIST thu trong cùng một buổi nên cho kết quả lạc quan.

## 2026-10-31. Cách bật feature flag
`NEXT_PUBLIC_INNOVATION` nhận 1/true/yes/on (trước chỉ nhận "1", nên owner đặt `true` thì không thấy gì). Backend đọc `.env` ở thư mục gốc repo, vì cú pháp `KEY=1 python ...` không chạy được trên PowerShell hay cmd của Windows.

## 2026-11-01. L1 làm bằng gọng kính, chia hai giai đoạn
- **Dạng tấn công:** gọng kính in hoa văn (không dán thẳng lên mặt), gọng chiếm 15–30% diện tích ảnh crop mặt; lớn hơn vẫn chấp nhận nếu cần.
- **Giai đoạn A:** owner là kẻ tấn công, nạn nhân là danh tính ảo (VoxCeleb1, hoặc phần verification của dataset môn học). Chỉ đo digital, để kiểm chứng phương pháp.
- **Giai đoạn B:** người hỗ trợ là kẻ tấn công, owner là nạn nhân. Có đo vật lý, control và giọng.
- **Thêm dodging** làm bài kiểm tra dễ hơn trước.
- **Lịch người hỗ trợ:** buổi 1 chụp frame và ghi giọng cùng kênh; buổi 2 thử vật lý.
- Chi tiết: `plan_L1.md`.

## 2026-11-01. Ngưỡng giọng 0.40 sai với giọng thật của owner
Trên 20 clip của owner (`results/innovation/20261101_140348_voice_scores/`), ngưỡng 0.40 cho FRR 65%; ngưỡng 0.25–0.30 cho FRR 5%, FAR 0%. Điểm genuine thấp chủ yếu vì **lệch kênh thu** (enroll qua trình duyệt, test bằng file .m4a), không phải vì cao độ: 5 clip owner cố ý đọc giọng cao có điểm trung bình 0.39, bằng 15 clip giọng thường (0.391). Hướng xử lý:
- enroll 5 mẫu chia 2 buổi;
- enroll và verify cùng kênh (trình duyệt);
- chốt ngưỡng trong `prereg.md` sau khi có clip impostor cùng kênh từ người hỗ trợ.

Tạm thời chưa đổi giá trị trong `config.json`.

## 2026-11-01. Độ phủ 15–30% không đạt được với gọng kính
Đo trên ảnh webcam thật của owner (crop của backend nới 15%): gọng thường phủ khoảng **4%** ảnh crop 112×112; gọng dày khoảng 8%; gọng ngoại cỡ, dày gần che kín vùng quanh mắt, khoảng **12%**. Muốn 15–30% thì gọng phải che cả má và trán, tức thành mặt nạ chứ không còn là kính.

Quyết định: độ phủ là tham số (`--thickness`, `--rim-scale`). Ablation 3 mức khoảng 4%, 8%, 12%; mặc định khoảng 8%. Mỗi lần chạy ghi độ phủ thật trong `attack.json`. Nếu owner vẫn muốn ≥ 15% thì phải đổi dạng phụ kiện (ví dụ kính kèm mặt nạ hoặc khẩu trang in), và đó là một thí nghiệm khác.

## 2026-11-01. Sửa sau review code L1 giai đoạn A
- **method5 là ViT, không phải FaceCNN.** Checkpoint method5 có 157 key (`backbone.cls_token`, `pos_embed`, `patch_embed`…), là `FaceViT_ArcFace` trong `training_max.ipynb` (commit đầu `39324a0`): ViT-B/16 của timm, Linear 768→512 không bias, BN, L2; đầu vào 224×224, chuẩn hoá ImageNet. Trước đây `FaceRecognizer` load mọi checkpoint vào `FaceCNN` với `strict=False`, nên method5 chạy bằng trọng số ngẫu nhiên mà không báo gì. Nay `FaceRecognizer` nhận dạng kiến trúc theo key (ViT / ArcFace / Classifier / Triplet), dùng đúng class, đúng preprocessing (`Preprocess(size, mean, std)`), suy `num_classes` từ shape.
- **Load strict khi nghiên cứu.** Ở research mode (và luôn luôn trong các script L1) thiếu checkpoint hoặc có `missing_keys`/`unexpected_keys` thì dừng hẳn. App thường vẫn chạy như cũ nhưng in cảnh báo kèm danh sách key.
- **Kết luận transfer khác kiến trúc chỉ dựa trên method5.** method1/2/4 là cùng kiến trúc `FaceCNN`, khác cách train.
- **Max mềm bỏ log(n).** `τ·(logsumexp − log n)` lệch theo số ảnh: 10 ảnh trừ 0.115, 5 ảnh trừ 0.080 (τ = 0.05), tức margin lệch 0.035 giữa các người. Thay bằng trung bình có trọng số softmax `Σ softmax(cos/τ)·cos`: luôn ≤ max thật, tiến về max khi τ → 0, thêm ảnh xấu gần như không đổi điểm (test: < 0.001, bản cũ 0.035).
- **Gallery 5 ảnh mỗi người**, đúng 5 góc như Register (trước: tới 10). Người có ít hơn 5 ảnh dùng được bị cảnh báo và ghi `gallery_report.uneven`.
- **Liveness trong evaluate đi qua `backend.processing._liveness_decision`** (cổng khoảng cách "Move closer"/"Move back", rồi smoother), thay cho tự gọi `live.check` + `smoother.update`. Smoothing tắt trong config thì đi nhánh `live.check` như `process_attendance_frame`.
- **Cổng sang giai đoạn B chỉ có kết luận khi có liveness**; không có thì `pass = null`.
- **Tập đánh giá:** ngày khác, cùng ánh sáng bình thường. Frame tối (enhancer có tác động) báo cáo riêng ở tầng `…@dim`, không gộp vào tỉ lệ chính hay cổng.
- **Nạn nhân và lớp mô phỏng lấy từ `verification_data`**: ID trong tập train đã ẩn danh nên không kiểm tra trùng theo tên được.
- **Khoảng cách trong D_EOT** lấy từ `eot.UNSEEN.distance_px` (cạnh crop 60–120 px), giống `eot.photometric`, thay cho hệ số co cứng 0.5–0.8.

## Hướng đã loại
Giữ nguyên danh sách ở spec §10 (rPPG, YOLO bắt điện thoại, blink thụ động, screen-flash, motion challenge, depth/CDCN, sonar, face–voice không enroll, multi-face).
