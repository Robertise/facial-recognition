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


## 2026-11-03. Dựng gallery từ LFW, chọn nạn nhân theo trung vị

- **verification_data không dùng được:** Khảo sát tập HW2P2 (CMU 11-785) cho thấy các nhóm trong `verification_pairs_val.txt` đều chỉ có đúng 2 ảnh, không ai có đủ 5 ảnh; ảnh 64×64 crop sát mặt khiến MediaPipe khó bắt và khác hẳn phân bố webcam backend.
- **Chọn LFW:** Dataset public chuẩn, ảnh 250×250, lọc lấy ứng viên có ≥ 5 ảnh hợp lệ (tâm bbox lệch không quá 20% tâm ảnh).
- **Kiểm tra chưa thấy lúc train:**
  - Đối chiếu tên với VGGFace2 (tập train gốc của method3): lần chạy đầu không có đối chiếu tên (thiếu meta); lần chạy lại có đối chiếu tên với 4000 ID train qua identity_meta.csv (nạp thành công 4 000 tên và loại 37 ứng viên LFW trùng tên).
  - Hiệu chỉnh tâm ArcFace trên miền 64×64: đo trên 200 ID member (`test_data`) và 200 cặp non-member (`verification_pairs_val.txt`), xác định ngưỡng $T$ tại phân vị 5% của member ($T = 0.4783$, $\text{AUC} = 0.8061$). Loại các ứng viên có `agree ≥ 3` và `med_top1 ≥ T`.
- **Luật chọn nạn nhân:** Chọn `LFW_victim` có điểm tương đồng baseline với frame của owner (`s1_day1`, unenhanced) gần trung vị nhất ($s = 0.0648$, p50.13%). Tránh chọn người quá giống (tấn công dễ bất thường) hoặc quá khác (khó bất thường).
- **Lớp mô phỏng:** 50 người ngẫu nhiên `LFW_0001` ... `LFW_0050` (seed 0) từ các ứng viên còn lại, không đụng vào `owner/`.

## 2026-11-03. Đổi lưới texture mặc định 75×200 → 15×40 + upsample mượt

- **Vấn đề với lưới 75×200 cũ:** Khởi tạo nhiễu ngẫu nhiên `0.5 * randn` trên lưới 75×200 tạo ra các chấm nhiễu tần số cao. Tuy nhiên, trên ảnh webcam 640×480, gọng kính chỉ rộng khoảng 100–120 px. Chi tiết nhỏ hơn độ phân giải camera sẽ bị cảm biến webcam làm nhòe/mất chi tiết, khiến pattern tối ưu kỹ thuật số bị vô hiệu hóa khi thử vật lý. Ngoài ra, máy in tiệm không thể in chính xác nhiễu hạt rời rạc.
- **Quyết định:**
  - Thêm cờ `--tex-res H W` với mặc định `15 40`.
  - Tối ưu logits ở lưới thô 15×40, khởi tạo `logits = 0` (màu xám) cộng nhiễu nhỏ (`std = 0.05`) thay cho `0.5 * randn`.
  - Phóng lên template 150×400 bằng nội suy bicubic mượt ngay trong vòng tối ưu để gradient chảy qua bước phóng.
  - Tổng biến thiên (TV) giảm mạnh từ ~0.080 xuống ~0.0028 (>28 lần), tạo thành các mảng màu loang mượt in được và camera thấy rõ.
  - Lưu `tex_res` vào `attack.json` và `texture.pt`. `evaluate.py` và `export.py` đọc lại `tex_res` (mặc định `[75, 200]` cho run cũ để tương thích ngược).
  - Xuất ảnh preview "camera thấy" (`results/<run>/camera_preview.png`) để xem gọng ở độ phân giải webcam và phóng to 4x kiểm tra.
  - Hỗ trợ in nhiều gọng trên một tờ A4 (`export.py --sheet`) kèm controls và thước chuẩn 50 mm.
  - Yêu cầu in tiệm: Cùng máy in và cùng loại giấy cho cả bảng màu và kính thật.

## 2026-11-03. control_random xáo theo ô ở tex_res thay vì xáo pixel ở 150×400

- **Vấn đề:** Bản `control_random` ban đầu xáo độc lập từng pixel ở độ phân giải template 150×400 (60 000 pixel). Khi texture adversarial chuyển sang mảng loang mượt (lưới `tex_res 15×40` phóng bicubic lên 150×400), việc xáo pixel tạo ra nhiễu hạt siêu mịn. Khi qua webcam (với độ mờ quang học và kích thước gọng thực chỉ ~100–120 px), toàn bộ nhiễu mịn này bị cảm biến hòa trộn quang học thành một màu trung bình đồng nhất, khiến `control_random` có hành vi gần như trùng lặp với `control_plain`. Do đó, control không còn giữ được cùng thang độ loang không gian với texture thật.
- **Quyết định:**
  - Viết hàm dùng chung `control_random_texture(logits, seed)`: nhận `logits` ở độ phân giải tham số `tex_res` (15×40 hoặc 75×200 đối với run cũ), hoán vị ngẫu nhiên vị trí các ô không gian ($H 	imes W$ ô, giữ nguyên toàn vẹn 3 kênh màu của từng ô) với seed cố định (mặc định 0), sau đó mới phóng bicubic lên 150×400 và áp dụng sigmoid, hoàn toàn tương đồng với chuỗi xử lý của texture thật.
  - Cả `evaluate.py` và `export.py` dùng chung hàm này để tạo `control_random`. `control_plain` giữ nguyên (màu trung bình không gian).
  - Ghi rõ `tex_res` và `control_seed` vào `summary.json` (đánh giá số) và nhãn dưới mỗi gọng trên tờ in A4 (`export.py --sheet`).
  - Kết quả: `control_random` giữ nguyên chính xác multiset màu sắc ô ở `tex_res` và có cùng bậc biến thiên toàn phần (TV) với texture đối kháng thật (~0.0028 thay vì tăng vọt lên ~0.025 khi xáo pixel).

## Hướng đã loại
Giữ nguyên danh sách ở spec §10 (rPPG, YOLO bắt điện thoại, blink thụ động, screen-flash, motion challenge, depth/CDCN, sonar, face–voice không enroll, multi-face).
