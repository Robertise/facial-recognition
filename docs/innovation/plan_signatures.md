# Plan: đo khả thi chữ ký phòng thủ S1/S2/S3 (không train)

Viết bởi Claude, 2026-11-06. Dành cho coding agent. **Không commit, không push. Không đổi tên CLI arg, key JSON, cột CSV, thư mục run có sẵn.** Chỉ thêm file mới.

## 1. Câu hỏi cần trả lời

Trước khi xây phòng thủ (detector, model làm sạch, fusion), kiểm tra rẻ: các dấu vết mà phụ kiện adversarial để lại có **tách được** frame bị tấn công khỏi frame mặt thật không? Chỉ dùng model và dữ liệu đã có, không train gì.

- **S1 danh tính còn sót:** gọng không xoá được mặt kẻ tấn công, nên trong 1:N, ngoài top-1 (nạn nhân) còn một danh tính khác (kẻ tấn công) có điểm cao bất thường. Mặt thật thì danh tính thứ hai thường thấp.
- **S2 bằng chứng tập trung:** danh tính giả đến từ một dải nhỏ (vùng gọng); danh tính thật được nhiều vùng của mặt củng cố. Đo bằng che từng dải ngang.
- **S3 bất ổn theo thời gian:** danh tính giả chập chờn giữa các frame liên tiếp.

## 2. Dữ liệu

Model: `checkpoints/method3_best.pth` qua `backend.face_recognizer.FaceRecognizer` (strict), gallery `data/innovation/gallery` dựng bằng `faces.build_gallery` (5 ảnh/người, như `evaluate.py`), đúng đường xử lý của `evaluate.py`: detect → crop nới 15% → enhancer → embed. Chỉ dùng frame ánh sáng thường (enhancer không bật); frame `@dim` bỏ ra và đếm riêng.

**Frame bị tấn công (dương tính):** dựng lại bằng chính hàm vẽ của `evaluate.py` (`paint`, texture qua `upsample_logits` + `cmap`, mask từ `attack.json` → `frame_shape`), JPEG 95, trên các session **chưa dùng để tối ưu texture đó**:
- `A_C3`: texture `results/innovation/20261104_140557_advglasses_impersonate` (gọng tròng mở, train trên s1) vẽ lên `s2_day2`, `s5_vary`, `s6_vary`.
- `A_R2b`: texture `results/innovation/20261106_132526_advglasses_impersonate` (dải gần như kín mắt, train trên s1 + s2) vẽ lên `s5_vary`, `s6_vary`.
- Mỗi bộ có thêm bản EOT (như tầng `D_EOT`: `unseen_frame` + `A_unseen`, seed như `evaluate.py`).
- Chỉ giữ frame mà tấn công **thành công ở mức nhận dạng** (top-1 = nạn nhân, cos ≥ 0.5). Frame tấn công thất bại không phải mối đe doạ; đếm và báo riêng.

**Frame mặt thật (âm tính), mọi frame có top-1 khớp đúng người với cos ≥ 0.5:**
- `R_owner`: baseline (không vẽ gì) của `s2_day2`, `s5_vary`, `s6_vary`, cùng bản EOT (`unseen_frame` trên frame gốc).
- `R_owner_controls`: tầng `control_random` và `control_plain` của cả hai texture trên cùng các session (mặt thật đeo gọng thường cùng hình dạng, để tách "có gọng" với "gọng adversarial").
- `R_lfw`: với mỗi danh tính LFW trong gallery, lần lượt giữ lại 1 ảnh làm query, gallery của người đó còn 4 ảnh (leave-one-out), các người khác giữ nguyên 5. Cho ~53 danh tính, để S1/S2 không chỉ học "mặt owner".

## 3. Chữ ký (mỗi frame một giá trị, quy ước: giá trị cao hơn = nghi tấn công hơn)

Gọi score(id) = max cosine của embedding với 5 ảnh gallery của id; top-1, top-2, top-3 theo id (khác nhau).
- `s1_second`: score của top-2.
- `s1_gap_inv`: −(score top-1 − score top-2).
- `s1_ratio`: score top-2 / score top-1.

S2: chia crop 112×112 (sau enhancer, trước chuẩn hoá) thành **8 dải ngang** cao 14 px. Với mỗi dải k, thay dải bằng giá trị trung tính (0 sau chuẩn hoá [−1, 1]), embed, lấy score của **id top-1 gốc**.
- `s2_maxdrop`: max_k (score_gốc − score_che_k) / score_gốc.
- `s2_flips`: số dải mà khi che thì top-1 đổi sang id khác hoặc rơi dưới 0.5.
- `s2_conc`: phần lớn nhất của tổng độ giảm nằm ở một dải (max_k drop_k / Σ_k max(drop_k, 0)).
- Lưu cả vector 8 độ giảm vào CSV (`s2_drop_0` … `s2_drop_7`).

S3 (chỉ cho frame thuộc session liên tục, không áp dụng `R_lfw`): cửa sổ trượt 1.5 s (như smoother), trong cửa sổ:
- `s3_top1_switch`: tỉ lệ cặp frame liên tiếp có top-1 khác nhau.
- `s3_score_std`: độ lệch chuẩn của score id top-1 của frame hiện tại trong cửa sổ.

## 4. Đầu ra

Thư mục `results/innovation/<ts>_signatures/` qua `innovation.eval.runs.new_run` (giống các run khác):
- `frames.csv`: một dòng/frame, cột: `set` (A_C3, A_C3_EOT, A_R2b, A_R2b_EOT, R_owner, R_owner_EOT, R_owner_controls, R_lfw), `source` (session hoặc id LFW), `frame`, `t`, `label` (1 = tấn công), `top1_id`, `top1_score`, `top2_id`, `top2_score`, rồi toàn bộ chữ ký ở §3.
- `summary.json`: với mỗi chữ ký và mỗi cặp so sánh, AUC + CI 95% bootstrap (2000 lần, seed 0) + số mẫu mỗi bên:
  - chính: `A_C3` (gộp EOT) so với âm tính gộp (`R_owner` + `R_owner_EOT` + `R_owner_controls` + `R_lfw`); `A_R2b` (gộp EOT) so với âm tính gộp;
  - phụ: mỗi bộ tấn công so với riêng `R_owner_controls`, và riêng `R_lfw`.
- `hist_<signature>.png`: histogram dương/âm cho từng chữ ký (bộ chính).
- In ra màn hình bảng AUC.

Đặt thành module mới `innovation/attacks/adv_glasses/signatures.py` (CLI `python -m innovation.attacks.adv_glasses.signatures`, mặc định đúng các đường dẫn ở §2). Tái dùng hàm của `evaluate.py`/`faces.py`, không chép lại logic vẽ. Thêm test nhỏ trong `tests/innovation/` cho: chia 8 dải che đúng vùng; leave-one-out bỏ đúng ảnh query khỏi gallery; quy ước dấu (giá trị cao = nghi tấn công) cho từng chữ ký trên dữ liệu giả.

Không chạy trên `s1_day1` (dữ liệu train của cả hai texture). Không train, không tối ưu gì. Chạy pytest `tests/innovation/`, in `git diff --stat`. **Không chạy script chính**; owner chạy.

## 5. Luật quyết định (đã ghi trong `experiment_log.md` trước khi chạy)

Xem mục "2026-11-06 Luật đặt trước: khả thi chữ ký S1/S2/S3".
