# Plan v2: tối ưu phòng thủ chứng nhận (recognizer học đặc trưng bất biến với mask)

Viết bởi Claude, 2026-11-16. Dành cho coding agent. Luật: `experiment_log.md` mục 2026-11-16. Nền: `plan_certified.md` (v1, đã chạy).

**Ràng buộc:** không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục có sẵn; mọi tuỳ chọn mới có **mặc định = hành vi cũ**. Không sửa file ngoài `innovation/defense/{eval_certified,train_inpainter}.py`, file mới `innovation/defense/train_student.py`, và test. Không test nào ghi vào `results/innovation/` hay `data/`. Không dùng frame owner/p01 để train.

## 0. Chẩn đoán (vì sao v1 trượt)
Trên owner_dev (222 frame, bỏ liveness) và LFW val, lỗi chính **không phải ngưỡng** (frame có top-1 đúng nhưng sim < 0.5 chỉ 0–7%) mà là **khớp nhầm sang người khác**: G0/gray nhận owner thành người khác ở 22% frame, G48/gray 46%; trên LFW G0/gray 71%. Nguyên nhân: gallery cũng bị che cùng chỗ, nên mọi mặt đều có chung một khối xám/vá, embedding của các người khác nhau dồn lại gần nhau. Recognizer chưa từng học mặt bị che. Inpainter chỉ giảm một phần (LFW G0: 71% → 26%) và gần như không giúp trên webcam.

Hướng sửa (theo masked face recognition): **học một student recognizer nhận mặt đã che và đưa embedding về đúng không gian của mặt sạch** (template-level knowledge distillation, Huber et al. FG 2021; ý "unmasking embedding", Boutros et al. Pattern Recognition 2022), rồi **so với gallery sạch** thay vì gallery cũng bị che. Chứng nhận vẫn giữ: nhánh m vẫn chỉ là hàm của pixel ngoài mask; gallery không chứa pixel của kẻ tấn công.

## 1. `eval_certified.py` (chỉ thêm)
- `--gallery-mode {symmetric,clean}` (mặc định `symmetric` = như cũ). `clean`: mọi nhánh (m, fill) so với embedding gallery của nhánh `none` (method3 trên ảnh gallery sạch).
- `--student-ckpt PATH` (mặc định không): nhánh (m, fill) lấy embedding bằng student thay cho method3. Nhánh `none` luôn là method3.
- `--thresholds PATH` (mặc định không = 0.5 cho mọi nhánh): JSON `{branch: tau}`; `match_id` dùng ngưỡng của nhánh. Cột mới `threshold` trong `frames.csv`.
- Mọi chỉ số GAR / certified / attack trong `summary.json` có thêm bản **không gate liveness**, key mới `gar_recog`, `certified_rate_recog`, `rate_recog` (cạnh key cũ). `summary.json` ghi `config`: `gallery_mode`, `student_ckpt`, `thresholds`, `inpainter_ckpt`, `liveness`.
- `lfwval --val-part {all,A,B}` (mặc định `all`): chia danh tính LFW val đủ điều kiện thành A/B theo `int(sha1(identity).hexdigest(), 16) % 2` (0 → A, 1 → B). Nhận cùng `--gallery-mode`, `--student-ckpt`, `--thresholds`.
- Subcommand mới `calibrate`: trên LFW **val A**, với cấu hình đưa vào (`--inpainter-ckpt`, `--student-ckpt`, `--gallery-mode`), mỗi probe lấy điểm impostor = max sim trên gallery val A **trừ chính danh tính đó**. FAR_none = tỉ lệ impostor nhánh none ≥ 0.5. Mỗi nhánh b: τ_b = ngưỡng nhỏ nhất trong lưới 0.30 → 0.90 bước 0.005 sao cho FAR_b(τ_b) ≤ FAR_none. Ghi `thresholds.json` (τ, FAR từng nhánh, số impostor) vào thư mục run mới `<ts>_certified_calib/`.
- `summarize --select`: thêm `--metric {gar,gar_recog}` (mặc định `gar` = như cũ) và đọc tập LFW theo key `lfw_val_B` khi run lfwval là `--val-part B` (set name ghi `lfw_val_B`).
- `run --no-liveness` không cần (đã có: bỏ `--liveness`). Dev v2 chạy **không** `--liveness`.

## 2. `train_student.py` (mới)
- Teacher: method3 đóng băng (`FrozenMethod3Embedder`, `checkpoints/method3_best.pth`). Student: cùng kiến trúc, **khởi tạo từ method3**, train toàn bộ; cùng tiền xử lý khả vi như `FrozenMethod3Embedder`. Không ghi đè file method3.
- Dữ liệu: `LFWMultiMaskDataset` (crop + 6 mask), split train. Val: LFW val **A** (cùng cách chia §1).
- Mỗi mẫu: crop sạch c; ảnh student = webcam-aug(c) rồi che mask m ngẫu nhiên trong 6 mask (xác suất 0.15 không che) rồi fill (`--fill gray` = 128; `--fill inpaint` = inpainter v1 đóng băng `--inpainter-ckpt`). Teacher nhận c sạch, không aug.
  - webcam-aug (trước khi che): JPEG q ∈ [50, 95]; Gaussian blur σ ∈ [0, 1.2]; thu nhỏ hệ số [0.5, 1] rồi phóng lại; nhiễu Gaussian σ ≤ 0.02; sáng/tương phản ±0.1; lật ngang (lật cả mask). Mỗi phép xác suất 0.5.
- Loss: `(1 − cos(s_i, t_i)) + λ · InfoNCE(s_i; {t_j} trong batch, τ = 0.05)`, λ = 1.0 (`--lambda-nce`). s, t chuẩn hoá L2.
- Tối ưu: AdamW lr 1e-4, weight decay 1e-4, warmup 1 epoch rồi cosine, batch 128, tối đa 40 epoch, **early stopping** patience 6 theo chỉ số val.
- Chỉ số val (mỗi epoch, xác định, không aug): LFW val A, gallery = ảnh đầu mỗi danh tính qua **teacher sạch**; probe = các ảnh còn lại, qua student với từng mask trong 6 mask (fill như train). Top-1 đúng **và** sim ≥ 0.5. Báo theo từng mask và TB; chọn checkpoint theo TB. Ghi thêm val cos(s, t).
- Output `results/innovation/<ts>_student_train/`: `best.pt` (state_dict + tên kiến trúc + tiền xử lý), `train_log.csv` (ghi mỗi epoch), `meta.json`, `summary.json` (best epoch, chỉ số từng mask). In một dòng mỗi epoch (`flush=True`).
- CLI: `--fill {gray,inpaint}`, `--inpainter-ckpt`, `--epochs 40`, `--batch 128`, `--lr 1e-4`, `--lambda-nce 1.0`, `--patience 6`, `--seed 0`, `--device`, `--limit`, `--output-dir`, cùng `--data/--landmarks/--maskset-dir/--checkpoint` như `train_inpainter`.
- Hàm `load_student(ckpt, device)` trả đối tượng có `extract_embedding(crop112_bgr_uint8) -> (512,)` chuẩn hoá L2, cùng tiền xử lý với `FaceRecognizer.extract_embedding`.

## 3. `train_inpainter.py` (chỉ thêm tuỳ chọn, mặc định như cũ)
`--cosine` (lr cosine), `--patience N` (early stopping theo val cos TB, mặc định tắt), `--webcam-aug` (như §2), `--base-width {32,64}` (kênh U-Net 32-64-128-256 hoặc 64-128-256-512; cần `PurifierUNet(base_channels=...)` đã có tham số). Checkpoint ghi `base_channels` để `load_purifier_checkpoint` dựng đúng (nếu hàm hiện tại không đọc được thì thêm hàm nạp mới `load_inpainter_checkpoint`, không sửa hàm cũ).

## 4. Test (`tests/innovation/test_certified_v2.py`, mới)
- Bất biến end-to-end với student: hai crop chỉ khác trong `region_m` (độ phân giải gốc) → embedding nhánh (m, fill) bằng student giống hệt.
- `gallery-mode clean` dùng đúng embedding gallery của nhánh none.
- `calibrate` trên số giả: τ chọn đúng theo FAR.
- Chia A/B tất định, không giao nhau.
- `train_student --limit 64 --epochs 1 --device cpu` chạy được, ghi đủ file vào `tmp_path`.
- Mặc định của mọi tuỳ chọn mới cho kết quả trùng v1 trên fixture.

## 5. Agent chạy (không chạy thật, không commit)
pytest `tests/innovation/`; `git status`; `git diff --stat`; chạy thử `train_student --limit 128 --epochs 1 --device cpu --fill gray --output-dir <tạm>`; `eval_certified calibrate` và `lfwval --val-part B` trên `--limit` nhỏ nếu có, hoặc fixture; xoá thư mục tạm.

## 6. Owner chạy (Claude đưa runbook sau verify)
Trên vast.ai (GPU, không liveness): train `student_gray`, `student_inpaint`; rồi lưới cấu hình dev (mỗi cấu hình: `lfwval --val-part B` + `run --split dev`, có thể thêm `calibrate`):
- C1: method3, gallery clean.
- C2: C1 + ngưỡng hiệu chỉnh.
- C3: student_gray, gallery clean.
- C4: student_inpaint, gallery clean.
- C5: cấu hình tốt hơn trong C3/C4 + ngưỡng hiệu chỉnh.
- (C6, nếu còn thời gian: inpainter v2 `--cosine --webcam-aug --base-width 64 --epochs 60 --patience 8` + student_inpaint trên inpainter v2.)
Test (`run --split test --liveness`) chỉ chạy **một lần** trên laptop với cấu hình và tập S chọn theo luật.

## 7. Sửa sau verify (2026-11-16) — trước khi chạy thật
1. **Lật ảnh lệch giữa student và teacher** (`train_student.py`): `apply_webcam_aug` lật ngang ảnh student (và mask) nhưng teacher vẫn nhận crop chưa lật, nên mục tiêu embedding sai với một nửa số mẫu lật. Sửa: quyết định lật một lần cho mỗi mẫu, áp cho **cả** crop sạch của teacher lẫn ảnh student và mask (các phép aug khác chỉ áp cho student như cũ).
2. **Chọn S theo đúng fill của student** (`summarize --select`): đang luôn đọc nhánh `inpaint`. Thêm `--fill {inpaint,gray}` (mặc định `inpaint` = như cũ); `selection.json` ghi `fill` và `metric`. `train_student` lưu `fill` vào `best.pt`; `eval_certified` ghi `student_fill` vào `config` của `summary.json` (lấy từ ckpt).
3. **Nạp inpainter có `base_channels`**: `eval_certified.py` (run, lfwval, calibrate) và `train_student.py` đang dùng `load_purifier_checkpoint` (luôn dựng kênh 32), sẽ lỗi với inpainter v2 `--base-width 64`. Đổi sang `train_inpainter.load_inpainter_checkpoint` (nạp được cả ckpt cũ).
4. `load_student`: suy `num_classes` (và `embed_dim`) từ shape trong `state_dict` thay vì cố định 4000; lưu `preprocess` từ `recognizer.preprocess` thay vì hằng số.
5. Ghi nhận, không sửa: ở `gallery-mode symmetric`, gallery nhánh che vẫn dùng method3 dù có `--student-ckpt` (v2 chỉ dùng student với `clean`). Thêm cảnh báo khi gặp tổ hợp `symmetric` + `--student-ckpt`.
pytest lại; chạy thử `train_student --limit 128 --epochs 1 --device cpu --fill gray --output-dir <tạm>` và `--fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt`; xoá thư mục tạm.
