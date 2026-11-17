# Plan v3: student đối xứng + dữ liệu lớn (VGGFace2 test) cho phòng thủ chứng nhận

Viết bởi Claude, 2026-11-17. Dành cho coding agent. Luật: `experiment_log.md` mục 2026-11-17 "Luật v3". Nền: v1 (`plan_certified.md`), v2 (`plan_certified_v2.md`).

**Ràng buộc:** không commit; không đổi tên CLI arg / key / cột / thư mục; mọi tuỳ chọn mới có mặc định = hành vi cũ. Chỉ sửa `innovation/defense/{eval_certified,train_student}.py`, thêm `innovation/defense/prepare_faces.py` và test mới. Không dùng frame owner/p01 để train. Không test nào ghi vào `results/innovation/` hay `data/`.

## 0. Vì sao v3
- v2: so mặt che với gallery sạch tệ hơn che đối xứng (C0). Student kéo lại được một phần khi so với gallery sạch (owner G0 0.56 → 0.79), tức student **có** học được cách đọc mặt che, nhưng đem dùng sai chế độ so.
- v3 làm hai việc: (A) dùng student ở **chế độ đối xứng** (probe và gallery cùng che, cùng qua student); (B) train student trên **dữ liệu lớn hơn ~15 lần, có nhãn danh tính, cùng miền với method3** (VGGFace2 test: 500 danh tính, không giao 4000 danh tính train của method3), với loss đối xứng có giám sát: hai ảnh **khác nhau** của cùng một người, che **cùng mask**, phải gần nhau; khác người phải xa.

## 1. `eval_certified.py` (chỉ thêm)
- `--gallery-encoder {method3,student}` (mặc định `method3` = như cũ). `student` + `--gallery-mode symmetric`: ảnh gallery của nhánh (m, fill) đi qua che → fill → **student** (cùng pipeline với probe). Bỏ cảnh báo symmetric+student khi `--gallery-encoder student`. Ghi `gallery_encoder` vào `config`. Áp cho `run`, `lfwval`, `calibrate`.
- Cache gallery: khoá cache phải gồm đường dẫn ckpt student khi `--gallery-encoder student`.

## 2. `prepare_faces.py` (mới): dựng shard crop + landmark từ thư mục ảnh có nhãn
- Đầu vào: `--images-root` (cấu trúc `<root>/<class_id>/<img>.jpg`, như VGGFace2), `--meta data/vggface2/identity_meta.csv` (cột `Class_ID, Name, ...`), `--per-id 60` (lấy tối đa 60 ảnh đầu theo tên file mỗi danh tính), `--out-dir data/innovation/defense/vgg_test` (tạo `crops/` và `landmarks/`), `--workers` (multiprocessing, mỗi tiến trình tự tạo detector + FaceMesh).
- **Loại trùng tên**: chuẩn hoá tên (bỏ dấu nháy, khoảng trắng → `_`, chữ thường). Loại mọi danh tính VGGFace2 có tên trùng (a) bất kỳ danh tính LFW trong gallery (dùng `prepare.resolve_gallery_lfw_names(data/innovation/gallery)`), (b) bất kỳ danh tính trong LFW val (`identities` của `lfw_crops/val_*.npz`). Ghi danh sách loại vào report.
- Mỗi ảnh: giống `prepare.py` + `prepare_landmarks.py`: detector → `faces.crop` → `faces.enhance` → resize 112 (crop lưu); eyes + 468 landmark + window; `ok = 1` khi có mặt, có mesh, `band_in_crop` dựng được.
- Ghi shard 1000 dòng: `crops/train_XXX.npz` (`crops`, `identities` = class_id, `paths`) và `landmarks/train_XXX.npz` (`eyes`, `window`, `mesh`, `ok`, `paths`) cùng tên, cùng thứ tự (đúng định dạng `LFWMultiMaskDataset` đọc được). `report.json`: số danh tính dùng/loại, số ảnh, số ok, thời gian. In tiến độ mỗi shard (`flush=True`).

## 3. `train_student.py` (chỉ thêm, mặc định như cũ)
- `--extra-data DIR` (có thể lặp): thư mục có `crops/` + `landmarks/` như §2, gộp với LFW train. Nhãn danh tính lấy từ `identities` (tiền tố nguồn để không trùng giữa LFW và VGG).
- `--init-ckpt PATH`: khởi tạo student từ một student đã train (mặc định từ method3).
- `--sym-weight λs` (mặc định 0 = tắt): bật loss đối xứng có giám sát.
  - Sampler: mỗi batch P danh tính × 2 ảnh (`--pk 64`, tức 128 ảnh), chỉ danh tính có ≥ 2 ảnh ok.
  - Mỗi batch chọn **một** mask m (đều trong 6 mask, hoặc không che với xác suất 0.15) dùng cho **mọi** ảnh của batch (đối xứng như lúc eval); webcam-aug riêng từng ảnh; fill như `--fill`.
  - Loss = KD hiện có (cos + InfoNCE với teacher sạch) + λs · SupCon(student embedding của cả 2P ảnh, positive = cùng danh tính, τ = 0.05).
- `--num-workers N` cho DataLoader (render mask trên CPU là nút cổ chai).
- Val (early stopping): thêm chỉ số **đối xứng** trên LFW val A: gallery = ảnh đầu mỗi danh tính, che mask m, fill, qua student; probe che cùng m, qua student; top-1 đúng và sim ≥ 0.5; TB 6 mask. Khi `--sym-weight > 0`, chọn checkpoint và early stop theo chỉ số này (ghi cả chỉ số cũ).

## 4. Test (`tests/innovation/test_certified_v3.py`)
- `--gallery-encoder student` dùng student cho gallery (embedding gallery khác method3, khớp student trên cùng ảnh).
- `prepare_faces` trên fixture 3 danh tính giả: loại đúng tên trùng, shard crop/landmark cùng tên cùng thứ tự, `LFWMultiMaskDataset` đọc được.
- Sampler P×2 đúng; mọi ảnh trong batch cùng một mask.
- Mặc định mọi tuỳ chọn mới cho kết quả trùng v2 trên fixture.

## 5. Agent chạy (không chạy thật, không commit)
pytest; `git diff --stat`; `train_student --limit 128 --epochs 1 --device cpu --sym-weight 1.0 --extra-data <fixture>`; `eval_certified lfwval --val-part B --gallery-mode symmetric --gallery-encoder student --student-ckpt <ckpt tạm> --limit 20`; xoá thư mục tạm.

## 6. Cấu hình dev (Claude đưa runbook, chạy tiền cảnh trên vast)
- C7: symmetric + student inpaint v2 có sẵn (`20261117_123128`) + `--gallery-encoder student`. Không train.
- C8: student mới: `--fill inpaint --inpainter-ckpt <v1> --init-ckpt 20261117_123128 --extra-data data/innovation/defense/vgg_test --sym-weight 1.0 --epochs 15 --patience 4 --num-workers 8`, đánh giá symmetric + gallery student.
- C9: C8 + ngưỡng hiệu chỉnh (calibrate cùng cấu hình).
