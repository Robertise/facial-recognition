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

## 7. Sửa sau verify (Claude, 2026-11-18). Làm hết trước khi đóng gói vast. Không commit.
Lý do: C8 chỉ train **một lần** theo luật v3, nên mọi lỗi im lặng hoặc lỗi loss phải sửa trước khi có số.

1. **InfoNCE có âm tính giả trong batch đối xứng (`train_student.py`, vòng train).** Với `SymmetricBatchSampler`, mỗi hàng i có ảnh thứ hai cùng người ở cột `i ^ 1`; InfoNCE hiện coi teacher của ảnh đó là âm tính, tức đẩy student(ảnh a) xa teacher(ảnh b) cùng người, ngược chiều SupCon. Sửa: khi batch đến từ sampler đối xứng (`sym_weight > 0`), đặt `sim_mat[i, i ^ 1] = -inf` trước `cross_entropy` (giữ đích là đường chéo). Chế độ cũ (`sym_weight = 0`) không đổi.
2. **Không được lùi về im lặng. Các trường hợp sau phải `SystemExit` với thông báo rõ:**
   - `--init-ckpt` có giá trị nhưng file không tồn tại (hiện lặng lẽ khởi tạo từ method3 mà `meta` vẫn ghi `init_ckpt`).
   - `--sym-weight > 0` mà không có danh tính nào ≥ 2 ảnh ok (hiện lùi về DataLoader ngẫu nhiên nhưng SupCon vẫn coi hai mẫu kề nhau là cùng người).
   - `eval_certified` `run`/`lfwval`/`calibrate`: `--gallery-encoder student` mà thiếu `--student-ckpt` (hiện lặng lẽ dùng method3 nhưng `config` ghi `student`).
   - `prepare_faces.py`: thư mục gallery tồn tại nhưng `prepare.resolve_gallery_lfw_names` lỗi (bỏ nhánh lùi về tên thư mục `LFW_0001...`); file `--meta` không tồn tại; `--images-root` không có thư mục danh tính nào.
3. **Loại trùng tên mở rộng sang LFW train (`prepare_faces.py`).** Đo trên laptop (Claude, 18/11): 500 danh tính VGGFace2 test trùng 0 tên với gallery, 3 với LFW val, **25 với LFW train** (5 124 danh tính). 25 người này sẽ mang hai nhãn (`lfw:<tên>` và `vgg_test:<class_id>`), thành âm tính giả cho SupCon/InfoNCE. Loại thêm mọi tên có trong `identities` của `lfw_crops/train_*.npz`. `report.json` thêm (chỉ thêm key) `excluded_by_source`: số danh tính loại theo `gallery`, `lfw_val`, `lfw_train`. Kỳ vọng trên VGGFace2 test: 0 / 3 / 25 (có thể chồng nhau), dùng 472–475 danh tính.
4. **Cân bằng nguồn trong sampler (`SymmetricBatchSampler`).** Hiện chọn danh tính đều nhau: LFW có 1 472 danh tính đủ điều kiện (7 736 ảnh, trung bình ~5 ảnh) so với ~475 danh tính VGG (~60 ảnh). Kết quả: ~75% cặp trong batch là LFW, lặp lại các cặp cũ (danh tính 2 ảnh luôn ra đúng một cặp), còn ảnh VGG mỗi epoch chỉ được dùng ~0.3 lần, tức dữ liệu mới chủ yếu bị pha loãng. Thêm `--pk-weighting {uniform,count}` (mặc định `uniform` = như code hiện tại): `count` chọn P danh tính không lặp với xác suất tỉ lệ số ảnh ok của danh tính. Ghi `pk_weighting` vào `meta`. C8 sẽ dùng `count` (khi đó VGG chiếm ~80% lượt chọn).
5. **Kiểm khớp shard (`_load_training_shards`).** Khôi phục kiểm như `LFWMultiMaskDataset`: shard landmark thiếu shard crop → `SystemExit`; `paths` của crop và landmark phải trùng từng dòng, số dòng phải bằng nhau, nếu không thì `SystemExit`. Áp cho cả LFW lẫn `--extra-data`.
6. **Test thêm** (`tests/innovation/test_certified_v3.py`): (a) InfoNCE batch đối xứng không phạt cột `i ^ 1` (so loss tính tay trên embedding giả); (b) ba trường hợp `SystemExit` của mục 2 và `prepare_faces` với gallery hỏng; (c) fixture có tên trùng LFW train bị loại, `excluded_by_source` đúng; (d) `--pk-weighting count` trên fixture lệch (một danh tính 50 ảnh, chín danh tính 2 ảnh) chọn danh tính lớn nhiều hơn rõ rệt qua nhiều batch, vẫn P danh tính khác nhau mỗi batch; `uniform` cho kết quả trùng code hiện tại với cùng seed; (e) shard lệch `paths` → `SystemExit`.
7. Chạy lại: `pytest tests/innovation` (toàn bộ), smoke test như §5 thêm `--pk-weighting count`. Ghi vào `verification_log.md` **đúng ký tự**: lần trước các chuỗi `\v`, `\b`, `\r`, `\t` trong đường dẫn bị python hiểu thành ký tự điều khiển (`val_` thành `\x0bal_`...); dùng raw string hoặc ghi file bằng editor. Không commit.

Cấu hình C8 cập nhật (thay dòng C8 ở §6): `--fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --init-ckpt results/innovation/20261117_123128_student_train/best.pt --extra-data data/innovation/defense/vgg_test --sym-weight 1.0 --pk 64 --pk-weighting count --epochs 15 --patience 4 --num-workers 8`.
