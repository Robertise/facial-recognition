# Plan: phòng thủ có chứng nhận bằng đồng thuận nhiều mask neo theo mắt

Viết bởi Claude, 2026-11-14. Dành cho coding agent. Luật: `experiment_log.md` mục 2026-11-14 "Luật đặt trước: phòng thủ có chứng nhận". Quyết định: `decisions.md` 2026-11-14.

**Ràng buộc:** không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục có sẵn. **Không sửa file có sẵn** (`band.py`, `purifier.py`, `train_purifier.py`, `eval_purifier.py`, `adaptive.py`, `glasses.py`, `faces.py`...): chỉ import và dùng lại. Chỉ THÊM module mới trong `innovation/defense/` và test mới trong `tests/innovation/`. Không test nào ghi vào `results/innovation/` hay `data/` (dùng `tmp_path`, `--output-dir`).

## 0. Ý tưởng (để agent hiểu)
Purifier hiện tại xoá **một** vùng cố định (band) rồi vẽ lại; đầu ra không phụ thuộc pixel trong band (đã có test bất biến). Kẻ tấn công thích ứng đặt hoa văn ra ngoài band (vành r48) và thắng 56%.

Mở rộng: một **tập mask** S, mỗi mask che một vùng phụ kiện có thể có, neo theo **hai tâm mắt** (cùng phép similarity như band). Mỗi frame chạy |S| nhánh; nhánh m: che vùng m → enhancer → inpainter vẽ lại → method3 → khớp 1:N với gallery đã qua cùng nhánh. **Chỉ chấp nhận danh tính X khi mọi nhánh cùng ra X** (sim ≥ 0.5). Nếu phụ kiện nằm gọn trong một mask m*, nhánh m* chỉ thấy mặt sạch của kẻ tấn công, nên không có đồng thuận về nạn nhân → không giả danh được, với **mọi** thuật toán tấn công. Cái giá: người thật phải được nhận ra ở mọi nhánh (GAR giảm).

## 1. Landmark cho LFW: `innovation/defense/prepare_landmarks.py` (mới)
Shard crop hiện có (`data/innovation/defense/lfw_crops/{train,val}_*.npz`: `crops`, `bands`, `identities`, `paths`) không lưu landmark. Không ghi đè shard cũ.
- Với mỗi dòng của mỗi shard: đọc ảnh gốc `data/public/lfw/lfw_home/lfw_funneled/<paths[i]>`; `faces.FaceDetector().detect` → bbox; `faces.crop_window(bbox, img.shape)` → window; MediaPipe FaceMesh (`static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5`, như `faces.EyeLandmarks`) → 468 điểm (px ảnh gốc); `eyes` = trung bình `faces.LEFT_EYE` / `faces.RIGHT_EYE` đúng như `EyeLandmarks.eyes`.
- Kiểm khớp: `band_in_crop(img, bbox, eyes)` so với `bands[i]` đã lưu (IoU). `ok = 1` nếu có mặt, có mesh và IoU ≥ 0.95; còn lại `ok = 0`.
- Ghi `data/innovation/defense/lfw_landmarks/{split}_{k:03d}.npz`, khớp 1–1 với shard crop cùng tên: `eyes` (N,2,2) float32, `window` (N,4) int32, `mesh` (N,468,2) float32, `ok` (N,) uint8, `paths`. `report.json`: tổng ảnh, số ok, số band lệch (IoU < 0.95), IoU min/TB, thời gian.
- CLI: `--crops-dir` (mặc định `data/innovation/defense/lfw_crops`), `--lfw-dir`, `--out-dir` (mặc định `data/innovation/defense/lfw_landmarks`), `--limit`. Chạy CPU.

## 2. Tập mask: `innovation/defense/maskset.py` (mới)
### 2.1 Canvas và hàm render
- Canvas 800×900 (W×H). Template 400×150 của `glasses` đặt tại offset (200, 300), tức mắt template ở (310, 380) và (490, 380). Hằng số mới trong `maskset.py`: `CANVAS_W, CANVAS_H, CANVAS_OFF = 800, 900, (200, 300)`, `DELTA = 8` (lề, px template), `Q = 0.05`.
- `similarity_canvas(eyes)` = `glasses.similarity(eyes, src_eyes=glasses.EYES + CANVAS_OFF)`.
- `mask_in_window(window, eyes, canvas_mask, size=112)`: như `band_in_crop` (grid như `adaptive._grid_padded` nhưng với W, H của canvas; `grid_sample` bilinear, `padding_mode="zeros"`, `align_corners=False`; resize 112; ngưỡng `> 0.01`) → uint8 {0,1}. `mask_in_crop(img_shape, bbox, eyes, canvas_mask)` = `mask_in_window(faces.crop_window(bbox, img_shape), ...)`. Thiếu bbox/eyes → None.
### 2.2 `python -m innovation.defense.maskset build`
- **Họ kính G_r**, r ∈ {0, 16, 32, 48}: `band_template.png` dán vào canvas tại offset. Vùng `region_Gr` = dilate(band, ellipse bán kính r) (cùng kiểu kernel `adaptive.make_ring`); mask `Gr` = dilate(band, ellipse r + DELTA).
- **L (nửa dưới mặt: khẩu trang, khăn)** và **F (trán: mũ, băng đô)** dựng từ LFW **train**, `ok = 1`. Với mỗi ảnh, đưa 468 điểm về canvas bằng nghịch đảo `similarity_canvas(eyes)`, rồi:
  - `oval` = đa giác theo `FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]`.
  - `L_i` = (fill(oval) ∪ fill(oval dịch xuống 27 px)) ∩ {y ≥ y của landmark 6}.
  - `F_i` = {y ≤ min y của `BROW_TOP = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]`} ∩ {x_min(oval) ≤ x ≤ x_max(oval)} (kéo tới mép trên canvas).
  - Bản đồ tần suất trên mọi ảnh; `region_L` / `region_F` = pixel có tần suất ≥ Q; mask `L` / `F` = dilate(region, ellipse DELTA).
- Ghi vào `data/innovation/defense/maskset/`: `G0.png, G16.png, G32.png, G48.png, L.png, F.png` (mask, canvas 800×900, 0/255), `region_*.png` cùng tên, `maskset_meta.json` (canvas, offset, DELTA, Q, chỉ số landmark, số ảnh LFW dùng, độ phủ mỗi mask trên crop 112 của `owner/s1_day1`: TB/min/max, giao nhau từng cặp), `maskset_preview.png` (overlay 6 mask, mỗi mask một màu, trên 4 crop LFW val + 2 frame `owner/s1_day1`).
- Kiểm (in ra và ghi meta): trên `owner/s1_day1`, `band_in_crop` ⊆ render `G0` ở mọi frame có mắt (đếm pixel band nằm ngoài G0, phải = 0 trừ viền nội suy ≤ 1 px).

## 3. Train inpainter: `innovation/defense/train_inpainter.py` (mới)
- Dataset mới: crop từ `lfw_crops` + landmark từ `lfw_landmarks` (chỉ `ok = 1`). Mỗi mẫu: chọn mask m đều ngẫu nhiên trong {G0, G16, G32, G48, L, F}; render bằng `mask_in_window(window, eyes, ...)`; dịch mask ±3 px (dùng lại `train_purifier.shift_band_mask`); lật ngang (lật cả crop và mask); jitter sáng/tương phản như `LFWCropBandDataset`.
- Model `PurifierUNet` khởi tạo mới; loss `purifier_loss` (w_band 1.0 trên vùng mask, w_full 0.1, w_id 0.5); `FrozenMethod3Embedder` với `checkpoints/method3_best.pth`; Adam lr 2e-4, 30 epoch, batch 64, seed 0 (giống purifier).
- Val: mỗi ảnh val × 6 mask, không jitter, không lật. Ghi cos(emb(pred), emb(clean)) và cos của gray (cùng mask) **theo từng mask**. Checkpoint tốt nhất theo **TB cos val trên 6 mask**.
- Output `results/innovation/<ts>_inpainter_train/` (`new_run("inpainter_train")`): `best.pt` (cùng định dạng checkpoint purifier, nạp được bằng `load_purifier_checkpoint`), `train_log.csv` (mỗi epoch: các loss, val cos TB và từng mask, gray từng mask), `meta.json`, `summary.json` (best epoch, val cos từng mask).
- CLI: `--data`, `--landmarks`, `--maskset-dir`, `--epochs 30`, `--batch 64`, `--lr 2e-4`, `--device`, `--seed 0`, `--limit` (chỉ để chạy thử), `--output-dir`. Chạy được CPU và CUDA.

## 4. Đánh giá: `innovation/defense/eval_certified.py` (mới)
### 4.1 Nhánh và pipeline
- Nhánh `none`: đúng luồng của `eval_purifier` mode none (crop → enhance → embed → khớp gallery none).
- Nhánh (m, fill), m ∈ {G0, G16, G32, G48, L, F}, fill ∈ {gray, inpaint}: `faces.crop(frame, bbox)` → `mask112 = mask_in_crop(frame.shape, bbox, eyes_của_chính_frame)` → pixel trong mask = 128 → `faces.enhance(crop_đã_che)` (enhancer chỉ thấy pixel ngoài mask) → gray: giữ nguyên; inpaint: `apply_purifier(inpainter, crop, mask112)` → resize 112 nếu cần → `extract_embedding` → `score_against_gallery` với gallery (m, fill).
- Gallery (m, fill): mỗi ảnh gallery đi qua **đúng** pipeline nhánh đó (dùng eyes và bbox của ảnh gallery). Cache theo (m, fill, tên ckpt, hash mask) trong `results/innovation/_cache`.
- Thiếu landmark mắt → mọi nhánh (m, fill) từ chối frame (fail-closed); nhánh none không đổi.
- Liveness: như `eval_purifier` (trên frame gốc, smoother theo session, không warmup). Phân loại ánh sáng thường / `@dim` theo cờ enhancer của **nhánh none** (cố định tập frame cho mọi nhánh), số chính chỉ trên ánh sáng thường như `eval_purifier`.
### 4.2 `python -m innovation.defense.eval_certified run --split {dev,test}`
- `dev`: người thật `owner/s3_vary`, `owner/s4_vary`, `owner/s5_vary` (đích owner). Không tấn công.
- `test`: người thật `owner/s6_vary` (đích owner), `p01/s2` (đích p01; cần `--extra-gallery data/innovation/gallery_people`). Tấn công tier D dán digital lên `owner/s6_vary`, nạn nhân `LFW_victim`: `H09_s0=20261107_214538`, `H09_s1=20261112_112957`, `H09_s2=20261112_115550`, `R2b=20261106_132526` (dán như `eval_purifier`), `ADA_r8=20261113_074527`, `ADA_r16=20261113_074700`, `ADA_r32=20261113_074829`, `ADA_r48=20261113_075000` (dán như `adaptive` eval: `texture.pt` + `mask.png` + `render_padded`). Gọi `check_session_not_in_train_dev` cho mọi run.
- `--extra-genuine NAME=DIR:TARGET` (tuỳ chọn, cho `owner/acc_*` nếu quay).
- `frames.csv` (dạng dài, một dòng mỗi (set, frame, branch)): `set, kind (genuine/attack), frame, target_id, branch (none | <m>/<fill>), eyes_ok, match_id ("" nếu top1 sim < 0.5), top1_id, top1_sim, target_sim, is_real, light (normal/dim)`. Với frame tấn công thêm `eye_shift_px` (lệch lớn nhất của hai tâm mắt giữa frame sạch và frame bị tấn công cùng tên, đổi sang px template: × 180 / khoảng cách hai mắt trên frame sạch) và `bbox_iou` (bbox sạch vs bị tấn công).
- `summary.json`: GAR (genuine) / tỉ lệ thành công (attack, khớp nạn nhân + liveness) cho `none` và mọi nhánh đơn (m, fill), Wilson 95%; và cho mọi tập S trong `--sets` (mặc định: mọi tập trong danh sách ưu tiên của luật, xem §4.4) với cả hai fill.
- Thư mục `results/innovation/<ts>_certified_eval/` (`new_run("certified_eval")`), có `--output-dir`.
### 4.3 `python -m innovation.defense.eval_certified lfwval`
- Dùng crop + landmark LFW **val** (`ok = 1`), danh tính có ≥ 2 ảnh: gallery = ảnh đầu tiên theo thứ tự `paths`, probe = các ảnh còn lại; 1:N trên toàn bộ gallery val. Không liveness. Crop đã qua enhancer lúc prepare: nhánh (m, fill) chỉ che + (gray | inpaint), không chạy lại enhancer. Mask render bằng `mask_in_window`.
- GAR = probe khớp đúng danh tính với sim ≥ 0.5. Cùng `frames.csv` / `summary.json` như §4.2 (set = `lfw_val`). Thư mục `<ts>_certified_lfwval/`.
### 4.4 `python -m innovation.defense.eval_certified summarize`
- `--run DIR [DIR ...]`, `--sets "G48+L+F" ...`: tính lại từ `frames.csv`, ghi `sets_summary.json` vào từng run.
- Định nghĩa (chỉ trên frame ánh sáng thường):
  - S chấp nhận X ở một frame ⇔ mọi m ∈ S có `match_id(m, fill) == X` (khác rỗng); với tập camera thêm `is_real == 1`.
  - GAR(S) = tỉ lệ frame người thật S chấp nhận đúng `target_id`. Thành công tấn công(S) = tỉ lệ frame tấn công S chấp nhận `LFW_victim`.
  - Được chứng nhận (frame người thật) ⇔ mọi m ∈ S có `match_id(m, fill) ∈ {"", target_id}`.
- `--select --dev-runs <run owner dev> <run lfwval>`: áp **đúng** luật chọn S trong `experiment_log.md` 2026-11-14 (fill = inpaint; Pass(S) ⇔ GAR(S) ≥ 0.90 × GAR(none) trên cả hai run; thứ tự: {G_r, L, F} với r = 48, 32, 16, 0 → {G_r, L} / {G_r, F} với r = 48 → 0, cùng r cả hai đạt thì lấy min GAR cao hơn → {G_r} với r = 48 → 0 → không tập nào). In tập chọn và bảng Pass của mọi tập đã thử; ghi `selection.json`.

## 5. Test (`tests/innovation/test_certified.py`, mới)
- `mask_in_window` với canvas chỉ chứa band (DELTA = 0) trùng `band_in_crop` (IoU ≥ 0.99) trên frame giả.
- **Bất biến end-to-end:** với mỗi m, hai crop chỉ khác nhau bên trong mask m → đầu ra nhánh (m, inpaint) **giống hệt** (gồm cả `faces.enhance` thật trên crop đã che và inpainter khởi tạo ngẫu nhiên). Thêm trường hợp crop tối (enhancer bật) để kiểm enhancer cũng bất biến.
- Logic đồng thuận / chứng nhận / thành công trên `frames.csv` giả (bảng tính tay).
- `select` áp đúng thứ tự ưu tiên trên số giả (ít nhất 4 tình huống: chọn được tập đầy đủ; chỉ được {G, L}; hoà {G, L} vs {G, F}; không tập nào).
- Dataset train trả mask đúng loại, lật đồng bộ crop và mask, dịch ±3 px.
- `eval_certified run` chạy trên fixture nhỏ (frame giả, gallery giả, inpainter ngẫu nhiên) với `--output-dir` trong `tmp_path`.

## 6. Chạy
**Agent** (không chạy thật, không commit):
- `pytest tests/innovation/` (báo pass/fail, `lastfailed` rỗng); `git status`; `git diff --stat` (chỉ file mới).
- `python -m innovation.defense.prepare_landmarks --limit 200 --out-dir <thư mục tạm ngoài data/>`; báo `report.json`.
- `python -m innovation.defense.maskset build` trỏ vào landmark tạm đó, ghi ra thư mục tạm; gửi đường dẫn `maskset_preview.png` và `maskset_meta.json` tạm.
- `python -m innovation.defense.train_inpainter --limit 256 --epochs 1 --device cpu --output-dir <tạm>`.
- `python -m innovation.defense.eval_certified run --split dev --limit-frames 5 --inpainter-ckpt <ckpt tạm> --output-dir <tạm>` (thêm `--limit-frames` chỉ cho chạy thử).
- Xoá mọi thư mục tạm sau khi báo. Kiểm `results/innovation/` và `data/innovation/defense/` không có gì mới.

**Owner** (sau khi Claude verify; Claude sẽ đưa runbook đầy đủ):
1. Laptop: `python -m innovation.defense.prepare_landmarks` → Claude kiểm `report.json`.
2. Laptop: `python -m innovation.defense.maskset build` → Claude xem preview + meta **trước khi train**.
3. vast.ai: `python -m innovation.defense.train_inpainter --device cuda`.
4. Laptop: `eval_certified lfwval` và `eval_certified run --split dev` → Claude chạy `summarize --select`, ghi tập S vào log.
5. Laptop: `eval_certified run --split test` **một lần** → Claude áp luật.

## 8. Sửa sau verify (2026-11-14) — bắt buộc trước mọi lần chạy thật
Chỉ sửa 4 module mới và `test_certified.py`. Không đổi tên CLI arg / key / cột đã có trong các module này (chỉ thêm).

**Lỗi nghiêm trọng**
1. `prepare_landmarks.py`: thân vòng lặp shard (từ `data = np.load(sf, ...)` đến `remaining_limit -= n_process`) đang nằm **ngoài** `for sf in shards:` (thụt lề sai), nên mỗi split chỉ xử lý shard **cuối** (`train_011`, `val_001`). Đưa toàn bộ thân vào trong vòng lặp. `report.json` thêm số ảnh và số ok **theo từng shard**. Cuối hàm kiểm: mọi shard crop có shard landmark cùng tên và cùng số dòng (khi không dùng `--limit`), sai thì `SystemExit`.
2. Ghép shard crop ↔ landmark (`LFWMultiMaskDataset` trong `train_inpainter.py` và `cmd_lfwval` trong `eval_certified.py`): đang `zip` theo thứ tự sắp xếp, thiếu shard là ghép lệch im lặng. Sửa: ghép theo **tên file**; mỗi cặp phải cùng số dòng và `paths` trùng từng dòng; thiếu hoặc lệch → `SystemExit` (in tên shard).
3. `compute_sets_summary`: GAR đang gộp mọi tập người thật (owner/s6 + p01/s2 thành một số), trong khi luật cần **từng tập**. Sửa:
   - GAR, `certified_rate` (và Wilson 95%) **theo từng `set`** người thật. Với split dev, thêm một mục gộp `owner_dev` = s3 + s4 + s5 (luật chọn S dùng mục gộp này và `lfw_val`).
   - Thêm GAR của nhánh `none` và của **từng nhánh đơn** (m, fill) theo từng set (người thật) và tỉ lệ thành công theo từng set tấn công, ghi trong `summary.json` mục mới `branches`.
   - Bỏ fallback "không có frame ánh sáng thường thì dùng mọi frame": nếu một set không còn frame ánh sáng thường thì ghi `null` và cảnh báo.
   - `--select` đọc GAR gộp `owner_dev` và `lfw_val` từ cấu trúc mới; GAR(none) lấy từ mục `branches` (bỏ `get_none_gar` tự tính lại hoặc giữ nhưng phải trùng).
4. `cmd_run`: đang `continue` im lặng khi thiếu thư mục session hoặc run tấn công, làm kiểm hợp lệ (V) có thể rỗng. Sửa: thiếu bất kỳ session/run nào trong danh sách của split → `SystemExit`. `summary.json` ghi danh sách set đã chấm và số frame mỗi set.
5. `maskset.py`: khi không có ảnh LFW nào dùng được (`n_used == 0`) đang tạo mask L/F **rỗng** im lặng. Sửa: `SystemExit`. Ghi thêm `n_used` theo shard vào meta.

**Cần sửa**
6. `cmd_lfwval`: bắt buộc `--inpainter-ckpt` (không có thì `SystemExit`); nếu `mask_in_window` trả None ở nhánh (m, fill) thì từ chối mẫu đó (không dùng crop không che); tính điểm bằng `score_against_gallery`.
7. Test bất biến ở độ phân giải gốc: hiện test đưa crop 112 vào nên bước resize là đồng nhất. Thêm test: `face_raw` kích thước thật (vd 240×260, window tương ứng), phụ kiện nằm trong `region_m` (mask co lại δ) vẽ ở độ phân giải gốc, hai giá trị phụ kiện khác nhau → đầu ra nhánh (m, inpaint) giống hệt, cho ít nhất G0, G48, L, F. Trong ca crop tối, assert enhancer thực sự bật (`meta["applied"]`).
8. `maskset_preview.png`: thêm 4 crop LFW val (như §2.2), không chỉ 2 frame owner.
9. `cmd_run`: thêm `--extra-genuine NAME=DIR:TARGET` (nhiều giá trị, tuỳ chọn) như §4.2.

**Tuỳ chọn (tốc độ)**: `LFWMultiMaskDataset` với split val: render sẵn 6 mask cho mọi ảnh một lần lúc khởi tạo (val đánh giá mỗi epoch).

**Agent chạy lại §6** (thư mục tạm, không chạy thật, không commit), báo thêm: `report.json` tạm có số dòng theo từng shard và đủ cả `train_000` lẫn `val_000`; test mới qua; `summary.json` của eval chạy thử có mục `branches` theo từng set.

## 9. Bổ sung nhỏ (2026-11-14, sau verify §8)
- `train_inpainter.py`: in một dòng mỗi epoch (`flush=True`): epoch, loss TB, val cos TB, val cos 6 mask, thời gian; và ghi lại `train_log.csv` sau **mỗi** epoch (hiện chỉ ghi cuối), để theo dõi khi chạy nền trên vast.
- `prepare_landmarks.py`: in một dòng sau mỗi shard (tên, số ảnh, số ok, thời gian, `flush=True`).
- Không đổi gì khác. pytest lại.

## 10. Sửa hình mask L (2026-11-15, trước khi train, trước mọi số GAR)
Lý do: ở canvas, mắt template nằm ở y = 380. Landmark 6 trên LFW train có trung vị y = 389, p5 = 373 (gần như ngang tầm mắt). Vì vậy vùng L dựng theo đường ngang qua landmark 6, sau khi lấy hợp 95% và nới δ, bắt đầu từ y = 365, **phủ cả hai mắt**. Nhánh L khi đó chỉ còn trán và lông mày, không ai được nhận ra, và vùng đó không còn là "khẩu trang" nữa. Đây là lỗi đặc tả của Claude ở §2.2. Phát hiện trên canvas LFW, không dựa trên số nào của owner/p01.
- `maskset.py`: đường cắt trên của `L_i` dùng **landmark 195** (giữa sống mũi; LFW train: p5 403, trung vị 431, p95 454) thay cho landmark 6. Mọi thứ khác giữ nguyên: dịch oval xuống 27 px, Q = 0.05, δ = 8, F và họ G không đổi.
- Thêm hằng `L_TOP_LANDMARK = 195`. Trong `maskset_meta.json`: `landmarks.nose_bridge` ghi đúng chỉ số đang dùng (195) và thêm `landmarks.l_top_landmark: 195`.
- Thêm kiểm bắt buộc trong `build`: vùng mắt template = hai hình chữ nhật x ∈ [270, 350] và [450, 530], y ∈ [366, 394] (tâm mắt ± 40 px ngang, ± 14 px dọc). Mask `L` và `F` (sau δ) **không được giao** vùng mắt; giao → `SystemExit` (in số pixel giao). Ghi kết quả kiểm vào meta (`eye_zone_check`).
- Thêm vào `maskset_preview.png` một ảnh canvas (mỗi mask một màu, đánh dấu hai tâm mắt) để thấy hình mask không phụ thuộc mặt cụ thể.
- Test: dựng maskset trên landmark giả có landmark 195 ở dưới mắt → L không giao vùng mắt; landmark giả đặt 195 ngang mắt → `SystemExit`.
- Không chạy lại `prepare_landmarks`. Owner chỉ chạy lại `python -m innovation.defense.maskset build`.
