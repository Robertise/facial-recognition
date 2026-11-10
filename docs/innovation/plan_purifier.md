# Plan: model làm sạch vùng gọng (purifier) + kiểm nhất quán danh tính

Viết bởi Claude, 2026-11-09. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục có sẵn. Chỉ THÊM** (module mới `innovation/defense/`). Luật: `experiment_log.md` mục 2026-11-09 "Luật đặt trước: purifier".

## 0. Ý tưởng (để agent hiểu, không phải để báo cáo)
Phụ kiện adversarial nằm trong một vùng biết trước quanh mắt (vùng gọng). Purifier **luôn xoá toàn bộ vùng đó** trên crop 112 rồi vẽ lại bằng U-Net inpainting. Vì đầu vào vùng đó bị xoá hết, hoa văn bên trong vùng **không ảnh hưởng được** đầu ra (đúng với mọi kẻ tấn công, kể cả thích ứng, miễn phụ kiện nằm trong vùng). Cái giá là mặt thật cũng mất vùng mắt → phải đo người thật có bị từ chối nhầm không. Gallery cũng đi qua purifier (đối xứng), để người đeo kính cận khi enroll không bị lệch.

## 1. Vùng gọng `band`
- `innovation/defense/band.py`: dựng `band_template.png` (400×150, template của `glasses.py`) = hợp của: `FrameShape` của C3 và R2b (đọc từ `attack.json` của `20261104_140557` và `20261106_132526`), `PhotoFrame` owner với `grow_px=16, clearance_mm=0` (fill-lens), rồi dilate 4 px. Lưu `data/innovation/defense/band_template.png` + `band_meta.json` (diện tích template, các nguồn).
- Hàm `band_in_crop(img, bbox, eyes) -> mask112`: warp `band_template` vào frame bằng `glasses.similarity(eyes)` (như `paint`), rồi cắt theo `faces.crop_window(bbox, ...)` và resize 112 (như `glasses.coverage`). Nếu không có landmark mắt: trả về None (frame đó đánh dấu `band_ok=0`).
- In độ phủ trung bình của band trên crop 112 cho `owner/s1_day1` (để ghi report).

## 2. Dữ liệu train (chỉ LFW, không ảnh owner/p01)
- `innovation/defense/prepare.py`: duyệt `data/public/lfw/lfw_home/lfw_funneled`. **Loại** mọi danh tính LFW đã dùng ở `data/innovation/gallery/` (đọc tên gốc từ `_build_report.json` / `_candidates.csv`; nếu không map được tên nào thì SystemExit). Chia **theo danh tính**: 90% train / 10% val, seed 0; ghi `split_ids.json`.
- Mỗi ảnh: detector + crop + enhance như backend (`faces.crop`, `faces.enhance`), landmark mắt, `band_in_crop`. Bỏ ảnh không có mặt/mắt (đếm lại). Lưu shard `.npz` (crop uint8 112×112×3, band uint8) vào `data/innovation/defense/lfw_crops/`. Ghi `prepare_report.json`.
- Augment lúc train: lật ngang (lật cả band), jitter sáng/tương phản nhẹ, dịch band ±3 px (bền với sai số landmark).

## 3. Model và train
- `innovation/defense/purifier.py`: U-Net nhỏ (4 mức, kênh 32-64-128-256, ~2–4M tham số). Đầu vào 4 kênh: crop với vùng band thay bằng xám 0.5 + mask band. Đầu ra 3 kênh; **ghép**: ngoài band giữ nguyên pixel gốc, trong band lấy output.
- Loss: L1 trong band ×1.0 + L1 toàn ảnh ×0.1 + **identity loss** = 1 − cos(emb(method3, output), emb(method3, clean)) ×0.5. Method3 đóng băng (`checkpoints/method3_best.pth`, nạp qua `backend.face_recognizer.FaceRecognizer`, dùng đúng tiền xử lý của `extract_embedding` nhưng differentiable).
- `innovation/defense/train_purifier.py`: `--data`, `--epochs 30`, `--batch 64`, `--lr 2e-4` (Adam), `--device`, `--seed 0`, `--out`. Lưu checkpoint tốt nhất theo identity cos trên val, `train_log.csv` (loss các thành phần, val cos), `meta.json`. Chạy được trên CPU (chậm) và CUDA. Thư mục kết quả `results/innovation/<ts>_purifier_train/`.
- Baseline không học: `gray` = chỉ thay band bằng xám 0.5 (cùng band). Đây là mốc so sánh.

## 4. Đánh giá `innovation/defense/eval_purifier.py`
Chạy qua đúng luồng backend: frame → detector → crop → enhance → **[purifier hoặc gray hoặc none]** → `extract_embedding` → `find_match` (ngưỡng 0.5). Liveness tính trên frame gốc như check-in, smoother theo session, không warmup (như `--eval-session`). Gallery: ảnh gallery cũng đi qua cùng chế độ (none/gray/purifier) trước khi embed; cache theo chế độ.
- `--modes none gray purifier`, `--purifier-ckpt`.
- **Tập tấn công (digital, dán như `evaluate.paint`, tier D):** `H09` trên `owner/s6_vary` (run `20261107_214538`), `R2b` trên `owner/s6_vary` (run `20261106_132526`), và tuỳ chọn `--attack NAME=RUN:SESSIONDIR` (dùng cho p01 khi có). Gallery 1:N = gallery chính (+ `--extra-gallery`).
- **Tập người thật:** `owner/s6_vary` baseline (đích = owner), `p01/s2` (đích = p01, cần `--extra-gallery data/innovation/gallery_people`), owner đeo control_random/control_plain của H09 trên s6 (đích = owner), và `p01/acc_*` nếu có.
- Mỗi frame ghi `frames.csv`: mode, set, frame, band_ok, match_id, match_sim, top1 trước/sau (so với mode none), sim đích, is_real, success.
- `summary.json` theo (mode, set): tỉ lệ thành công tấn công (khớp nạn nhân + liveness), **GAR** = tỉ lệ frame người thật khớp đúng danh tính (+ liveness), Wilson 95%, tỉ lệ `band_ok`.
- **Bộ phát hiện nhất quán** (`consistency`): điểm = sim với top1 của mode none trừ sim với chính danh tính đó ở mode purifier (giảm điểm danh tính khi làm sạch). Báo AUC (tấn công vs người thật) cho mode purifier và gray, ngưỡng tại TPR 95% trên tấn công, FPR trên từng tập người thật.
- Kết quả: `results/innovation/<ts>_purifier_eval/`.

## 5. Test
- `band_in_crop` trên frame giả: mask khớp vùng `paint` của PhotoFrame fill-lens (mọi pixel gọng H09 nằm trong band).
- Ghép đầu ra: pixel ngoài band không đổi.
- **Bất biến:** hai crop chỉ khác nhau trong band → đầu ra purifier giống hệt (assert bằng nhau).
- `prepare` loại đúng danh tính gallery; split theo danh tính không giao nhau.
- Eval chạy được trên fixture nhỏ với purifier ngẫu nhiên.

## 6. Chạy
Agent: pytest `tests/innovation/`; `git diff --stat`; chạy `band.py` (in độ phủ band) và `prepare.py --limit 200` thử. **Không train thật.**
Owner:
```
python -m innovation.defense.band
python -m innovation.defense.prepare
python -m innovation.defense.train_purifier --data data/innovation/defense/lfw_crops --device cuda     # vast.ai; CPU rất chậm
python -m innovation.defense.eval_purifier --modes none gray purifier --purifier-ckpt results/innovation/<run train>/best.pt --extra-gallery data/innovation/gallery_people --liveness
```
Nếu train trên vast.ai: chỉ cần mang `innovation/`, `backend/` (phần recognizer), `checkpoints/method3_best.pth`, `data/innovation/defense/lfw_crops/`. Không mang ảnh owner/p01.

## 7. Sửa sau verify (2026-11-10): fail-closed khi không dựng được band
- Hiện tại `transform_crop112` trả nguyên crop khi `band112 is None` → frame không có landmark mắt đi thẳng vào nhận dạng không qua làm sạch (fail-open). Hoa văn adversarial có thể làm hỏng landmark để lách purifier.
- Sửa trong `eval_purifier.py` (không đổi tên cột/khoá có sẵn): ở mode `gray` và `purifier`, nếu `band_ok == 0` thì **từ chối frame**: `match_id = ""`, `success = 0` (cho cả tấn công lẫn người thật), `top1_after_id = ""`. Mode `none` giữ nguyên. Thêm cột mới `rejected_no_band` (0/1).
- Bộ phát hiện nhất quán: frame `band_ok == 0` được coi là **nghi ngờ tối đa** (điểm `top1_drop` thay bằng `+inf` khi tính AUC/ngưỡng; trong CSV ghi `top1_drop = ""` và `rejected_no_band = 1`).
- `summary`: thêm `rejected_no_band_rate` cho mỗi (mode, set).
- Nếu thiếu tập `p01_s2` thì `rule_check.purifier_pass = null` (luật cần cả hai tập người thật), không tự hạ xuống chỉ owner.
- Test: frame không có mắt ở mode purifier → `success = 0`, `rejected_no_band = 1`; ở mode none không đổi.
