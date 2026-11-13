# Plan: tấn công thích ứng chống purifier (vành ngoài vùng gọng)

Viết bởi Claude, 2026-11-12. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục có sẵn. Không sửa hằng số hay hàm có sẵn trong `glasses.py`, `optimize.py`, `eval_purifier.py` (chỉ import và dùng lại). Chỉ THÊM** module mới `innovation/defense/adaptive.py` (+ test). Luật: `experiment_log.md` mục 2026-11-12 "Luật đặt trước: tấn công thích ứng".

## 0. Ý tưởng
Purifier luôn xoá vùng band, nên hoa văn trong band vô dụng (đã có test bất biến). Kẻ tấn công thích ứng biết toàn bộ pipeline (band, U-Net, recognizer, bộ phát hiện nhất quán) và đặt hoa văn lên **vành bao ngoài band** rộng r px (đơn vị px template, 180 px = 63 mm, tức ~2.86 px/mm). Đo tỉ lệ thành công theo r để ra "cái giá" của kẻ tấn công.

## 1. Khung mở rộng (không đụng `glasses.TEMPLATE_W/H`)
- Band phủ 95% khung 400×150, nên vành sẽ tràn khung. Dùng khung đệm P = 56 px mỗi phía: `W' = 512, H' = 262`, mắt template tại `(110+P, 80+P)`, `(290+P, 80+P)`.
- Hàm riêng trong `adaptive.py`: `similarity_padded(eyes)` = `glasses.similarity(eyes, src_eyes=EYES + P)`; `grid_padded(A, window)` và `render_padded(region, texture, mask, A, window)` như `glasses._grid`/`glasses.render` nhưng dùng `W'`, `H'`; `coverage_padded(mask, A, window)` như `glasses.coverage`.
- `band_padded` = `band_template.png` dán vào giữa khung đệm. `ring(r) = dilate(band_padded, ellipse r) AND NOT band_padded`. Lưu `ring_r{r}.png` và độ phủ vành trên crop 112 (owner/s1, TB/min/max) vào `ring_meta.json`.
- Kiểm: `ring(r)` warp vào crop 112 không giao với `band_in_crop` (trừ sai số nội suy ≤ 1 px biên, đếm và in ra).

## 2. Tối ưu `python -m innovation.defense.adaptive optimize`
- Cùng dữ liệu và cách chia như H09: `--attacker-frames data/innovation/frames/owner`, `--eval-sessions s3_vary s4_vary s5_vary s6_vary`, train s1+s2, dev block 20%, attacker owner, victim LFW_victim, gallery chính (không extra), method3, `--steps 3000 --lr 0.03 --seed 0`, EOT bật (dùng lại `eot` như optimize), M = identity (digital, kẻ tấn công mạnh nhất), TV và NPS giữ trọng số mặc định của optimize. Tham số texture: lưới `--tex-res 20 51` (cùng mật độ ô với 15×40 trên 150×400), upsample lên `H'×W'`.
- `--ring-px r` (bắt buộc), `--purifier-ckpt` (mặc định seed 0: `results/innovation/20261112_004047_purifier_train/best.pt`).
- Mỗi bước, với mỗi frame trong batch: dán hoa văn lên vành (`render_padded`), lấy crop 112 như optimize, rồi hai nhánh:
  - **nhánh purifier:** band của frame (tính trước bằng `band_in_crop` trên frame gốc, giữ cố định), U-Net đóng băng, method3 → loss giả danh như optimize (soft-max cos tới victim, margin với người khác gồm attacker);
  - **nhánh none:** crop đã dán, method3 → cùng loss.
  - Tổng loss = loss_purifier + loss_none + TV + NPS. (Muốn qua bộ phát hiện nhất quán thì phải là victim ở cả hai nhánh.)
- Ghi như optimize: `attack.json` (thêm `adaptive: true`, `ring_px`, `canvas_pad`, `purifier_ckpt`, `tex_res`), `split.json`, `texture.pt`, `mask.png` (vành, khung đệm), `loss.csv`, `meta.json`. Thư mục `results/innovation/<ts>_adaptive_optimize/`.
- Chạy được CPU và CUDA (`--device`).

## 3. Đánh giá `python -m innovation.defense.adaptive eval`
- `--run <adaptive run>`, `--eval-session s6_vary`, `--liveness`, `--purifier-ckpt` (giống lúc tối ưu), `--consistency-threshold` (mặc định đọc `consistency.purifier.threshold_tpr95` từ `results/innovation/20261112_103208_purifier_eval/summary.json` = 0.4253; ghi giá trị dùng vào summary).
- Kiểm rò rỉ như `eval_purifier.check_session_not_in_train_dev`.
- Tầng: `D` (hoa văn tối ưu trên vành) và `control_random` (hoán vị ô, như optimize). Mỗi tầng chấm 3 mode none / gray / purifier, gallery qua cùng mode, fail-closed khi thiếu band (giống eval_purifier §7). Dùng lại hàm của `eval_purifier` (transform_crop112, build_gallery_for_mode, điểm nhất quán).
- Mỗi frame: success_recog (khớp victim ở mode đó + liveness), flagged (điểm nhất quán ≥ ngưỡng, so mode purifier với none), **success_full = success_recog ở mode purifier AND NOT flagged**.
- `summary.json`: theo (tầng, mode) rate + Wilson 95%; riêng mode purifier thêm `success_full_rate`, `flag_rate`; `ring_coverage_s6` (độ phủ vành trên crop 112 của s6, TB/min/max). Thư mục `<ts>_adaptive_eval/`.

## 4. Test
- `render_padded` với P = 0 và mask 400×150 cho kết quả trùng `glasses.render`.
- `ring(r)` rời band; r tăng thì độ phủ tăng.
- Gradient của loss nhánh purifier theo texture **bằng 0** ở các ô chỉ nằm trong band (xác nhận bất biến từ đầu đến cuối).
- Eval chạy trên fixture nhỏ (purifier ngẫu nhiên), có các key ở §3.

## 5. Chạy
Agent: pytest `tests/innovation/`; `git diff --stat`; chạy `adaptive ring` (dựng vành + in độ phủ) cho r ∈ {8, 16, 32, 48}; chạy thử optimize `--steps 5` trên CPU. **Không chạy tối ưu thật.**
Owner (mỗi r một lần, r ∈ {8, 16, 32, 48}):
```
python -m innovation.defense.adaptive optimize --ring-px 8 --device cuda
python -m innovation.defense.adaptive eval --run results/innovation/<run r8> --eval-session s6_vary --liveness
```

## 6. Sửa sau verify (2026-11-12)
- `run_eval_adaptive`: thêm tham số `output_dir: Optional[Path] = None`; nếu có thì ghi vào đó thay vì `new_run`. CLI `eval` thêm `--output-dir` (mặc định không có = như hiện tại).
- `test_adaptive.py`: truyền `output_dir=tmp_path / "eval_out"`; không test nào được ghi vào `results/innovation/`.
- Không đổi gì khác.

## 7. Sửa sau kết quả 11-12: nhánh purifier phải chấm với gallery mode purifier (2026-11-13)
Lý do: `experiment_log.md` mục 2026-11-13. Trong `cmd_optimize`, nhánh purifier đang so với `E` (gallery sạch từ `faces.build_gallery`), còn `run_eval_adaptive` so với gallery đi qua purifier. Kẻ tấn công vì vậy tối ưu tới template sai.

Chỉ sửa `innovation/defense/adaptive.py` (hàm `cmd_optimize`, có thể thêm một hàm phụ mới) và `tests/innovation/test_adaptive.py` (chỉ thêm test). **Không commit. Không đổi tên** CLI arg, key JSON, cột `loss.csv` / `frames.csv`. **Không đổi** `run_eval_adaptive`, nhánh none, tham số mặc định, hay file nào khác.

### 7.1 `cmd_optimize`
- Sau khi nạp `purifier_model` (đã `.eval()`), dựng gallery mode purifier bằng **đúng lời gọi như trong `run_eval_adaptive`**:
```
gal_pur, g_pur_report = build_gallery_for_mode(
    root=Path(args.gallery), recognizer=recognizer, detector=detector, eyes_finder=eyes_finder,
    mode="purifier", band_template=band_tmpl, purifier_model=purifier_model,
    purifier_tag=purifier_ckpt.stem, device=args.device,
    max_per_id=args.max_gallery, cache_dir=run.parent / "_cache",
)
E_pur = {k: torch.from_numpy(v).float().to(args.device) for k, v in gal_pur.embeddings.items()}
```
- Kiểm: `set(E_pur) == set(E)`, khác thì `SystemExit` in hai phần chênh; `args.victim` và `args.attacker_id` phải có trong `E_pur`.
- Nhánh purifier: `s_target_pur` dùng `E_pur[target]`; `s_others_pur` dùng `E_pur[k]` với cùng danh sách `others`. Trong khối log: `hard_pur` dùng `E_pur`. Nhánh none, `hard_none`, `target`, `others` giữ nguyên dùng `E`.
- In một dòng lúc khởi động: số danh tính trong `E_pur`, `g_pur_report[victim]`, `g_pur_report[attacker_id]`.
- `attack.json`: **thêm** hai key (không đổi key cũ): `"purifier_branch_gallery": "purifier"` và `"gallery_pur_report": g_pur_report`.
- Ghi chú, không sửa: nhánh none dùng `faces.build_gallery` còn eval none dùng `build_gallery_for_mode(mode="none")`; hai đường chỉ khác bước resize, và nhánh none đã chuyển sang s6 gần nguyên (log 11-13), nên giữ để không đổi cấu hình nhánh none.

### 7.2 Test (`tests/innovation/test_adaptive.py`, chỉ thêm)
- Test A: trên fixture nhỏ (purifier khởi tạo ngẫu nhiên như test eval hiện có), chạy `cmd_optimize` với `--steps 2 --device cpu`, thư mục run nằm trong `tmp_path` (monkeypatch `new_run` hoặc tương đương). Không test nào được ghi vào `results/innovation/`. Kiểm `attack.json["purifier_branch_gallery"] == "purifier"` và có `gallery_pur_report` chứa victim.
- Test B: embedding target mà nhánh purifier dùng phải trùng (`allclose`, atol 1e-5) với `build_gallery_for_mode(mode="purifier")` gọi bằng cùng đối số như `run_eval_adaptive`, và phải khác embedding gallery sạch (purifier ngẫu nhiên làm đổi ảnh). Nếu cần để test được, tách phần dựng `E` / `E_pur` thành một hàm mới trong `adaptive.py` (tên mới, chỉ thêm) và `cmd_optimize` gọi lại hàm đó.
- Mọi test cũ (kể cả gradient bằng 0 trong band) phải vẫn qua.

### 7.3 Agent chạy và báo
- `pytest tests/innovation/`: báo số pass / fail; `.pytest_cache/v/cache/lastfailed` phải rỗng.
- `git status`, `git diff --stat`, `git diff --ignore-cr-at-eol innovation/defense/adaptive.py tests/innovation/test_adaptive.py`.
- Chạy thử `python -m innovation.defense.adaptive optimize --ring-px 8 --steps 5 --device cpu`. Báo tên thư mục, `prepare_report`, `purifier_branch_gallery`, `gallery_pur_report["LFW_victim"]`, `gallery_pur_report["owner"]`. Sau đó xoá đúng thư mục chạy thử đó.
- Kiểm `results/innovation/` không có thư mục mới nào khác (do pytest hay chạy thử).
- **Không chạy tối ưu thật. Không chạy eval. Không commit.**
