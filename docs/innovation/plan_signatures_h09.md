# Plan: thêm tấn công H09 vào `signatures.py` (S2 trên gọng thật)

Viết bởi Claude, 2026-11-09. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / set có sẵn (A_C3, A_R2b, R_owner, ...). Chỉ THÊM.** Luật: `experiment_log.md` mục 2026-11-09 "Luật đặt trước: S2 trên H09".

## 1. CLI
- Thêm `--extra-attack NAME=RUN_DIR [NAME=RUN_DIR ...]` (mặc định không có). NAME khớp `[A-Za-z0-9]+`, không trùng `C3`/`R2b`.
- Mỗi extra attack dùng `load_texture_assets(RUN_DIR)` như C3/R2b (đã hỗ trợ PhotoFrame qua `shape_from_attack`).

## 2. Áp dụng lên session nào (chống rò rỉ)
- Một extra attack chỉ được dán lên session owner mà nó **không** dùng để tối ưu: đọc `RUN_DIR/split.json`, lấy tên session có trong `eval` (bỏ hết session có trong `train`/`dev`). Trong các session owner mà script đang xử lý (s2_day2, s5_vary, s6_vary), chỉ áp lên những session thuộc danh sách eval đó. Với H09 (`20261107_214538`): s5_vary, s6_vary. In ra session được áp cho từng NAME.
- Thêm stream: `A_<NAME>` (tier D), `A_<NAME>_EOT` (tier D_EOT), và control của nó vào `R_owner_controls` với tier `control_random_<NAME>`, `control_plain_<NAME>`. Với people sessions (âm tính), thêm control `control_random_<NAME>`/`control_plain_<NAME>` vào `R_<person>_<session>_controls` như C3/R2b.
- Quy tắc lọc frame tấn công giữ nguyên như C3/R2b (frame tấn công thất bại bị bỏ, đếm vào `attack_failed`).

## 3. Chỉ số mới (giữ nguyên mọi chỉ số cũ)
- Comparisons: `A_<NAME>_vs_pooled_negatives`, `A_<NAME>_vs_R_owner_controls`, `A_<NAME>_vs_R_lfw`, `A_<NAME>_vs_R_<tag>_all` cho mỗi people session.
- Operating point riêng: ngưỡng tại TPR 95% trên `A_<NAME>` + `A_<NAME>_EOT`, báo FPR trên từng tập âm tính (owner, controls, LFW, mỗi people session). Ghi vào key mới `operating_points_extra[NAME]`.
- `rule_check_extra[NAME]` cho `s2_maxdrop` và `s2_conc`: `auc_vs_primary` (AUC so với `R_<primary-session>_all`), `fpr_primary_at_tpr95`, `pass` = AUC ≥ 0.90 và FPR ≤ 0.10. `rule_check` cũ giữ nguyên.
- Histogram: thêm một hình riêng cho mỗi NAME (không sửa hình cũ).

## 4. Test
- Không có `--extra-attack`: `summary`/`frames.csv` như cũ (so key và số dòng trên fixture).
- Extra attack có `train` chứa s2 → không áp lên s2.
- NAME trùng C3 → SystemExit.

## 5. Chạy
Agent: `pytest tests/innovation/`; `git diff --stat`. **Không chạy study thật.** Owner chạy:
```
python -m innovation.attacks.adv_glasses.signatures --extra-gallery data/innovation/gallery_people --people-sessions p01/s2 p01/s1 --primary-session p01/s2 --extra-attack H09=results/innovation/20261107_214538_advglasses_impersonate
```
