# Plan: signatures v2, thêm âm tính người thứ hai (p01)

Viết bởi Claude, 2026-11-07. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV có sẵn. Chỉ thêm.** Luật quyết định đã ghi trong `experiment_log.md` (mục 2026-11-07 "Luật đặt trước: S2 với mặt thật của người khác").

## Thay đổi trong `innovation/attacks/adv_glasses/signatures.py`
1. Thêm `--extra-gallery DIR` (mặc định không có). Nếu có: dựng gallery của DIR bằng `faces.build_gallery` (cùng checkpoint, cùng `GALLERY_PER_ID`) và **gộp** vào `gallery_embs` (id không được trùng; trùng thì dừng). Ghi danh sách id thêm vào `summary.json` (`extra_gallery_ids`).
2. Thêm `--people-sessions NAME/SESSION ...` (ví dụ `p01/s1`), đường dẫn tương đối với thư mục cha của `--frames-dir` (tức `data/innovation/frames`). Với mỗi session: chạy như `process_continuous_session` nhưng **chỉ các stream âm tính**: baseline, baseline_EOT, và control_random/control_plain của cả hai texture (C3, R2b); `target_id` = tên person (phần trước `/`), set = `R_<person>_<session>` (ví dụ `R_p01_s2`) cho baseline/EOT và `R_<person>_<session>_controls` cho controls. **Không gộp các session của cùng một người** (s1 và s2 của p01 khác điều kiện đeo kính). Không tạo stream tấn công trên frame người khác.
3. Khi có `--extra-gallery` hoặc `--people-sessions`, tạo run mới như v1 (`new_run("signatures", ...)`) và thêm vào `summary.json`:
   - `comparisons`: giữ các so sánh v1; thêm `A_C3_vs_R_<person>_<session>_all` và `A_R2b_vs_R_<person>_<session>_all` (âm tính = `R_<person>_<session>` + `R_<person>_<session>_controls`), cho mọi chữ ký như v1.
   - `operating_points`: cho `s2_maxdrop` và `s2_conc`: ngưỡng tại TPR 95% trên dương tính gộp (A_C3 + A_C3_EOT + A_R2b + A_R2b_EOT; ngưỡng = phân vị 5% của dương tính), rồi FPR trên từng nhóm âm tính: `R_owner` (+EOT), `R_owner_controls`, `R_lfw`, `R_<person>_<session>`, `R_<person>_<session>_controls`, `R_<person>_<session>_all`. Kèm số mẫu.
   - `rule_check`: cho mỗi biến thể S2: min AUC (C3, R2b) so với `R_<person>_<session>_all`, FPR `R_<person>_<session>_all` tại TPR 95%, cho **từng session**, và `pass` theo luật (AUC ≥ 0.90 và FPR ≤ 0.10); `s2_holds_for_other_people` = có ít nhất một biến thể pass **trên session được khai báo bằng `--primary-session`** (thêm tham số này, ví dụ `p01/s2`); các session khác chỉ báo số, không vào kết luận. Báo thêm cho mỗi session: số frame được nhận đúng người (top-1 = person, cos ≥ 0.5) trên tổng frame có mặt.
4. `frames.csv` thêm các dòng mới với cùng cột như v1.
5. Nếu nhiệm vụ gọng ảnh đã thêm `glasses.shape_from_attack`, dùng nó trong `load_texture_assets`; nếu chưa có thì giữ nguyên.

## Test
Trong `tests/innovation/test_signatures.py`: gộp gallery báo lỗi khi trùng id; session người khác chỉ sinh stream âm tính với `target_id` đúng; ngưỡng TPR 95% và FPR tính đúng trên dữ liệu giả.

## Chạy (owner chạy, agent không chạy)
`python -m innovation.attacks.adv_glasses.signatures --extra-gallery data/innovation/gallery_people --people-sessions p01/s2 p01/s1 --primary-session p01/s2`
