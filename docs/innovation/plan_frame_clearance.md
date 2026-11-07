# Plan: mask gọng "fill-lens + lỗ chừa đồng tử" (H-ladder)

Viết bởi Claude, 2026-11-07. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục / cấu hình có sẵn (gồm G0–G15, C0–C4, R2*). Chỉ THÊM.** Luật: `experiment_log.md` mục 2026-11-07 "Luật đặt trước: H-ladder".

## 1. `glasses.PhotoFrame`: thêm clearance, giữ hành vi cũ
- Thêm tham số khởi tạo `clearance_mm: float = 0.0` và `ipd_mm: float = 63.0`. Khi `clearance_mm <= 0`: `mask()` **giữ nguyên hành vi hiện tại** (outward-only, không đụng tròng) — bắt buộc, để mọi run cũ tái lập.
- Khi `clearance_mm > 0`: `mask()` =
  1. `sil = rim ∪ lens` (đọc từ `mask_path` và `lens_path`);
  2. `grown = dilate(sil, grow_px)` (giữ `grow_px` làm g_out);
  3. trừ hai đĩa bán kính `clearance_mm * (180.0 / ipd_mm)` px ở tâm mắt template `(110,80)` và `(290,80)`;
  4. anti-alias như cũ (supersample 4). Tròng (phần trong hai đĩa) để trống.
- `shape_from_attack`: đọc thêm `clearance_mm`, `ipd_mm` từ attack khi `frame_source=="photo"`. `optimize.py`/`evaluate.py`: thêm `--clearance-mm` (mặc định 0.0) và `--ipd-mm` (mặc định 63.0); ghi vào `attack.json`/`texture.pt` cùng `grow_px`. Không có `--clearance-mm` thì như cũ.

## 1b. Cắt ngàm + regenerate base mask (trong `frame_mask_from_photo.py`)
- Trước khi lưu `owner_frame_mask.png`/`owner_frame_lens.png`: cắt **chỉ hai ngàm ở mép ngoài**. Chỉ xử lý các cột nằm ngoài phạm vi ngang của tròng (x < mép trái tròng trái, x > mép phải tròng phải); trong hai dải đó, bỏ phần mỏng theo chiều dọc (vertical opening kernel 1×51). **Không áp lên phần giữa**: cầu mũi chỉ cao ~18 px nên opening toàn ảnh sẽ xoá mất nó (lỗi đã gặp ở prototype của Claude). Gán lại rim = (silhouette_cut AND NOT lens). In x-range trước/sau (kỳ vọng ~0–399 → ~22–381) và xác nhận cột giữa (x ≈ 200) vẫn còn pixel gọng.
- Sao lưu mask hiện tại sang `data/innovation/frame/_v2/` trước khi ghi đè. Tham số `--keep-hinges` (mặc định tắt) để bỏ qua bước cắt nếu cần.
- Tùy chọn `--side-thicken PX` (mặc định 0): sau cắt ngàm, nở ngang mép ngoài (band 45 px mỗi bên) thêm PX px. Chỉ thẩm mỹ, bù ~0.1% độ phủ.

## 2. Bảng độ phủ
`frame_mask_from_photo.py --measure-coverage` (và/hoặc `ladder --measure-coverage`): thêm chế độ đo theo `clearance_mm ∈ {16,14,13,12,11,10,9}` với `grow_px` cố định (tham số `--g-out`, mặc định 8). Mỗi dòng: độ phủ TB [min,max] trên s1 thường, **% diện tích tròng còn hở**, % gọng chạm biên template, clearance ra mm và px. Xác nhận lại các số prototype (g_out 8: clr14→~9.9%, clr11→~11.0%).

## 3. Cấu hình H-ladder trong `ladder.py`
- Thêm `H16,H13,H11,H09` vào `LADDER_CONFIGS`: `frame_mask`/`frame_lens` như G-ladder, `grow_px` = `--g-out` (CLI mới, mặc định 8), `clearance_mm` = 16/13/11/9, `tex_res` [15,40], `no_eot` False. Truyền `--clearance-mm`/`--grow-px`/`--ipd-mm` cho optimize.
- `H_ORDER = ["H16","H13","H11","H09"]`; luật chọn dùng `select_config` hiện có với thứ tự này. Thêm cột `clearance_mm` vào `ladder.csv` (cột mới cuối). In lệnh eval s6 cho cả 4 (cấu hình chọn kèm transfer), như G-ladder.

## 4. Test
`PhotoFrame` với `clearance_mm=0` cho mask y hệt trước (so mảng); với clearance>0 thì hai đĩa ở tâm mắt trống và diện tích tròng-hở khớp bán kính; cầu mũi (cột x≈200, y 60–100) vẫn có pixel gọng **sau khi cắt ngàm**; ladder truyền đúng cờ; cột `clearance_mm` có trong csv.

## 5. Chạy
Agent: pytest `tests/innovation/`; `python -m innovation.attacks.adv_glasses.ladder --measure-coverage --frame-mask data/innovation/frame/owner_frame_mask.png --frame-lens data/innovation/frame/owner_frame_lens.png --g-out 8` (in bảng clearance); in `git diff --stat`. **Không chạy ladder tối ưu.** Owner gửi bảng cho Claude chốt g_out + 4 clearance rồi chạy:
`python -m innovation.attacks.adv_glasses.ladder --configs H16 H13 H11 H09 --g-out 8`
