# Plan: sửa mask gọng ảnh + thang độ phủ G-ladder

Viết bởi Claude, 2026-11-07. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục có sẵn. Chỉ thêm.** Luật quyết định: `experiment_log.md`, mục 2026-11-07 "Luật đặt trước ... G-ladder".

## 1. Sửa mask (`innovation/tools/frame_mask_from_photo.py`)
Overlay hiện tại: mép trên dính bóng đổ trên giấy (bóng xám, sáng hơn gọng rất nhiều), viền răng cưa theo vân giấy.
1. Phân vùng gọng chặt hơn: gọng thật rất tối (đo lại trên ảnh; gọng ≈ L < 60–70 trong Lab, bóng đổ ≈ L 120–170). Thêm tham số `--rim-max-l` (mặc định chọn sao cho tách gọng khỏi bóng; in giá trị đã dùng). Giữ thành phần liên thông lớn nhất làm gọng.
2. Làm sạch: close rồi open với kernel tỉ lệ theo px/cm (≈ 0.5 mm), lấp lỗ nhỏ bên trong viền, làm mượt đường bao (ví dụ Gaussian blur mask rồi threshold, hoặc `approxPolyDP` với epsilon ≈ 0.3 mm). Tròng giữ nguyên phân vùng hiện tại.
3. Ghi đè 4 artifact trong `data/innovation/frame/` như cũ, **trước đó sao lưu bản cũ** sang `data/innovation/frame/_v1/`. Overlay phải cho thấy mép trên không còn bóng.
4. In lại số đo mm (tròng, cầu mũi, tổng rộng, cao) để so với lần trước (48.0×35.7 mm, 23.2 mm, 150.4 mm, 45.7 mm).

## 2. Bảng độ phủ mở rộng
`--measure-coverage` đo grow ∈ {0, 4, 8, …, 64} px. Mỗi dòng: độ phủ TB [min, max] trên frame s1 thường, % tròng còn mở, **grow quy ra mm** (grow × ipd_mm / 180), và **% bóng gọng đã nới bị cắt ở biên template 400×150**. In thêm: grow có độ phủ TB gần 11.0% nhất và gần 15.0% nhất. Lưu bảng vào `owner_frame_meta.json` (`s1_coverage_ladder`).

## 3. Cấu hình G-ladder trong `ladder.py`
- Thêm 4 config vào `LADDER_CONFIGS`: `G0`, `G16`, `G11`, `G15`, mỗi cái có `frame_mask` = `data/innovation/frame/owner_frame_mask.png`, `frame_lens` = `data/innovation/frame/owner_frame_lens.png`, `grow_px`, `tex_res` [15, 40], `no_eot` False. Grow của G11/G15 đọc từ tham số CLI mới `--g11-grow` và `--g15-grow` (bắt buộc khi chạy G11/G15; không đoán).
- Khi config có `frame_mask`, `ladder.py` truyền `--frame-mask/--frame-lens/--grow-px` cho optimize thay cho `--thickness/--rim-scale`.
- Thêm `GROW_ORDER = ["G0", "G16", "G11", "G15"]`; khi các config chạy đều là G*, luật chọn dùng thứ tự này (cùng `select_config` hiện có: đầu tiên ≥ 80%, không có thì cao nhất).
- Thêm cột `grow_px` vào `ladder.csv` (thêm cột mới ở cuối, không đổi cột cũ).
- Cuối ladder, in sẵn lệnh eval trên s6 cho **cả 4 run**: config được chọn có `--eval-session s6_vary --liveness --transfer method1=… method2=… method4=… method5=…`; 3 config còn lại chỉ `--eval-session s6_vary --liveness`. Không tự chạy.

## 4. Test
Mask sau khi sửa: tròng vẫn mở 100% ở mọi grow; ladder truyền đúng cờ PhotoFrame cho config G*; cột `grow_px` có trong `ladder.csv`; luật chọn theo `GROW_ORDER`.

## 5. Chạy
Agent chạy: pytest `tests/innovation/`; `python -m innovation.tools.frame_mask_from_photo --image data/innovation/frame/owner_frame_photo.png --px-per-cm 152 --ipd-mm 63 --measure-coverage`; in `git diff --stat`. **Không chạy ladder.** Owner gửi bảng độ phủ cho Claude chốt grow G11/G15 rồi mới chạy:
`python -m innovation.attacks.adv_glasses.ladder --configs G0 G16 G11 G15 --g11-grow <x> --g15-grow <y>`
