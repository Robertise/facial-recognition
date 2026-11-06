# Plan: mask gọng từ ảnh gọng thật + vòng tối ưu "gọng vuông" (giai đoạn A, digital)

Viết bởi Claude, 2026-11-06. Dành cho coding agent. **Không commit, không push. Không đổi tên CLI arg / key JSON / cột CSV / thư mục run có sẵn. Giữ nguyên `glasses.FrameShape` (khuôn elip) để tái lập các run cũ — chỉ THÊM.**

## 0. Bối cảnh
- Khuôn elip hiện tại (`FrameShape`): làm dày thì tròng đóng theo chiều dọc trước (R2b: tròng co về khe 2 px → thực chất là dải che mắt, không phải kính). Owner đổi sang **gọng thật** của owner (ảnh đính kèm trong chat, kiểu vuông/wayfarer, 2 tròng gần chữ nhật bo góc).
- Mục tiêu: tạo mask gọng từ ảnh đó trong hệ template (400×150, tâm 2 tròng = `glasses.EYES` = (110,80),(290,80), cách 180 px), **tròng luôn để trống**, chỉ cho phép nới gọng **ra ngoài**.

## 1. Tách mask từ ảnh (một lần, lưu artifact)
Thêm `innovation/tools/frame_mask_from_photo.py` (CLI). Input: đường dẫn ảnh gọng. Các bước:
1. Đọc ảnh. Phân vùng theo màu (HSV hoặc Lab):
   - `rim` = vùng tối (gọng navy/đen) trên nền giấy trắng;
   - `lens` = vùng xanh dương trong (hai tròng);
   - nền trắng = phần còn lại.
   Dùng ngưỡng + `cv2.morphologyEx` làm sạch; lấy 2 thành phần liên thông lớn nhất của `lens` làm hai tròng (bỏ bóng đổ).
2. Tìm tâm (centroid) hai tròng. Dựng affine map đưa tâm trái→(110,80), tâm phải→(290,80) (đường nối tâm nằm ngang, scale theo khoảng cách tâm→180 px). Warp vùng `frame_silhouette = rim ∪ lens` và vùng `lens` sang template 400×150.
3. `mask_base = frame_silhouette_warped AND NOT lens_warped` (viền gọng, tròng trống). Lưu:
   - `data/innovation/frame/owner_frame_mask.png` (mask template, 0/255);
   - `data/innovation/frame/owner_frame_lens.png` (vùng tròng phải để trống, để bước nới trừ ra);
   - `data/innovation/frame/owner_frame_meta.json`: px/cm đọc từ thước (nếu đọc được; owner xác nhận sau), độ phủ base, kích thước tròng/cầu mũi quy ra mm để đối chiếu số đo thô của owner (tròng ≈50×38–40 mm, cầu ≈18–20 mm, tổng ≈130 mm). In các số này ra, **không tự sửa nếu lệch** — báo để owner kiểm.
4. Lưu ảnh overlay (mask phủ lên ảnh gọng đã warp ngược) để owner nhìn mắt thường kiểm mask có đúng viền không: `data/innovation/frame/owner_frame_overlay.png`.

## 2. Mask có tham số nới ("grow outward only")
Thêm vào `glasses.py` một lớp/đường mới, KHÔNG sửa `FrameShape`:
- `class PhotoFrame`: `__init__(mask_path, lens_path, grow_px=0.0)`. `mask()` trả về template float32 [0,1]:
  - `grown = dilate(frame_silhouette, grow_px)`  (nới toàn bộ bóng gọng ra ngoài);
  - `mask = grown AND NOT lens`  (tròng luôn trống, không bao giờ bị nới đè);
  - anti-alias như `FrameShape` (supersample 4).
- `grow_px` là số px template nới ra ngoài (0 = đúng gọng thật). Đây là tham số độ phủ mới, thay cho `thickness/rim-scale` cho gọng ảnh.
- `optimize.py` và `evaluate.py`: thêm `--frame-mask PATH` + `--frame-lens PATH` + `--grow-px FLOAT`. Khi có `--frame-mask`, dùng `PhotoFrame` thay `FrameShape`; `attack.json` ghi `frame_source: "photo"`, đường dẫn mask, `grow_px`, và `coverage_train` như cũ. Khi không có, hành vi cũ nguyên vẹn. `evaluate.py` đọc hình gọng từ `attack.json` như hiện tại (đã đọc `frame_shape`); thêm nhánh đọc `PhotoFrame` khi `frame_source == "photo"`.

## 3. Đo độ phủ (trước khi chọn config)
`ladder.py --measure-coverage` (hoặc script tương đương) đo độ phủ TB trên frame s1 thường cho `grow_px ∈ {0, 4, 8, 12, 16}` với `PhotoFrame`, in bảng [min, mean, max]. Mục tiêu tham chiếu: tìm các grow cho độ phủ TB ~11% (ngang R2a, để so trực tiếp) và ~15% (ngang R2b, nhưng lần này tròng vẫn mở). **In kèm: với mỗi grow, phần trăm diện tích tròng còn trống** (phải ~100%, xác nhận tròng không bị đóng).

## 4. Test
`tests/innovation/`: (a) `PhotoFrame.mask()` với mask giả hình chữ nhật có 2 lỗ tròng → nới grow_px làm tăng diện tích rim nhưng **không đổi** diện tích lỗ tròng; (b) `--frame-mask` đẩy `frame_source="photo"` vào attack.json và evaluate đọc lại đúng; (c) đường cũ (`FrameShape`) không đổi.

## 5. Không làm
Không chạy optimize/ladder. Không in. Chạy pytest `tests/innovation/`, chạy riêng `frame_mask_from_photo.py` để sinh 3 artifact + bảng độ phủ, in `git diff --stat`. Owner xem overlay + bảng độ phủ rồi mới chốt grow cho vòng tối ưu.
