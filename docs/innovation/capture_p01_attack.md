# Hướng dẫn quay cho người hỗ trợ (p01) — 2026-11-09

Gửi nguyên phần **A** cho người hỗ trợ. Phần **B** là việc của owner.

## A. Phần người hỗ trợ đọc

Cảm ơn bạn đã giúp lần nữa. Lần này mình cần bạn quay **4 đoạn bắt buộc + 3 đoạn tuỳ chọn**, mỗi đoạn khoảng 15 giây, tổng khoảng 10–15 phút. Làm y như lần trước: mở link mình gửi, vào trang **Frame capture**, cho phép webcam.

### Chuẩn bị (giữ nguyên cho mọi đoạn)
- **Chỗ ngồi:** cùng một chỗ, cùng laptop/webcam, quay hết trong một lần ngồi.
- **Ánh sáng:** nguồn sáng ở **phía trước mặt** (cửa sổ hoặc đèn hướng vào mặt). **Không** ngồi quay lưng ra cửa sổ hay đèn sáng phía sau. Tránh đèn rọi thẳng làm loá tròng kính hoặc bóng da.
- **Khung hình:** ngồi cách webcam khoảng **50–70 cm**; mặt chiếm khoảng 1/4–1/3 bề ngang khung, thấy đủ trán và cằm; webcam ngang tầm mắt.
- **Phía sau:** không có người khác, không có ảnh/poster/màn hình có mặt người lọt vào khung.
- **Kính:** đeo **kính cận bình thường của bạn** (như lúc enroll) cho các đoạn a1–a4. Không đội mũ, không khẩu trang, tóc không che lông mày.
- **Cài đặt trên trang:** `Person` = `p01_attack` (đoạn tuỳ chọn dùng `p01`), `Frames` = **75**, `Every (ms)` = **200**.
- Mỗi tên session **chỉ bấm Capture một lần**. Nếu lỡ dừng giữa chừng hoặc quay hỏng, **đừng quay đè**: quay lại với tên mới thêm `_r2` (ví dụ `a1_day_r2`) và nhắn mình.

### Các đoạn bắt buộc (`Person` = `p01_attack`)
| Thứ tự | Session | Cử động |
|---|---|---|
| 1 | `a1_day` | Nhìn webcam như đang điểm danh. Cử động tự nhiên nhẹ: chớp mắt, gật nhẹ, quay đầu rất ít (≤ 10–15°), có thể nói hoặc cười nhẹ. |
| 2 | `a2_day` | Đứng dậy, ngồi lại chỗ cũ (lệch tư thế một chút là được), rồi làm y như a1. |
| 3 | `a3_vary` | Thay đổi vừa phải, **chậm**: quay đầu trái rồi phải khoảng 20° (không quay nghiêng hẳn), ngẩng nhẹ, cúi nhẹ, cười, nói vài câu, nghiêng người tới gần/lùi xa một chút nhưng mặt vẫn trong khung. Không đưa tay che mặt, không quay nhanh. |
| 4 | `a4_vary` | Làm lại y như a3 (đây là đoạn dự phòng). |

### Các đoạn tuỳ chọn, nếu tiện (`Person` = `p01`)
Cử động như a1 (điểm danh bình thường), chỉ thay phụ kiện:
| Session | Phụ kiện |
|---|---|
| `acc_sun` | Kính râm thay kính cận |
| `acc_mask` | Khẩu trang y tế (vẫn đeo kính cận) |
| `acc_hat` | Mũ lưỡi trai (vẫn đeo kính cận) |

Xong thì nhắn mình là đã quay đoạn nào. Cảm ơn bạn!

## B. Phần owner
1. Mở tunnel như lần thu p01 trước (HANDOFF_20261107 §5): backend bật research mode + `EXTRA_CORS_ORIGINS`, frontend với `NEXT_PUBLIC_API_BASE`, cloudflared quick tunnel cho cổng 3000 và 8000. **Tắt tunnel ngay khi quay xong** (backend không có auth).
2. Kiểm tra trên máy: `data/innovation/frames/p01_attack/{a1_day,a2_day,a3_vary,a4_vary}` mỗi thư mục 75 ảnh, `meta.csv` 76 dòng. Nếu có thư mục `_r2`, báo Claude trước khi chạy gì.
3. Báo Claude kiểm contact sheet + độ sáng trước khi chạy optimize.
