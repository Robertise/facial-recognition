# Quyết định (innovation)

Mỗi mục: ngày, quyết định, lý do, hệ quả. Hướng đã loại thì ghi ở cuối, để không đề xuất lại.

## 2026-10-31. Trả lời các câu hỏi §11 của spec

| # | Quyết định | Hệ quả |
|---|---|---|
| Q1 | Repo đã tách khỏi team; owner có toàn quyền sửa, nâng cấp mọi phần | Bỏ ràng buộc "không sửa file của người khác". Claude commit lên nhánh riêng, owner review và merge |
| Q2 | Challenge đọc chữ số bằng tiếng Anh | ASR dùng model `.en`; dataset đọc số tiếng Anh (Speech Commands, AudioMNIST) |
| Q3 | Chỉ owner tham gia, cộng người ngoài khoảng một buổi mỗi tuần, cộng dữ liệu public | Thiết kế lại vai trò và cách chia tập (plan §2) |
| Q4 | GPU: vast.ai (5070 Ti, 3090, 4090, 5090), vài giờ mỗi job; hoặc Kaggle chạy qua đêm | Job P4 phải chạy xong trong vài giờ trên một GPU |
| Q5 | Không cần báo trước cho giảng viên (dự án cá nhân) | Đạo đức vẫn giữ: chỉ sinh deepfake của owner, không công bố texture adversarial |
| Q6 | Hạn: vài tuần; làm càng nhanh càng tốt, owner luôn sẵn sàng | Lịch khoảng 3–4 tuần (plan §5) |
| Q7 | Chốt các ngưỡng ở spec §6. Trượt thì chuyển sang phương án dự phòng; mọi lần trượt ghi vào `experiment_log.md` để đưa vào report | Bảng dự phòng ở plan §6 |

## 2026-10-31. Giao diện tiếng Anh
Thay cho spec §8.7 (tiếng Việt). Theme Console, đã có từ commit d930530.

## 2026-10-31. Nạn nhân L1 là danh tính ảo từ dataset public
Owner là kẻ tấn công nên không thể đồng thời là nạn nhân. Nạn nhân B lấy mặt và giọng của cùng một người trong VoxCeleb1, enroll vào DB bằng script. Không sinh media mới của B; kính chỉ tối ưu theo embedding của B.

## 2026-10-31. Train trên dữ liệu public, test trên owner
Thay cho việc chia tập theo người trong team: train và dev trên danh tính public, test trên owner (sau khi có prereg) và người ngoài.

## 2026-10-31. Ngưỡng giọng tạm: 0.40 (thay cho 0.25)
Đo trên AudioMNIST (12 người, giọng thật): ngưỡng 0.25 cho FAR 21%. Điểm EER ≈ 0.44; impostor p99 = 0.40; genuine p05 = 0.49. Chọn 0.40 làm giá trị tạm, ưu tiên ít từ chối nhầm người thật. **Ngưỡng chính thức chốt trong `prereg.md`**, dựa trên dữ liệu dev của owner (khác buổi, khác micro), vì AudioMNIST thu trong cùng một buổi nên cho kết quả lạc quan.

## 2026-10-31. Cách bật feature flag
`NEXT_PUBLIC_INNOVATION` nhận 1/true/yes/on (trước chỉ nhận "1", nên owner đặt `true` thì không thấy gì). Backend đọc `.env` ở thư mục gốc repo, vì cú pháp `KEY=1 python ...` không chạy được trên PowerShell hay cmd của Windows.

## Hướng đã loại
Giữ nguyên danh sách ở spec §10 (rPPG, YOLO bắt điện thoại, blink thụ động, screen-flash, motion challenge, depth/CDCN, sonar, face–voice không enroll, multi-face).
