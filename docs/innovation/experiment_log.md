# Nhật ký thí nghiệm (innovation)

Ghi mọi lần chạy, kể cả lần hỏng hay bỏ dở, theo thứ tự thời gian. Số liệu thô nằm ở `results/innovation/<run_id>/` (có `meta.json`: git commit, config, seed, máy chạy). Không sửa entry cũ; nếu sai thì thêm entry mới nói rõ.

Mẫu:

```
## <YYYY-MM-DD> <run_id> · <E-id hoặc phase> · <pass | fail | aborted | info>
- Giả thuyết:
- Cấu hình: (model, dữ liệu, tham số, máy chạy)
- Kết quả: (số liệu, link tới results/innovation/<run_id>/)
- Phân tích: (vì sao ra kết quả này)
- Quyết định tiếp theo: (giữ, chuyển sang dự phòng nào, hay dừng)
```

---

## 2026-10-31 p0-acceptance · P0 · pass
- **Giả thuyết:**
  - Khi tắt flag, app chạy y hệt trước.
  - Một clip ghi từ trình duyệt (dùng chung stream camera với luồng WebSocket, thêm track micro) được backend decode ra ≥ 20 fps thực và audio 16 kHz.
- **Cấu hình:**
  - Chromium headless (Playwright): webcam giả 640×480 ở 30 fps phát lại từ ảnh mẫu, micro giả của Chromium.
  - MediaRecorder `video/webm;codecs=vp8,opus`, clip 4 s; decode bằng PyAV 18.1.0.
  - Python 3.11, torch 2.2.2. Máy: container cloud, 4 vCPU, không GPU.
- **Kết quả:**
  - Tắt flag: 37 test cũ pass; `/lab` và `/security` trả 404; thanh điều hướng không có link mới; `/api/innovation/*` trả 404.
  - Bật flag: 43 test pass (gồm 6 test P0). Clip 4 s cho 120 frame, **30.0 fps đo từ timestamp**, audio **16 000 Hz mono, 3.96 s**, −21 dBFS. Decode 432 ms; file webm 302 KB.
- **Phân tích:**
  - Webcam giả có tốc độ khung hình đều. Webcam thật có thể cho fps thay đổi theo ánh sáng (driver tự kéo dài thời gian phơi sáng). Cần đo lại trên máy owner, đặc biệt trên Windows.
- **Quyết định tiếp theo:**
  - Owner chạy Lab trên máy thật (webcam và micro thật, Windows) và gửi lại các số liệu trên.
  - Nếu fps thực < 20 thì tính đến hạ độ phân giải ghi, hoặc tách frame theo timestamp thay vì giả định tốc độ cố định.
  - Sang P1.
