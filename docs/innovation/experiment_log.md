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

## 2026-10-31 p1-enrolment · P1 · partial
- **Giả thuyết:** enroll giọng (3 chuỗi 6 số) chạy trọn trên giao diện. Các bản ghi kém được trả về đúng lỗi, và chỉ bản ghi đạt mới được lưu embedding.
- **Cấu hình:**
  - Backend dùng **encoder giả** thay cho ECAPA, vì môi trường cloud của Claude vẫn bị proxy chặn huggingface.co (403), kể cả sau khi owner đã mở quyền; cài đặt mạng mới có thể chỉ áp dụng cho session sau.
  - Micro giả của Chromium phát một file wav giống giọng nói (tông hài có nhịp, xen khoảng lặng).
- **Kết quả:**
  - 57 test pass (gồm 12 test voice, 2 test metric). Luồng People → Enrol → đồng ý → 3 lần ghi → "Voice enrolled (3 of 3)" chạy trọn trong trình duyệt; `/api/voice/status` trả `{"SV002": 3}`.
  - Test tổng hợp xác nhận phát hiện được các lỗi TOO_SHORT, TOO_QUIET, CLIPPING và NOISY.
  - Khi tắt flag, cột Voice và `/people/<id>/voice` không xuất hiện.
- **Phân tích:**
  - Lần đầu, clip nhiều nhiễu bị báo thành TOO_SHORT + TOO_QUIET, vì VAD không tìm thấy giọng lẫn trong nhiễu. Hướng dẫn đó sai cho người dùng. Đã sửa: nếu tìm thấy ít giọng nhưng cả clip đủ to thì báo NOISY.
- **Còn thiếu (chưa đạt nghiệm thu P1):**
  - Latency ECAPA và Whisper trên máy owner: `python -m innovation.eval.latency`.
  - Bảng điểm owner so với impostor kèm EER: `python -m innovation.eval.voice_scores ...`.
  - Cả hai cần trọng số thật từ Hugging Face.
- **Quyết định tiếp theo:**
  - Owner enroll giọng thật và chạy hai script trên (hoặc Claude chạy khi session mới có quyền HF).
  - Song song, bắt đầu P2 (challenge + ASR), vì phần code không phụ thuộc vào số đo.

## 2026-10-31 p1-real-models · P1 · pass (dữ liệu thay thế)
- **Giả thuyết:** ECAPA tách được giọng một người khỏi người khác khi đọc chuỗi 6 số, đủ để chặn L1 (kẻ tấn công giống mặt nhưng khác giọng). Thời gian embed và ASR đủ nhỏ cho mục tiêu E4 (≤ 3 s cho cả bước verify).
- **Cấu hình:**
  - Model thật: ECAPA (speechbrain 1.0.3), Whisper tiny.en / base.en (CTranslate2, int8, CPU).
  - Dữ liệu: AudioMNIST, 12 người. Mỗi người 6 bản ghi cho mỗi chữ số, ghép thành 12 chuỗi 6 số (`innovation/tools/audiomnist.py`).
  - Đánh giá chéo: mỗi người lần lượt làm owner, enroll bằng 3 chuỗi; 9 chuỗi còn lại là genuine; 11 người kia là impostor.
  - Máy: container cloud, 4 vCPU, không GPU.
- **Kết quả:**
  - `results/innovation/20261031_125000_voice_crossval/`: EER gộp **0.16%**, AUC 0.9999. EER theo từng người: trung bình 0.09%, cao nhất 0.76%. Genuine: trung bình 0.67, p05 0.49. Impostor: trung bình 0.16, p99 0.40. Ở ngưỡng tạm 0.25 thì FAR 21% (FRR 0%). Embed mất 180 ms (trung vị).
  - `results/innovation/20261031_125032_latency/` (clip giọng thật 4.9 s):
    - ECAPA 114 ms.
    - Whisper tiny.en 386 ms, ra "0, 2, 6, 2, 8, 1.".
    - Whisper base.en 544 ms, ra "026281".
    - Tải model: 0.5–1.7 s.
  - `results/innovation/20261031_124632_latency/` (clip tổng hợp, không phải tiếng người): base.en mất 16.5 s và bịa ra "Oh, oh, oh…".
- **Phân tích:**
  - Tách giọng rất tốt, nhưng **lạc quan**: AudioMNIST thu trong một buổi, cùng micro, phòng yên tĩnh. Giọng owner ở buổi khác hoặc micro khác sẽ có điểm genuine thấp hơn.
  - Ngưỡng 0.25 quá lỏng; đổi tạm sang 0.40 (`decisions.md`).
  - Whisper "bịa" chữ khi gặp âm thanh không phải giọng người. P2 phải lọc theo VAD trước ASR và giới hạn thời gian. Bằng chứng: lần chạy ở trên mất 16.5 s.
  - Latency ECAPA + ASR < 1 s trên 4 vCPU: E4 khả thi, còn phải cộng thêm decode và AV.
- **Quyết định tiếp theo:**
  - P1 đạt về kỹ thuật. Bảng điểm trên giọng **owner thật** vẫn cần làm khi owner enroll, ghi khoảng 20 clip ở buổi khác, rồi chạy `voice_scores`.
  - Bắt đầu P2: dùng sẵn AudioMNIST để đo độ chính xác ASR.
