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

## 2026-11-01 p0-owner-webcam · P0 · pass (có hạn chế)
- **Giả thuyết:** webcam thật của owner (laptop Windows) đạt ≥ 20 fps thực khi ghi trong `/lab`.
- **Cấu hình:** Chrome trên Windows, 640×480, clip 4 s, 3 lần:
  1. đủ sáng, bật 3 tuỳ chọn xử lý âm thanh;
  2. thiếu sáng, bật 3 tuỳ chọn;
  3. đủ sáng, tắt cả 3.
- **Kết quả:**
  1. 119 frame, **29.9 fps**, audio 16 kHz −19.6 dBFS, decode 911 ms.
  2. 60 frame, **15.0 fps** (trình duyệt vẫn báo 30), −18.7 dBFS, decode 489 ms.
  3. 119 frame, **29.9 fps**, −20.3 dBFS, decode 889 ms.
- **Phân tích:**
  - Thiếu sáng thì driver kéo dài thời gian phơi sáng, fps thực giảm một nửa. Đây là rủi ro cho P3 (khớp môi cần khoảng 25 fps).
  - Bật hay tắt xử lý âm thanh gần như không đổi mức âm lượng.
  - Decode chiếm 0.5–0.9 s trên ngân sách 3 s của E4.
- **Quyết định tiếp theo:** ở P3 xử lý trường hợp fps < 20: cảnh báo thiếu sáng trên giao diện, thử ép `frameRate.min`, hoặc nội suy theo timestamp. Ghi điều kiện ánh sáng vào mọi lần thu dữ liệu.

## 2026-11-01 p1-owner-voice · P1 · pass (ngưỡng phải sửa)
- **Giả thuyết:** giọng owner đã enroll (3 mẫu qua trình duyệt) tách được khỏi người khác ở ngưỡng tạm 0.40.
- **Cấu hình:**
  - 20 clip `.m4a` của owner (clip 2–6 cố ý đọc giọng cao hơn), so với 240 clip của 12 người AudioMNIST.
  - Máy Windows của owner; `voice_scores` và `latency`.
- **Kết quả:**
  - `results/innovation/20261101_140348_voice_scores/`: EER **0.63%**, AUC 0.999. Genuine: trung bình 0.39, thấp nhất 0.20 (`huy-2`), cao nhất 0.60. Impostor: trung vị 0.04, cao nhất 0.23.
  - **Ở ngưỡng 0.40: FAR 0%, FRR 65%.** Ở ngưỡng 0.25–0.30: FAR 0%, FRR 5%.
  - `results/innovation/20261101_140519_latency/`:
    - ECAPA 153 ms.
    - Whisper tiny.en 314 ms, ra "7, 6, 0, 3, 8, 1.".
    - Whisper base.en 612 ms, ra "760-381". Cả hai đúng chuỗi số.
- **Phân tích:**
  - Ngưỡng 0.40 (đặt từ AudioMNIST) quá chặt với giọng thật: AudioMNIST thu cùng buổi, cùng micro nên lạc quan.
  - Điểm genuine thấp do lệch kênh thu: enroll bằng webm/opus qua trình duyệt, test bằng `.m4a`. Không phải do cao độ: 5 clip giọng cao có trung bình 0.39, bằng nhóm giọng thường.
  - FAR 0% cũng lạc quan, vì impostor AudioMNIST khác hẳn kênh thu và ngôn ngữ.
  - base.en viết chuỗi số kiểu "760-381", nên P2 cần chuẩn hoá chỉ lấy chữ số.
- **Quyết định tiếp theo:** xem `decisions.md` (2026-11-01). Ngưỡng chốt trong prereg sau khi có impostor cùng kênh (người hỗ trợ, buổi 1 của L1).

## 2026-11-01 l1a-tooling · L1 giai đoạn A · info (kiểm tra code, chưa phải thí nghiệm)
- **Mục đích:** kiểm tra bộ công cụ tấn công chạy trọn từ đầu đến cuối trước khi owner chạy với checkpoint thật.
- **Đã kiểm tra:**
  - Pipeline torch so với backend (trọng số ngẫu nhiên, 20 crop thật cùng crop ngẫu nhiên): cosine nhỏ nhất **1.00000**.
  - Crop khớp `crop_face_from_bbox` trên 200 bbox ngẫu nhiên.
  - JPEG kiểu BPDA cho gradient bằng 1.
  - Bảng màu chạy vòng tròn (giả lập chụp nghiêng, lệch màu, nhiễu): sai số fit khoảng 1/255.
  - Gọng đặt đúng theo mắt trên ảnh webcam thật của owner.
- **Độ phủ đo được** trên ảnh owner: gọng thường 3.7%; độ dày 35, `rim_scale` 1.1 cho 8.2%; độ dày 55, `rim_scale` 1.2 cho 11.6% (xem `decisions.md`).
- **Chạy thử optimize, evaluate, export trên CPU bằng trọng số ngẫu nhiên:** chạy hết luồng, nhưng **không có ý nghĩa về hiệu quả**. Model ngẫu nhiên cho mọi ảnh cosine khoảng 0.98, gradient gần bằng 0, texture hội tụ về màu xám. Đã xoá các thư mục kết quả chạy thử.
  - Phát hiện và sửa: logsumexp chưa trừ τ·log(n) nên hàm max mềm vượt quá 1.
- **Quyết định tiếp theo:** owner chạy `runbook_L1.md` với checkpoint thật. Bước đầu tiên là kiểm tra parity trên crop thật.
