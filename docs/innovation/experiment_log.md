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

## 2026-11-01 l1a-review-fixes · L1 giai đoạn A · info (sửa code sau review, chưa phải thí nghiệm)
- **Mục đích:** sửa các lỗi review chỉ ra trước khi owner chạy với checkpoint thật (chi tiết trong `decisions.md`, cùng ngày).
- **Lỗi nghiêm trọng đã xác nhận:** method5 (ViT-B/16, 157 key) bị load vào `FaceCNN` với `strict=False`, tức chạy bằng trọng số ngẫu nhiên. Mọi kết quả transfer sang method5 trước bản sửa này (nếu có) đều không dùng được.
- **Đã kiểm tra:**
  - Dựng lại `FaceViT_ArcFace` theo `training_max.ipynb`: state_dict có đúng **157 key**, tên key khớp (`backbone.cls_token`, `backbone.pos_embed`, `backbone.patch_embed.proj.weight`, …, `weight`).
  - Load strict cho cả 4 bố cục (ViT, CNN+ArcFace, CNN+classifier, CNN triplet); key thiếu hoặc thừa thì dừng.
  - Parity pipeline torch so với backend trên ViT (224, chuẩn hoá ImageNet), crop thật của owner: cosine nhỏ nhất **0.99999**.
  - Max mềm mới: thêm 5 ảnh xấu làm điểm đổi < 0.001 (bản cũ đổi 0.035).
  - `liveness_step` trả "Move closer" khi mặt quá nhỏ; cổng giai đoạn B là `null` khi không có liveness.
  - Chạy thử optimize (4 bước) và evaluate có `--transfer method5=<ViT ngẫu nhiên>` trên CPU: chạy hết luồng; kết quả tách tầng thường và `@dim`. Trọng số ngẫu nhiên nên **không có ý nghĩa về hiệu quả**; đã xoá các thư mục kết quả chạy thử. `--liveness` dừng đúng với lỗi thiếu `liveness_best.pth` (môi trường này không có checkpoint thật).
  - Test: 80 pass (venv innovation có timm); 60 pass, 4 skip (venv cơ bản, không có timm/av).
- **Quyết định tiếp theo:** owner chạy `runbook_L1.md` (bản mới), bắt đầu bằng parity cho cả method3 và method5.


## 2026-11-03 build-gallery · L1 gallery construction · pass
- **Giả thuyết:** LFW cung cấp đủ ứng viên không trùng tập train để dựng gallery chuẩn 5 ảnh/người (1 nạn nhân trung vị, 50 người lớp mô phỏng) cho bài toán 1:N của L1.
- **Cấu hình:**
  - LFW Funneled (5 760 người, 250×250).
  - Detector MediaPipe `model_selection=0`, lọc tâm bbox lệch ≤ 20% tâm ảnh.
  - ArcFace calibration: 200 members từ `test_data` (4 000 ID, 2 ảnh/ID), 200 non-member pairs từ `verification_pairs_val.txt`.
  - Baseline frame owner: 53 frame hợp lệ (unenhanced) từ `data/innovation/frames/owner/s1_day1`.
  - Máy chạy: Windows 11, Intel Core, Python 3.12 (`swin` env).
- **Kết quả:**
  - Hiệu chỉnh ArcFace: ngưỡng $T = 0.4783$ (phân vị 5% member), phân tách member/non-member đạt $\text{AUC} = 0.8061$.
  - 420 người trong LFW có đủ 5 ảnh hợp lệ qua detector backend.
  - 35 người bị loại do nghi ngờ trùng tập train (`agree ≥ 3` và `med_top1 ≥ T`); 385 người đủ điều kiện (yêu cầu tối thiểu 53).
  - Nạn nhân chính `LFW_victim`: Yashwant_Sinha ($s_{\text{baseline}} = 0.0648$, phân vị 50.13%).
  - Nạn nhân phụ (extra victims): `LFW_victim_easy`: Yoko_Ono ($s_{\text{baseline}} = 0.1986$, phân vị 90.13%); `LFW_victim_hard`: William_Donaldson ($s_{\text{baseline}} = -0.0355$, phân vị 10.13%).
  - Lớp mô phỏng: 50 người `LFW_0001` ... `LFW_0050` được chọn ngẫu nhiên (seed 0).
  - Folder `owner/` được giữ nguyên vẹn 100%. Báo cáo lưu tại `_build_report.json` và `_candidates.csv`.
- **Phân tích:** Phân bố $s_{\text{baseline}}$ tập trung quanh 0.065, cho thấy nạn nhân trung vị được chọn có khoảng cách đặc trưng tự nhiên với owner, không quá dễ cũng không quá khó. Tầng lọc ArcFace loại bỏ chính xác 35 trường hợp có đặc trưng quá khớp với tập train.
- **Quyết định tiếp theo:** Tiến hành chạy dodging và impersonation (runbook §5 và §6).

## 2026-11-03 build-gallery-rerun · L1 gallery construction (with VGGFace2 meta) · pass
- **Mục tiêu:** Chạy lại `build_gallery` có đối chiếu tên với metadata VGGFace2 (`data/vggface2/identity_meta.csv`), loại bỏ triệt để các danh tính LFW trùng với 4 000 class tập train của method3.
- **Cấu hình:**
  - LFW Funneled (5 749 người đã xét, 250×250).
  - Detector MediaPipe `model_selection=0`, lọc tâm bbox lệch ≤ 20% tâm ảnh (`center_tol=0.20`), seed 0.
  - Cờ bổ sung: `--vggface2-meta data/vggface2/identity_meta.csv --replace --extra-victims`.
  - ArcFace calibration: 200 members từ `test_data` (4 000 ID, 2 ảnh/ID), 200 non-member pairs từ `verification_pairs_val.txt`.
  - Baseline frame owner: 53 frame hợp lệ (unenhanced) từ `data/innovation/frames/owner/s1_day1`.
- **Kết quả:**
  - Nạp thành công 4 000 tên train từ `identity_meta.csv` (đọc qua `csv.reader(..., skipinitialspace=True)`).
  - 420 người trong LFW có đủ ≥ 5 ảnh hợp lệ qua detector backend.
  - Phân loại loại trừ:
    - 37 người bị loại vì `name_in_train` (trùng tên với tập train VGGFace2; run 1 là 0).
    - 15 người bị loại vì `arcface_member` (`agree ≥ 3` và `med_top1 ≥ T`; run 1 là 35 do chưa lọc theo tên trước).
    - 3 người bị loại vì `fewer_than_5_valid`.
    - Tổng còn 368 người đủ điều kiện (n_eligible; yêu cầu tối thiểu 53).
  - Nạn nhân chính `LFW_victim`: Yashwant_Sinha ($s_{\text{baseline}} = 0.0648$, phân vị 50.0%) — **không đổi so với run 1**.
  - Nạn nhân phụ:
    - `LFW_victim_easy`: Scott_McClellan ($s_{\text{baseline}} = 0.1991$, phân vị 89.95%, target p90 là 0.1997) — thay cho Yoko_Ono (do Yoko_Ono trùng tên train VGGFace2).
    - `LFW_victim_hard`: Wayne_Ferreira ($s_{\text{baseline}} = -0.0390$, phân vị 10.33%, target p10 là -0.0393) — thay cho William_Donaldson (do William_Donaldson trùng tên train VGGFace2).
  - Lớp mô phỏng: 50 người `LFW_0001` ... `LFW_0050` (49/50 người đổi danh tính so với run 1 do pool ứng viên thay đổi từ 385 xuống 368; chỉ LFW_0002 Alastair_Campbell giữ nguyên).
  - Cache gallery (`faces.build_gallery`) được bổ sung hash danh sách file ảnh thực tế (đường dẫn tương đối, size, mtime), đảm bảo cache tự động bị vô hiệu hóa khi thay đổi ảnh hoặc dựng lại gallery.
- **Ghi chú kỹ thuật:** `test_data` chỉ có 2 ảnh/ID nên lúc hiệu chỉnh agree tối đa là 2; điều kiện "agree ≥ 3" chưa được hiệu chỉnh, chỉ có T được hiệu chỉnh ($T = 0.4783$, $\text{AUC} = 0.8061$).
- **Quyết định tiếp theo:** Tiến hành chạy dodging và impersonation trên gallery mới.


## 2026-11-03 20261103_185645 / 20261103_185805 · advglasses_dodge · aborted
- **Mục tiêu:** Chạy thử nghiệm dodging L1 giai đoạn A (`--mode dodge --steps 300 --batch 6`).
- **Cấu hình:**
  - Lệnh: `python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --mode dodge --color-map data/innovation/chart/colormap.json --steps 300 --batch 6`
  - Checkpoint: `checkpoints/method3_best.pth`, máy Windows owner CPU.
- **Kết quả:** Cả 2 run dừng ngay khi khởi động, không có kết quả (`meta.json` được ghi nhận nhưng script thoát vì FileNotFoundError do `data/innovation/chart/colormap.json` chưa tồn tại).
- **Phân tích:** Tham số `--color-map data/innovation/chart/colormap.json` được truyền theo hướng dẫn cũ, trong khi ở giai đoạn A (kiểm chứng digital) bảng màu chưa được in và fit ánh xạ webcam thực tế.
- **Quyết định tiếp theo:** Ở giai đoạn A, chạy tối ưu không cần `--color-map` (script tự động dùng $M = \text{identity}$ và ghi log `color_map_calibrated: false`, như đã cập nhật trong `runbook_L1.md`). Bảng màu chỉ in và fit trước giai đoạn B.


## 2026-11-04 20261104_121206 / 20261104_122938 · advglasses_dodge + eval · info (chưa hội tụ)
- **Mục tiêu:** Kiểm chứng digital (giai đoạn A) tấn công dodging với gọng kính đối kháng ở lưới tham số mới `--tex-res 15 40`, không dùng `--color-map` ($M = \text{identity}$).
- **Cấu hình:**
  - Tối ưu (`20261104_121206_advglasses_dodge`): `--attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --mode dodge --steps 300 --batch 6`. Lưới $15 \times 40$, 300 bước, CPU Intel Windows 11.
  - Đánh giá (`20261104_122938_advglasses_eval_dodge`): `--run results/innovation/20261104_121206_advglasses_dodge --liveness`, tập test là `s2_day2` (75 frame).
- **Kết quả:**
  - Tối ưu: chạy 300 bước mất 59.0 s trên CPU; loss giảm từ 0.650 xuống 0.396; độ tương đồng với owner (`target_cos`) giảm từ 0.656 xuống 0.365; mô hình đang trên đà giảm nhưng **chưa hội tụ** hoàn toàn.
  - Đánh giá trên tập test (normal light):
    - `baseline`: 0/54 thành công (0.0%, Wilson [0.00, 0.07]) — đúng mong đợi, kẻ tấn công chưa đeo gì thì luôn bị nhận diện là chính mình.
    - `D`: 28/63 thành công (44.4%, Wilson [0.33, 0.57]) — gọng kính đối kháng giúp né nhận diện thành công 44% số frame.
    - `control_random` (xáo theo ô $15 \times 40$): 0/60 thành công (0.0%, Wilson [0.00, 0.06]).
    - `control_plain` (màu trung bình): 0/59 thành công (0.0%, Wilson [0.00, 0.06]).
    - `D_EOT`: 39/61 thành công (63.9%, Wilson [0.51, 0.75]).
    - Liveness pass: đạt $\sim 97\%$ trên mọi tầng (baseline 96.3%, D 96.8%, control_random 96.7%, control_plain 96.6%, D_EOT 98.4%).
    - Lighting enhancer: tự động kích hoạt trên 21/75 frame của session `s2_day2` (các frame này được tách riêng sang các tầng `@dim`, ví dụ `baseline@dim` 21 frame, `D@dim` 12 frame; không gộp vào tỉ lệ normal light).
- **Phân tích:**
  - Cả hai tầng control (`control_random` và `control_plain`) đều đạt 0%, chứng minh hiệu ứng dodging không phải do việc đeo một vật thể che mắt bất kỳ mà do pattern đối kháng tạo ra.
  - Tỉ lệ thành công của `D_EOT` (64%) cao hơn `D` (44%): Hiện tượng này có thể do các biến đổi xuống cấp ảnh của EOT (mờ quang học, khoảng cách xa, nén JPEG, gain màu) tự nó làm giảm điểm nhận diện khuôn mặt dẫn đến né nhận diện (false dodging).
- **Ghi chú kỹ thuật:** Chưa thể kết luận kính chịu EOT tốt hay do suy giảm chất lượng ảnh tự gây dodge cho đến khi có tầng đối chứng `baseline_EOT` (đã bổ sung vào `evaluate.py`).
- **Quyết định tiếp theo:** Chạy lại `evaluate.py` với bộ tầng EOT đầy đủ (`baseline_EOT`, `control_random_EOT`, `control_plain_EOT`) để so sánh chuẩn xác; tăng số bước tối ưu lên 1000–1500 bước để hội tụ sâu hơn.


## 2026-11-04 20261104_123824 / 20261104_124845 · advglasses_impersonate (E1a run 1) · fail (trượt cổng E1a & kích hoạt ladder)
- **Mục tiêu:** Kiểm chứng digital (giai đoạn A) tấn công mạo danh (impersonation) L1 nhắm vào nạn nhân ảo `LFW_victim` (Yashwant Sinha) trên mô hình mục tiêu `method3_best.pth`.
- **Cấu hình:**
  - Tối ưu (`20261104_123824_advglasses_impersonate`): `--attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --victim LFW_victim --mode impersonate --steps 1500 --batch 8 --lr 0.01`. Gọng chuẩn (`thickness 35`, `rim-scale 1.1`, lưới $15 \times 40$), $M = \text{identity}$, 1500 bước trên CPU Windows 11.
  - Đánh giá (`20261104_124845_advglasses_eval_impersonate`): `--run results/innovation/20261104_123824_advglasses_impersonate --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth`. Tập kiểm thử là session `s2_day2` (75 frame).
- **Kết quả:**
  - Quá trình tối ưu: 1500 bước mất 528.9 s; `loss = 0.9511`; `main = 0.6243`; `margin = 0.2081`; `margin_vs_attacker = 0.1532`; `target_cos = 0.3978`; `batch_success = 0.0` (loss **chưa hội tụ** sau 1500 bước). Độ phủ gọng trên ảnh crop: trung bình 7.9% (min 4.4%, max 11.3%).
  - Đánh giá trên session `s2_day2` (normal light, 62–65 frame hợp lệ):
    - `baseline`: 0/54 thành công (0.0%, Wilson [0.00, 0.07]), cosine với nạn nhân 0.0717, cosine với attacker 0.6559.
    - `D`: 0/62 thành công (0.0%, Wilson [0.00, 0.06]), cosine với nạn nhân tăng từ 0.0717 lên 0.2946, nhưng chưa đủ vượt ngưỡng 0.50 để khớp nạn nhân; `margin_vs_attacker` = 0.15 (hệ thống vẫn nhận diện là attacker).
    - `control_random`: 0/65 thành công (0.0%, Wilson [0.00, 0.06]), cosine nạn nhân 0.0503.
    - `control_plain`: 0/60 thành công (0.0%, Wilson [0.00, 0.06]), cosine nạn nhân 0.0713.
    - `D_EOT`: 0/59 thành công (0.0%, Wilson [0.00, 0.06]), cosine nạn nhân 0.2244.
    - `baseline_EOT`: 0/59 thành công (0.0%, Wilson [0.00, 0.06]), cosine nạn nhân 0.0219.
    - Cổng sang giai đoạn B (`gate_to_stage_B`): **TRƯỢT** (`D_rate` = 0.0, yêu cầu $\ge 80\%$).
    - Liveness pass: $\sim 95\% - 97\%$ trên mọi tầng (baseline 96.3%, D 96.8%, control_random 96.9%, control_plain 96.7%, D_EOT 94.9%).
    - Lighting enhancer: tự động kích hoạt trên 13–21 frame của `s2_day2` (tách sang các tầng `@dim`).
  - Transfer sang các mô hình khác (tầng D so với baseline):
    - `method2` (Triplet): cosine nạn nhân tăng từ 0.3822 lên 0.5224 (+0.1402); 0/62 frame match.
    - `method4` (Triplet): cosine nạn nhân tăng từ 0.2841 lên 0.4597 (+0.1756); 0/62 frame match.
    - `method1` (Classifier): cosine nạn nhân 0.0441 → 0.0679 (không đổi đáng kể); 0/62 frame match.
    - `method5` (ViT-B/16 + ArcFace): cosine nạn nhân 0.0685 → 0.0666 (không đổi); 0/62 frame match.
    - Không có bất kỳ frame nào match được nạn nhân trên các model transfer.
- **Phân tích:**
  - Vòng tối ưu 1500 bước với learning rate 0.01 chưa đủ để hội tụ; loss còn ở mức 0.9511 và cosine mục tiêu mới đạt 0.3978 trong train và 0.2946 trên test.
  - Vùng gọng kính ban đầu tương đối mảnh (độ phủ 7.9%), kết hợp với EOT ngẫu nhiên và độ phân giải lưới thấp $15 \times 40$ làm hạn chế capacity của bề mặt đối kháng trong số bước có hạn.
- **Áp dụng dự phòng (1) và Luật đặt trước (Preregistered rule):**
  - Kích hoạt phương án dự phòng (1) theo `plan.md` §6: tăng số bước tối ưu lên 3000 bước, tăng learning rate lên 0.03, tăng diện tích gọng (`thickness 55`, `rim-scale 1.2`), và khảo sát thang cấu hình (ladder) 5 mức để cách ly từng yếu tố ràng buộc:
    - `C0`: `--tex-res 75 200 --no-eot --thickness 55 --rim-scale 1.2` (kỹ thuật số không ràng buộc)
    - `C1`: `--tex-res 75 200 --thickness 55 --rim-scale 1.2` (lưới mịn + EOT)
    - `C2`: `--tex-res 30 80 --thickness 55 --rim-scale 1.2` (lưới trung bình + EOT)
    - `C3`: `--tex-res 15 40 --thickness 55 --rim-scale 1.2` (lưới in được $15 \times 40$ + EOT, gọng to)
    - `C4`: `--tex-res 15 40 --thickness 35 --rim-scale 1.1` (lưới in được $15 \times 40$ + EOT, gọng chuẩn E1a ban đầu)
  - Toàn bộ thang cấu hình C0–C4 được chọn lọc và đánh giá trên tập `dev` (tách ngẫu nhiên 20% từ `s1_day1`), tuyệt đối KHÔNG chạy và không tinh chỉnh trên `s2_day2`.
  - **Luật đặt trước (Preregistered selection rule):**
    - Chọn cấu hình có ràng buộc nhiều nhất (thứ tự ràng buộc tăng dần: $C0 < C1 < C2 < C3 < C4$) đạt tỉ lệ thành công trên tập dev $\ge 80\%$.
    - Cấu hình được chọn sẽ được chạy đúng 1 lần duy nhất trên session kiểm thử `s2_day2` với `--liveness` và `--transfer` làm kết quả chính thức của thí nghiệm E1a.
    - Nếu không cấu hình nào đạt dev $\ge 80\%$ nhưng C0 đạt: kết luận ràng buộc vật lý (in ấn, EOT, kích thước gọng thực tế) làm mất năng lực tấn công, chuyển sang phương án dự phòng (2) (phụ kiện lớn hơn kiểu AdvHat).
    - Nếu ngay cả C0 cũng không đạt dev $\ge 80\%$: kết luận vùng gọng kính hoàn toàn không đủ capacity để mạo danh trên hệ thống này, chuyển sang phương án dự phòng (2).

## 2026-11-04 20261104_131718_ladder · L1 thang cấu hình C0–C4 trên dev · info + sửa đổi luật đo (ghi TRƯỚC khi chạy s2)

- **Dữ liệu:** dev = 15 frame ngẫu nhiên (20%) của `s1_day1`, 13 frame ánh sáng bình thường (2 frame `@dim`). 3000 bước, lr 0.03, seed 0, `LFW_victim`.
- **Lỗi đo phát hiện khi đọc `frames.csv`:** trên dev, `liveness_pass` chỉ 0.20–0.25 ở **mọi** tầng, kể cả `baseline` (không kính), trong khi `live_score` thô ≈ 1.0. Nhãn hầu hết là `Checking`. Nguyên nhân: smoother liveness (cửa sổ 1.5 s) cần frame liên tiếp ở ~5 fps; dev là 15 frame lấy ngẫu nhiên rải trên ~15 s nên smoother không bao giờ tích đủ cửa sổ. Đây là lỗi của cách lấy mẫu dev, không phải liveness từ chối gọng kính. Vì vậy cột `dev_success_frames` trong `ladder.csv` (yêu cầu liveness Real) **không dùng được** để chọn cấu hình.
- **Kết quả đúng trên dev** (match 1:N: top-1 = `LFW_victim`, cosine ≥ 0.5, bỏ qua liveness), method3, 13 frame thường:

| Cấu hình | Lưới | EOT | Độ phủ | D match | D_EOT match | Control (random / plain) | Baseline | Cos nạn nhân D (TB) | Cos train cuối |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 75×200 | không | 11.1% | 1/13 | 1/15 | 0 / 0 | 0 | 0.33 | 0.68 |
| C1 | 75×200 | có | 11.1% | 10/13 | 8/15 | 0 / 0 | 0 | 0.59 | 0.54 |
| C2 | 30×80 | có | 11.1% | 12/13 | 8/15 | 0 / 0 | 0 | 0.61 | 0.55 |
| C3 | 15×40 | có | 11.1% | **11/13 (85%)** | 7/15 | 0 / 0 | 0 | 0.61 | 0.57 |
| C4 | 15×40 | có | 7.8% | 0/13 | 0/15 | 0 / 0 | 0 | 0.38 | 0.33 |

- **Nhận xét:**
  - C0 (không EOT) đạt cos 0.68 trên ảnh train nhưng chỉ 1/13 trên dev: quá khớp vào đúng các frame train. EOT cần thiết để tổng quát hoá ngay cả giữa các frame cùng buổi, không chỉ cho môi trường vật lý.
  - Ép texture mượt (75×200 → 30×80 → 15×40) gần như không làm mất năng lực (10, 12, 11 trên 13; khác biệt nằm trong nhiễu với n = 13). Texture in được không phải trả giá.
  - Độ phủ là yếu tố quyết định: 7.8% → 0/13; 11.1% → 11/13.
  - Control luôn 0 ở mọi cấu hình: hiệu ứng đến từ hoa văn tối ưu, không phải việc đeo gọng.
- **Sửa đổi luật đo (ghi trước khi chạy s2, lý do ở trên):** tiêu chí chọn cấu hình trên dev dùng **match 1:N bỏ qua liveness** (`match_ignoring_liveness`), vì liveness đã smooth không đo được trên frame rời rạc. Liveness được đánh giá trên s2, nơi frame liên tục (75 frame, 0.2 s/frame) và smoother hoạt động đúng như backend. Ngưỡng 80% và thứ tự ràng buộc giữ nguyên.
- **Áp dụng luật:** cấu hình ràng buộc nhiều nhất đạt dev ≥ 80% là **C3** (11/13 = 85%, Wilson 95% ≈ [0.58, 0.96]; khoảng tin cậy rộng vì n nhỏ). C4 không đạt. → Chạy **C3 đúng một lần** trên `s2_day2` với `--liveness` và transfer; đó là kết quả E1a chính thức (cổng: D ≥ 80% và cả hai control ≤ 20%, ánh sáng thường, có liveness).
- **Việc cần sửa (không chặn):** `evaluate.py --split dev` nên báo liveness ở dạng thô theo frame hoặc ghi `null`, và `ladder.py` nên chọn theo `match_ignoring_liveness`.

## 2026-11-04 20261104_181712 · advglasses_eval_impersonate (E1a chính thức, C3 trên s2) · fail (trượt cổng 80%)

- **Cấu hình:** C3 (`20261104_140557_advglasses_impersonate`): lưới 15×40, EOT, `--thickness 55 --rim-scale 1.2` (độ phủ TB 11.1%), 3000 bước, lr 0.03, `LFW_victim`. Chạy đúng một lần trên `s2_day2`, `--liveness`, transfer method1/2/4/5. Màu: M = identity (chưa có bảng màu).
- **method3, s2, ánh sáng thường:**

| Tầng | Thành công (liveness Real + top-1 nạn nhân, cos ≥ 0.5) | Wilson 95% | Match bỏ qua liveness | Liveness pass | Cos nạn nhân TB | Cửa sổ 3 s có ≥ 1 frame thành công |
|---|---|---|---|---|---|---|
| baseline | 0/54 | [0, 0.07] | 0.00 | 0.96 | 0.07 | 0/5 |
| **D** | **36/60 (60%)** | **[0.47, 0.71]** | 0.67 | 0.93 | 0.56 | **4/5 (80%)** |
| control_random | 0/72 | [0, 0.05] | 0.00 | 0.97 | 0.06 | 0 |
| control_plain | 0/62 | [0, 0.06] | 0.00 | 0.68 | 0.14 | 0 |
| D_EOT | 27/62 (44%) | [0.32, 0.56] | 0.45 | 0.94 | 0.50 | 4/5 |
| baseline_EOT | 0/59 | [0, 0.06] | 0.00 | 0.98 | 0.02 | 0 |

- **Cổng sang giai đoạn B:** D = 0.60 < 0.80, control max = 0.00 → `pass False`. **E1a trượt theo luật đặt trước.**
- **Ghi nhận (không đổi kết luận):**
  - Theo frame thì trượt, nhưng theo đơn vị vận hành của Quick mode (backend ghi điểm danh khi một frame khớp), 4/5 cửa sổ 3 s có ít nhất một frame giả danh thành công.
  - Dev (C3) 11/13 = 85% so với s2 60%: khoảng cách dev→s2 dù s2 cùng setup; dev chỉ 13 frame nên ước lượng dev lạc quan.
  - Ánh sáng tối (`@dim`, enhancer bật): D 0/15, match 4/15, liveness pass 0.47 (nhãn Fake xuất hiện). Lighting enhancer và liveness cùng làm hỏng tấn công khi tối.
  - Control trơn làm liveness pass giảm còn 0.68 (thường) và 0.00 (`@dim`): gọng dán digital một màu có thể trông "giả" với model liveness. Cần đo lại ở giai đoạn vật lý.
  - Transfer (cùng texture, model khác): method1 (FaceCNN + classifier) 9/60 = 15%; method2 1/60, method4 3/60 (cos nạn nhân tăng mạnh 0.38→0.70 và 0.28→0.63 nhưng hiếm khi top-1); method5 (ViT) 0/60, cos không đổi (0.07→0.08).

## 2026-11-05 Quyết định sau E1a + luật đặt trước cho vòng 2 (ghi TRƯỚC khi có dữ liệu `s3_vary`)

- **Quyết định của owner:** tiếp tục cải thiện (vòng 2). Dù vòng 2 có đạt cổng 80% hay không, **vẫn chuyển sang giai đoạn B**. Lý do: mục tiêu chính của L1 là E1c (kênh giọng chặn tấn công), chỉ cần tấn công có lúc thành công; ở Quick mode một frame khớp là đủ ghi điểm danh. Kết quả E1a (vòng 1 và vòng 2) vẫn được báo cáo đúng như đo, kể cả khi trượt.
- **Nút thắt đã đo (ladder + E1a vòng 1):** độ phủ gọng (7.8% → 0/13 dev; 11.1% → 11/13 dev) và lượng/độ đa dạng dữ liệu tối ưu (chỉ ~60 frame, cùng setup; dev 85% → s2 60%).
- **Dữ liệu vòng 2:**
  - Tối ưu trên `s1_day1` + `s2_day2` (s2 đã dùng xong cho E1a vòng 1, giờ được phép làm dữ liệu train). Chỉ frame enhancer bỏ qua.
  - Dev: **khối liên tục** 20% cuối của mỗi session train (không lấy ngẫu nhiên), để smoother liveness hoạt động như backend. Tiêu chí dev = thành công đầy đủ (liveness Real + top-1 nạn nhân, cos ≥ 0.5).
  - Đánh giá: session mới `s3_vary` (≈75 frame liên tục, chụp sau khi ghi mục này), cố ý thay đổi khoảng cách (±10–15 cm), vị trí ngồi lệch trái/phải, nguồn đèn; giữ mức sáng thường, cùng kính và kiểu tóc. Chạy đúng một lần.
- **Cấu hình vòng 2** (3000 bước, lr 0.03, seed 0, `LFW_victim`, EOT bật):
  - R2a: lưới 15×40, `--thickness 55 --rim-scale 1.2` (độ phủ TB đo trên s1: 11.24%, như C3).
  - R2b: lưới 15×40, `--thickness 85 --rim-scale 1.42` (đo độ phủ trên 53 frame s1 thường: trung bình **15.02%**, min 8.37%, max 21.38%; đạt yêu cầu ≈15% và không quá 16%).
  - R2c: lưới 30×80, `--thickness 85 --rim-scale 1.42` (cùng gọng với R2b, độ phủ TB 15.02%).
  - Bảng khảo sát đo độ phủ trên 53 frame s1 thường (`python -m innovation.attacks.adv_glasses.ladder --measure-coverage`):
    - `thickness 55, rim-scale 1.20`: 11.24% [6.27%, 16.05%] (R2a)
    - `thickness 65, rim-scale 1.30`: 12.95% [7.22%, 18.49%]
    - `thickness 75, rim-scale 1.35`: 14.12% [7.86%, 20.12%]
    - `thickness 80, rim-scale 1.40`: 14.80% [8.24%, 21.06%]
    - `thickness 80, rim-scale 1.42`: 14.92% [8.32%, 21.26%]
    - **`thickness 85, rim-scale 1.42`**: **15.02%** [8.37%, 21.38%] (cặp đã chọn cho R2b và R2c)
    - `thickness 100, rim-scale 1.40`: 15.00% [8.34%, 21.36%]
- **Luật chọn:** chọn cấu hình ràng buộc nhiều nhất (thứ tự: độ phủ nhỏ hơn trước, rồi lưới thô hơn: R2a < R2b... cụ thể ưu tiên R2a, rồi R2b, rồi R2c) có dev ≥ 80%. Không cấu hình nào đạt thì chọn cấu hình có dev cao nhất. Cấu hình được chọn chạy **một lần** trên `s3_vary` với `--liveness` và transfer.
- **Cổng (giữ nguyên):** D ≥ 80% và cả hai control ≤ 20%, ánh sáng thường, có liveness. Ghi `pass/fail`. Báo thêm tỉ lệ cửa sổ 3 s có ít nhất một frame thành công. Không có vòng 3 tối ưu trên dữ liệu đánh giá.

## 2026-11-06 Sửa đổi luật đo dev vòng 2: làm nóng smoother (ghi TRƯỚC khi chạy ladder vòng 2, chưa có số)

- **Vấn đề:** smoother liveness cần `min_frames` frame mới thoát trạng thái `Checking`. Trong frames.csv của E1a (`20261104_181712`), frame 0000 và 0001 luôn là `Checking` (`is_real=0`); ở tầng D hai frame này có `match_ok=1` nhưng vẫn bị tính thất bại. Dev block chỉ có ~15 frame/session nên mất 2/15 mỗi session: trần dev ≈ 26/30 = 87%, trong khi eval 75 frame chỉ mất ~3%. Sai lệch ~13 điểm phần trăm, đủ đảo luật chọn R2a/R2b/R2c và thiên về cấu hình phủ to hơn.
- **Sửa:** khi `evaluate.py --split dev` với `dev_mode == "block"` và `--liveness`, smoother của mỗi session được làm nóng bằng `max(2, min_frames)` frame liền ngay trước frame dev đầu tiên (frame train, đi qua đúng đường của tầng: vẽ gọng/EOT, JPEG 95). Các frame này **không** ghi vào frames.csv và không tính điểm. Số frame làm nóng ghi ở `summary.json` → `smoother_warmup_frames`. Mô phỏng socket check-in chạy liên tục. Không đổi hành vi `--split eval`.
- Kèm theo: `optimize.py` dừng (`SystemExit`) nếu session trong `--eval-sessions` không tồn tại, để không chạy ladder khi chưa chụp `s3_vary`.
- Luật chọn, cấu hình R2a–R2c và cổng giữ nguyên như mục 2026-11-05.

## 2026-11-06 20261106_130834_ladder · L1 vòng 2, thang R2a–R2c trên dev block · info (ghi TRƯỚC khi chấm `s3_vary`)

- Split (kiểm tra trên `split.json` cả 3 run): train s1 60 + s2 60, dev block 15 + 15 (20% cuối mỗi session), eval `s3_vary` 75; không trùng frame giữa train/dev/eval. Tối ưu chỉ dùng 78/120 frame train (42 frame enhancer bật bị loại). Warmup smoother 6 frame (3/session), liveness_pass tầng D = 1.00.
- Dev (method3, ánh sáng thường, thành công đầy đủ):
  - R2a (15×40, phủ TB 11.6%): D 21/30 = 70% [0.52, 0.83], cos nạn nhân 0.57; D_EOT 13/30.
  - **R2b** (15×40, phủ TB 15.5%, max 21.7%): **D 28/30 = 93%** [0.79, 0.98], cos 0.68; D_EOT 22/30.
  - R2c (30×80, phủ TB 15.5%): D 29/30 = 97% [0.83, 0.99], cos 0.67; D_EOT 22/30.
  - Mọi cấu hình: baseline 0/29, control random 0/30, control plain 0/30 (cả bản EOT).
- **Luật chọn:** R2a < 80% → R2b là cấu hình đầu tiên theo thứ tự ưu tiên đạt ≥ 80% → **chọn R2b** (`20261106_132526_advglasses_impersonate`). R2c cao hơn 1 frame nhưng luật ưu tiên lưới thô hơn (in được) nên không chọn.
- Nhận xét trước khi có số s3: dev cùng ngày và cùng điều kiện với train nên lạc quan (vòng 1: dev 85% → s2 60%). Lưới 15×40 không mất năng lực so với 30×80 (93% so với 97%), khớp kết quả vòng 1: độ phủ là nút thắt, độ phân giải texture thì không.
- Bước tiếp: chấm R2b **một lần** trên `s3_vary` với `--liveness` và transfer method1/2/4/5, cổng giữ nguyên.

## 2026-11-06 20261106_143847 · advglasses_eval_impersonate (E1a vòng 2, R2b trên `s3_vary`) · không hợp lệ (session nằm ngoài vùng hoạt động của hệ thống)

- Chạy đúng một lần, cùng tham số E1a (`--liveness`, transfer method1/2/4/5, seed 1, control seed 0). Cổng: `pass False` (D 0/75).
- **Nguyên nhân:** cả 75/75 frame ở **mọi tầng, kể cả baseline** đều mang nhãn liveness `Move closer`: distance gate (`min_ratio` 0.22, tức mặt rộng < ~141 px trên khung 640) chặn trước smoother. Model liveness thô vẫn thấy mặt thật (p_live trung vị 1.0). Nghĩa là chính mặt thật của owner cũng không check-in được trong session này. Kết quả 0% **không đo tấn công**, chỉ đo việc session được chụp quá xa.
- **Đo tấn công bỏ qua liveness (thông tin phụ, không phải cổng):** method3 D khớp nạn nhân 71/75 = 95% (cos TB 0.69; cos với owner 0.22); D_EOT 85%; baseline, control random và control plain đều 0. Tức là hoa văn R2b vẫn giả danh được ở khoảng cách xa hơn và điều kiện khác train, nhưng đây là vùng hệ thống từ chối.
- Transfer (match bỏ qua liveness): method1 11%, method2 11% (cos nạn nhân 0.76 nhưng hiếm top-1), method4 1%, method5 ViT 0% (cos 0.15). Khớp xu hướng E1a vòng 1.
- **Lỗi thiết kế luật (Claude):** luật vòng 2 không có điều kiện hợp lệ cho session đánh giá (ví dụ baseline phải qua liveness), và hướng dẫn chụp `s3_vary` không nhắc vùng khoảng cách của distance gate.
- Quyết định tiếp theo: chờ owner (xem bên dưới nếu có).

## 2026-11-06 Luật đặt trước: đánh giá lại R2b đã đóng băng trên `s4_vary` (ghi TRƯỚC khi chạy bất kỳ eval nào trên s4)

- **Quyết định của owner:** thay session không hợp lệ `s3_vary` bằng session mới `s4_vary` (75 frame, chụp 2026-11-06 sau khi có kết quả s3). Texture R2b (`20261106_132526_advglasses_impersonate`) **giữ nguyên, không tối ưu lại**. Kết quả s3 vẫn được báo cáo nguyên trạng.
- **Bước A: kiểm tra hợp lệ, không nhìn tấn công.** Chạy eval trên s4 với **chỉ tầng `baseline`** (`--tiers baseline --liveness`, không transfer). Session hợp lệ khi đồng thời: (1) method3/baseline (ánh sáng thường) có `liveness_pass ≥ 0.80`; (2) `no_face ≤ 0.10`; (3) số frame ánh sáng thường ≥ 50/75. Bước A không chứa bất kỳ tầng nào có gọng, nên không lộ kết quả tấn công.
- Nếu bước A trượt: được chụp lại (s5) và lặp bước A, vì chưa nhìn tấn công. Tối đa 2 lần chụp lại; sau đó dừng, E1a vòng 2 báo "không có session hợp lệ".
- **Bước B: chấm một lần.** Chỉ khi bước A đạt: eval đầy đủ (mọi tầng, `--liveness`, transfer method1/2/4/5, seed 1, control seed 0) trên đúng session đó. Cổng giữ nguyên: D ≥ 80% và cả hai control ≤ 20%, ánh sáng thường, có liveness. Không có lần chấm thứ hai.
- Ghi chú minh bạch: đây là lần đánh giá thứ hai của R2b trên dữ liệu mới; đã biết từ s3 (bỏ qua liveness) rằng hoa văn khớp 95% ở khoảng cách xa. Report phải nêu cả s3 và s4.

## 2026-11-06 20261106_164210 · bước A trên `s4_vary` (chỉ baseline) · không hợp lệ

- Code `--eval-session` (agent) đã verify trên diff: chỉ thêm, chặn trùng train/dev, summary ghi `eval_session_override` + `eval_session_names`. Run chỉ có tầng `baseline`, model method3, không transfer: không lộ tấn công.
- method3/baseline: liveness_pass **0.49** (< 0.80) → **trượt điều kiện hợp lệ (1)**. no_face 0.00, 75/75 frame ánh sáng thường (enhancer 0%). Nhận dạng mặt thật vẫn đúng: owner 73/75.
- Phân rã nhãn liveness theo thời gian: Real 37, `Move closer` 17 (t ≈ 2.0–4.2 s và 13.8–14.8 s, mặt nhỏ hơn 22% khung), **Fake 17 liên tục (t ≈ 4.6–7.8 s, p_live trung vị 0.001)**, Checking 4 (smoother reset sau mỗi lần qua cổng khoảng cách).
- Diễn giải: session vẫn ra ngoài vùng khoảng cách ở hai đoạn, và có một đoạn ~3 s model liveness (run3) chắc chắn coi mặt thật là giả, nhiều khả năng do điều kiện ánh sáng đoạn đó (giới hạn đã biết của texture cue). Đây là giới hạn của hệ thống gốc, không liên quan tấn công.
- Theo luật: được chụp lại `s5_vary` (lần chụp lại 1/2) rồi lặp bước A.

## 2026-11-06 20261106_170156 · bước A trên `s5_vary` (lần chụp lại 1/2, chỉ baseline) · không hợp lệ

- method3/baseline (ánh sáng thường, 72 frame): liveness_pass **0.62** (< 0.80) → trượt điều kiện (1). no_face 0, 3 frame `@dim`. Nhận dạng owner 75/75.
- Lỗi khoảng cách đã hết (không còn `Move closer`/`Move back`). Toàn bộ frame bị từ chối là **Fake** trên mặt thật, ba đoạn: t ≈ 3.4–4.6 s (7), 6.4–7.4 s (6), 12.6–14.8 s (12). Ngay cả đoạn Real, điểm smoothed chủ yếu chỉ 0.12–0.29, sát ngưỡng `t_on` 0.2 / `t_off` 0.05: model liveness run3 kém tự tin với mặt thật của owner trong điều kiện s5.
- Phát hiện phụ cho report (giới hạn hệ thống gốc): tỉ lệ chấp nhận mặt thật của liveness giảm khi điều kiện thay đổi trong vùng khoảng cách hợp lệ: s1/s2 (tĩnh) ~0.97 → s4 0.49 (gồm lỗi khoảng cách) → s5 0.62.
- Theo luật: còn đúng một lần chụp lại (`s6_vary`, 2/2). Nếu s6 trượt bước A: dừng, E1a vòng 2 báo "không có session hợp lệ", vẫn sang giai đoạn B theo quyết định đã có.

## 2026-11-06 So sánh điều kiện chụp s1–s6 (đo trên ảnh, TRƯỚC bước A/B của `s6_vary`)

- Owner chụp `s6_vary` (lần chụp lại 2/2) theo hướng: tránh quay đầu mạnh và tránh đèn hắt sáng mạnh lên da (hai nguyên nhân owner nghi gây Fake ở s4/s5). Owner đặt thêm điều kiện: chỉ chạy eval khi s6 tương đồng s1/s2 về độ ổn định, ánh sáng và góc đầu.
- Đo bằng MediaPipe (face detection `model_selection=0` như backend + face mesh), script tạm ngoài repo. Chỉ số: bề rộng mặt / bề rộng khung (cổng 22–70%), yaw proxy (độ lệch mũi theo trục mắt / khoảng cách hai mắt), pitch proxy, |roll| p95 (độ), độ sáng TB vùng mặt (0–255), glare = % pixel mặt ≥ 245, dịch chuyển tâm mặt giữa 2 frame / bề rộng mặt (p95), biên độ ngang của tâm mặt (tỉ lệ khung).

| session | mặt rộng % p5/TV/p95 | yaw p5/TV/p95 | pitch TV | roll p95 | sáng mặt TV/p95 | glare % TV/p95 | rung p95 | biên độ ngang |
|---|---|---|---|---|---|---|---|---|
| s1 | 23.4/24.9/26.8 | −0.49/−0.04/0.19 | 1.19 | 5.8 | 105/126 | 0.00/0.06 | 0.06 | 0.08 |
| s2 | 24.0/26.0/27.7 | −0.51/−0.02/0.15 | 1.24 | 11.4 | 109/122 | 0.00/0.06 | 0.05 | 0.07 |
| s3 | 16.2/19.1/21.5 | −0.29/0.06/0.55 | 1.02 | 5.5 | 127/132 | 0.50/1.81 | 0.08 | 0.07 |
| s4 | 19.6/25.0/28.7 | −0.31/0.01/0.39 | 1.01 | 4.2 | 90/98 | 0.99/2.38 | 0.07 | 0.07 |
| s5 | 24.0/27.8/29.5 | −0.40/0.03/0.33 | 1.11 | 15.3 | 110/124 | 0.00/0.10 | 0.08 | 0.06 |
| **s6** | 25.1/28.5/29.5 | −0.28/0.00/0.16 | 1.20 | 6.5 | 100/111 | 0.03/0.22 | 0.07 | 0.05 |

- Kết luận: s6 nằm trong vùng của s1/s2 ở mọi chỉ số (mặt gần hơn ~2–3 điểm %, tối hơn nhẹ, quay đầu và dịch ngang ít hơn s1/s2). → Đạt điều kiện tương đồng của owner.
- Hệ quả diễn giải: s6 **không** phải session "thay đổi điều kiện" như thiết kế `s3_vary`; nó là session mới, khác thời điểm, cùng loại điều kiện với dữ liệu train. Bước B trên s6 đo khả năng tổng quát sang session mới cùng điều kiện, so sánh trực tiếp được với E1a vòng 1 (C3 trên s2: 60%). Độ bền theo khoảng cách chỉ có bằng chứng phụ từ s3 (bỏ qua liveness 95%).
- Quan sát phụ: s3 và s4 có glare cao hơn hẳn (0.5–1.0% TV, p95 ~2%), khớp giả thuyết của owner về hắt sáng; s5 glare thấp nhưng roll p95 15.3° (nghiêng đầu nhiều nhất), khớp giả thuyết quay/nghiêng đầu. Chỉ là tương quan trên vài session, không phải phép thử.

## 2026-11-06 20261106_191956 (bước A) / 20261106_192143 (bước B) · E1a vòng 2: R2b đóng băng trên `s6_vary` · **pass**

- **Bước A** (chỉ baseline): liveness_pass **0.97** (73 Real, 2 Checking đầu session), no_face 0, 75/75 ánh sáng thường → session hợp lệ.
- **Bước B** (chạy đúng một lần, mọi tầng, `--liveness`, transfer method1/2/4/5, seed 1, control seed 0):
  - method3 baseline 0/75; **D 62/75 = 82.7% [0.73, 0.90]**; control random 0/75; control plain 0/75 → `gate_to_stage_B.pass = True` (D ≥ 80%, control max 0).
  - Bỏ qua liveness: D khớp nạn nhân 74/75 = 99%, cos nạn nhân TB 0.71 (cos với owner 0.24). D_EOT 52/75 = 69% (khớp 89%). Cửa sổ 3 s có ≥ 1 frame thành công: 5/5.
  - **Toàn bộ 13 frame D thất bại là do liveness, không phải nhận dạng:** 12 frame đầu (t < 2.4 s) đã khớp nạn nhân nhưng smoother ở Checking/Fake; điểm smoothed của D bắt đầu ~0.00–0.02 rồi tăng dần, qua `t_on` 0.2 ở frame ~12, sau đó giữ Real nhờ hysteresis (`t_off` 0.05). Baseline, control random (cùng màu, xáo ô) và control plain cùng thời điểm đều ~1.00. → Liveness run3 phản ứng với **hoa văn adversarial dán digital** ở đầu session, không phải với gọng. Chưa biết có xảy ra với kính in thật hay không (có thể là dấu vết ghép ảnh digital).
  - Nhận xét thẳng: điểm ước lượng qua cổng nhưng cận dưới Wilson 0.73 < 0.80; con số 82.7% phụ thuộc phần nào vào hysteresis của smoother (một khi đã Real thì khó rơi về Fake).
- Transfer (thành công đầy đủ): method1 0/75, method2 3/75, method4 2/75, method5 ViT 0/75. Cos nạn nhân method2/method4 cao (0.71/0.69) nhưng hiếm top-1. Transfer method1 giảm so với vòng 1 (15% → 0%): hoa văn R2b chuyên biệt hơn cho method3.
- So với E1a vòng 1 (C3 trên s2, cùng loại điều kiện "session mới, cùng điều kiện"): 60% [0.47, 0.71] → 82.7% [0.73, 0.90]. Hai thay đổi cùng lúc (độ phủ 11% → 15.5% và dữ liệu s1 + s2); dev vòng 2 (R2a 70% so với R2b 93%, cùng dữ liệu) cho thấy độ phủ là phần chính.
- Chuỗi đánh giá đầy đủ phải nêu trong report: s3 không hợp lệ (quá xa; bỏ qua liveness 95%), s4 trượt bước A (0.49), s5 trượt bước A (0.62), s6 hợp lệ và pass. Tấn công chỉ được chấm trên s3 và s6.
- Bước tiếp: E1a vòng 2 đóng. Sang bảng màu + giai đoạn B (kính R2b in thật, người hỗ trợ), song song đo khả thi chữ ký S1/S2 trên frame D của R2b.

## 2026-11-06 Phát hiện sau E1a vòng 2: hình dạng gọng R2b gần như kín, không còn tròng kính

- Kiểm tra `mask.png` của các run (template 400×150, tâm mắt cách 180 px ≈ IPD): theo `glasses.FrameShape.mask`, tròng trong = elip (rx·rim_scale − thickness, ry·rim_scale − thickness), với `RIM_RX, RIM_RY = 80, 58`.
  - C3 / R2a (`--thickness 55 --rim-scale 1.2`): tròng mở ≈ 82×29 template px mỗi bên (≈ 29×10 mm với IPD 63 mm); phần gọng chiếm 67% template; tròng 1897 px/bên.
  - **R2b / R2c (`--thickness 85 --rim-scale 1.42`): bán trục dọc tròng trong = 58·1.42 − 85 = −2.6 → kẹp về 1 px.** Tròng chỉ còn một khe ngang ~57×2 px (108 px/bên, 0.4% template); gọng chiếm 89% template và tràn hết chiều cao template. Về thực chất R2b là **một dải che kín hai mắt có khe hẹp**, không phải kính.
- Hệ quả:
  - Bảng khảo sát độ phủ (ghi 2026-11-05) bão hoà quanh 15%: 80/1.42 → 14.92%, 85/1.42 → 15.02%, 100/1.40 → 15.00%. Độ phủ ngừng tăng chính vì tròng đã đóng. Mức "≈15%" đạt được bằng cách bịt tròng, không phải bằng gọng to hơn.
  - Kết luận "độ phủ là nút thắt" bị lẫn yếu tố: từ R2a lên R2b vừa tăng diện tích, vừa **che hẳn hai mắt** (vùng mang nhiều thông tin danh tính nhất) của kẻ tấn công. Không tách được hai phần này từ dữ liệu hiện có.
  - Kết quả E1a vòng 2 (82.7% trên s6) vẫn đúng như đo, nhưng phải được mô tả là "dải che mắt adversarial (khe hẹp)", không phải "kính adversarial". Về vật lý, người đeo gần như không nhìn thấy, và người quan sát sẽ dễ nhận ra.
- **Lỗi review (Claude):** luật vòng 2 chỉ ràng buộc độ phủ (≈15%, ≤16%), không có ràng buộc tròng phải mở; Claude đã duyệt cặp tham số mà không xem hình gọng. Code đúng theo tham số, lỗi nằm ở đặc tả.
- Quyết định về hình dạng gọng cho giai đoạn B: chờ owner. Lưu ý: giai đoạn B vốn phải tối ưu lại (có `--color-map` đã hiệu chỉnh), nên chọn lại hình dạng không tốn thêm một vòng riêng.

## 2026-11-06 Luật đặt trước: khả thi chữ ký S1/S2/S3 (ghi TRƯỚC khi có code/số; plan: `docs/innovation/plan_signatures.md`)

- Mục đích: kiểm tra rẻ (không train) xem hướng phòng thủ dựa trên chữ ký của phụ kiện adversarial có nền không.
- Chỉ số quyết định cho mỗi họ chữ ký (S1, S2, S3) = AUC tốt nhất trong các biến thể của họ đó, trên **so sánh chính** (tấn công gộp EOT so với âm tính gộp), lấy **min giữa `A_C3` và `A_R2b`** (chữ ký phải chạy được trên cả gọng tròng mở và dải kín mắt).
  - AUC ≥ 0.90: họ chữ ký **có nền**, đưa vào detector/fusion làm đặc trưng chính.
  - 0.75 ≤ AUC < 0.90: giữ làm đặc trưng phụ cho fusion, không đứng một mình.
  - AUC < 0.75: **bỏ** họ chữ ký đó, ghi lý do vào `decisions.md`.
- Nếu không họ nào ≥ 0.75: hướng "phòng thủ dựa trên chữ ký" không có nền với dữ liệu hiện có; quay lại cân nhắc hướng (model làm sạch / adversarial training) trước khi xây tiếp.
- Hạn chế biết trước, phải nêu kèm kết quả: chỉ có một kẻ tấn công (owner), nên S1 có thể học "dấu vết owner" thay vì "dấu vết tấn công" (so sánh phụ với `R_lfw` và `R_owner_controls` để soi); chưa có âm tính đeo phụ kiện thật (kính cận, kính râm, khẩu trang); tấn công chưa thích ứng (không biết phòng thủ), nên AUC là cận trên.

## 2026-11-06 Quyết định hình dạng gọng cho giai đoạn tiếp theo (owner)

- Owner chọn phương án (b): ràng buộc tròng phải mở. Việc in bảng màu tạm hoãn (lý do ngoài kỹ thuật); làm phần digital trước.
- Owner chỉ ra khuôn elip (từ P0, chưa từng được biện minh) là lựa chọn kém: làm dày thì tròng đóng theo chiều dọc trước, diện tích hữu ích ít. Gọng demo owner có sẵn là kiểu vuông/wayfarer (thanh mày dày, tròng gần chữ nhật).
- Hướng đã thống nhất: tạo mask từ **ảnh chụp gọng thật của owner** (có thước, quy đổi mm theo IPD), tròng giữ đúng kích thước thật, chỉ cho phép nới gọng **ra ngoài**. Khuôn elip giữ nguyên trong code để tái lập các run cũ. Giai đoạn B dán hoa văn in lên chính gọng này.
- Cần một vòng tối ưu digital mới (luật đặt trước sẽ ghi khi có ảnh gọng và IPD). Đề xuất đánh giá cuối trên `s6_vary` (đã hợp lệ ở bước A, chưa dùng để chọn gì cho gọng mới; ghi rõ là lần dùng thứ hai) hoặc session mới `s7` qua bước A: chờ owner chọn.
- Đang chờ: ảnh gọng có thước, IPD owner, số đo tròng/cầu mũi/tổng bề rộng.

## 2026-11-06 Quyết định: dùng gọng thật của owner (kiểu vuông) + tái dùng s6 cho eval cuối

- Owner gửi 2 ảnh gọng thật đặt cạnh thước cm (giấy ăn làm nền): gọng kiểu vuông/wayfarer, hai tròng gần chữ nhật bo góc, cầu mũi hẹp, thanh mày và viền dưới dày vừa. Đây là vật mang cho giai đoạn B (dán hoa văn in lên gọng này).
- Số đo thô đọc từ thước (ảnh 1, để agent kiểm chứng lại khi tách mask; không phải số chốt): bề rộng một tròng ≈ 50 mm, cao tròng ≈ 38–40 mm, cầu mũi ≈ 18–20 mm, tổng bề rộng gọng (không tính càng) ≈ 130 mm.
- Với bước tối ưu digital, chỉ cần hình gọng trong hệ template (đặt theo tâm hai tròng = `EYES`, cách 180 px). IPD chỉ cần ở bước in (`export.py`), hiện hoãn cùng việc in bảng màu.
- **Tái dùng `s6_vary` cho eval cuối của gọng mới:** owner đồng ý, khung nhìn đúng: đây là đổi cách tiếp cận do yếu tố ngoại cảnh (không in được + khuôn elip sai hình), không phải đổi vì đã thấy điểm. Tính hợp lệ: texture gọng mới tối ưu lại hoàn toàn trên s1 + s2, không chạm s6; s6 không dùng để chọn hình gọng hay tham số. Rủi ro còn lại (Claude đã thấy hành vi s6 của R2b) là nhỏ và được ghi minh bạch ở đây và trong report. Không chụp s7.

## 2026-11-06 Luật đặt trước: vòng tối ưu gọng ảnh (PhotoFrame), ghi TRƯỚC khi có mask/số (plan: `docs/innovation/plan_frame_mask.md`)

- Hình gọng: `PhotoFrame` tách từ ảnh gọng thật owner; tròng luôn trống, chỉ nới ra ngoài (`grow_px`). Không dùng `thickness/rim-scale` cho gọng này.
- Dữ liệu & quy trình giống E1a vòng 2: tối ưu trên s1 + s2 (frame enhancer bỏ qua), dev khối liên tục 20% cuối mỗi session (có warmup smoother), M = identity (chưa in bảng màu), tex-res 15×40, 3000 bước, lr 0.03, seed 0, victim LFW_victim, EOT bật.
- Config (chọn sau khi xem bảng độ phủ của bước 3, agent đo; đặt trước ràng buộc chứ chưa đặt số grow cuối):
  - F0 = `grow_px` cho độ phủ TB ~11% (so trực tiếp R2a 70% và C3 60% — cùng độ phủ, khác hình gọng).
  - F1 = `grow_px` cho độ phủ TB ~15% (so R2b 82.7% — cùng độ phủ, nhưng tròng mở).
  - Ràng buộc bắt buộc mọi config: phần diện tích tròng còn trống ≥ 98% (nếu một config đóng tròng thì loại, không coi là "gọng").
- Luật chọn: ưu tiên F0 trước F1 (độ phủ nhỏ hơn = thực tế hơn); chọn config đầu tiên đạt **dev ≥ 80%**. Không config nào đạt → chọn dev cao nhất. Config được chọn chấm **một lần** trên s6_vary (`--frame-mask` tương ứng, `--liveness`, transfer method1/2/4/5). Cổng giữ nguyên: D ≥ 80% và cả hai control ≤ 20%, ánh sáng thường, có liveness.
- So sánh định hướng report (không phải tiêu chí pass/fail): nếu ở cùng độ phủ ~15%, gọng-vuông-tròng-mở đạt gần R2b (dải kín mắt) thì kết luận "độ phủ là nút thắt, không cần bịt mắt"; nếu thấp hơn nhiều thì phần điểm của R2b đến từ việc che mắt, phải nói rõ.
- Không có vòng tối ưu thứ hai trên s6. Nếu cả F0 và F1 trượt cổng: báo cáo đúng như đo; vẫn sang giai đoạn B với config dev cao nhất khi in được.

## 2026-11-06 Review code PhotoFrame + signatures (chưa có run nào)

- Trạng thái thật: `signatures.py` đã commit (`3db04b5`) nhưng **chưa chạy** (không có `*_signatures` trong results). `frame_mask_from_photo.py` + `PhotoFrame` viết xong nhưng **chưa sinh artifact**: ảnh gọng chỉ có trong chat. Claude đã ghi ảnh vào `data/innovation/frame/owner_frame_photo.png` (2552×1152). Tỉ lệ thước đo trên ảnh: chu kỳ vạch 1 cm ≈ 152 px (vạch mm ≈ 15.2 px) → `px_per_cm ≈ 152`.
- **PhotoFrame (đúng):** nới bóng gọng (silhouette = rim ∪ lens) rồi trừ lens → tròng không bao giờ bị đóng, đúng ràng buộc (b). `optimize`/`evaluate` chỉ thêm nhánh `--frame-mask`, đường `FrameShape` cũ nguyên vẹn; `attack.json` ghi `frame_source`, `frame_mask`, `frame_lens`, `grow_px`.
- **Lỗi hình học (lỗi đặc tả của Claude trong `plan_frame_mask.md` §1.2):** tool đặt **tâm hai tròng** trùng `EYES` (cách 180 px = IPD). Nhưng khoảng cách tâm tròng của gọng thật (≈ bề rộng tròng + cầu mũi ≈ 68–70 mm) lớn hơn IPD người đeo (thường ~60–66 mm). Hệ quả: gọng digital bị thu nhỏ ~8–12% so với gọng thật trên mặt, độ phủ bị đo thấp, và khi in (`export.py` co giãn theo IPD) gọng in sẽ không khớp gọng thật. Sửa: tỉ lệ phải lấy từ thước (`px_per_cm`) và IPD (template px/mm = 180/IPD); đặt trung điểm hai tròng ở x = 200; mắt nằm lệch vào phía mũi trong tròng như thực tế. **Cần IPD ngay bây giờ** (Claude đã nói sai trước đó rằng IPD chỉ cần lúc in).
- **Ngoài phạm vi, phải sửa trước khi dùng:** `export.py` (dòng ~63) và `signatures.load_texture_assets` (dòng ~294) đều cứng `glasses.FrameShape(**frame_shape)` → sẽ lỗi với run `frame_source: "photo"`.
- **signatures.py (đúng theo plan):** chia session đúng (C3: s2, s5, s6; R2b: s5, s6); dương tính chỉ giữ frame tấn công thành công ở mức nhận dạng; âm tính gồm owner, owner + control, LFW leave-one-out; S2 che dải bằng 0 trong không gian chuẩn hoá, embedding cùng đường `_preprocess_face` + `get_embedding` (FaceCNN đã L2-normalize trong `forward`) nên cosine hợp lệ; S3 giữ cả frame thất bại trong lịch sử (đúng, vì đo bất ổn).
- **Rủi ro trong signatures:** leave-one-out bỏ embedding theo **vị trí file** (`query_idx = q_idx`), giả định gallery giữ đúng thứ tự file. `build_gallery` bỏ qua ảnh không có mặt / enhancer bật, nên nếu một ảnh bị bỏ, LOO xoá nhầm embedding và query tự khớp với chính nó (cos ≈ 1) → âm tính LFW trông "rất thật", thổi phồng AUC. Gallery LFW hiện có đủ 53 người × 5 ảnh nên nhiều khả năng không xảy ra; kiểm tra hậu kiểm: mọi dòng `R_lfw` phải có `top1_score < 0.99`.
- pytest cache (13:59 UTC) ghi `tests/innovation/test_build_gallery.py` **thất bại**: chưa rõ nguyên nhân, cần owner chạy lại và gửi lỗi.

## 2026-11-06 20261106_210957_signatures · khả thi chữ ký S1/S2/S3 · S2 có nền, S1 và S3 bị loại (theo luật đặt trước)

- Mẫu: dương tính A_C3 227 (131 + 96 EOT), A_R2b 278 (147 + 131 EOT); âm tính gộp 1099 (R_owner 201, R_owner_EOT 181, R_owner_controls 667, R_lfw 50). Bỏ qua: 124 frame dim, 203 frame tấn công thất bại ở mức nhận dạng, 96 âm tính không khớp đúng người. Hậu kiểm LOO: `top1_score` lớn nhất của R_lfw = 0.895 (< 0.99) → không có tự khớp.
- AUC so sánh chính (tấn công gộp EOT so với âm tính gộp), C3 / R2b:
  - S1: second 0.645 / 0.622; gap_inv 0.720 / 0.455; ratio 0.712 / 0.516.
  - S2: **maxdrop 0.997 / 0.997**; flips 0.817 / 0.681; **conc 0.965 / 0.973**.
  - S3: top1_switch 0.582 / 0.507; score_std 0.699 / 0.678.
- **Áp luật:** S1 = min(0.720, 0.622) = 0.62 < 0.75 → **loại**. S3 = min(0.699, 0.678) = 0.68 < 0.75 → **loại**. S2 = min(0.997, 0.997) ≥ 0.90 → **có nền, đặc trưng chính**. (Cả hai cách đọc luật, "tốt nhất theo biến thể rồi min theo tấn công" hay ngược lại, cho cùng kết quả.)
- Cơ chế S2 (kiểm trên frames.csv): ở frame tấn công, dải gây giảm điểm lớn nhất gần như luôn là **dải 2** (hàng 28–42 của crop 112, vùng mắt/gọng): C3 129/131, R2b 141/147. Che dải này làm sụp danh tính giả: maxdrop trung vị 0.73–0.79. Mặt thật owner: maxdrop trung vị 0.22–0.24, dải lớn nhất phân tán (2–6). Owner + gọng control: 0.23–0.26.
- **Cảnh báo lớn (gây nhiễu domain/danh tính):** âm tính gộp có 95% là frame của owner. Riêng với R_lfw: maxdrop trung vị 0.50, dải lớn nhất cũng thường là dải 2 (36/50); AUC maxdrop so với R_lfw chỉ 0.930 [0.87, 0.97] (C3) / 0.935 (R2b), và **ở ngưỡng 0.5 báo nhầm 48% LFW (owner 0%)**; ngưỡng 0.6: 22% LFW. Không tách được nguyên nhân là "người khác" hay "domain ảnh khác" (LFW là ảnh báo đã funnel, khớp LOO khó hơn: top1 TB 0.64). `s2_conc` bền hơn qua domain: AUC so với R_lfw 0.980 / 0.989.
- Hạn chế (như đã ghi trước): một kẻ tấn công; tấn công không thích ứng; chưa có âm tính đeo phụ kiện thật; tấn công dán digital. Thêm: với gọng cao (gọng vuông của owner có thể phủ 3–4 dải), kẻ tấn công thích ứng có thể rải bằng chứng ra nhiều dải để mỗi lần che một dải không làm sụp danh tính → che một dải là chưa đủ; hướng tự nhiên là che/khôi phục **toàn bộ vùng gọng** (model làm sạch).
- Bước tiếp đề xuất: thu âm tính webcam của **người khác** (thành viên team / người hỗ trợ, mỗi người enroll 5 góc + ~75 frame), để tách "khác người" với "khác domain" trước khi xây detector.

## 2026-11-07 Dữ liệu người thứ hai `p01` (chụp từ xa qua tunnel, webcam của p01)

- `p01_enroll/s0`: 75 frame 640×480, nhịp 200 ms (1 lần trễ 429 ms), đủ 5 góc (trái 12 / phải 20 / ngẩng 17 / cúi 9 frame theo proxy MediaPipe), mặt rộng 22–28% khung. Sáng hơn owner (trung vị 129, p95 166), 6 frame glare > 2%.
- `p01/s1`: 75 frame, ngồi điểm danh bình thường. Mặt rộng 29–31.5%, yaw −0.02…0.05, sáng 110–118, glare p95 0.3%, pitch trung vị 1.41 (hơi cúi so với owner 1.20; có thể do vị trí camera). Nhịp 213 ms, chậm nhất 832 ms (trễ tunnel, không có khoảng hở > 1 s).
  - Sự cố: p01 chụp 2 lần vào cùng session (lần 1 dừng ở index 31, lần 2 đủ 75 và ghi đè ảnh). Cả 75 ảnh khớp byte với lần 2. Claude sửa `meta.csv` chỉ giữ 75 dòng của lần 2; bản gốc lưu `meta_raw_two_runs.csv`.
- Chưa có `p01/s2` (đeo kính thật).
- Gallery p01: `data/innovation/gallery_people/p01/` (front/left/right/up/down), chọn tự động theo luật trong `_selection.json` (glare ≤ 1%, sáng ≤ 150; trái/phải = yaw nhỏ/lớn nhất, ngẩng/cúi = pitch nhỏ/lớn nhất, thẳng = gần yaw 0 và pitch trung vị). **Tách khỏi `data/innovation/gallery/`** để không đổi phép so 1:N của các run L1 trước.
- Khác biệt cần ghi trong report: p01 khác owner cả người lẫn camera/phòng (webcam của p01 qua tunnel). Chỉ 1 người.

## 2026-11-07 Luật đặt trước: S2 với mặt thật của người khác (signatures v2; ghi TRƯỚC khi có code/số; plan `docs/innovation/plan_signatures_v2.md`)

- Câu hỏi: S2 tách tấn công khỏi mặt thật **của người khác trên webcam**, hay chỉ tách khỏi mặt owner? (v1: báo nhầm 48% LFW ở ngưỡng 0.5, owner 0%.)
- Âm tính mới `R_p01`: `p01/s1` gốc + bản EOT + p01 đeo gọng control (random/plain, cả hai hình gọng C3 và R2b), gallery = gallery cũ + `gallery_people/p01`. Dương tính: A_C3, A_R2b như v1 nhưng chấm lại với gallery mở rộng.
- Chỉ số: cho `s2_maxdrop` và `s2_conc`: (a) AUC tấn công (C3, R2b, gộp EOT) so với `R_p01`; (b) ngưỡng đặt trên **dương tính** tại TPR 95% (gộp C3 + R2b), báo FPR trên `R_p01`, trên owner và trên LFW.
- Luật: **S2 đứng vững với người khác** nếu có ít nhất một biến thể S2 đạt đồng thời: min(AUC C3, AUC R2b) so với `R_p01` ≥ 0.90 **và** FPR trên `R_p01` tại TPR 95% ≤ 10%. Nếu không: S2 bị coi là phụ thuộc người/domain; không xây detector chỉ dựa trên S2 mà chuyển trọng tâm sang model làm sạch (che/khôi phục toàn vùng gọng), S2 chỉ còn là đặc trưng phụ.
- Hạn chế biết trước: 1 người, 1 session; tấn công vẫn chỉ từ owner, dán digital, không thích ứng.

## 2026-11-07 Sửa luật signatures v2 (TRƯỚC khi có code/số): p01 đeo kính lúc enroll

- Owner báo: `p01_enroll/s0` p01 **đeo kính thật** (như bình thường), `p01/s1` **không đeo kính**. Chỉ số MediaPipe của Claude không phát hiện được điều này; mục dữ liệu p01 ở trên thiếu thông tin đó.
- Hệ quả: gallery p01 (chọn từ s0) có kính, nên chấm s1 là so **chéo điều kiện** (enroll có kính, check-in không kính); báo nhầm trên s1 sẽ lẫn giữa "người khác" và "lệch kính", và nhiều frame s1 có thể không khớp p01 ≥ 0.5 (bị loại khỏi âm tính).
- **Luật sửa:** âm tính **chính** = p01 cùng điều kiện với gallery (đeo kính), cũng là âm tính "mặt thật đeo kính thật" mà S2 đang thiếu:
  - nếu có `p01/s2` (75 frame, ngồi điểm danh, đeo kính): dùng `p01/s2`;
  - nếu không: dùng 16 frame nhìn gần thẳng của `p01_enroll/s0` (|yaw| ≤ 0.15, pitch 1.0–1.4, bỏ 5 ảnh gallery và ±2 frame lân cận: index 0, 1, 2, 3, 9, 17, 18, 19, 32, 33, 43, 44, 54, 60, 61, 62), ghi rõ cỡ mẫu nhỏ (mỗi frame ≈ 6% FPR).
- Ngưỡng pass giữ nguyên (min AUC C3/R2b ≥ 0.90 **và** FPR tại TPR 95% ≤ 10%), áp trên âm tính chính (gốc + EOT + gọng control).
- `p01/s1` (không kính) báo riêng như phép thử chéo điều kiện, kèm tỉ lệ frame p01 được nhận đúng; **không** dùng để quyết định pass/fail.

## 2026-11-07 `p01/s2` (đeo kính, ngồi điểm danh) → âm tính chính cho signatures v2

- 75 frame, meta sạch (không trùng index), nhịp 200 ms (chậm nhất 334 ms), 14.8 s. Mặt rộng 26.5–30%, yaw −0.00…0.04 (rất thẳng), pitch trung vị 1.44 (hơi cúi, giống s1).
- Phơi sáng mạnh **toàn mặt**, không riêng tròng kính: glare vùng mắt/mày trung vị 1.6% (p95 4.6%), phần mặt còn lại 1.9% (p95 4.3%), độ sáng phần còn lại 165 (s1: 136; owner s6: 127). Cùng kiểu với bộ enroll `p01_enroll/s0` (trung vị 129, p95 166, glare p95 4.35%), nên s2 cùng điều kiện với gallery p01 cả về kính lẫn ánh sáng. Ghi nhận như một khác biệt domain so với owner.
- Theo luật đã sửa: `p01/s2` là âm tính chính; không cần dùng 16 frame dự phòng từ s0. `p01/s1` (không kính) chỉ báo riêng.
- Sửa `plan_signatures_v2.md` (trước khi có code/số): tập âm tính đặt tên theo person + session (`R_p01_s2`, `R_p01_s1`), không gộp; thêm `--primary-session p01/s2` để kết luận chỉ dựa trên session chính; báo tỉ lệ frame được nhận đúng người cho mỗi session.

## 2026-11-07 20261107_124516_signatures · S2 với người thứ hai (p01) · **S2 đứng vững với p01 (pass theo luật)**

- Code verify trên diff: gộp gallery báo lỗi khi trùng id (`extra_gallery_ids = [p01]`); session người khác chỉ sinh stream âm tính (`include_attacks=False`), tập đặt tên theo person + session; LOO giờ bỏ embedding có cosine lớn nhất ≥ 0.99 (`loo_mismatch = 0`). Dương tính chấm lại với gallery mở rộng: số mẫu không đổi (A_C3 227, A_R2b 278).
- Âm tính chính `R_p01_s2_all` = 332 (75 gốc + 65 EOT + 192 gọng control). p01/s2 nhận đúng p01: 75/75.
- Kết quả trên session chính p01/s2:
  - `s2_maxdrop`: AUC 0.9996 (C3) / 0.9996 (R2b); ngưỡng TPR 95% = 0.594 → **FPR p01 = 0/332**. **pass**.
  - `s2_conc`: AUC 0.975 / 0.978; FPR 18/332 = 5.4%. **pass**.
  - → `s2_holds_for_other_people = true`.
- Phân bố maxdrop: owner trung vị 0.22 (max 0.42); p01/s2 0.26 (p95 0.39, max 0.52); p01 + gọng control 0.35 (max 0.53); tấn công C3 0.73, R2b 0.75. Biên an toàn với p01 + gọng control khá hẹp (max 0.53 so với ngưỡng 0.594). Dải gây giảm lớn nhất ở p01/s2 thường là dải 1 hoặc 5, không phải dải 2 như tấn công.
- Cùng ngưỡng, LFW vẫn bị báo nhầm 11/50 = 22%. Vì mặt thật webcam của **người khác** (p01) không bị báo nhầm, nhiều khả năng điểm yếu LFW đến từ domain ảnh (ảnh báo, khớp LOO giữa các ảnh khác nhau) chứ không phải từ "người khác". Chỉ là suy luận trên 1 người.
- p01/s1 (không kính, gallery có kính): **0/75 frame được nhận là p01** (model của team không nhận ra p01 khi bỏ kính), nên không có âm tính gốc. Nhưng p01/s1 **đeo gọng control dán digital** thì 127 frame được nhận là p01: với model này, có gọng hay không chi phối danh tính rất mạnh. Hạn chế của recognizer gốc, đáng nêu trong report.
- Hạn chế: 1 người khác, 1 session chính; tấn công chỉ từ owner, dán digital, không thích ứng.

## 2026-11-07 Mask gọng ảnh (PhotoFrame) đã sinh: độ phủ thực tế thấp hơn nhiều so với luật F0/F1

- Code sửa theo IPD đã verify: IPD 63 mm, 152 px/cm, khoảng cách hai tâm tròng 72.5 mm, mắt lệch vào trong 4.7 mm mỗi bên; số đo: tròng 48.0×35.7 / 47.3×36.3 mm, cầu mũi 23.2 mm, tổng rộng 150.4 mm (gồm chốt bản lề), cao 45.7 mm; 1.4% bóng gọng bị cắt ngoài template (chốt bản lề hai bên). `export.py`, `evaluate.py`, `signatures.py` dùng chung `glasses.shape_from_attack`.
- Lỗi nhỏ trên mask (xem overlay): mép trên dính bóng đổ trên giấy (thanh trên dày hơn thật), viền răng cưa theo vân giấy.
- **Độ phủ trên 53 frame s1 thường:** grow 0 → 4.0% [2.2, 5.7]; 4 → 5.3%; 8 → 6.4%; 12 → 7.3%; 16 → 8.1% [4.5, 11.6]; tròng mở 100% ở mọi mức. Gọng thật ở độ dày thật chỉ phủ ~4% (khớp đo trước: gọng thường ~4%, dày ~8%). Mức ~11% (F0) và ~15% (F1) trong luật 2026-11-06 cần nới ước chừng ~28 px và ~45 px (~1.0 và ~1.6 cm giấy quanh gọng) — chưa đo. Chưa có optimize nào chạy; chờ owner quyết định lại cấu hình trước khi đặt số grow.

## 2026-11-07 Luật đặt trước (thay luật F0/F1 ngày 2026-11-06): thang độ phủ trên gọng thật (G-ladder), ghi TRƯỚC khi có mask sửa và trước mọi optimize

- Owner chọn phương án C: đo đường "tỉ lệ thành công theo độ phủ" trên đúng gọng thật, tròng luôn mở. Lý do đổi luật: đo được gọng thật chỉ phủ ~4% (grow 16 px mới 8.1%), nên F0 ~11% / F1 ~15% cần viền giấy ~1–1.6 cm, không còn là "kính bình thường". Chưa có số tấn công nào trên gọng ảnh.
- Mask sửa trước (plan `docs/innovation/plan_frame_ladder.md`): bỏ bóng đổ, làm mượt viền; đo lại bảng độ phủ grow 0–64 px bước 4. Grow cho G11/G15 = mức trong bảng có độ phủ TB gần 11.0% / 15.0% nhất (ghi số grow và mm vào log trước khi optimize). Nếu bị cắt ở biên template làm không đạt 15%, G15 = grow lớn nhất trong bảng, ghi rõ.
- Cấu hình (PhotoFrame, tex-res 15×40, s1 + s2, dev block 20% có warmup, M = identity, 3000 bước, lr 0.03, seed 0, LFW_victim, EOT):
  - G0: grow 0 (gọng thật, ~4%); G16: grow 16 (~8%, ~5.6 mm); G11: grow cho ~11%; G15: grow cho ~15%.
  - Ràng buộc: tròng mở 100% ở mọi cấu hình.
- **Luật chọn cho giai đoạn B:** thứ tự ưu tiên G0 → G16 → G11 → G15 (nhỏ trước); chọn cấu hình đầu tiên có dev ≥ 80% (thành công đầy đủ, method3, ánh sáng thường). Không cái nào đạt → chọn dev cao nhất.
- **Đánh giá trên s6:** sau khi việc chọn đã cố định từ dev, chấm **cả 4 cấu hình**, mỗi cái đúng một lần, trên `s6_vary` (`--eval-session s6_vary --liveness`); chỉ cấu hình được chọn chạy thêm transfer method1/2/4/5 và chỉ cổng của nó được tính (D ≥ 80% và hai control ≤ 20%). 3 cấu hình còn lại là điểm mô tả của đường cong, không phải cổng.
- Phân tích đặt trước cho report: đường tỉ lệ thành công (dev và s6) theo độ phủ; so ở cùng độ phủ với kết quả gọng elip (C4 7.8% → 0/13 dev; R2a 11.6% → 70% dev; R2b 15.5% dải kín mắt → 93% dev) để tách "độ phủ" khỏi "che mắt".

## 2026-11-07 Mask gọng ảnh v2 + chốt grow cho G-ladder (ghi TRƯỚC khi chạy ladder; chỉ dựa trên bảng độ phủ, chưa có số tấn công)

- Mask v2 (`rim_max_l` = 70; bản cũ ở `data/innovation/frame/_v1/`): hết bóng đổ ở mép trên; viền mượt. Số đo mới: tròng 49.3×37.9 / 49.3×38.4 mm (gần số đo thô owner ~50×38–40), cầu mũi 22.4 mm, rộng 150.3 mm, cao 43.2 mm; khoảng cách tâm tròng 73.7 mm, mắt lệch vào trong 5.4 mm. Còn vài mảng nhỏ ở đáy tròng bị tính là gọng (nhiều khả năng là gờ tròng nhìn qua kính); ảnh hưởng nhỏ, chấp nhận.
- Code `ladder.py` (G0/G16/G11/G15, `--g11-grow/--g15-grow`, cột `grow_px`, luật theo `GROW_ORDER`, in lệnh eval s6 cho cả 4 run) verify trên diff. pytest không còn lỗi (kể cả `test_build_gallery.py`).
- **Bảng độ phủ (53 frame s1 thường), tròng mở 100% mọi mức:** grow 0 → 2.6% (thấp hơn 4.0% của mask v1 vì bỏ bóng đổ); 8 → 5.0%; 16 → 6.8%; 24 → 8.3%; 28 (9.8 mm) → 8.7% (cắt biên 17.6%); 32 → 9.0%; 40 → 9.3%; 48 (16.8 mm) → 9.4% (cắt 37%); ≥ 52 → 9.43% (bão hoà). Gọng cao 43 mm ≈ 124 px trong template cao 150 px, nên nới thêm chủ yếu bị cắt ở biên trên/dưới.
- **Hệ quả cho luật G-ladder:** 11% và 15% **không đạt được** trong template hiện tại. Luật đã dự trù cho 15% (dùng grow lớn nhất), chưa dự trù cho 11% (gần 11% nhất cũng là mức bão hoà, trùng G15). Sửa bắt buộc:
  - Slot `G11` chạy với `--g11-grow 28` (8.7%, ~1 cm, mức trước khi cắt biên tăng nhanh); slot `G15` chạy với `--g15-grow 48` (9.4%, mức nhỏ nhất đạt trần). **Tên G11/G15 giữ nguyên trong code nhưng không còn nghĩa 11%/15%**; report dùng grow (px, mm) và độ phủ đo được, không dùng tên.
  - Thang thực tế: G0 2.6% → G16 6.8% → G11(=grow 28) 8.7% → G15(=grow 48) 9.4%. Luật chọn, đánh giá s6 và cổng giữ nguyên.
- Kỳ vọng ghi trước (không phải tiêu chí): với gọng elip, 7.8% (C4) cho 0/13 dev và 11.6% (R2a) cho 70% dev; gọng thật tròng to (49×38 mm) chỉ đạt tối đa ~9.4%, nên nhiều khả năng không cấu hình nào đạt 80% dev. Nếu vậy đó là kết quả chính: **với tròng kính kích thước thật, độ phủ tối đa của một gọng (kể cả nới ~1.7 cm) không đủ cho tấn công giả danh trên model này ở mức ≥ 80%**.
- Lệnh chạy: `python -m innovation.attacks.adv_glasses.ladder --configs G0 G16 G11 G15 --g11-grow 28 --g15-grow 48`

## 2026-11-07 20261107_133518_ladder · G-ladder trên gọng thật (G0/G16/G11/G15, dev block) · info + chọn cấu hình

- Dev (method3, ánh sáng thường, thành công đầy đủ), độ phủ đo trên frame train:
  - G0 grow 0, phủ 2.9%: 0/74 (cos nạn nhân 0.10).
  - G16 grow 16, phủ 7.7%: 0/74 (cos 0.35).
  - G11 grow 28, phủ 9.7%: 20/74 = 27% (cos 0.43).
  - G15 grow 48, phủ 10.5%: 27/74 = 36% (cos 0.47).
- **Luật chọn:** không cấu hình nào đạt dev ≥ 80% → chọn dev cao nhất = **G15** (grow 48, 36.5%).
- Đọc kết quả: trên gọng thật tròng to (49×38 mm), độ phủ tối đa ~10.5% (đã nới ~1.7 cm, bị cắt biên template) chỉ cho 36% dev, không đủ cổng 80%. Đối chiếu gọng elip cùng độ phủ: R2a 11.6% → 70% dev, R2b 15.5% (dải kín mắt) → 93%. Cùng ~10–11% độ phủ nhưng gọng thật kém hơn elip R2a: tròng to giữ vùng mắt hở, nên hoa văn ít tiếp giáp vùng quyết định danh tính (dải 2) hơn. Củng cố kết luận: **độ phủ quanh vùng mắt là yếu tố, không phải tổng diện tích gọng**; và tròng kính kích thước thật là ràng buộc vật lý thật cho kẻ tấn công.
- Theo luật G-ladder: chấm cả 4 cấu hình trên s6 (một lần mỗi cái); chỉ G15 chạy transfer và tính cổng. Dù G15 gần như chắc trượt cổng, vẫn chấm để có đường cong s6 và để nhất quán với luật.
- Bảng màu: fit hai frame (0040, 0050) cho MAE 9.60 / 9.63 (8-bit), max ~22; ổn định giữa hai frame. `colormap.json` đã ghi (calibrated). Dùng cho bản in giai đoạn B.

## 2026-11-07 Lỗi phương pháp mask gọng ảnh (outward-only) + thiết kế lại (fill-lens + lỗ chừa đồng tử). Ghi TRƯỚC khi chạy lại.

- **Lỗi (owner phát hiện, Claude xác nhận):** `PhotoFrame` chỉ nới gọng **ra ngoài**, giữ nguyên toàn bộ tròng thật. Hai hậu quả: (1) ở độ phủ cao (G11/G15) viền đâm vào khung template 400×150 và bị cắt phẳng — đo pixel chạm biên: G11 trên/dưới/trái/phải 0.72/0.81/0.73/0.83; G15 ~1.0 cả bốn cạnh. Nên độ phủ bão hoà ~10.5% là **giới hạn của khung template, không phải của kính**; gọng in ra sẽ có mép cắt vuông, không đeo được (khớp quan sát owner về G15 mất cầu mũi). (2) chừa trống cả vùng tròng to một cách vô ích, đúng vùng mắt mà S2 cho thấy danh tính tập trung; và làm việc đo IPD thành vô nghĩa (đặt tâm mắt nhưng không dùng để khoét lỗ chừa mắt). Claude overcorrect sau phê bình R2b "che kín mắt".
- Lưu ý: con số độ phủ cũ **không bị thổi phồng** (`coverage()` dùng `padding_mode="zeros"`, phần ngoài template tính 0), nhưng hình dạng sai và trần là giả tạo. → Run `20261107_133518_ladder` (G0–G15) **không dùng làm kết quả vật lý**; không chấm trên s6. Giữ lại chỉ để ghi nhận lỗi.
- **Thiết kế lại (owner đồng ý hướng):** adv = `dilate(rim ∪ lens, g_out)` TRỪ hai đĩa chừa đồng tử bán kính `clearance_mm` đặt ở tâm mắt (template (110,80),(290,80), quy đổi 180 px / IPD 63 mm). Tức lấp tròng, chỉ chừa lỗ vừa đủ cho mắt nhìn; `g_out` nhỏ để giữ cầu mũi và không chạm biên. Tròng-mở giờ đo bằng % diện tích tròng còn hở (lỗ chừa), không phải 100%.
- **Độ phủ đo thử (prototype, warpAffine ~ grid_sample, 75 frame s1 thường):**
  - g_out 8, clearance 14 mm: 9.9%, tròng hở 38%, chạm biên 0.8%.
  - g_out 8, clearance 11 mm: 11.0%, hở 23%, biên 0.8%.
  - g_out 16, clearance 11 mm: 13.0%, hở 23%, biên 1.4%.
  - g_out 16, clearance 9 mm: 13.5%, hở 16%, biên 1.4%.
  - Trần thực tế với lỗ chừa dùng được (đường kính ≥ 18 mm) ≈ 13.5%; muốn 15% phải bịt gần kín mắt (về lại vùng R2b). Preview: `data/innovation/frame/_preview/mask_*.png`.
- Nhận định cần ghi cho report: thiết kế này là "kính có tròng phủ hoa văn, chỉ hở một lỗ ~22 mm ở đồng tử" — ranh giới giữa "kính thường" và "miếng dán adversarial lên tròng". Owner chủ động chọn hướng này; report phải mô tả trung thực, không gọi là kính thường.

## 2026-11-07 Luật đặt trước: H-ladder (fill-lens + clearance) — thay G-ladder cho gọng vật lý. Ghi TRƯỚC khi có số tấn công.

- Tham số mask mới trong `PhotoFrame`: thêm `clearance_mm` (đĩa chừa ở tâm mắt) và giữ `grow_px` (nay là g_out nhỏ). Khuôn elip và đường outward-only cũ giữ nguyên để tái lập run cũ.
- g_out cố định (chờ owner chốt: 8 px ~2.8 mm, an toàn cầu mũi; hoặc 16 px). Thang theo `clearance_mm`:
  - H18 clearance 18 mm (~8%), H14 14 mm (~10%), H11 11 mm (~11%), H09 9 mm (~11.5–13.5% tuỳ g_out).
  - Ràng buộc: tròng còn hở ≥ 15% (lỗ chừa đủ để nhìn/liveness thấy mắt); cầu mũi còn nguyên; chạm biên ≤ 3%.
- Cấu hình: tex-res 15×40, s1 + s2, dev block 20% có warmup, M = identity, 3000 bước, lr 0.03, seed 0, LFW_victim, EOT. Luật chọn: ưu tiên clearance lớn trước (ít phủ, dễ nhìn hơn) → H18 → H14 → H11 → H09; chọn cái đầu đạt dev ≥ 80%; không có thì dev cao nhất. Đánh giá s6 như cũ (cả thang một lần mỗi cái; chỉ cấu hình chọn chạy transfer + tính cổng).
- Phân tích report: đường tỉ lệ thành công theo độ phủ, cho **cùng kích thước tròng thật**, so với elip R2a (11.6% → 70% dev) và R2b (dải kín mắt). Lần này độ phủ đến từ việc lấn vào tròng quanh mắt, nên là phép đo sạch hơn cho giả thuyết "vùng mắt quyết định".

## 2026-11-07 Chốt thiết kế H-ladder (bổ sung luật đặt trước ở trên; vẫn TRƯỚC khi có số tấn công)

- Owner chốt: g_out = 8 px cố định (không nới lớn hơn để khỏi phải phóng khung template, vốn sẽ làm lệch hình học mọi run cũ). Độ phủ cao hơn lấy từ lỗ chừa nhỏ hơn.
- **Cắt ngàm hai bên (temple hinge):** ngàm ở x < 22 và x > 381 (cao 18–40 px so với thân 100+), cắt bằng vertical opening (kernel 1×51). Đo: ngàm nằm gần như ngoài crop 112, nên cắt chỉ mất ~0.1% độ phủ trong crop (fill-lens 10.38% → 9.59% ở template, nhưng cover-trong-crop H11 g8: 11.0% → 10.3%). Side-thickening để bù: chỉ thêm ~0.1% (ngàm vốn ngoài crop), nên là tùy chọn thẩm mỹ.
- Bảng độ phủ cuối (g_out 8, nub-cut, đo prototype): clr 14 mm → 9.1%; clr 11 mm → 10.3%; clr 9 mm → 10.8%. Trần ~10.8% với lỗ chừa ≥ 18 mm đường kính.
- **Luật H-ladder chốt:** 4 cấu hình g_out 8 + nub-cut, clearance = 16 / 13 / 11 / 9 mm (đặt lại rung để phủ dải 8–11% đều hơn; con số cuối chờ bảng đo của agent xác nhận). Ưu tiên clearance lớn trước. Chọn cấu hình đầu đạt dev ≥ 80%; không có thì dev cao nhất. Đánh giá s6 như cũ.
- Ghi chú trung thực cho report giữ nguyên: đây là "kính phủ hoa văn lên tròng, chỉ chừa lỗ ~22 mm ở đồng tử", ranh giới giữa kính thường và miếng dán adversarial lên tròng.
- Sửa (2026-11-07, owner phát hiện): prototype cắt ngàm của Claude dùng vertical opening trên toàn ảnh nên **xoá luôn cầu mũi** (cao ~18 px). Đúng phải là chỉ cắt trong hai dải ngoài phạm vi tròng; cầu mũi giữ nguyên. Các số độ phủ "nub-cut" ghi ở trên đo trên mask thiếu cầu mũi nên hơi thấp; agent đo lại sau khi sửa. Đã sửa `plan_frame_clearance.md` §1b.

## 2026-11-07 Kết quả dev H-ladder (run `20261107_173748_ladder`) và áp luật chọn

- Cấu hình đúng luật: g_out 8, nub-cut, tex-res 15×40, 3000 bước, lr 0.03, seed 0, dev block, LFW_victim, EOT, color_map None (Claude kiểm attack.json/meta.json cả 4 run).
- Dev (tầng không EOT, method3, liveness gated):

| Cấu hình | clearance | độ phủ train | D success | Wilson95 | match bỏ liveness | D_EOT |
|---|---|---|---|---|---|---|
| H16 | 16 mm | 9.4% | 36/75 = 48.0% | 0.37–0.59 | 52.0% | 14/75 |
| H13 | 13 mm | 10.8% | 40/75 = 53.3% | 0.42–0.64 | 60.0% | 21/75 |
| H11 | 11 mm | 11.6% | 43/73 = 58.9% | 0.47–0.69 | 74.0% | 30/75 |
| H09 | 9 mm | 12.3% | 45/74 = 60.8% | 0.49–0.71 | 81.1% | 37/75 |

- Control (random, plain) 0 match ở cả 4. Baseline 0 match.
- **Áp luật: không cấu hình nào đạt dev ≥ 80% → chọn dev cao nhất = H09.** CI của H11 và H09 chồng nhau gần hết; vẫn theo luật, không đổi.
- Quan sát: liveness là nút thắt thứ hai trong khối dev. Baseline chỉ 49/75 frame được coi là thật (0.65), D khoảng 60/75. H09 có 15 frame khớp danh tính nhưng bị liveness chặn. Hoa văn không làm liveness tệ hơn baseline trong khối này (khác với hiện tượng đầu session ở R2b), nên phần chặn này chủ yếu do điều kiện khối dev.
- "Độ phủ train" (cột `coverage` trong ladder.csv) cao hơn bảng đo s1 (8.4–10.9%) vì đo trên tập train s1+s2; dùng bảng s1 khi so với R2a/R2b cho nhất quán.
- Đường cong: độ phủ tăng khoảng 3 điểm % (lỗ chừa 32 → 18 mm) thì match bỏ liveness tăng 52 → 81%. Danh tính tập trung quanh mắt, khớp với S2.
- Tiếp: chấm cả 4 trên s6 một lần mỗi cái; chỉ H09 chạy transfer và tính cổng.

## 2026-11-07 HUỶ kết quả dev H-ladder `20261107_173748_ladder` (rò rỉ split)

- Chấm s6 bị `evaluate --eval-session s6_vary` chặn: frame s6 nằm trong train/dev. Kiểm `split.json` của 4 run H: train = s1, s2, s4, s5, s6 (mỗi session 60 frame), dev = khối 15 frame cuối của cả 5 session, eval = s3_vary. Nguyên nhân: ladder mặc định `--eval-sessions s3_vary`; lúc chạy R2 (11-06) chỉ có s1–s3 nên train đúng là s1+s2 (đã kiểm split R2b `20261106_132526`: train s1/s2 60+60, eval s3). Sau khi chụp s4–s6, mặc định này đẩy chúng vào train. Lỗi do plan H-ladder (Claude) không ghi rõ `--eval-sessions`.
- Hệ quả: (1) run H đã tối ưu trên s6, nên không được chấm s6; (2) số dev ở mục trên đo trên khối dev gồm cả s4/s5 (liveness kém, đã trượt bước A), không phải dev s1+s2 như luật. Baseline liveness 0.65 trong dev là do vậy, không phải do "điều kiện khối dev". **Bảng dev và việc chọn H09 ở mục trên bị huỷ.** G-ladder `20261107_133518` cũng dính lỗi này (vốn đã bỏ).
- R2b (cổng s6 82.7%) không bị ảnh hưởng.
- Chạy lại, luật không đổi (cùng 4 cấu hình, cùng luật chọn), chỉ sửa split cho đúng luật đặt trước: `ladder --configs H16 H13 H11 H09 --g-out 8 --eval-sessions s3_vary s4_vary s5_vary s6_vary`. Sau đó mới chấm s6. Kiểm split.json trước khi đọc số dev.
- Ghi chú minh bạch: owner và Claude đã thấy số dev của run lỗi. Luật chọn đã cố định từ trước nên không bị số này tác động; không đổi gì dựa trên chúng.

## 2026-11-07 Kết quả dev H-ladder chạy lại (run `20261107_210423_ladder`, split đúng)

- Claude kiểm `split.json` cả 4 run: train = s1_day1 60 + s2_day2 60; dev = khối 15 cuối của s1 và s2; eval = s3–s6. attack.json: g_out 8, clearance 16/13/11/9; meta: 3000 bước, lr 0.03, seed 0, dev block. Đúng luật.
- Dev (method3, tầng không EOT, liveness gated; liveness_pass = 1.0 ở mọi tầng):

| Cấu hình | clearance | phủ s1 | D success | Wilson95 | D_EOT |
|---|---|---|---|---|---|
| H16 | 16 mm | 8.4% | 8/28 = 28.6% | 0.15–0.47 | 2/30 |
| H13 | 13 mm | 9.7% | 7/27 = 25.9% | 0.13–0.45 | 2/30 |
| H11 | 11 mm | 10.4% | 17/30 = 56.7% | 0.39–0.73 | 12/29 |
| H09 | 9 mm | 10.9% | 18/30 = 60.0% | 0.42–0.75 | 17/30 |

- Baseline 0/29, control random/plain 0 ở cả 4.
- **Áp luật: không cấu hình nào đạt dev ≥ 80% → chọn dev cao nhất = H09.**
- Nhận xét: dev chỉ ~30 frame nên CI rộng; H16/H13 và H11/H09 không tách được nhau, chỉ thấy bước nhảy giữa 13 mm và 11 mm. Khớp với R2a (elip, phủ 11.6% → dev 70%): với độ phủ ~11% tấn công dừng quanh 60–70% dev.
- Tiếp: chấm cả 4 trên s6 một lần mỗi cái; chỉ H09 (`20261107_214538_advglasses_impersonate`) chạy transfer và tính cổng.

## 2026-11-08 Đánh giá s6 H-ladder (mỗi cấu hình đúng 1 lần, `--eval-session s6_vary --liveness`)

- Claude kiểm: mỗi run H (`210429`/`211930`/`213245`/`214538`) có đúng một eval s6 (`20261108_215634`/`215438`/`215327`/`215113`). Baseline liveness s6 0.973.

| Cấu hình | phủ s1 | dev | **s6 D** | Wilson95 | match bỏ liveness | D_EOT | control max |
|---|---|---|---|---|---|---|---|
| H16 | 8.4% | 28.6% | 15/75 = 20.0% | 0.13–0.30 | 20.0% | 4/75 | 0 |
| H13 | 9.7% | 25.9% | 18/75 = 24.0% | 0.16–0.35 | 25.3% | 5/75 | 0 |
| H11 | 10.4% | 56.7% | 52/75 = 69.3% | 0.58–0.79 | 72.0% | 47/74 | 0 |
| **H09 (chọn)** | 10.9% | 60.0% | **65/75 = 86.7%** | 0.77–0.93 | 89.3% | 51/75 | 0 |

- **Cổng L1 (chỉ H09 tính): D 86.7% ≥ 80%, control 0/0 ≤ 20% → PASS** (`gate_to_stage_B.pass = true`). Dự báo trước của Claude (khó qua cổng) sai: s6 cao hơn dev; dev chỉ 30 frame nên nhiễu lớn.
- Transfer H09 (s6): method1 0/75, method2 5/75 (6.7%), method4 0/75, method5 (ViT) 0/75. Tấn công vẫn white-box đặc thù method3, giống R2b.
- Liveness của frame D = baseline (0.973). Hiện tượng "liveness phản ứng với hoa văn dán digital" ở R2b **không lặp lại** với H09; không dùng làm tín hiệu phòng thủ.
- Ý nghĩa: gọng thật của owner (tròng phủ hoa văn, chừa lỗ đồng tử 18 mm đường kính, 16% tròng hở) đủ để giả danh với độ phủ ~11%, vẫn là dán digital. Bước nhảy rõ giữa clearance 13 mm và 11 mm (24% → 69%) ủng hộ giả thuyết danh tính tập trung quanh mắt (cùng cơ chế S2).
- Giới hạn: một kẻ tấn công (owner), một nạn nhân, một session đánh giá, digital chưa in.

## 2026-11-08 Luật đặt trước: thử vật lý H09 có giới hạn (giai đoạn B rút gọn). Ghi TRƯỚC khi có số vật lý.

- Bối cảnh: owner chốt phương án 1 (đánh giá digital, phòng thủ là trọng tâm) + **đúng một lần thử in** H09. Xem `decisions.md` 2026-11-08.
- **Giới hạn:** tối đa 2 ngày tính từ lúc tối ưu bản in. Một bản in (A4, kèm control_random và control_plain), một session mỗi loại, chấm một lần.
- **Bản in:** tối ưu lại H09, giữ y hệt mọi tham số của `20261107_214538` (s1+s2, `--eval-sessions s3_vary s4_vary s5_vary s6_vary`, dev block, g_out 8, clearance 9, 3000 bước, lr 0.03, seed 0, EOT), chỉ thêm `--color-map data/innovation/chart/colormap.json`. In cùng máy và cùng giấy mờ với `chart.png`, tỉ lệ 100%, tắt tự chỉnh màu. Phần hoa văn phủ lên tròng; khoét hai lỗ đồng tử theo mask.
- **Kiểm digital phụ (mô tả, không phải cổng):** run có color-map chấm s6 một lần. Nếu D < 60%, vẫn in (luật không đổi) nhưng ghi lại.
- **Quay:** owner tự đeo, `person = owner_physical`, ba session: `p_adv` (H09 in), `p_rand` (control_random in), `p_plain` (control_plain in); mỗi session ~75 frame, ánh sáng thường, khoảng cách như s6, cùng ngày, cùng chỗ. Không để dưới `frames/owner/` (tránh lọt vào split của ladder).
- **Cổng vật lý:** tầng P (method3, liveness gated, frame enhancer tách sang `@dim`): P ≥ 80% VÀ cả hai control vật lý ≤ 20%. Transfer method1/2/4/5 chỉ mô tả.
- **Nếu trượt:** được sửa đúng một lần trong ≤ nửa ngày, chỉ được chọn một trong: (a) in lại đúng kích thước nếu thước 50 mm sai ≥ 2 mm; (b) quay lại nếu liveness_pass của p_plain < 0.80 (lỗi điều kiện quay, như bước A); (c) không có lỗi đo nào → dừng luôn. Sau đó dừng hẳn, ghi khoảng cách digital/vật lý làm kết quả, quay về phương án 1.
- Rủi ro biết trước: liveness model học kết cấu giấy in/màn hình nên có thể chặn khuôn mặt đeo gọng giấy; nếu P trượt chủ yếu do liveness (match bỏ liveness ≥ 80%) thì ghi là phát hiện, không tính là qua cổng.

## 2026-11-09 H09 có bảng màu (run `20261109_094721`) — kiểm digital phụ trên s6

- Claude kiểm meta: khác run `20261107_214538` đúng một tham số (`color_map = data/innovation/chart/colormap.json`); split train s1+s2, dev khối s1/s2; `color_map_calibrated = true`.
- Tối ưu hội tụ kém hơn: cuối 3000 bước target_cos 0.53 (không bảng màu: 0.64), batch_success 0.5 (không bảng màu: 1.0).
- **s6 (eval `20261109_105917`, 1 lần): D 8/75 = 10.7% [0.06–0.20], match bỏ liveness 10.7%, D_EOT 8/74, control 0/0, baseline liveness 0.973.** So với 86.7% khi M = identity.
- Diễn giải: bảng màu đo được cho thấy giấy in qua camera chỉ phủ dải sáng ~0.06–0.69 mỗi kênh (`observed_range`), tức mất ~40% tương phản. Trong dải màu in được, tấn công H09 gần như mất tác dụng. Đây là ước lượng bằng mô phỏng cho kết quả vật lý.
- Theo luật đặt trước (D < 60% vẫn in, ghi lại): **vẫn in và quay một lần**, không đổi tham số tối ưu. Kết quả vật lý dùng để kiểm mô phỏng bảng màu có đúng không. Không được tối ưu lại với cấu hình khác (bước, tex-res, trọng số NPS) trong lần thử này: luật chỉ cho sửa lỗi đo.
- Lưu ý: `observed_range` trần 0.69 có thể một phần do phơi sáng của ảnh chụp bảng màu (frame 0050), nên mô phỏng có thể bi quan hơn thực tế. Chính vì vậy số vật lý mới có giá trị.

## 2026-11-09 Quay vật lý lần 1: kiểm trước khi chấm (chưa chấm)
- Owner quay 3 session lúc ~21:02–21:04 nhưng cả ba đều đeo gọng D (hoa văn mượt); chưa in/đeo control. Claude kiểm bằng contact sheet. `p_adv` hợp lệ (75 frame). `p_rand`/`p_plain` sai gọng → chuyển sang `owner_physical/_wrong_pattern_20261109/` (không xoá, không chấm). Không phải lỗi đo được tính vào lần sửa duy nhất: chưa có số nào.
- Độ sáng `p_adv`: L* vùng nửa dưới mặt TB 49.9 (p10 41.4), s6 50.1 (p10 46.1), s1 46.5. Đủ sáng, tương đương s6; vài frame tối hơn sẽ bị enhancer tách sang `@dim`.
- Quan sát: hoa văn in ra nhạt hơn hẳn bản trên màn hình, đúng như bảng màu dự báo.
- Tiếp: quay lại `p_rand` (gọng giữa tờ in, đốm ngẫu nhiên) và `p_plain` (gọng dưới, màu nâu trơn), cùng chỗ, cùng ánh sáng, cùng ngày nếu được; rồi chấm một lần cả ba.

## 2026-11-09 Kết quả vật lý H09 (run chính thức `20261109_212933_advglasses_eval_physical`)

- Contact sheet đã kiểm: `p_adv` gọng D (quay 21:02), `p_rand` đốm ngẫu nhiên (21:27), `p_plain` nâu trơn (21:25); mỗi session 75 frame, cùng phòng, cùng đèn. L* nửa dưới mặt: 49.9 / 48.1 / 50.0 (s6: 50.1). Nguồn: run có bảng màu `20261109_094721`; `smoother_warmup_frames = 0`.
- Run `20261109_210500_advglasses_eval_physical` (21:05) **không hợp lệ**: chạy khi `p_rand`/`p_plain` còn là gọng D quay nhầm. Tầng P của nó dùng đúng các frame `p_adv` như run chính thức nên số P giống hệt; không có lựa chọn nào dựa trên run này.

| Tầng (method3, liveness gated) | success | Wilson95 | match bỏ liveness | liveness_pass |
|---|---|---|---|---|
| **P** (gọng H09 in) | **4/75 = 5.3%** | 0.02–0.13 | 5.3% | 0.973 |
| P_control_random | 0/75 | 0–0.05 | 0 | 0.973 |
| P_control_plain | 0/75 | 0–0.05 | 0 | 0.760 |

- **Cổng vật lý: FAIL** (P 5.3% < 80%). Transfer P: method1 0, method2 0, method4 4/75, method5 0.
- Áp luật sửa: (a) không có báo cáo lệch thước; (b) `p_plain` liveness 0.76 < 0.80, nhưng P trượt do **nhận dạng** (match bỏ liveness chỉ 5.3%, liveness P 0.973), quay lại control không thể đổi kết luận cổng → không dùng lần sửa. **Dừng thử vật lý, đóng giai đoạn B.**
- Kết luận: mô phỏng bảng màu dự báo đúng. Chuỗi kết quả cùng gọng H09: digital M=identity 86.7% → digital có bảng màu 10.7% [0.06–0.20] → vật lý 5.3% [0.02–0.13] (hai CI sau chồng nhau). Nút thắt là dải màu in được qua camera (~0.06–0.69 mỗi kênh), không phải liveness hay hình học gọng. Liveness không chặn gọng giấy (P 0.973).

## 2026-11-09 Luật đặt trước: S2 trên H09 (plan `plan_signatures_h09.md`; TRƯỚC khi có code/số)
- Câu hỏi: chữ ký S2 (che dải, độ giảm điểm) còn tách được tấn công khi gọng là gọng thật fill-lens H09, không phải elip C3/R2b?
- Dương tính: `A_H09` + `A_H09_EOT`, dán lên s5_vary và s6_vary (session eval của H09; không dán lên s2). Âm tính chính `R_p01_s2_all`; báo thêm owner, controls, LFW. Gallery = chính + `gallery_people`.
- Luật: **S2 đứng vững với H09** nếu `s2_maxdrop` hoặc `s2_conc` đạt AUC(A_H09 vs R_p01_s2_all) ≥ 0.90 VÀ FPR trên p01/s2 ≤ 10% tại ngưỡng TPR 95% của A_H09. Không đạt: S2 chỉ còn là đặc trưng phụ, purifier + kiểm nhất quán là phòng thủ chính.

## 2026-11-09 Luật đặt trước: p01 giả danh owner (kẻ tấn công thứ hai; plan `plan_p01_attack.md`; TRƯỚC khi quay)
- Mục đích: kiểm tấn công gọng H09 với người tấn công khác, camera/phòng khác; tạo dương tính thứ hai cho phòng thủ.
- Dữ liệu: `frames/p01_attack/` gồm a1_day, a2_day (tối ưu), a3_vary, a4_vary (giữ ra). Hướng dẫn quay: `capture_p01_attack.md`. Gallery 1:N = gallery chính + `gallery_people/p01`.
- Cấu hình **y hệt H09** (g_out 8, clearance 9, tex-res 15×40, 3000 bước, lr 0.03, seed 0, EOT, M = identity, dev block 20%), `--eval-sessions a3_vary a4_vary`. Một lần tối ưu, **không chỉnh tham số theo kết quả**.
- Bước A (chỉ tầng baseline, trước mọi đánh giá tấn công) trên a3_vary: giống bước A của s4–s6: liveness_pass ≥ 0.80, no_face ≤ 0.10, ≥ 50/75 frame ánh sáng thường; thêm: tỉ lệ baseline khớp owner ≤ 20%. Trượt → bước A trên a4_vary. Cả hai trượt → không có session hợp lệ, báo nguyên trạng.
- Bước B: chấm đầy đủ **một lần** trên session đạt bước A, có `--liveness` và transfer method1/2/4/5. Cổng: D ≥ 80% VÀ hai control ≤ 20%. Trượt vẫn báo nguyên trạng và các frame tấn công vẫn dùng làm dương tính cho phòng thủ (ghi rõ tỉ lệ thành công).

## 2026-11-09 Luật đặt trước: purifier (plan `plan_purifier.md`; TRƯỚC khi có code/số)
- Một lần train duy nhất với siêu tham số trong plan; checkpoint chọn theo identity cos trên **LFW val** (danh tính tách riêng), **không bao giờ** theo số trên owner/p01.
- Đánh giá trên ánh sáng thường, liveness gated, 3 chế độ none / gray / purifier, gallery đi qua cùng chế độ.
- **(1) Chặn tấn công:** tỉ lệ thành công ở mode purifier ≤ 10% cho H09 và R2b trên s6 (và p01→owner nếu đã có).
- **(2) Không hại người thật:** GAR(purifier) ≥ 0.90 × GAR(none) trên owner/s6 VÀ trên p01/s2.
- **Purifier đạt** nếu (1) và (2). **Hơn gray** nếu GAR cao hơn gray ≥ 5 điểm % trên cả hai tập người thật, hoặc gray trượt (2) mà purifier đạt.
- **Bộ phát hiện nhất quán** là đặc trưng chính nếu AUC(tấn công vs người thật) ≥ 0.90 VÀ FPR ≤ 10% tại TPR 95% trên từng tập người thật (owner/s6, p01/s2, owner đeo control).
- Trượt (2): báo đánh đổi nguyên trạng. Hướng sau (chưa làm, cần luật mới): fine-tune recognizer trên mặt đã làm sạch.
- Biết trước: bảo đảm "hoa văn trong band không ảnh hưởng đầu ra" chỉ đúng khi phụ kiện nằm trong band; kẻ tấn công thích ứng phải dùng phụ kiện lớn hơn vùng gọng (ngoài mô hình đe doạ "kính"), sẽ đo ở bước tấn công thích ứng.
