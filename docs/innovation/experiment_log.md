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
