# Kế hoạch triển khai: "Đúng người, không chỉ người thật" (Attack Ladder)

Ngày lập: 2026-10-31. Spec gốc: [`spec_v1.md`](spec_v1.md). Quyết định đã chốt: [`decisions.md`](decisions.md). Nhật ký thí nghiệm (cả thất bại): [`experiment_log.md`](experiment_log.md).

Kế hoạch này giữ nguyên luận điểm, attack ladder (L0 → L4), ngưỡng §6 và các phase P0 → P6 của spec. Nó sửa lại những gì spec giả định nhưng không còn đúng: repo nay tách riêng khỏi team, và **chỉ có một người tham gia** (owner), cộng với một người ngoài thỉnh thoảng (khoảng một buổi mỗi tuần), cùng dữ liệu public.

---

## 1. Những gì thay đổi so với spec

| Spec giả định | Thực tế | Cách xử lý |
|---|---|---|
| Nhiều thành viên, mỗi người vừa là genuine vừa là impostor | Chỉ owner, cộng người ngoài khoảng một buổi mỗi tuần | Vai trò mới ở §2; impostor lấy từ người ngoài và dữ liệu public |
| Chia train/test theo người trong team | Một người | **Train trên danh tính public, test trên owner và người ngoài** (người chưa từng thấy lúc train) |
| L1: bạn A giả làm bạn B | A và B phải là hai người khác nhau | Owner là kẻ tấn công A; nạn nhân B là một danh tính ảo (§2.2) |
| Fake L3 do cả team tự sinh trên chính mình | Chỉ sinh fake của owner | Train detector trên dataset deepfake public, test trên fake của owner (cross-dataset) |
| Không được sửa file của người khác, không commit | Toàn quyền với repo | Claude commit lên nhánh riêng; owner review rồi merge vào master |
| Giao diện tiếng Việt (§8.7) | Giao diện tiếng Anh | Mọi chuỗi giao diện bằng tiếng Anh; challenge đọc số tiếng Anh |
| `test_liveness_*.py`, `plans/*.md` | Đã chuyển hoặc chưa có | Demo nằm ở `liveness/tools/`; log nằm ở `docs/innovation/` |

## 2. Vai trò và dữ liệu khi chỉ có một người

### 2.1 Vai trò

| Vai trò | Ai | Dùng cho |
|---|---|---|
| Owner, genuine | Owner | BPCER, E5, enroll giọng, dữ liệu genuine cho AV |
| Owner, kẻ tấn công | Owner | L1 vật lý (đeo kính để bị nhận thành B) |
| Nạn nhân L2 và L3 | Owner | Replay bản ghi của chính owner; clone giọng và lip-sync của chính owner (chỉ owner, đúng nguyên tắc đạo đức) |
| Nạn nhân L1 | Danh tính ảo B: mặt và giọng của một người trong dataset public (mặc định VoxCeleb1, có cả mặt lẫn giọng của cùng một người), enroll thẳng vào DB bằng script | Mục tiêu tối ưu kính. Không sinh media mới nào của B, chỉ dùng embedding |
| Người ngoài (khoảng một buổi mỗi tuần) | Người ngoài có đồng ý | Impostor zero-effort thật (E0), genuine thứ hai (BPCER, E5), control L1 |
| Lớp học mô phỏng | Khoảng 50 đến 100 danh tính public enroll vào DB | Nhận diện 1:N giống lớp thật, FAR với danh sách có distractor |

### 2.2 Nguồn dữ liệu public [KIỂM CHỨNG license và cách tải ở P0]

| Nguồn | Nội dung | Dùng cho |
|---|---|---|
| Dataset Kaggle của môn (classification_data, verification_pairs_val.txt) | Ảnh mặt, cặp verification | E0 (FAR/FRR của method3), lớp mô phỏng |
| VoxCeleb1 test (40 người) | Video mặt và giọng của cùng một người | Danh tính ảo B cho L1; impostor giọng |
| Google Speech Commands v2 (CC BY 4.0) | Chữ số "zero" đến "nine" của hàng nghìn người | Impostor đọc số, kiểm tra ASR, thử ghép chuỗi số |
| AudioMNIST | 60 người đọc chữ số | Impostor đọc số (bổ sung) |
| LAV-DF / FakeAVCeleb / AV-Deepfake1M | Video lip-sync thật và giả | Train AV detector (cross-dataset) |
| LRS3 hoặc VoxCeleb2 (nếu được phép) | Video nói thật | Genuine bổ sung cho AV detector |

Nguyên tắc: không sinh deepfake mới của người ngoài đời thật. Fake tự sinh chỉ dùng mặt và giọng của owner. Fake của người khác chỉ lấy từ dataset đã công bố.

### 2.3 Chia tập
- **Train/dev:** chỉ danh tính public cùng các buổi dev của owner.
- **Test:** các buổi của owner thu **sau** khi commit `prereg.md`, các buổi của người ngoài, và fake L3 sinh bằng cấu hình chưa dùng lúc train.
- Báo cáo riêng hai kết quả: trên public held-out và trên owner/người ngoài, để thấy khả năng tổng quát hoá.

## 3. Nguyên tắc làm việc

1. **Git:** Claude làm trên nhánh `claude/relaxed-thompson-0br7xw`. Trước khi viết code và trước khi push đều fetch master; nếu master có commit mới thì merge vào trước. Owner review rồi merge.
2. **Nhật ký là một phần của sản phẩm.** Mọi lần chạy, kể cả chạy hỏng hay bỏ dở, đều ghi vào `experiment_log.md` theo mẫu: giả thuyết, cấu hình, kết quả, quyết định. Số liệu thô nằm ở `results/innovation/<run_id>/` kèm `meta.json` (git commit, config, seed, máy chạy).
3. **Quyết định** (chọn hoặc loại một hướng) ghi vào `decisions.md`, kèm lý do, để không đề xuất lại.
4. **[KIỂM CHỨNG]** (license, API, số liệu trích dẫn) được kiểm tra rồi ghi vào `verification_log.md` trước khi dùng.
5. **Preregistration:** `prereg.md` chốt metric, ngưỡng, cách chia tập và luật fallback. Commit **trước** khi thu tập test, không sửa sau đó; mọi thay đổi ghi thành amendment có ngày.
6. **Feature flag:** `NEXT_PUBLIC_INNOVATION=1` ở frontend và `INNOVATION_ENABLED=1` ở backend. Tắt flag thì app chạy y hệt hiện tại, và test cũ vẫn phải pass.

## 4. Kiến trúc đích

Như spec §7. Các điểm cụ thể:

- `innovation/` (package Python): `config.py`, `challenge.py`, `media.py`, `voice.py`, `asr.py`, `avsync.py`, `fusion.py`, `pipeline.py`, `api.py` (APIRouter), `db.py` (3 bảng mới), `eval/`, `attacks/adv_glasses/`, `attacks/av_deepfake/`.
- `tests/innovation/`: test cho từng module, dùng mock.
- `innovation/requirements.txt`: dependency riêng. Vì repo đã thuộc quyền mình, có thể nâng `torch`/`numpy` nếu speechbrain hoặc faster-whisper cần; việc nâng phải ghi vào `decisions.md`, và test cũ phải pass.
- **Frontend:** mở rộng `CheckLadder` từ 4 lên 6 kênh. Thêm `lib/media.ts` (`useAVRecorder`), `lib/innovation.ts` (client có type theo schema §7.4), các trang `/security`, `/lab`, và Secure mode trong `/attendance`.
- **DB:** chỉ thêm `voice_embeddings`, `challenges`, `verification_attempts`. Bảng `attendance` giữ nguyên, thêm cột `method` (`quick`/`secure`) với mặc định `quick`.

## 5. Các phase

Thời lượng tính theo ngày làm việc liên tục, với owner luôn sẵn sàng. Cột "Owner" là việc chỉ owner làm được.

### P0. Hạ tầng và kiểm chứng (1–2 ngày)
- **Claude:**
  - Skeleton `innovation/`, config, 3 bảng DB, router rỗng, hai feature flag.
  - `useAVRecorder` dùng chung stream camera với WebSocket.
  - `media.py`: decode webm thành frame cộng audio 16 kHz.
  - Trang khung `/security` và `/lab`.
  - Kiểm chứng license và cài đặt của: speechbrain ECAPA, faster-whisper, SyncNet, Wav2Lip, OpenVoice, các dataset ở §2.2.
- **Owner:** chạy app trên máy thật, ghi một clip 4 s; chép checkpoint thật (liveness, method3, emotion) lên máy GPU khi tới P4.
- **Nghiệm thu:**
  - Tắt flag thì test cũ pass.
  - Clip 4 s decode được ≥ 20 fps thực và audio 16 kHz (log in ra).
  - `verification_log.md` có kết luận cho từng thư viện và dataset.

### P1. Giọng nói (1–2 ngày)
- **Claude:**
  - `voice.py`: ECAPA, VAD, kiểm tra chất lượng (độ dài, SNR, clipping).
  - API enroll (`start`/`submit`/`status`); bước Voice enrollment ở `/register`; badge giọng ở `/people`.
  - Script enroll danh tính public (mặt và giọng) vào DB.
- **Owner:** enroll giọng của mình.
- **Nghiệm thu:**
  - Bảng điểm: owner với chính owner, so với owner với khoảng 40 người VoxCeleb và khoảng 100 người Speech Commands.
  - Self-score tách rõ khỏi impostor (báo cáo EER).
  - Latency ECAPA đo trên máy owner.

### P2. Challenge, ASR, Secure mode (2 ngày)
- **Claude:**
  - `challenge.py`: `secrets`, hết hạn, dùng một lần.
  - `asr.py`: faster-whisper, chuẩn hoá chữ số, edit distance.
  - API challenge; Secure mode trên `/attendance` (toggle, các bước, chuỗi số lớn, đếm ngược, thanh âm lượng, crop miệng).
  - **Chặn tự ghi điểm danh ở Secure mode.**
  - Test tự động: Secure mode mà face match nhưng chưa qua challenge thì không có dòng `attendance`; challenge hết hạn hoặc dùng lại bị từ chối.
- **Owner:** thử khoảng 20 lần đọc challenge, ghi nhận độ chính xác ASR trên giọng thật.
- **Nghiệm thu:** test trên pass; ASR khớp ≥ 90% với chuỗi 6 số trên giọng owner.

### P3. AV sync, fusion R1, thẻ kết quả (3 ngày)
- **Claude:**
  - `avsync.py`: crop miệng, SyncNet, đường cong conf(offset).
  - `fusion.py`: R1 cùng `reason_codes`.
  - `pipeline.py` kèm `latency_ms`.
  - Thẻ kết quả 6 kênh.
  - `/research/analyze_file` (research mode).
- **Owner:** quay vài clip có audio lệch cố ý (dùng script tạo lệch 0.5 s).
- **Nghiệm thu:**
  - Đủ 6 kênh hiển thị.
  - Clip lệch 0.5 s cho offset khác 0.
  - Tổng latency được đo. **Điểm dừng:** nếu vượt xa 3 s hoặc decode webm trên Windows không ổn định thì dừng lại bàn.

### P4. Tooling red-team offline (song song với P2–P3, dùng GPU)
- **P4a. L1, kính adversarial**
  - Notebook hoặc script cho vast.ai: mask kính trên mặt 112×112, EOT, NPS, tối ưu theo embedding B.
  - Đo tầng D (digital) và D+EOT. Transfer sang method1/2/4/5 cùng một model public.
  - **Nghiệm thu:** bảng success tầng D (E1a).
- **P4b. L3, deepfake của owner**
  - Pipeline clone giọng (OpenVoice V2 hoặc XTTS-v2) cộng Wav2Lip, chỉ trên dữ liệu của owner.
  - Đo latency sinh deepfake (E3d).
  - **Nghiệm thu:** ≥ 5 clip fake.
- **P4c. Harness file-mode:** chạy pipeline trên cả thư mục và xuất `results/innovation/*.csv`.
- **Owner:**
  - Chạy job GPU (Claude chuẩn bị script đến mức chỉ cần một lệnh).
  - Ghi 30–60 ảnh webcam để tối ưu kính, cùng 10–30 s giọng làm nguồn clone.
  - In bảng màu để đo NPS.

### P5. Dữ liệu, prereg, train, thí nghiệm (4–5 ngày, xen kẽ)
- **Claude:**
  - Viết `prereg.md`.
  - Train AV detector (LR/GBM, rồi GRU/1D-CNN nếu cần) trên dataset public.
  - Train fusion R2 trên vector điểm dựng từ dữ liệu public.
  - Chạy E0–E5 và xuất figure.
- **Owner:**
  - Commit hoặc duyệt prereg.
  - Thu các buổi test qua chế độ thu dữ liệu của `/lab`: genuine, L1 vật lý (kính adversarial và kính control), L2 replay, L3 qua điện thoại và qua virtual camera.
  - Hẹn người ngoài cho E0, E5 và control.
- **Nghiệm thu:** CSV và figure cho E0–E5; Claude đối chiếu với prereg.

### P6. Demo mode, dashboard, báo cáo (2 ngày)
- **Claude:**
  - Dashboard `/security`: bảng, cảnh báo nghi vấn, biểu đồ, xuất CSV.
  - Lab: trực quan AV và L1.
  - Demo mode: chia đôi màn hình, chọn kịch bản L0 → L3, phím tắt bật/tắt từng kênh.
  - README innovation, gom figure cho report.
- **Owner:** chạy thử kịch bản demo L0 → L3.
- **Nghiệm thu:** demo chạy trọn không lỗi.

### Lịch dự kiến (khoảng 3–4 tuần)

| Tuần | Đường chính | Song song (GPU / owner) |
|---|---|---|
| 1 | P0 → P1 → bắt đầu P2 | Kiểm chứng license; owner ghi ảnh và giọng nguồn |
| 2 | P2 xong → P3 | P4a (kính) và P4b (deepfake) trên vast.ai; người ngoài buổi 1 |
| 3 | P4c, prereg, thu dữ liệu test, train detector và R2 | In và thử kính vật lý; người ngoài buổi 2 |
| 4 | Chạy E0–E5, fallback nếu cần, P6 | Người ngoài buổi 3 (E5); quay demo |

## 6. Thí nghiệm, ngưỡng đã chốt và phương án dự phòng

Ngưỡng là ngưỡng ở spec §6, đã chốt. Nếu trượt thì chạy phương án dự phòng theo thứ tự trong bảng, và **mọi lần trượt đều ghi vào `experiment_log.md`** (giả thuyết, đã làm gì, vì sao trượt).

| ID | Dữ liệu khi chỉ có một người | Ngưỡng | Dự phòng, theo thứ tự |
|---|---|---|---|
| E0 | Cặp verification của dataset môn học; owner và người ngoài so với lớp mô phỏng | Chỉ báo cáo | Không cần |
| E1a | Ảnh của owner, dán texture kính, tầng D | ≥ 80% | (1) kiểm tra crop và align giữa lúc tối ưu và backend giống nhau; tăng bước tối ưu và vùng mask. (2) phụ kiện lớn hơn (miếng dán trên mũ kiểu AdvHat). (3) rút L1 về phân tích transfer và độ rơi digital → physical; demo defense bằng frame tấn công digital đưa qua virtual camera |
| E1b | Owner đeo kính thật, có kính control | Báo cáo IAPMR | Hợp lệ ở mọi kết quả; nếu IAPMR thấp thì phân tích khoảng cách digital → physical |
| E1c | Tấn công L1 (vật lý, hoặc virtual camera nếu E1b thấp) so với fusion R3 | IAPMR ≤ 5%, BPCER tăng ≤ 5 điểm % | (1) chỉnh ngưỡng giọng trên tập dev. (2) cho phép một lần retry với chuỗi số mới. (3) báo cáo đánh đổi giữa IAPMR và BPCER |
| E2 | Owner replay bản ghi của chính mình (câu cố định và chuỗi số cũ) | Câu cố định pass ≥ 80%; chuỗi ngẫu nhiên IAPMR ≤ 2% | ASR kém thì: (1) prompt hoặc giới hạn từ vựng chữ số. (2) **tự train bộ nhận dạng chữ số** trên Speech Commands (thêm một model tự train) |
| E3a | Giọng clone của owner so với ECAPA | Báo cáo | Không cần |
| E3b | Ngưỡng SyncNet trên Wav2Lip của owner và dataset public | Dự đoán AUC < 0.75 | Không cần: kết quả nào cũng là một phát hiện |
| E3c | Detector train trên public, test trên owner | AUC ≥ 0.90 | (1) thêm đặc trưng và dữ liệu. (2) chuyển từ LR/GBM sang GRU/1D-CNN. (3) đặc trưng visual-speech (đọc khẩu hình chữ số). (4) nếu < 0.75 thì ghi là giới hạn; defense L3 dựa vào thời gian (E3d) |
| E3d | Latency sinh deepfake so với hạn challenge | Báo cáo phân bố | Không cần |
| E4 | Toàn bộ bước verify trên máy owner | ≤ 3 s | (1) Whisper tiny.en int8. (2) chạy các kênh song song. (3) cắt cửa sổ SyncNet. (4) dùng GPU nếu có |
| E5 | Owner nhiều buổi, người ngoài | BPCER ≤ 10%, ≤ 1 lần retry (yên tĩnh) | (1) chỉnh ngưỡng trên dev. (2) hướng dẫn trên giao diện tốt hơn. (3) cho 2 lần retry, ghi lại cái giá phải trả |

Ablation bắt buộc giữ nguyên như spec: thêm dần từng kênh, so R1/R2/R3, hạn challenge 5/8/12 s, chuỗi 4/6 số, có và không có noise suppression.

## 7. Phần tự train (để đáp ứng yêu cầu "train your own model")
- **AV-inconsistency detector:** phần train chính.
- **Fusion R2:** logistic regression trên điểm các kênh.
- **Bộ nhận dạng chữ số** (nếu kích hoạt dự phòng E2).
- **Tối ưu texture kính adversarial:** tối ưu chứ không phải train, nhưng là phần tự build.
- **ECAPA và Whisper dùng pretrained làm bộ trích đặc trưng.** Phương án nếu cần: fine-tune nhẹ đầu ECAPA trên VoxCeleb (quyết định ở P1, ghi vào `decisions.md`).

## 8. Rủi ro chính
- **Dataset deepfake public cần đăng ký hoặc có license hạn chế** (FakeAVCeleb, AV-Deepfake1M). Dự phòng: LAV-DF, hoặc chỉ dùng fake của owner kèm cross-validation theo buổi.
- **Một người tham gia làm khả năng tổng quát hoá yếu.** Báo cáo rõ; người ngoài là bằng chứng bổ sung.
- **Decode webm/opus trên Windows** (P0 kiểm tra sớm).
- **Dependency xung đột với torch 2.2.2 / numpy 1.26.** Được phép nâng phiên bản, nhưng phải chạy lại test.
- **Checkpoint thật không nằm trong repo.** Owner phải chép lên máy GPU cho P4a, và chạy đánh giá cuối trên máy có checkpoint.

## 9. Việc owner cần chuẩn bị
- [ ] Checkpoint thật: `liveness_best.pth`/`liveness_config.json`, `method3_best.pth` (và method1/2/4/5 cho transfer), emotion.
- [ ] Tài khoản vast.ai hoặc Kaggle cho P4.
- [ ] Đăng ký dataset cần form (FakeAVCeleb, AV-Deepfake1M) nếu kiểm chứng cho thấy cần.
- [ ] Gọng kính nhựa rẻ, máy in màu, giấy in.
- [ ] Điện thoại để replay; phần mềm virtual camera và virtual audio cable (OBS Virtual Camera, VB-CABLE) cho kênh (b) của L3.
- [ ] Lịch người ngoài: 3 buổi, mỗi buổi khoảng 30–45 phút.
