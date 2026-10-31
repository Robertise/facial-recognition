# SPEC — Individual Innovation: "Đúng người, không chỉ người thật" (Attack Ladder đa phương thức)

Owner: Robertiz. Ngày viết: 2026-10-31. Trạng thái: **DRAFT, chờ user chốt các mục ở §11 trước khi agent bắt đầu.**
Đường dẫn tương đối tính từ `PROJECT/code/` (repo), trừ khi ghi khác.

Quy tắc cho agent (bắt buộc):
- Không commit. User tự review và commit.
- Không đổi tên thứ đã có (CLI args, key JSON/config, cột CSV, tên thư mục run, key trong JSON trả về của WebSocket). Chỉ **thêm** key mới.
- Không sửa file của người khác khi chưa có lead duyệt: `backend/face_recognizer.py`, `emotion/*`, `extreme_cond_enhancer/*`, `test_liveness_*.py`. Frontend và `backend/main.py`, `backend/processing.py` là file dùng chung: chỉ sửa theo đúng §7 và §8, sau khi user xác nhận lead đồng ý (mục Q1 ở §11).
- Mọi thứ ngoài scope này: ghi vào `plans/innovation_out_of_scope.md`, không tự sửa.
- Các mục đánh dấu **[KIỂM CHỨNG]** là thông tin Claude chưa xác minh (license, số liệu, API). Agent phải kiểm tra trước khi dùng và ghi kết quả vào `plans/innovation_verification_log.md`.

---

## 1. Luận điểm (thesis) của innovation

Anti-spoofing chỉ trả lời câu hỏi "đây có phải **người thật** không". Nó không trả lời câu hỏi "đây có phải **đúng người** không". Một bạn A đứng trước camera, đeo gọng kính in hoa văn adversarial, và được điểm danh thành bạn B. Liveness pass vì A là người thật. Face recognition pass vì hoa văn đã lừa được embedding.

Innovation này xây một **chuỗi tấn công leo thang (attack ladder)** lên chính hệ thống điểm danh của team. Ở mỗi mức có bốn phần: tấn công, đo xem hệ thống fail ra sao, phân tích vì sao fail, và thêm defense đúng cho lỗi đó. Defense chính là **xác thực đa phương thức có thách thức ngẫu nhiên**: khuôn mặt, giọng nói đã enroll, và kiểm tra rằng giọng đó phát ra từ chính miệng đang ở trong khung hình lúc này (audio-visual sync). Mức cao nhất (clone giọng + lip-sync deepfake) được tấn công thật và báo cáo giới hạn một cách trung thực.

Chiều sâu đến từ ba thứ: (1) mỗi defense được biện minh bằng lỗi đo được ở mức trước; (2) mỗi lựa chọn có mục "vì sao không phải X"; (3) mọi luật quyết định được đặt trước khi có số (§6).

## 2. Threat model

- **Kẻ tấn công:** một sinh viên trong lớp (insider). Mục tiêu là điểm danh hộ bạn B (impersonation). Kẻ tấn công có: ảnh/video công khai của B (mạng xã hội), vài giây đến vài chục giây giọng của B, một GPU thuê (vast.ai), máy in màu, điện thoại.
- **Không nằm trong threat model:** chiếm quyền máy điểm danh, sửa code client, tấn công DB/mạng. Virtual camera injection chỉ được dùng **trong thí nghiệm L3** như một kênh đưa media vào, và được ghi rõ là giả định mạnh hơn thực tế.
- **Hệ thống mục tiêu:** đúng hệ thống của team. Recognizer `FaceCNN_ArcFace` (`checkpoints/method3_best.pth`, input 112×112, embedding 512, cosine, threshold 0.5, so khớp max theo từng angle đã enroll). Liveness run3 + smoothing/gate. Webcam laptop 640×480.
- **Đạo đức:** chỉ thành viên team có consent mới là "nạn nhân" và "kẻ tấn công". Không clone giọng hay mặt của người ngoài. Dữ liệu lưu cục bộ, xóa sau môn học. Không công bố texture adversarial. Đề xuất báo trước cho lecturer (Q5).

## 3. Attack ladder

| Mức | Tấn công | Dự đoán hệ thống hiện tại | Defense được thêm | Phần tự build/train |
|---|---|---|---|---|
| L0 | Zero-effort impostor (A đứng trước camera, không làm gì) + ảnh/video B trên điện thoại | Đo FAR giữa các thành viên team. Replay ở xa: liveness fail (đã đo p_live ≈ 0.98 khi JPEG + mặt nhỏ) | Không thêm, đây là baseline | Không |
| L1 | **Gọng kính adversarial** in ra, A đeo để bị nhận thành B | Liveness pass (người thật). Recognition bị lừa (cần đo) | **Giọng đã enroll** như kênh độc lập + fusion | Tối ưu texture kính (EOT); train fusion |
| L2 | Replay bản ghi B (video có tiếng) đọc tên hoặc câu cố định | Nếu dùng câu cố định: pass | **Chuỗi số ngẫu nhiên do server sinh**, dùng một lần, có hạn thời gian + ASR khớp nội dung | Không train (logic) |
| L3 | Clone giọng B (TTS) + lip-sync (Wav2Lip) đọc đúng chuỗi số mới, phát qua điện thoại hoặc virtual camera | Speaker verification có thể bị clone lừa. ASR pass. Face pass | **Kiểm tra AV sync / phát hiện lip-sync deepfake** + giới hạn thời gian phản hồi | **Train AV-inconsistency detector** trên dữ liệu team tự sinh |
| L4 | Adaptive: kẻ tấn công biết detector và tối ưu để qua mặt | Có thể pass | Không cam kết defense; phân tích giới hạn | Chỉ phân tích |

L0 là baseline. Phần lõi là **L1 + L2 + L3**. L4 chỉ phân tích, có thì làm, hết thời gian thì là mục Discussion.

## 4. Thiết kế kỹ thuật từng thành phần

### 4.1 L1 — Gọng kính adversarial (attack)

**Cơ sở:** Sharif et al., "Accessorize to a Crime", CCS 2016 (gọng kính in được, impersonation + dodging). AdvHat (Komkov & Petiushko, 2019, sticker trên mũ lừa ArcFace). EOT (Athalye et al., ICML 2018). [KIỂM CHỨNG: số liệu physical success trong từng paper trước khi trích.]

**Pipeline** (offline, chạy trên vast.ai, nằm trong `innovation/attacks/adv_glasses/`):
1. Mask gọng kính cố định trên template mặt đã align 112×112. Mask là ảnh PNG, vẽ một lần, có thể chỉnh.
2. Ảnh của kẻ tấn công A: khoảng 30–60 ảnh từ webcam ở nhiều góc và ánh sáng (dùng lại capture tool của liveness nếu được).
3. Đích: embedding trung bình của B (lấy từ các angle đã enroll trong DB).
4. Tối ưu texture T trong vùng mask. Loss = (1 − cos(f(A ⊕ T), e_B)) + λ_tv·TV(T) + λ_nps·NPS(T).
   - EOT: random affine nhỏ (±5° xoay, ±5% scale, ±3 px dịch), brightness/contrast, gaussian blur, **JPEG q 0.6–0.95**, resize về 80–200 px rồi lên 112 (mô phỏng khoảng cách).
   - NPS: non-printability score với bảng màu in được, đo bằng cách in một bảng màu rồi chụp lại bằng webcam.
5. Xuất texture ở độ phân giải in (ví dụ 300 dpi), in lên giấy, dán lên một gọng kính nhựa rẻ.

**Ba tầng đo** (độ rơi giữa các tầng chính là phần phân tích):
- **D (digital):** dán texture lên ảnh A trong bộ test, không qua camera.
- **D+EOT (simulated physical):** như D nhưng áp biến đổi EOT chưa thấy lúc train.
- **P (physical):** A đeo kính thật trước webcam, chạy qua đúng pipeline backend.

**Transfer study** (phân tích "vì sao"): tấn công white-box trên `method3`, đo trên `method1/2/4/5` (5 checkpoint của team đã có) và một model public (ví dụ `facenet-pytorch` InceptionResnetV1 [KIỂM CHỨNG license]). Câu hỏi cần trả lời: attack có khai thác thứ gì chung giữa các model, hay chỉ khai thác điểm yếu riêng của `method3`?

**Vì sao không phải X:**
- *Patch dán lên má/trán:* dễ bị crop mất và không tự nhiên. Kính có sẵn trong không gian ảnh mặt và hợp lý về mặt xã hội.
- *Makeup adversarial:* khó lặp lại và khó in.
- *Digital perturbation toàn ảnh (PGD):* không khả thi về mặt vật lý trước webcam, chỉ dùng làm upper bound ở tầng D.

### 4.2 Voice enrollment + speaker verification (defense cho L1)

- Model: ECAPA-TDNN pretrained (`speechbrain/spkrec-ecapa-voxceleb`) [KIỂM CHỨNG: license, latency trên CPU máy yếu nhất, kích thước]. Embedding 192 chiều, cosine.
- Enrollment: 3 câu, mỗi câu là một chuỗi 6 chữ số ngẫu nhiên. Lưu embedding trung bình và từng embedding. Không lưu audio thô trừ khi user bật "lưu cho nghiên cứu" (cần cho thí nghiệm).
- Tiền xử lý: resample 16 kHz mono, VAD cắt im lặng, loại mẫu quá ngắn (< 1.5 s có tiếng nói) hoặc clipping.
- Khi verify: embedding từ clip challenge so với embedding đã enroll của **ứng viên mà face recognition đưa ra** (không phải tìm toàn DB). Đây là verification 1:1, nên giọng và mặt bị ràng buộc với cùng một danh tính.

**Vì sao không phải X:**
- *Face–voice cross-modal matching không cần enroll* (Nagrani et al., CVPR 2018): accuracy quá thấp để làm biometric, và dễ học shortcut (giới tính, tuổi). Chỉ giữ làm mục phân tích tùy chọn.
- *Train speaker model từ scratch:* không đủ dữ liệu và thời gian. Phần train của innovation nằm ở fusion và AV detector.

### 4.3 Random challenge + ASR (defense cho L2)

- Server sinh chuỗi 6 chữ số bằng `secrets`. Mỗi challenge có `challenge_id`, gắn với `candidate_id`, `issued_at`, `expires_at` (cửa sổ T_window, mặc định 8 s, cấu hình được), và chỉ dùng một lần (`used`).
- Ngôn ngữ đọc số: tiếng Anh mặc định (Q2). Cấu hình `challenge_lang`.
- ASR: Whisper tiny/base qua `faster-whisper` [KIỂM CHỨNG: chạy CPU int8, latency với clip 4 s]. Có thể dùng prompt hoặc grammar hạn chế về chữ số. Chuẩn hóa transcript thành chuỗi số rồi so bằng edit distance. Pass nếu khoảng cách ≤ 1 (cấu hình được).
- **Thời gian phản hồi** = thời điểm bắt đầu có tiếng nói − `issued_at`. Ghi lại, dùng cho phân tích L3 (kẻ tấn công cần thời gian để sinh deepfake).

**Vì sao không phải X:** câu cố định (tên, MSSV) bị replay ngay. Ablation ở §6 chứng minh điều này.

### 4.4 AV sync + phát hiện lip-sync deepfake (defense cho L3)

- Tín hiệu nền: SyncNet (Chung & Zisserman, ACCV-W 2016). Đầu vào là mouth crop khoảng 25 fps cộng MFCC. Đầu ra là đường cong confidence theo offset, cho biết offset tốt nhất và độ "nhọn" của đỉnh. [KIỂM CHỨNG: weights public `joonson/syncnet_python`, license, CPU latency.]
- **Phần tự train (trọng tâm build & train):** một classifier nhẹ phân biệt "giọng thật từ miệng thật" với "lip-sync tổng hợp". Đặc trưng:
  - đường cong conf(offset) của SyncNet, tính theo cửa sổ trượt (thống kê: max, độ nhọn, phương sai offset theo thời gian);
  - năng lượng chuyển động vùng miệng so với envelope âm thanh (cross-correlation);
  - đặc trưng vùng miệng (độ mờ, biên ranh giới vùng ghép của Wav2Lip). [KIỂM CHỨNG: nghiên cứu về dấu vết Wav2Lip, ví dụ LipForensics CVPR 2021, Chugh et al. 2020 "Not made for each other".]
  - Model bắt đầu từ logistic regression hoặc GBM, rồi đến GRU nhỏ hoặc 1D-CNN nếu cần. Chia train/val/test theo người.
- Dữ liệu: genuine là các thành viên team đọc challenge (§5). Fake do team tự sinh bằng pipeline L3 trên chính mình. Public bổ sung nếu xin được: FakeAVCeleb [KIỂM CHỨNG: form đăng ký, license].
- **Vì sao không phải X:**
  - *Chỉ dùng threshold trên SyncNet conf:* Wav2Lip được train với SyncNet làm discriminator, nên khả năng cao là output của nó **sync tốt theo SyncNet**. Đây là dự đoán cần đo, và là một insight hay nếu đúng: defense dùng đúng thước đo mà kẻ tấn công đã tối ưu thì sẽ vô dụng. Đó là lý do phải train detector riêng.
  - *Deepfake detector tổng quát pretrained:* generalize kém sang generator và điều kiện mới (vấn đề đã biết). Detector train trên dữ liệu đúng của hệ thống phù hợp hơn với scope này, nhưng phải báo cáo giới hạn generalization.

### 4.5 Phía tấn công L3 (offline, `innovation/attacks/av_deepfake/`)

- Voice clone: OpenVoice V2 (MIT) hoặc XTTS-v2 (Coqui Public Model License, phi thương mại) [KIỂM CHỨNG cả hai]. Nguồn là 10–30 s giọng của B.
- Lip-sync: Wav2Lip [KIỂM CHỨNG license, chỉ dùng cho nghiên cứu]. Nguồn là một video hoặc ảnh của B.
- Hai kênh đưa vào hệ thống:
  - (a) phát trên điện thoại hoặc laptop thứ hai trước webcam và mic (thực tế, nhưng liveness có thể bắt được ở gần);
  - (b) virtual camera + virtual audio cable (mạnh hơn, bỏ qua liveness texture, và được ghi rõ là ngoài threat model gốc).
- **Đo latency sinh deepfake** từ lúc thấy chuỗi số đến lúc phát xong. So với T_window. Đây là phân tích "thời gian là tham số bảo mật": T_window càng ngắn thì càng khó tấn công nhưng người thật càng dễ fail.

### 4.6 Fusion và quyết định

- Vector điểm cho mỗi lần thử: `s_face` (cosine), `p_live` (liveness đã smooth), `s_voice` (cosine ECAPA với ứng viên), `asr_ok` + `asr_dist`, `s_av` (output AV detector), `t_response`.
- So sánh ba luật (ablation chính):
  - **R1 AND-threshold:** mỗi kênh có ngưỡng riêng;
  - **R2 score-level logistic regression** (train được);
  - **R3 R2 + luật cứng:** `asr_ok` và challenge còn hạn là bắt buộc.
- Ngưỡng chọn trên tập dev rồi **khóa** trước khi chạy tập test.
- Output quyết định: `accept`, `reject`, hoặc `retry` (chất lượng âm thanh kém, không thấy miệng, quá thời gian). Luôn kèm `reason_codes`, ví dụ `VOICE_MISMATCH`, `AV_DESYNC`, `ASR_MISMATCH`, `EXPIRED`, `FACE_MISMATCH`, `SPOOF`, `LOW_AUDIO_QUALITY`.

## 5. Dữ liệu

- **Người tham gia:** tối thiểu các thành viên team có consent (Q3). Mỗi người làm cả genuine lẫn impostor cho người khác.
- **Genuine:** mỗi người khoảng 30 lần challenge, chia ra 2–3 buổi, 2 điều kiện ánh sáng, 2 mức ồn (phòng yên tĩnh, có tiếng nói nền).
- **L0:** mọi cặp (A trước camera, so với enroll của B), cộng với replay điện thoại ở gần và ở xa.
- **L1:** kính adversarial cho ít nhất 2 cặp (A→B), mỗi cặp khoảng 30 lần thử vật lý, 2 khoảng cách. Cộng với **kính thường có texture ngẫu nhiên** làm control (để tách hiệu ứng "đeo kính" khỏi hiệu ứng "adversarial").
- **L2:** replay video B đọc câu cố định và đọc một chuỗi số cũ.
- **L3:** khoảng 30 clip deepfake cho mỗi nạn nhân, qua cả hai kênh (a) và (b).
- **Lưu trữ:** `data/innovation/<session_id>/` gồm clip (webm), `meta.json` (người, điều kiện, attack_level, kênh, challenge), và điểm từng kênh. Không đẩy dữ liệu lên git (thêm vào `.gitignore` — đây là file dùng chung, hỏi user trước).
- **Chia tập:** theo người. Tập test thu **sau** khi commit `PREREG_innovation.md`.

## 6. Thí nghiệm và luật quyết định đặt trước

Metric theo ISO/IEC 30107-3: APCER theo từng loại tấn công, BPCER, và **IAPMR** (impostor attack presentation match rate: tỉ lệ tấn công vừa qua PAD vừa match đúng nạn nhân). Cộng thêm FAR/FRR cho recognition và latency.

**Các con số dưới đây là đề xuất. User sửa hoặc chốt, rồi commit `PREREG_innovation.md` trước khi thu tập test.**

| ID | Câu hỏi | Luật đặt trước |
|---|---|---|
| E0 | Baseline zero-effort FAR giữa các thành viên team ở threshold 0.5 | Chỉ báo cáo. Nếu FAR > 10% thì ghi rằng threshold hiện tại đã yếu ngay cả khi chưa có attack, và L1 phải so với baseline này |
| E1a | Kính adversarial, tầng D, white-box | Success ≥ 80%. Nếu thấp hơn, coi là lỗi implement và debug trước khi đi tiếp |
| E1b | Tầng P (vật lý) | Báo cáo IAPMR, có so với control kính ngẫu nhiên. **Không có ngưỡng pass**: IAPMR thấp thì câu chuyện chuyển thành phân tích khoảng cách digital→physical, vẫn hợp lệ |
| E1c | Fusion chặn L1 | Với R3, IAPMR của L1 ≤ 5%, đồng thời BPCER tăng không quá 5 điểm % so với face-only |
| E2 | Random challenge chặn replay | Câu cố định: dự đoán replay pass ≥ 80% (cho thấy cần challenge). Chuỗi ngẫu nhiên: IAPMR replay ≤ 2% |
| E3a | Clone giọng có lừa được ECAPA không | Chỉ báo cáo tỉ lệ accept. Dự đoán là cao |
| E3b | SyncNet threshold có bắt được Wav2Lip không | Dự đoán **không** (AUC < 0.75). Nếu đúng, đó là lý do train detector |
| E3c | Detector tự train | AUC ≥ 0.90 trên tập test chia theo người. Nếu từ 0.75 đến 0.90: báo cáo kèm phân tích lỗi. Nếu < 0.75: ghi là giới hạn, và khi đó defense L3 dựa vào T_window (E3d) |
| E3d | Latency sinh deepfake so với T_window | Báo cáo phân bố. Nếu p50 latency tấn công > T_window thì ghi rằng thời gian chặn được L3 ở cấu hình này, kèm cảnh báo rằng GPU nhanh hơn sẽ phá điều kiện này |
| E4 | Latency hệ thống trên máy yếu nhất team | Toàn bộ bước verify (ECAPA + ASR + AV) ≤ 3 s sau khi ghi xong clip. Nếu vượt: hạ model (Whisper tiny, cắt SyncNet window) trước khi bỏ kênh |
| E5 | Trải nghiệm người thật | Ở ánh sáng văn phòng, yên tĩnh: BPCER ≤ 10% và tối đa 1 lần retry. Ở môi trường ồn: báo cáo, không có ngưỡng |

**Ablation bắt buộc:** face-only, face+liveness, +voice, +challenge, +AV. Ba luật fusion R1, R2, R3. T_window ∈ {5, 8, 12} s. Độ dài chuỗi ∈ {4, 6} chữ số. Có và không có noise suppression của trình duyệt.

Mọi đề xuất bị loại trong quá trình làm được ghi vào `plans/innovation_decisions.md` kèm lý do, để không đề xuất lại.

## 7. Thay đổi hệ thống (backend, DB, API)

### 7.1 Cấu trúc thư mục mới (không đụng module cũ)
```
innovation/
  __init__.py
  config.py              # mọi ngưỡng/tham số, đọc từ innovation/config.json
  challenge.py           # sinh/kiểm tra challenge (secrets, expiry, one-time)
  media.py               # decode webm -> frames (fps gốc) + audio 16k (PyAV/ffmpeg)
  voice.py               # ECAPA embed, VAD, quality check
  asr.py                 # faster-whisper, chuẩn hoá số
  avsync.py              # SyncNet features + detector inference
  fusion.py              # R1/R2/R3, reason_codes
  pipeline.py            # verify_attempt(clip, challenge, candidate) -> dict
  attacks/adv_glasses/   # notebook vast.ai + script eval D / D+EOT
  attacks/av_deepfake/   # script sinh clone+lipsync (offline)
  eval/                  # harness chạy file-mode trên data/innovation, xuất CSV/figures
  tests/
```

### 7.2 DB (chỉ thêm bảng, không sửa bảng cũ)
- `voice_embeddings(id, employee_id FK, embedding BLOB, source TEXT, created_at)`
- `challenges(id TEXT PK, candidate_id, digits, issued_at, expires_at, used INT, mode TEXT)`
- `verification_attempts(id, challenge_id, candidate_id, decision, reason_codes TEXT(json), s_face, p_live, s_voice, asr_text, asr_dist, s_av, t_response, latency_ms, attack_label TEXT NULL, session_tag TEXT NULL, created_at)`

`attack_label` và `session_tag` chỉ được điền ở chế độ thu dữ liệu hoặc nghiên cứu.

### 7.3 API mới (REST, thêm vào `backend/main.py` bằng một `APIRouter` trong `innovation/api.py`)
- `POST /api/voice/enroll/start {employee_id}` → trả về 3 chuỗi số cần đọc.
- `POST /api/voice/enroll/submit` (multipart: employee_id, index, audio webm) → `{ok, quality: {duration_s, snr_db, clipping}, reason}`.
- `GET /api/voice/status/{employee_id}` → `{enrolled, n_samples}`.
- `POST /api/challenge/start {candidate_id}` → `{challenge_id, digits, expires_at, window_s}`. Chỉ cấp khi face đã match ứng viên trong N frame gần nhất (lấy từ state của WebSocket session).
- `POST /api/challenge/{id}/submit` (multipart: clip webm có cả video và audio, `client_started_at`) → kết quả verify đầy đủ (§7.4).
- `POST /api/research/analyze_file` (multipart: video file, `claimed_id`, `digits`) → cùng kết quả nhưng bỏ qua kiểm tra expiry và one-time. **Chỉ bật khi `INNOVATION_RESEARCH_MODE=1`.** Dùng cho eval harness và trang Red-team Lab. Phải ghi rõ trong report rằng file-mode không phải đường điểm danh.
- `GET /api/attempts?limit=&decision=&attack_label=` → cho dashboard.
- `GET /api/innovation/config` → tham số hiện tại (chỉ đọc khi demo).

### 7.4 Schema kết quả verify (key mới, frontend dựa vào)
```json
{
  "decision": "accept|reject|retry",
  "reason_codes": ["AV_DESYNC"],
  "candidate": {"employee_id": "...", "name": "..."},
  "channels": {
    "face":     {"similarity": 0.71, "pass": true},
    "liveness": {"score": 0.93, "pass": true},
    "voice":    {"similarity": 0.62, "pass": true},
    "asr":      {"expected": "482915", "heard": "482915", "distance": 0, "pass": true},
    "av":       {"score": 0.12, "best_offset": 0, "curve": [[-15,0.1],[0,0.8]], "pass": false},
    "timing":   {"response_s": 1.4, "within_window": true}
  },
  "fusion": {"rule": "R3", "score": 0.31, "threshold": 0.5},
  "latency_ms": {"decode": 120, "voice": 300, "asr": 900, "av": 700, "total": 2100},
  "attendance_logged": false
}
```

### 7.5 Thay đổi bắt buộc ở luồng điểm danh hiện tại
- Hiện tại `ws_attendance` **tự log** điểm danh ngay khi face match (debounce 30 s). Khi bật **Secure mode**, luồng WebSocket chỉ được đánh dấu `candidate_ready` cho ứng viên, **không được log**. Chỉ `/api/challenge/{id}/submit` với `decision=accept` mới được log. Nếu thiếu điểm này thì mọi defense bị bypass. Đây là acceptance test số 1.
- Thêm key mới vào JSON của WebSocket: `secure_mode: bool`, `candidate_ready: bool`, `candidate: {...}`. Không đổi các key cũ.
- Bảng `attendance`: không đổi schema. Ghi thêm một dòng `verification_attempts` và liên kết qua timestamp/candidate. Nếu cần FK thì hỏi lead.

### 7.6 Dependency mới [KIỂM CHỨNG: chạy được với torch 2.2.2, numpy<2, Windows]
`speechbrain`, `faster-whisper`, `av` (PyAV) hoặc `ffmpeg` trên PATH, `webrtcvad` hoặc Silero VAD, `scipy`, `python_speech_features` (MFCC cho SyncNet) hoặc `torchaudio`. Thêm vào file `innovation/requirements.txt` riêng, không sửa requirements chung khi lead chưa duyệt.

## 8. Yêu cầu giao diện (UI) để phục vụ innovation

Nguyên tắc: thêm trang và component mới. Chỉ sửa trang cũ ở những điểm nối cần thiết. Mọi tính năng innovation nằm sau feature flag `NEXT_PUBLIC_INNOVATION=1`, để bản demo của team vẫn chạy như cũ khi tắt.

### 8.1 Hạ tầng frontend dùng chung
- **`lib/media.ts`**: hook `useAVRecorder()` dùng `getUserMedia({video: {width: 640, height: 480, frameRate: {ideal: 30}}, audio: {channelCount: 1, sampleRate: 48000, echoCancellation, noiseSuppression, autoGainControl}})` + `MediaRecorder` (webm/opus). API gồm `start()`, `stop() -> Blob`, mức âm lượng realtime (AnalyserNode), và `actualFps`. Các cờ echoCancellation/noiseSuppression/AGC phải cấu hình được, vì đây là một ablation (§6).
- **Dùng chung một stream camera** cho luồng WebSocket 5 fps và cho MediaRecorder, để tránh mở camera hai lần (Windows hay lỗi khi làm vậy).
- **Xử lý quyền mic/camera:** màn hình hướng dẫn khi bị từ chối, nút thử lại, chọn thiết bị (dropdown mic và camera).
- **`lib/api.ts`**: wrapper các endpoint ở §7.3, có type cho schema §7.4.

### 8.2 Trang Register (`app/register/page.tsx`): thêm bước "Voice enrollment"
- Bước này đứng sau bước chụp 5 góc mặt hiện có, và có thể bỏ qua nếu tắt flag.
- Checkbox consent (bắt buộc), kèm nội dung ngắn: dữ liệu dùng làm gì, lưu ở đâu, xóa thế nào.
- Ba thẻ, mỗi thẻ hiển thị một chuỗi số lớn. Mỗi thẻ có nút Ghi (đếm ngược 3-2-1), thanh mức âm lượng, nút nghe lại, và nút ghi lại.
- Sau mỗi lần ghi có phản hồi chất lượng từ backend: quá ngắn, quá nhỏ, rè/clipping, ồn. Hiển thị bằng ngôn ngữ dễ hiểu.
- Trạng thái cuối: "Đã enroll giọng (3/3)". Trên danh sách nhân viên có badge 🎙 đã/chưa enroll.

### 8.3 Trang Attendance (`app/attendance/page.tsx`): thêm Secure mode
- Toggle **Quick** (face-only, như hiện tại) / **Secure** (face + voice + AV). Hiển thị rõ đang ở chế độ nào.
- State machine hiển thị trên UI: `Tìm mặt → Nhận ra ứng viên (tên, có nút "Không phải tôi") → Challenge → Đang ghi → Đang xác minh → Kết quả`.
- **Màn hình challenge:**
  - chuỗi số rất lớn ở giữa, cạnh camera;
  - đếm ngược T_window (thanh tiến trình);
  - chấm đỏ báo đang ghi, thanh âm lượng realtime;
  - khung nhỏ hiển thị vùng miệng (crop từ bbox ở client) để người dùng biết cần đưa miệng vào khung;
  - cảnh báo realtime: "không thấy miệng", "quá nhỏ tiếng", "quá xa" (dùng gate có sẵn).
- Tự dừng ghi khi phát hiện hết tiếng nói (VAD phía client đơn giản theo năng lượng) hoặc khi hết giờ.
- **Thẻ kết quả:** Accept hoặc Reject lớn, kèm **breakdown từng kênh** (face, liveness, voice, ASR expected vs heard, AV, timing). Mỗi kênh có thanh điểm, vạch ngưỡng và icon pass/fail, cộng với lý do bằng lời từ `reason_codes`.
- Retry tối đa 2 lần, mỗi lần một chuỗi số mới. Quá số lần thì chuyển sang "Liên hệ giảng viên" và log lại.
- Log điểm danh (`AttendanceLog`) có thêm cột "Phương thức" (Quick/Secure) và icon 🎙/🛡.

### 8.4 Trang mới: Security Dashboard (`app/security/page.tsx`)
- Bảng `verification_attempts`: thời gian, ứng viên, quyết định, reason_codes, điểm từng kênh. Lọc theo quyết định, lý do và attack_label.
- **Cảnh báo nghi vấn:** các lần thử có face pass nhưng voice hoặc AV fail sẽ được tô màu. Đây là dấu hiệu impersonation, đúng thứ cần demo cho L1/L3.
- Biểu đồ: số lần reject theo reason_code; scatter `s_face` × `s_voice` với ngưỡng (genuine, impostor và từng mức attack có màu riêng nếu có nhãn).
- Xuất CSV.

### 8.5 Trang mới: Red-team Lab (`app/lab/page.tsx`), chỉ khi `INNOVATION_RESEARCH_MODE=1`
- **Analyze file:** upload video (có tiếng), chọn `claimed_id`, nhập chuỗi số. Gọi `/api/research/analyze_file` và hiển thị đầy đủ breakdown. Đây là chỗ dùng "input video file" mà lecturer cho phép.
- **Trực quan AV:**
  - đường cong SyncNet conf(offset);
  - offset theo thời gian;
  - waveform âm thanh chồng với độ mở miệng theo thời gian. Thật thì hai đường bám nhau, lip-sync thì lệch hoặc bị "quá mượt". Đây là hình chính cho report.
- **Trực quan L1:** xem texture kính, tỉ lệ success theo tầng D → D+EOT → P, cosine với B theo từng frame khi đeo kính thật. Dữ liệu đọc từ JSON do script offline xuất ra; UI không chạy tối ưu.
- **Chế độ thu dữ liệu:** chọn `attack_label` (genuine, L0_replay, L1_glasses, L1_control, L2_replay, L3_phone, L3_vcam), `session_tag`, người và điều kiện (ánh sáng, ồn, khoảng cách), rồi bấm thu. Mỗi lần thử được ghi kèm nhãn. Cái này thay cho việc tự đặt tên file bằng tay.
- **Chỉnh tham số tạm thời** (T_window, độ dài chuỗi, luật fusion) để chạy ablation live. Có nút "reset về config đã khóa".

### 8.6 Demo mode (phục vụ quay video)
- Bố cục chia đôi: trái là camera + challenge, phải là breakdown kênh và biểu đồ AV realtime ngay sau mỗi lần thử.
- Bộ chọn kịch bản L0 → L3 hiển thị tiêu đề mức tấn công trên màn hình, để video tự kể chuyện.
- Một phím tắt bật/tắt từng kênh defense (face-only → +voice → +challenge → +AV), để demo trực tiếp rằng tấn công pass rồi bị chặn.

### 8.7 Yêu cầu phi chức năng cho UI
- Không làm luồng WebSocket 5 fps chậm đi khi đang ghi clip.
- Phản hồi sau khi bấm dừng ≤ 3 s trên máy yếu nhất (E4). Có spinner kèm các bước "Đang giải mã → giọng → nội dung → khớp môi".
- Mọi chuỗi giao diện tiếng Việt (giữ code và key tiếng Anh).
- Accessibility tối thiểu: chữ số challenge đủ lớn, không chỉ dùng màu để báo pass/fail.

## 9. Các phase cho agent và tiêu chí nghiệm thu

Claude verify từng phase bằng diff thật và output thật, không dựa vào log của agent.

| Phase | Nội dung | Nghiệm thu |
|---|---|---|
| P0 | Hạ tầng: thư mục `innovation/`, config, DB tables, router rỗng, feature flags, `useAVRecorder`, `lib/api.ts`, trang khung Security/Lab | App cũ chạy y hệt khi tắt flag (chạy lại test liveness hiện có). Ghi một clip 4 s, tải lên, backend decode ra fps thực ≥ 20 và audio 16 kHz, in ra log |
| P1 | Voice enrollment + speaker verification + UI Register | Enroll 1 người. Self-score cao hơn người khác trong một bảng điểm nhỏ do agent in ra. Latency ECAPA đo trên máy Robertiz |
| P2 | Challenge + ASR + Secure mode UI + **chặn auto-log** | Test tự động: ở Secure mode, face match nhưng không qua challenge thì không có dòng nào trong `attendance`. Challenge hết hạn hoặc dùng lại bị reject |
| P3 | AV sync (SyncNet features) + `fusion.py` R1 + thẻ kết quả đầy đủ | Breakdown hiển thị đủ 6 kênh. Clip có audio lệch 0.5 s cố ý bị SyncNet báo offset khác 0 |
| P4 | Red-team tooling offline: notebook adv glasses (vast.ai), eval D/D+EOT; script clone + lip-sync; eval harness file-mode xuất CSV | Có texture, có bảng success tầng D. Có ít nhất 5 clip deepfake. Harness chạy trên thư mục và xuất `results/innovation/*.csv` |
| P5 | Thu dữ liệu theo §5 qua Lab, commit PREREG, train AV detector + fusion R2, chạy §6 | CSV và figure cho E0–E5. Claude đối chiếu với PREREG |
| P6 | Demo mode + polish UI + README innovation | Kịch bản demo chạy trọn từ L0 đến L3 không lỗi |

**Điểm dừng:** nếu cuối P3 tổng latency vượt xa 3 s, hoặc decode webm trên Windows không ổn định, thì dừng lại bàn trước khi làm P4. Nếu E1a (tầng D) không đạt sau khi đã debug, L1 rút xuống còn phân tích transfer, và trọng tâm chuyển hẳn sang L2–L3.

## 10. Đề xuất đã loại (không đề xuất lại)

| Ý tưởng | Lý do loại |
|---|---|
| rPPG | Video replay mang nhịp tim thật; cần 5–10 s tín hiệu |
| YOLOv8n "cell phone" để bắt viền điện thoại | 0/92 khung ảnh xa, 4/93 khung video, +41 ms/khung |
| Blink thụ động | Video replay có blink tự nhiên |
| Screen-flash liveness | Phụ thuộc frontend, gần như làm lại Face Flashing/FaceRevelio, không đủ mới (user đánh giá) |
| Motion challenge + homography | Cùng nhóm "active liveness", không trả lời câu hỏi "đúng người" |
| Monocular depth, CDCN làm hướng chính | Vẫn là RGB một khung, không lấy lại thông tin đã mất |
| Sonar/ultrasonic | Rủi ro phần cứng và trình duyệt quá cao |
| Face–voice matching không enroll làm biometric | Accuracy thấp, dễ học shortcut; chỉ giữ làm phân tích tùy chọn |
| Multi-face detection làm innovation chính | Nông, dễ trùng với đồng đội |

## 11. Câu hỏi user cần chốt trước khi agent chạy

- **Q1.** Lead và các teammate có đồng ý cho sửa `backend/main.py`, `backend/processing.py`, frontend (trang Register, Attendance) và `.gitignore` không? Nếu không, phương án thay thế là một app Next.js riêng tên `innovation-ui/` và một `APIRouter` cắm vào bằng 1 dòng.
- **Q2.** Đọc chuỗi số bằng tiếng Anh hay tiếng Việt? (Ảnh hưởng ASR và độ chính xác.)
- **Q3.** Có bao nhiêu thành viên đồng ý tham gia thu dữ liệu (cả giọng lẫn deepfake của chính họ)?
- **Q4.** Ngân sách GPU vast.ai cho P4 là bao nhiêu (adv glasses + sinh deepfake)?
- **Q5.** Có báo trước cho lecturer về phần red-team (kính adversarial, deepfake) không?
- **Q6.** Ngày nộp cụ thể (cuối Week 11), để chia lịch các phase.
- **Q7.** Chốt hoặc sửa các ngưỡng ở §6 trước khi commit PREREG.

## 12. Mạch kể cho report và video

1. Hệ thống của team và liveness (phần cơ bản) cùng các lỗi đã đo (JPEG, khoảng cách, thiếu sáng).
2. Câu hỏi bị bỏ sót: liveness không trả lời "đúng người". L0 baseline FAR.
3. L1: tấn công kính, độ rơi digital → physical, transfer giữa 5 model của team. Vì sao embedding bị lừa.
4. Defense: thêm giọng làm kênh độc lập, có ablation fusion.
5. L2: replay phá câu cố định, chuỗi ngẫu nhiên chặn được.
6. L3: clone + lip-sync. SyncNet bị chính thước đo của nó đánh lừa (nếu E3b đúng), nên phải train detector. Thời gian là tham số bảo mật.
7. L4 và giới hạn: adaptive attack, virtual camera, môi trường ồn. Trung thực về những gì chưa chặn được.
