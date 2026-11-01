# Tổng hợp dự án: từ dọn repo đến L1 giai đoạn A

Ngày tổng hợp: 2026-11-01. Phạm vi: mọi việc đã làm trong repo `robertise/facial-recognition` từ 2026-10-31 đến hết commit `7a3a35e` (2026-11-01). Nguồn: lịch sử git, các tài liệu trong `docs/innovation/` và các trao đổi giữa owner với Claude.

Tài liệu gốc để tra chi tiết:

| File | Nội dung |
|---|---|
| [`spec_v1.md`](spec_v1.md) | Spec gốc của owner (giữ nguyên văn) |
| [`plan.md`](plan.md) | Kế hoạch tổng thể P0–P6, ngưỡng, phương án dự phòng |
| [`plan_L1.md`](plan_L1.md) | Kế hoạch chi tiết L1 (gọng kính adversarial), owner đã sửa |
| [`runbook_L1.md`](runbook_L1.md) | Các lệnh owner chạy cho L1 giai đoạn A |
| [`decisions.md`](decisions.md) | Các quyết định và lý do |
| [`experiment_log.md`](experiment_log.md) | Nhật ký mọi lần chạy, kể cả thất bại |
| [`verification_log.md`](verification_log.md) | Kiểm chứng thư viện, license, dataset |

---

## Mục lục
1. [Bối cảnh và quy tắc làm việc](#1-bối-cảnh-và-quy-tắc-làm-việc)
2. [Dòng thời gian](#2-dòng-thời-gian)
3. [Giai đoạn 1: dọn và tổ chức lại repo](#3-giai-đoạn-1-dọn-và-tổ-chức-lại-repo)
4. [Giai đoạn 2: thiết kế lại giao diện](#4-giai-đoạn-2-thiết-kế-lại-giao-diện)
5. [Giai đoạn 3: spec innovation và kế hoạch](#5-giai-đoạn-3-spec-innovation-và-kế-hoạch)
6. [P0: hạ tầng innovation](#6-p0-hạ-tầng-innovation)
7. [P1: giọng nói](#7-p1-giọng-nói)
8. [Sửa feature flag và đo P1 bằng model thật](#8-sửa-feature-flag-và-đo-p1-bằng-model-thật)
9. [Kết quả trên máy owner (P0, P1)](#9-kết-quả-trên-máy-owner-p0-p1)
10. [L1: thảo luận và kế hoạch](#10-l1-thảo-luận-và-kế-hoạch)
11. [L1 giai đoạn A: bộ công cụ tấn công](#11-l1-giai-đoạn-a-bộ-công-cụ-tấn-công)
12. [Review code L1 và các bản sửa](#12-review-code-l1-và-các-bản-sửa)
13. [Tổng hợp kiểm chứng và số liệu](#13-tổng-hợp-kiểm-chứng-và-số-liệu)
14. [Các lỗi đã gặp và cách sửa](#14-các-lỗi-đã-gặp-và-cách-sửa)
15. [Tất cả quyết định đã chốt](#15-tất-cả-quyết-định-đã-chốt)
16. [Trạng thái hiện tại và việc còn lại](#16-trạng-thái-hiện-tại-và-việc-còn-lại)
17. [Phụ lục: cấu trúc code, API, config, lệnh chạy](#17-phụ-lục-cấu-trúc-code-api-config-lệnh-chạy)

---

## 1. Bối cảnh và quy tắc làm việc

### 1.1 Dự án
- **Môn học:** COS30082 Applied Machine Learning. Đề bài: `docs/project_requirements.pdf` (Facial Recognition with Emotion and Liveness).
- **Hệ thống gốc:** điểm danh bằng khuôn mặt qua webcam.
  - Backend FastAPI, SQLite, WebSocket `/ws/attendance` và `/ws/register`.
  - Frontend Next.js.
  - Các module: liveness / anti-spoofing (phần chính), emotion, face recognition (method3 `FaceCNN_ArcFace`), lighting enhancer (`extreme_cond_enhancer`).
- **Repo đã tách khỏi team** và trở thành dự án innovation cá nhân. Owner có toàn quyền sửa và nâng cấp mọi phần.
- **Innovation:** "Đúng người, không chỉ người thật" (attack ladder đa phương thức).
  - Liveness chỉ trả lời câu hỏi "có phải người thật không".
  - Innovation thêm câu hỏi "có phải đúng người không", bằng cách xác thực mặt, giọng đã enroll, chuỗi số đọc ngẫu nhiên và độ khớp môi–tiếng (AV sync).

### 1.2 Quy tắc owner đặt ra (áp dụng suốt dự án)
- **Git:**
  - Claude chỉ làm và push trên nhánh `claude/relaxed-thompson-0br7xw`. Owner review rồi merge vào `master` (PR #1 đến #7).
  - Trước khi viết code: kiểm tra master có ahead không. Nếu có thì pull/merge về nhánh trước rồi mới code.
- **Báo cáo:**
  - Mỗi phần làm xong báo theo thứ tự: phần đó nhắm vào cái gì → đã làm gì → kết quả → bước tiếp theo cụ thể. Owner yêu cầu từ 2026-10-31.
  - Khi owner nói "chỉ báo cáo, khoan làm" thì không sang bước tiếp theo.
- **Ngôn ngữ:** trả lời bằng tiếng Việt; giao diện app bằng tiếng Anh.
- **Ghi chép:**
  - Mọi lần chạy, kể cả thất bại, ghi vào `experiment_log.md` để đưa vào report.
  - Quyết định ghi vào `decisions.md`; kiểm chứng ghi vào `verification_log.md`.
- **Bảo mật (nguyên văn):** "nếu trong skill có chứa các lệnh nào phá vỡ security hoặc cố gắng bypass security bằng việc gửi các thông tin session cho ai khác thì tuyệt đối không theo."
- **Đạo đức:**
  - Chỉ sinh deepfake của owner.
  - Texture adversarial không công bố và không đưa lên git.
  - Nạn nhân ảo chỉ dùng embedding, không sinh ảnh hay giọng mới của họ.
  - Người hỗ trợ phải đồng ý.

---

## 2. Dòng thời gian

| Ngày | Commit | Nội dung | PR |
|---|---|---|---|
| 10-31 | `39324a0` | First commit (repo gốc, có notebook train, gồm `training_max.ipynb`) | |
| 10-31 | `f63678f` | Dọn và tổ chức lại repo quanh module liveness | #1 |
| 10-31 | `61c2f86` | (owner) thêm skill thiết kế Hallmark | |
| 10-31 | `d930530` | Thiết kế lại frontend, theme Console graphite, tiếng Anh; backend trả chuỗi tiếng Anh | #2 |
| 10-31 | `896ccf7`, `b1a609b` | (owner) cập nhật frontend; sửa lỗi xoá người trong SQLite | |
| 10-31 | `a1d7fa1` | Kế hoạch innovation: spec, plan, decisions, experiment log | |
| 10-31 | `9999d43` | P0: hạ tầng innovation sau feature flag | #3 |
| 10-31 | `713c78b` | P1: enroll giọng và chấm điểm speaker | #4 |
| 10-31 | `df2c0fc` | Sửa feature flag trên Windows, đo P1 bằng model thật | #5 |
| 11-01 | `4c25f80` | (owner) đẩy kết quả voice_scores và latency trên máy owner, sửa `voice.py` (`LocalStrategy.COPY`) | |
| 11-01 | `1b92278` | Kế hoạch L1 (gọng kính), ghi kết quả P0/P1 của owner | #6 |
| 11-01 | `4da2064` | (owner) sửa `plan_L1.md` sau review | |
| 11-01 | `e20af33` | L1 giai đoạn A: bộ công cụ gọng kính adversarial, chụp frame trong Lab | #7 |
| 11-01 | `7a3a35e` | Sửa sau review: method5 là ViT, load strict, liveness đúng backend, v.v. | (chưa merge) |

---

## 3. Giai đoạn 1: dọn và tổ chức lại repo

**Nhắm vào:** repo bừa bộn. Owner muốn giữ phần anti-spoofing (phần chính, kèm test); emotion, lighting và face recognition chỉ cần chạy được.

### 3.1 Khảo sát (trước khi sửa)
Đã đọc toàn bộ code Python, 9 notebook, frontend, `results/`, git history và PDF đề bài.

**File đề xuất xoá, kèm lý do:**
| File | Lý do |
|---|---|
| `start.sh` | Owner yêu cầu bỏ `.sh`; lệnh chạy chuyển vào README |
| `sqlcommand.py` | Script lẻ, xoá cứng nhân viên `NV002` |
| `attendance.db-x-embeddings-6-embedding.bin` | Blob 2 KB export nhầm |
| `liveness/silent_face` | Gitlink submodule hỏng, rỗng |
| `embeddings/*.npz` (173 MB) | Cache của `compare_models.ipynb` |
| `extreme_cond_enhancer/tests/sample_enhanced.jpeg` (9 MB) | Không phải JPEG thật |
| `comparison_check.py` và thư mục ảnh | Tool so 2 ảnh chạy lẻ |
| `frontend/public/*.svg`, `frontend/README.md` | Mặc định của create-next-app |
| 3 lần chạy dở trong `results/liveness_offline/` | Chỉ có `frames.csv`, thiếu `summary.json` |
| `backend/requirements.txt` | Trùng và lệch với `requirements.txt` gốc |

**Lỗi tìm thấy:**
- Test `test_make_smoother_enabled_with_params` assert `t_on == 0.10`, trong khi config là `0.20`.
- README liveness trỏ tới file không tồn tại.
- Tool liveness nằm rải rác ở `utils/` và root.
- Thiếu `ultralytics` và `pytest` trong requirements.
- Font trong `draw_utils.py` chỉ có trên macOS.
- Chưa có README ở root.

### 3.2 Owner chốt
"Các file notebook xoá hết, xoá comparison, 3 lần chạy dở xoá luôn, clean sạch, nhưng repo vẫn phải hoạt động."

### 3.3 Đã làm (commit `f63678f`)
- Xoá mọi thứ ở bảng trên, cộng toàn bộ notebook train và `compare_models.ipynb`.
- Xoá script GUI chạy tay của enhancer (làm `pytest` dừng giữa chừng) cùng ảnh mẫu của nó.
- **Cấu trúc mới:**
  - Tool liveness chuyển vào `liveness/tools/` (chạy bằng `python -m liveness.tools.<tên>`): `prepare_fas_data`, `capture`, `live_eval`, `offline_eval`, `fft_probe`, `demo_photo`, `demo_webcam`, `draw_utils`, `samples/`.
  - Test chuyển vào `tests/liveness/` và `tests/extreme_cond_enhancer/`, thêm `pytest.ini`.
  - Kết quả gom về `results/{face_recognition, liveness/{live,offline}, emotion}`.
  - PDF đề bài chuyển vào `docs/project_requirements.pdf`.
  - Một `requirements.txt` (runtime, giữ đúng các phiên bản đã ghim), thêm `requirements-dev.txt` (pytest, ultralytics, huggingface_hub, pyarrow).
  - Viết lại `liveness/README.md`; thêm README ở root.
- Sửa test `t_on` thành `0.20`; sửa tham chiếu cũ trong docstring của emotion và README.

### 3.4 Kiểm tra
| Kiểm tra | Trước | Sau |
|---|---|---|
| `pytest` | 36 pass, 1 fail | **37/37 pass** |
| Smoke end-to-end (checkpoint giả): `/health`, tạo nhân viên, `/ws/register`, lưu, `/ws/attendance` | OK | OK |
| `demo_photo` trên 2 ảnh mẫu, `--help` của 7 tool, `offline_eval` chạy lại | | OK |
| `next build` | | OK |

**Lưu ý đã báo owner:**
- `.git` vẫn khoảng 139 MB vì `embeddings/` nằm trong commit đầu; muốn giảm phải viết lại history (Claude không làm).
- Repo không còn code train. Về sau `training_max.ipynb` được lấy lại từ commit `39324a0` để dựng class method5 (§12).

---

## 4. Giai đoạn 2: thiết kế lại giao diện

**Nhắm vào:** giao diện mới clean, dễ dùng, hợp với innovation về sau. Owner yêu cầu thêm: **giao diện tối, dịu mắt để dùng lâu, toàn bộ chữ tiếng Anh**.

### 4.1 Vòng đề xuất 1 (trước khi có skill)
- Nhận xét giao diện cũ:
  - Rối mắt: mỗi thẻ một màu viền, có emoji.
  - Lặp thông tin: trang Đăng ký báo "đã đăng ký" 3 lần.
  - Không có điều hướng.
  - Không có chỗ cho các kênh kiểm tra mà spec cần.
- Đề xuất 4 layout (A Workspace, B Kiosk + Console, C Wizard, D Instrument) và 3 style.
- Ý chung: một thành phần **danh sách các lớp kiểm tra** (`CheckLadder`) dùng cho cả Quick, Secure và Demo mode.

### 4.2 Vòng đề xuất 2, theo skill Hallmark
- Owner push skill Hallmark lên master; Claude pull về, đọc và rà bảo mật.
- **Không chạy** 4 thứ tác động ra ngoài repo:
  - `install-hook.mjs` (ghi hook vào settings);
  - `install.mjs` (copy skill vào `~/.claude`);
  - `imagegen.mjs` (gửi prompt tới api.together.xyz);
  - `theme.mjs photos --save` (tải ảnh từ Wikimedia).
- Chỉ dùng phần chạy cục bộ: `theme.mjs deal/scaffold` và linter `sloplint.mjs`.
- Linter trên giao diện cũ: **30 FAIL, 3 REFLEX, 5 WARN** (23 chỗ emoji làm icon, `transition-all`, màu Tailwind thô, thiếu `:active`/`:disabled`).
- Theo quy tắc "app screen" của skill (dưới khoảng 7 trang thì dùng top bar), bỏ sidebar.
- 3 hướng: **1 Console** (graphite hoặc amber-room), 2 Midnight (kiosk ban đêm), 3 Hedgerow (dịu mắt nhất).

### 4.3 Owner chốt
Hướng 1 Console, màu graphite, thêm trang People và History ngay, **đổi luôn chuỗi backend sang tiếng Anh**.

### 4.4 Đã làm (commit `d930530`)
- **Theme:**
  - Nền graphite 3 bậc, panel viền mảnh, không glow.
  - Font: Host Grotesk cho chữ; Atkinson Hyperlegible Mono cho số đo (score, ID, giờ); Be Vietnam Pro chỉ để hiển thị chữ có dấu tiếng Việt.
  - Màu: amber cho người, xanh lạnh cho số đo.
  - Token nằm trong `tokens.css`, `base.css`.
- **Top bar:** Overview, Check-in, Register, People, History, có skip link.
- **Check-in:**
  - Camera đặt cạnh `CheckLadder` (Face found, Liveness, Identity, Emotion). Mỗi dòng có trạng thái bằng chữ, score và thanh đo.
  - Khung mặt vẽ theo kích thước thật của camera.
- **Register:** một panel gồm thông tin, 5 góc (front, left, right, up, down) và một thông báo duy nhất. Logic tự chụp, chặn mặt trùng và lưu giữ nguyên.
- **People và History (mới):**
  - Dùng API có sẵn `/api/employees` và `/api/attendance/history`.
  - Tìm kiếm bỏ dấu ("nguyen" ra "Nguyễn"), filter lưu trên URL.
  - Đủ trạng thái loading/empty/error; hộp xác nhận khi xoá; trên điện thoại, bảng xếp thành khối có nhãn.
- **Khác:** icon lucide thay emoji; timestamp DB đọc theo UTC.
- **Backend tiếng Anh:** gate khoảng cách trả `Move closer` / `Move back`; lỗi đăng ký trùng mặt và hướng dẫn góc đầu (`head_pose.py`) đổi sang tiếng Anh. Test và README cập nhật theo.

### 4.5 Kiểm tra
- `pytest` 37/37.
- `tsc` sạch; `next build` thành công cả 6 route.
- Linter Hallmark (render 375 đến 1920 px, 5 route): **0 FAIL, 0 WARN**, 1 mục bỏ qua có lý do (poster video, vì đây là luồng camera trực tiếp).
- Chạy thật với webcam giả: nhận ra người, ghi điểm danh, chặn đăng ký trùng mặt, mở hộp xoá.

### 4.6 Vấn đề báo owner
1. **Không xoá được người đã từng điểm danh:** bảng `attendance` thiếu `ON DELETE CASCADE`, nên API trả 500. Owner tự sửa ở `b1a609b`.
2. ESLint còn 1 lỗi có sẵn (setState trong effect ở Register).
3. "Hôm nay" có thể lệch quanh nửa đêm (UTC so với giờ local).
4. Trên điện thoại, nav phải cuộn ngang; `capture.py` vẫn in tiếng Việt ra console.

---

## 5. Giai đoạn 3: spec innovation và kế hoạch

### 5.1 Tóm tắt spec (`spec_v1.md`)
- **Luận điểm:** anti-spoofing ≠ đúng người. Dựng chuỗi tấn công leo thang; mỗi mức gồm tấn công → đo → phân tích vì sao fail → thêm defense đúng chỗ.
- **Threat model:**
  - Kẻ tấn công: sinh viên trong lớp, có ảnh/giọng của nạn nhân, GPU thuê, máy in, điện thoại.
  - Ngoài phạm vi: chiếm máy, sửa client, tấn công DB/mạng.
- **Attack ladder:**

| Mức | Tấn công | Defense |
|---|---|---|
| L0 | Zero-effort; ảnh/video trên điện thoại | Baseline |
| L1 | Gọng kính adversarial in ra | Giọng đã enroll + fusion |
| L2 | Phát lại câu cố định | Chuỗi số ngẫu nhiên, dùng một lần, có hạn, kèm ASR |
| L3 | Clone giọng + Wav2Lip | Phát hiện lệch AV + giới hạn thời gian; **AV detector tự train** |
| L4 | Kẻ tấn công thích nghi | Chỉ phân tích |

- **Thành phần:**
  - ECAPA-TDNN (192 chiều), so 1:1 với ứng viên mà mặt đưa ra.
  - Challenge 6 số (`secrets`, hạn 8 s); ASR faster-whisper, đạt nếu edit distance ≤ 1.
  - SyncNet + detector tự train.
  - Fusion R1 (AND), R2 (logistic regression), R3 (R2 + bắt buộc ASR), kèm `reason_codes`.
- **Thí nghiệm E0–E5** và ngưỡng đặt trước (xem §15.3). Metric: APCER, BPCER, IAPMR (ISO/IEC 30107-3), FAR/FRR, latency.
- **Phase P0–P6.** Acceptance quan trọng nhất: **Secure mode không được tự ghi điểm danh**.
- **Hướng đã loại:** rPPG, YOLO bắt điện thoại, blink, screen-flash, motion challenge, depth/CDCN, sonar, face–voice không enroll, multi-face.
- **Chỗ spec lệch với repo:** quy tắc "không commit"; tên `test_liveness_*.py` đã đổi; spec đòi UI tiếng Việt nhưng owner chọn tiếng Anh; `plans/*.md` chưa có.

### 5.2 Owner trả lời Q1–Q7
| # | Trả lời |
|---|---|
| Q1 | Repo tách riêng, toàn quyền quyết định |
| Q2 | Đọc số bằng tiếng Anh |
| Q3 | Chỉ owner, cộng người ngoài khoảng 1 buổi/tuần, cộng dữ liệu public |
| Q4 | GPU vast.ai (5070 Ti, 3090, 4090, 5090), vài giờ; hoặc Kaggle qua đêm |
| Q5 | Không cần báo giảng viên (dự án cá nhân) |
| Q6 | Vài tuần, càng nhanh càng tốt, owner luôn sẵn sàng |
| Q7 | Chốt ngưỡng; trượt thì có phương án thay thế; mọi thất bại ghi log để đưa vào report |

### 5.3 Kế hoạch (`plan.md`, commit `a1d7fa1`)
- **Vai trò khi chỉ có một người:**
  - Owner là genuine, là nạn nhân L2/L3 (replay và deepfake của chính mình), và là kẻ tấn công L1.
  - Nạn nhân L1 là danh tính ảo từ dataset public.
  - Lớp mô phỏng gồm 50–100 danh tính public.
  - Người ngoài làm impostor thật và genuine thứ hai.
- **Chia tập:** train/dev trên danh tính public; test trên owner (sau khi commit prereg) và người ngoài; AV detector kiểm tra cross-dataset.
- **Lịch 3–4 tuần**, có bảng dự phòng cho từng thí nghiệm (§15.3).
- **Phần tự train:** AV detector, fusion R2, có thể thêm bộ nhận dạng chữ số. Phần tự build: tối ưu texture kính.
- **Rủi ro:** dataset deepfake cần đăng ký; chỉ một người tham gia; decode webm trên Windows; xung đột dependency; checkpoint không nằm trong repo.

---

## 6. P0: hạ tầng innovation

**Nhắm vào:** dựng nền cho innovation sau feature flag, để app cũ không đổi. Gồm DB mới và khả năng ghi clip có hình lẫn tiếng trên trình duyệt rồi gửi backend tách thành frame và audio.

### 6.1 Đã làm (commit `9999d43`)
- **Backend** (`INNOVATION_ENABLED=1`), package `innovation/`:
  - `config.json`: ngưỡng tạm (§17.3).
  - `db.py`: 3 bảng mới.
    - `voice_embeddings`: xoá người thì xoá theo (cascade).
    - `challenges`.
    - `verification_attempts`: giữ lại làm dữ liệu nghiên cứu.
    - Thêm cột `attendance.method` (mặc định `quick`).
  - `media.py`: decode webm/opus bằng PyAV, ra frame có timestamp, fps đo thật và audio 16 kHz mono.
  - Router: `/api/innovation/config`, `/api/attempts`, `/api/innovation/media/probe`.
- **Frontend** (`NEXT_PUBLIC_INNOVATION=1`):
  - Hook `useAVRecorder` (`lib/media.ts`): ghi từ chính stream camera đang mở, chỉ thêm track micro; bật/tắt được echo cancellation, noise suppression, gain; có thanh âm lượng và hiển thị settings trình duyệt báo.
  - `/lab` (ghi 4 s, xem backend decode ra gì) và `/security` (danh sách attempts). Tắt flag thì hai trang trả 404 và không có trong nav.
- **Kiểm chứng** (`verification_log.md`):
  - speechbrain **1.0.3** (bản 1.1 đòi torch ≥ 2.7.1); thêm `requests` vì 1.0.3 import nhưng không khai báo.
  - faster-whisper 1.2.1, av 18.1.0, mediapipe 0.10.14 chạy chung với numpy 1.26.4. **Không cần nâng torch.**
  - License: SyncNet MIT; Wav2Lip chỉ cho nghiên cứu; **OpenVoice V2 MIT (chọn cho clone giọng L3)**; XTTS phi thương mại (dự phòng).
  - Dataset: AudioMNIST MIT (tải được); Speech Commands CC BY 4.0 (bị chặn từ cloud); **LAV-DF CC BY-NC (dataset deepfake chính, không cần form)**; AV-Deepfake1M và FakeAVCeleb cần EULA/form (tuỳ chọn); VoxCeleb1 bị chặn từ cloud.

### 6.2 Nghiệm thu (Chromium headless, webcam/micro giả)
- Tắt flag: 37 test cũ pass; `/lab`, `/security`, `/api/innovation/*` trả 404.
- Bật flag: 43 test pass (6 test P0 mới).
- Clip 4 s: **120 frame, 30.0 fps** (đo từ timestamp), audio **16 kHz mono 3.96 s**, −21 dBFS, decode 432 ms, file 302 KB.
- Trọng số HF bị proxy chặn (403), nên chưa đo latency ECAPA/Whisper ở bước này.

---

## 7. P1: giọng nói

**Nhắm vào:** "chữ ký giọng" cho mỗi người, để hệ thống hỏi được "mặt nói đây là B, vậy giọng có phải của B không". Đây là lớp chặn L1.

### 7.1 Đã làm (commit `713c78b`)
- **`innovation/voice.py`:**
  - VAD dựa trên năng lượng.
  - Kiểm tra chất lượng: `TOO_SHORT`, `TOO_QUIET`, `CLIPPING`, `NOISY`.
  - ECAPA tải khi dùng lần đầu; so 1:1 với centroid các mẫu đã enroll.
- **API enroll:**
  - `/api/voice/enroll/start` cấp 3 chuỗi số dùng một lần (lưu thành challenge loại `enroll`).
  - `/submit` chỉ lưu embedding khi bản ghi đạt; bản ghi kém thì giữ prompt mở để ghi lại.
  - `status`, `reset`.
  - Audio thô chỉ giữ khi research mode bật và người dùng chọn giữ.
- **UI `VoiceEnrollment`:** phần đồng ý, 3 thẻ chữ số lớn, đếm ngược 3-2-1, thanh âm lượng, nghe lại, hướng dẫn rõ cho từng lỗi. Hiện sau Register và ở `/people/<id>/voice`; trang People có thêm cột Voice.
- **Eval:**
  - `innovation/eval/runs.py`: thư mục run có `meta.json` (git commit, config, seed, máy).
  - `metrics.py`: EER, AUC, FAR, FRR.
  - `latency.py`, `voice_scores.py`.
- **Tool:** `innovation/tools/enroll_public.py` đưa danh tính public (mặt + giọng) vào DB.

### 7.2 Kiểm tra (encoder giả, vì HF vẫn bị chặn)
- 57 test pass (gồm 12 test voice, 2 test metric).
- Luồng People → Enrol → đồng ý → 3 lần ghi → "Voice enrolled (3 of 3)" chạy trọn; `/api/voice/status` trả `{"SV002": 3}`.
- **Lỗi đã sửa:** clip nhiều nhiễu bị báo `TOO_SHORT` + `TOO_QUIET` (hướng dẫn sai cho người dùng). Giờ: ít đoạn có tiếng nói nhưng cả clip đủ to thì báo `NOISY`.

---

## 8. Sửa feature flag và đo P1 bằng model thật

**Vấn đề owner gặp:** sau khi pull về, không thấy tab Lab và Security, gõ link trực tiếp cũng không vào được.

### 8.1 Nguyên nhân và cách sửa (commit `df2c0fc`)
1. Frontend chỉ nhận đúng `NEXT_PUBLIC_INNOVATION=1`, trong khi owner đặt `true`. Giờ nhận **1/true/yes/on** (`lib/flags.ts`).
2. Lệnh `INNOVATION_ENABLED=1 python ...` là cú pháp bash, không chạy trên PowerShell hay cmd. Giờ backend đọc file `.env` ở root repo (`backend/env.py`); thêm `.env.example` và `frontend/.env.example`.
3. Hai cảnh báo trong log owner gửi không phải lỗi:
   - hydration mismatch do extension trình duyệt chèn thuộc tính `bis_skin_checked`;
   - Next không tạo được font dự phòng cho Atkinson.

### 8.2 Model thật (sau khi HF mở được)
- Ghim **`huggingface_hub<0.26`**, vì speechbrain 1.0.3 truyền `use_auth_token`, tham số mà bản hub mới đã bỏ.
- `innovation/tools/audiomnist.py` ghép chuỗi 6 số từ AudioMNIST (người thật đọc chữ số tiếng Anh); `voice_crossval.py` cho mỗi người lần lượt làm owner.
- **Kết quả** (`results/innovation/20261031_125000_voice_crossval/`, 12 người, enroll 3 chuỗi, 9 chuỗi genuine):

| | Giá trị |
|---|---|
| EER gộp / AUC | **0.16% / 0.9999** |
| EER từng người | trung bình 0.09%, cao nhất 0.76% |
| Genuine | trung bình 0.67, p05 0.49 |
| Impostor | trung bình 0.16, p99 0.40 |
| Ở ngưỡng tạm 0.25 | **FAR 21%**, nên đổi tạm lên **0.40** |
| Embed | 180 ms (trung vị) |

- **Latency** (`20261031_125032_latency`, clip giọng thật 4.9 s, 4 vCPU): ECAPA 114 ms; Whisper tiny.en 386 ms ("0, 2, 6, 2, 8, 1."); base.en 544 ms ("026281"). Tải model 0.5–1.7 s.
- **Phát hiện:** trên clip tổng hợp không phải tiếng người, base.en mất 16.5 s và bịa ra "Oh, oh, oh…". Vì vậy P2 phải lọc theo VAD trước ASR và giới hạn thời gian.
- **Hạn chế:** AudioMNIST thu một buổi, cùng micro, nên kết quả lạc quan.

---

## 9. Kết quả trên máy owner (P0, P1)

### 9.1 P0: webcam thật (Chrome, Windows, 640×480, clip 4 s)
| Lần | Điều kiện | fps thực | Audio | Decode |
|---|---|---|---|---|
| 1 | Đủ sáng, bật 3 tuỳ chọn âm thanh | **29.9** | 16 kHz, −19.6 dBFS | 911 ms |
| 2 | Thiếu sáng, bật 3 tuỳ chọn | **15.0** (trình duyệt vẫn báo 30) | −18.7 dBFS | 489 ms |
| 3 | Đủ sáng, tắt 3 tuỳ chọn | **29.9** | −20.3 dBFS | 889 ms |

- Thiếu sáng thì driver kéo dài thời gian phơi sáng, fps giảm một nửa. Đây là rủi ro cho P3, vì SyncNet cần khoảng 25 fps.
- Hướng xử lý cho P3: cảnh báo khi fps < 20, ép `frameRate.min`, hoặc nội suy theo timestamp.
- Bật/tắt xử lý âm thanh gần như không đổi âm lượng. Decode chiếm 0.5–0.9 s trong ngân sách 3 s của E4.
- **Kết luận:** đạt khi đủ sáng; thiếu sáng là hạn chế đã đo được.

### 9.2 P1: giọng thật của owner
20 clip `.m4a` của owner (clip 2–6 cố ý đọc giọng cao) so với 240 clip của 12 người AudioMNIST. Enroll 3 mẫu qua trình duyệt.

| | Giá trị |
|---|---|
| Run | `results/innovation/20261101_140348_voice_scores/` |
| EER / AUC | **0.63% / 0.999** |
| Genuine | trung bình 0.39, thấp nhất 0.20 (`huy-2`), cao nhất 0.60 |
| Impostor | trung vị 0.04, cao nhất 0.23 |
| Ngưỡng 0.40 | FAR 0%, **FRR 65%** (13/20 bị từ chối) |
| Ngưỡng 0.25–0.30 | FAR 0%, FRR 5% |
| Latency (`20261101_140519_latency`) | ECAPA 153 ms; tiny.en 314 ms ("7, 6, 0, 3, 8, 1."); base.en 612 ms ("760-381") |

**Phân tích:**
- Ngưỡng 0.40 (đặt từ AudioMNIST) sai với giọng thật.
- Điểm genuine thấp chủ yếu do **lệch kênh thu**: enroll webm/opus qua trình duyệt, test bằng `.m4a`.
- **Giả thuyết giọng cao bị bác bỏ:** 5 clip giọng cao trung bình 0.39, bằng 15 clip giọng thường (0.391). Clip 6 còn cao điểm nhất (0.60). ECAPA khá bền khi đổi cao độ.
- FAR 0% vẫn lạc quan, vì impostor AudioMNIST khác hẳn kênh thu.
- base.en viết "760-381", nên P2 phải chuẩn hoá chỉ lấy chữ số.
- Hai cảnh báo vô hại: torchaudio backend (Claude decode bằng PyAV) và symlink HF trên Windows. Owner đã thêm `LocalStrategy.COPY` vào `voice.py`.

**Hướng xử lý (`decisions.md`):** enroll 5 mẫu chia 2 buổi; enroll và verify cùng kênh trình duyệt; chốt ngưỡng trong `prereg.md` sau khi có impostor cùng kênh (người hỗ trợ). Giá trị trong `config.json` tạm giữ 0.40.

---

## 10. L1: thảo luận và kế hoạch

### 10.1 Thảo luận (owner: "chỉ bàn luận trước, khoan làm")
- Ý ban đầu của owner: tối ưu "mã màu" rồi in ra dán lên mặt người hỗ trợ.
- Claude phân tích:
  - Pipeline backend thực tế không có bước căn landmark.
  - Thành công phải là top-1 trong 1:N.
  - Loss có TV và NPS; EOT có cả uốn cong theo da (kiểu AdvHat).
  - Cần bảng màu in và chụp lại để fit ánh xạ màu.
  - Đo 3 tầng D, D+EOT, P, có control; đo thêm liveness.
- Đưa ra 4 câu hỏi: dạng miếng dán (lên mặt hay gọng kính), kích thước tối đa, có làm dodging không, lịch người hỗ trợ.

### 10.2 Owner chốt
1. Dùng **gọng kính**. Trước hết thử với nạn nhân ảo (VoxCeleb) để chắc method đúng, rồi mới làm với người hỗ trợ.
2. Kích thước **15–30%** (lớn hơn cũng được).
3. Đồng ý làm dodging.
4. Chốt lịch người hỗ trợ: buổi 1 chụp frame và ghi giọng cùng kênh, buổi 2 thử vật lý.

### 10.3 `plan_L1.md` (commit `1b92278`; owner sửa ở `4da2064`)
- **Giai đoạn A (digital):**
  - Owner là kẻ tấn công; nạn nhân là danh tính ảo.
  - **Owner có trong DB** (1:N thật, đối thủ mạnh nhất trong margin chính là kẻ tấn công).
  - Đo D, D+EOT, transfer, control digital; dodging.
  - **Cổng sang B:** D ≥ 80% và control ≤ 20%.
- **Giai đoạn B (vật lý):** người hỗ trợ tấn công owner; đo P, control vật lý, liveness, giọng (E1c).
- **§3 Pipeline phải giống hệt backend:**
  - Webcam → JPEG 0.95 → MediaPipe `model_selection=0` → liveness đã smooth và distance gate → crop nới 15% → lighting enhancer → 112×112, [−1, 1] → method3 → `find_match` (max theo góc, ngưỡng 0.5).
  - Import thẳng code backend, không viết lại.
  - Phát hiện lại bbox trên ảnh đã đeo kính.
  - Thành công = liveness real và top-1 là nạn nhân, điểm ≥ 0.5 (IAPMR).
  - Đếm thêm "có ít nhất một frame thành công trong cửa sổ 3 s" (điều kiện điểm danh hộ được ở Quick mode).
  - Enroll public bằng detector `model_selection=0` (thêm cờ `--detector-model`).
  - Lighting enhancer: tối ưu trên ảnh ánh sáng bình thường; loại frame bị enhancer tác động.
- **§4 Gọng kính:** template khoảng 400×150, đặt theo FaceMesh, xuất PNG đúng kích thước 300 dpi. Owner đang đeo kính thật.
- **§5 Tối ưu:**
  - Chuỗi biến đổi: `T` (màu in) → `M(T)` → warp → dán → EOT → embedding.
  - Loss impersonation: `1 − S_B` + margin 1:N + TV + NPS. Dodging: giảm `S_A`.
  - EOT: hình học ±3 px, ±5°, ±5%; màu ±20%; blur; JPEG 0.6–0.95; khoảng cách 80–200 px.
  - Bảng màu 10×10 → NPS và `M`. `T` sau tối ưu chính là file in.
  - Tách 70/30, khoảng 120–150 frame, khoảng tin cậy Wilson.
- **§6 Đo:** baseline, D, control digital, D+EOT, transfer, P, control vật lý, liveness, giọng.
- **§9 Đạo đức** như §1.2.

---

## 11. L1 giai đoạn A: bộ công cụ tấn công

**Nhắm vào:** kiểm chứng ở dạng digital rằng phương pháp, pipeline và loss đều đúng, trước khi tốn công in.

### 11.1 Code (commit `e20af33`, thư mục `innovation/attacks/adv_glasses/`)
| File | Việc |
|---|---|
| `faces.py` | Phát hiện mặt bằng chính `LivenessDetector.detect_face` (gọi unbound qua `SimpleNamespace(_mp_fd=...)`, không load model liveness); `crop_window` / `crop` (= `crop_face_from_bbox`, pad 0.15); `enhance` (= `_enhance_inference_crop`); `EyeLandmarks` (FaceMesh, khoé mắt 33/133 và 362/263); `Gallery` và `build_gallery` (embedding qua đường backend, cache theo checkpoint) |
| `glasses.py` | Template 400×150, tâm mắt (110,80) và (290,80); `FrameShape(thickness, rim_scale)` (vành, cầu mũi, càng kính); `similarity` (đặt theo hai tâm mắt), `jitter`, `render` (dán lấy đạo hàm được), `coverage` |
| `pipeline.py` | `TorchPipeline`: resize bilinear (`align_corners=False`, `antialias=False`, giống `cv2.INTER_LINEAR`), BGR→RGB, chuẩn hoá, embedding; `parity`; CLI kiểm tra parity |
| `eot.py` | `EOTConfig` TRAIN và UNSEEN (rộng hơn); `jpeg_bpda` (JPEG thật ở forward, gradient identity ở backward); `blur`; `photometric` |
| `printability.py` | Bảng màu với 4 marker ArUco (`DICT_4X4_50`); `read_chart`; `ColorMap` (đa thức bậc 2, fit, save/load); `nps`; CLI `chart` / `fit` |
| `optimize.py` | Tách frame (session cuối để đánh giá); bỏ frame bị enhancer tác động; loss impersonation và dodging; Adam trên logits 3×75×200 qua sigmoid; xuất `texture_print.png`, `mask.png`, `texture.pt`, `colormap.json`, `split.json`, `attack.json`, `loss.csv` |
| `evaluate.py` | Các tầng baseline, D, control_random, control_plain, D_EOT và transfer; JPEG q95; liveness; `find_match`; Wilson; cửa sổ 3 s; cổng sang B |
| `export.py` | Tờ in 300 dpi, tỉ lệ theo IPD (180 px template = IPD), thước 50 mm |

**Thêm ngoài thư mục trên:**
- `FaceRecognizer` nhận thêm `ckpt_path`.
- `enroll_public.py` có thêm `--detector-model` (mặc định giữ 1 như cũ).
- **Lab → Frame capture** (`FrameCapture.tsx`, API `/api/innovation/frames`, chỉ ở research mode): chụp frame đúng như check-in gửi (640×480, JPEG 0.95, mỗi 200 ms), lưu `data/innovation/frames/<person>/<session>/NNNN.jpg` kèm `meta.csv`.
- `.gitignore` chặn texture, ảnh in và `_cache`.
- Viết `runbook_L1.md`.

### 11.2 Kiểm tra (không có checkpoint thật)
- Parity pipeline torch so với backend (trọng số ngẫu nhiên, 20 crop thật và crop ngẫu nhiên): cosine **1.00000**.
- Crop khớp `crop_face_from_bbox` trên 200 bbox ngẫu nhiên.
- JPEG BPDA cho gradient bằng 1.
- Bảng màu chạy vòng tròn (giả lập chụp nghiêng, lệch màu, gamma, nhiễu): sai số fit khoảng 1/255.
- Gọng đặt đúng trên ảnh webcam thật của owner.
- Test: 70 pass (venv innovation); 51 pass, 3 skip (venv cơ bản). Test frame capture skip khi thiếu `av`.
- Chạy thử optimize → evaluate → export bằng trọng số ngẫu nhiên: chạy hết luồng nhưng **không có ý nghĩa về hiệu quả** (cosine mọi ảnh khoảng 0.98, gradient gần 0, texture hội tụ về xám). Kết quả chạy thử đã xoá.
- **Lỗi phát hiện và sửa khi đó:** logsumexp chưa trừ τ·log(n), nên max mềm vượt quá 1 (về sau đổi hẳn công thức, xem §12).

### 11.3 Phát hiện: độ phủ 15–30% không làm được bằng gọng kính
Đo trên ảnh owner (crop backend nới 15%):

| Gọng | Tham số | Độ phủ crop 112 |
|---|---|---|
| Thường | `--thickness 14 --rim-scale 1.0` | **3.7%** |
| Dày (mặc định) | `--thickness 35 --rim-scale 1.1` | **8.2%** |
| Ngoại cỡ | `--thickness 55 --rim-scale 1.2` | **11.6%** |

Muốn 15–30% thì phải che má và trán, tức thành mặt nạ. **Quyết định:** độ phủ là tham số, ablation 4/8/12%, mặc định khoảng 8%; mỗi lần chạy ghi độ phủ thật trong `attack.json`. Muốn ≥ 15% thì phải đổi phụ kiện (ví dụ kính kèm khẩu trang in), và đó là một thí nghiệm khác.

---

## 12. Review code L1 và các bản sửa

### 12.1 Nội dung review (owner gửi)
- **Nghiêm trọng:**
  - method1/2/3/4 là `FaceCNN` (method1 có classifier, method3 có ArcFace head, method2/4 chỉ có backbone).
  - **method5 là ViT** (`backbone.cls_token`, `pos_embed`, `patch_embed`; 157 key; 353 MB).
  - `FaceRecognizer` load bằng `strict=False`, nên method5 âm thầm chạy bằng trọng số ngẫu nhiên.
  - Cần (a) load method5 đúng class và đúng preprocessing; (b) ở research mode kiểm tra `missing_keys` và `unexpected_keys`, sai thì dừng.
  - method5 chính là model "khác kiến trúc" cho kết luận transfer.
- **Cần sửa:**
  1. `evaluate.py` bỏ qua distance gate; phải gọi `backend.processing._liveness_decision`.
  2. Cổng sang B được tính cả khi không có liveness; phải `null` nếu `liveness_gated` không bật.
  3. Tập đánh giá `s2_dim` lẫn hiệu ứng của enhancer; phải dùng ngày khác cùng ánh sáng, còn frame tối thành tầng riêng.
  4. `soft_max_cos` trừ τ·log(n) nên lệch theo số ảnh (10 ảnh: 0.115, 5 ảnh: 0.080 ở τ = 0.05), trong khi `find_match` dùng max thật.
  5. Gallery tới 10 ảnh/người khác DB (5 góc); thống nhất 5 ảnh/người.
- **Nhỏ:**
  - Kiểm tra trùng người theo tên không làm được (ID ẩn danh), nên dùng thẳng `verification_data`.
  - Bỏ biến `CKPT_PATH` cục bộ trong `face_recognizer.py`.
  - `UNSEEN.distance_px` khai báo nhưng không dùng.

### 12.2 Đã làm (commit `7a3a35e`)
- **method5:**
  - Lấy lại `training_max.ipynb` từ commit `39324a0`. `FaceViT_ArcFace` = `timm.create_model('vit_base_patch16_224', num_classes=0)` → `Linear(768, 512, bias=False)` → `BatchNorm1d` → L2; ArcFace `weight` (s = 32, m = 0.35); train với lr 5e-5.
  - Preprocessing ViT: resize 224, ImageNet mean (0.485, 0.456, 0.406) và std (0.229, 0.224, 0.225). CNN: 112, mean/std 0.5 ([−1, 1]).
  - `backend/face_recognizer.py`:
    - `Preprocess(size, mean, std)`, `CNN_PREPROCESS`, `VIT_PREPROCESS`.
    - Class mới `FaceCNN_Triplet`, `FaceViT_ArcFace` (import timm lazily; dựng với `pretrained=False`).
    - `_unwrap_state` (xử lý `state_dict`/`model` lồng và tiền tố `module.`).
    - `build_model_for_state` nhận dạng kiến trúc theo key; `num_classes` lấy từ shape thay vì cố định 4000.
    - `FaceRecognizer(strict=None)`: mặc định strict khi `INNOVATION_RESEARCH_MODE` bật. Strict thì thiếu file hoặc lệch key là dừng (`FileNotFoundError` / `RuntimeError`); không strict thì in cảnh báo kèm danh sách key.
    - Thêm `load_report`, `arch`; dùng `self.ckpt_path`, bỏ biến cục bộ.
  - `pipeline.py` dùng preprocessing của recogniser đang dùng; mọi script L1 load với `strict=True`; `attack.json` và `summary.json` ghi `load_report` của từng model.
  - Ghim `timm==1.0.11` trong `innovation/requirements.txt` (chạy với torch 2.2.2 / torchvision 0.17.2).
- **Điểm 1:** `evaluate.liveness_step` gọi `_liveness_decision` (distance gate rồi smoother). Smoothing tắt thì đi nhánh `live.check`, giống `process_attendance_frame`. Mỗi session một smoother.
- **Điểm 2:** hàm `stage_b_gate` trả `pass = null` khi không có liveness; in "(no verdict: rerun with --liveness)".
- **Điểm 3:**
  - Frame bị enhancer tác động tách sang tầng `…@dim`, không gộp vào tỉ lệ chính hay cổng.
  - Runbook mới: `s1_day1`, `s2_day2` (ngày khác, cùng ánh sáng; tập đánh giá), `s3_dim` tuỳ chọn.
  - Có `s3_dim` thì optimize **bắt buộc** thêm `--eval-sessions s2_day2 s3_dim`.
- **Điểm 4:** `soft_max_cos` = `Σ softmax(cos/τ)·cos`: luôn ≤ max thật, không có hạng log(n).
- **Điểm 5:** `GALLERY_PER_ID = 5`, đúng 5 góc như Register. Người thiếu ảnh bị cảnh báo và ghi `gallery_report.uneven`; `attack.json` ghi `gallery_per_id`.
- **Nhỏ:**
  - Runbook: nạn nhân và lớp mô phỏng lấy từ `verification_data`; không đủ 5 ảnh/người thì ghép theo cặp "same" trong `verification_pairs_val.txt`; VoxCeleb chỉ là dự phòng.
  - `unseen_frame` dùng `UNSEEN.distance_px` (cạnh crop 60–120 px), chỉ thu nhỏ, không phóng to.
- **Tài liệu:**
  - `plan_L1.md` sửa công thức S, tập đánh giá, và câu sai "cả năm dùng chung backbone FaceCNN".
  - Cập nhật `runbook_L1.md`, `decisions.md`, `experiment_log.md`, `verification_log.md`.

### 12.3 Kiểm tra
- `FaceViT_ArcFace` dựng lại có đúng **157 key** (`weight`, `backbone.cls_token`, `backbone.pos_embed`, `backbone.patch_embed.proj.weight`, …), `num_features` = 768. Số key của các bố cục CNN: ArcFace 74, Triplet 73, Classifier 75.
- Load strict pass cho cả 4 bố cục; key thiếu hoặc thừa thì dừng; research mode bật strict.
- Parity ViT (224, ImageNet) trên crop thật của owner: cosine nhỏ nhất **0.99999**.
- Max mềm mới: thêm 5 ảnh xấu, điểm đổi < 0.001 (công thức cũ đổi > 0.03).
- `liveness_step` trả "Move closer" khi mặt rộng 40 px trong frame 640 px.
- Smoke run: optimize 4 bước, rồi evaluate có `--transfer method5=<ViT ngẫu nhiên>`. Chạy hết luồng, tách đúng tầng thường và `@dim`, `gate pass = None`. `--liveness` dừng đúng với lỗi thiếu `liveness_best.pth` (môi trường cloud không có checkpoint thật). Kết quả smoke đã xoá.
- Test: **80 pass** (venv có timm, av); **60 pass, 4 skip** (venv cơ bản).
- **Hệ quả:** mọi kết quả transfer sang method5 trước bản sửa này, nếu có, đều không dùng được.

---

## 13. Tổng hợp kiểm chứng và số liệu

### 13.1 Số test theo từng mốc
| Mốc | Test |
|---|---|
| Trước khi dọn | 36 pass, 1 fail |
| Sau khi dọn / UI | 37/37 |
| P0 | 37 (tắt flag); 43 (bật flag) |
| P1 | 57 |
| L1-A tooling | 70 (innovation); 51 + 3 skip (cơ bản) |
| Review fixes | **80** (innovation); **60 + 4 skip** (cơ bản) |

### 13.2 Bảng số liệu chính
| Hạng mục | Kết quả |
|---|---|
| Decode clip 4 s, webcam giả | 30.0 fps, 16 kHz, 432 ms |
| Decode clip 4 s, webcam owner | 29.9 fps (đủ sáng), 15.0 fps (thiếu sáng), 0.5–0.9 s |
| ECAPA AudioMNIST (12 người) | EER 0.16%, AUC 0.9999 |
| ECAPA owner (20 clip) | EER 0.63%; ngưỡng 0.40 → FRR 65%; ngưỡng 0.25–0.30 → FRR 5%, FAR 0% |
| Latency, cloud 4 vCPU | ECAPA 114 ms, tiny.en 386 ms, base.en 544 ms |
| Latency, máy owner | ECAPA 153 ms, tiny.en 314 ms, base.en 612 ms |
| Parity pipeline L1 (CNN / ViT) | 1.00000 / 0.99999 |
| Fit bảng màu (giả lập) | khoảng 1/255 |
| Độ phủ gọng | 3.7% / 8.2% / 11.6% |

### 13.3 Thư mục kết quả trong repo
- `results/innovation/20261031_124613_latency/`, `20261031_124632_latency/` (clip tổng hợp; base.en bịa chữ), `20261031_125032_latency/`
- `results/innovation/20261031_125000_voice_crossval/`
- `results/innovation/20261101_140348_voice_scores/`, `20261101_140519_latency/` (máy owner)

---

## 14. Các lỗi đã gặp và cách sửa

| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| Test smoother fail | Assert `t_on == 0.10` trong khi config 0.20 | Sửa assert (`f63678f`) |
| `pytest` dừng giữa chừng | Script GUI enhancer bị collect | Xoá script |
| Xoá người trả 500 | `attendance` thiếu cascade | Owner sửa (`b1a609b`) |
| Clip nhiễu báo "quá ngắn, quá nhỏ" | VAD không thấy tiếng nói trong nhiễu | Báo `NOISY` khi clip đủ to |
| Không thấy Lab/Security | Flag chỉ nhận "1"; cú pháp env kiểu bash | Nhận 1/true/yes/on; backend đọc `.env` |
| speechbrain lỗi `use_auth_token` | huggingface_hub mới bỏ tham số | Ghim `huggingface_hub<0.26` |
| Whisper bịa chữ, 16.5 s | Âm thanh không phải tiếng người | Ghi chú cho P2: VAD trước ASR, giới hạn thời gian |
| Ngưỡng giọng 0.40 sai | Đặt từ AudioMNIST; lệch kênh thu | Chốt lại trong prereg (§9.2) |
| SecurityView lặp vô hạn | Fetcher khai báo inline | Đưa fetcher ra cấp module |
| Trang server import module hook | Ranh giới client/server | Tách `lib/flags.ts` |
| `pkill` tự giết shell (môi trường thử) | Pattern trùng chính lệnh | Dùng `fuser -k` |
| Chromium lỗi WebSocket qua proxy (môi trường thử) | Proxy cloud | `--no-proxy-server`, dùng production build |
| Test frame capture fail ở venv cơ bản | Thiếu `av` | `importorskip("av")` |
| `crop_window` trả sai | Biểu thức return lỗi | Sửa trước khi commit |
| Max mềm > 1 | logsumexp chưa chuẩn hoá | Trừ τ·log(n), sau đó thay bằng softmax-weighted (§12) |
| method5 chạy trọng số ngẫu nhiên | Load vào `FaceCNN` với `strict=False` | Class ViT riêng và load strict (§12) |
| Liveness trong evaluate bỏ qua gate | Tự gọi `live.check` + smoother | Gọi `_liveness_decision` |
| Cổng B "pass" khi không có liveness | `is_real` mặc định 1 | `pass = null` |

---

## 15. Tất cả quyết định đã chốt

### 15.1 Chung
- Repo riêng, owner toàn quyền; Claude commit lên nhánh riêng, owner merge.
- Giao diện tiếng Anh, theme Console graphite; backend trả chuỗi tiếng Anh.
- Challenge đọc chữ số tiếng Anh; ASR model `.en`.
- Chỉ owner, cộng người ngoài khoảng 1 buổi/tuần, cộng dữ liệu public.
- Train trên dữ liệu public, test trên owner và người ngoài.
- Feature flag: frontend `NEXT_PUBLIC_INNOVATION` (1/true/yes/on); backend `INNOVATION_ENABLED`, `INNOVATION_RESEARCH_MODE`, đọc từ `.env` ở root.
- Không nâng torch/numpy (torch 2.2.2, numpy 1.26.4); speechbrain 1.0.3; `huggingface_hub<0.26`; `timm==1.0.11`.
- Clone giọng L3: OpenVoice V2 (MIT); dự phòng XTTS. Dataset deepfake chính: LAV-DF.

### 15.2 Giọng nói
- Ngưỡng 0.25 → 0.40 (tạm, từ AudioMNIST) → **0.40 sai với owner**. Chốt trong prereg sau khi có impostor cùng kênh; khả năng nằm quanh 0.25–0.30.
- Enroll 5 mẫu chia 2 buổi, enroll và verify cùng kênh trình duyệt.

### 15.3 Ngưỡng thí nghiệm (đã chốt) và dự phòng
| ID | Ngưỡng | Dự phòng chính |
|---|---|---|
| E0 | Chỉ báo cáo | |
| E1a | ≥ 80% tầng D | Kiểm tra crop/align → phụ kiện lớn hơn (AdvHat) → rút về phân tích transfer + demo qua virtual camera |
| E1b | Báo cáo IAPMR | Phân tích khoảng cách digital → physical |
| E1c | IAPMR ≤ 5%, BPCER tăng ≤ 5 điểm | Chỉnh ngưỡng giọng → 1 lần retry → báo cáo đánh đổi |
| E2 | Câu cố định ≥ 80%; chuỗi ngẫu nhiên IAPMR ≤ 2% | Giới hạn từ vựng → tự train bộ nhận dạng chữ số |
| E3a, E3d | Báo cáo | |
| E3b | Dự đoán AUC < 0.75 | Kết quả nào cũng là phát hiện |
| E3c | AUC ≥ 0.90 | Thêm đặc trưng → GRU/1D-CNN → đọc khẩu hình → ghi là giới hạn |
| E4 | ≤ 3 s | tiny.en int8 → chạy song song → cắt cửa sổ → GPU |
| E5 | BPCER ≤ 10%, ≤ 1 retry | Chỉnh ngưỡng → UI tốt hơn → 2 retry |

### 15.4 L1
- Gọng kính, hai giai đoạn (A digital với nạn nhân ảo; B vật lý với người hỗ trợ). Có thêm dodging. Owner có trong DB (1:N thật).
- Độ phủ: ablation 4/8/12% (15–30% không khả thi với gọng kính).
- Cổng sang B: D ≥ 80% và control ≤ 20%, trên frame ánh sáng bình thường, **chỉ khi có liveness**.
- Tập đánh giá: ngày khác, cùng ánh sáng bình thường; frame tối là tầng `@dim` riêng.
- Gallery 5 ảnh/người (5 góc). Max mềm dạng softmax-weighted, không có log(n).
- Nạn nhân và lớp mô phỏng lấy từ `verification_data`.
- Transfer: method1/2/4 là cùng kiến trúc, khác cách train; **method5 (ViT) là khác kiến trúc**. Load strict, đúng class và đúng preprocessing.
- Liveness trong đánh giá đi qua `_liveness_decision` của backend.

### 15.5 Hướng đã loại
rPPG, YOLO bắt điện thoại, blink thụ động, screen-flash, motion challenge, depth/CDCN, sonar, face–voice không enroll, multi-face.

---

## 16. Trạng thái hiện tại và việc còn lại

### 16.1 Trạng thái các phase
| Phase | Trạng thái |
|---|---|
| Dọn repo, UI | Xong, đã merge |
| P0 | Đạt (giới hạn khi thiếu sáng) |
| P1 | Đạt về khả năng tách giọng; ngưỡng chưa chốt |
| P2 (challenge, ASR, Secure mode, chặn auto-log) | **Chưa làm** (owner chọn làm L1 trước) |
| P3 (AV sync, fusion R1, thẻ 6 kênh) | Chưa làm |
| P4a, L1 giai đoạn A | Code xong và đã sửa sau review; **chưa có kết quả với checkpoint thật** |
| P4a, L1 giai đoạn B | Chưa làm |
| P4b (deepfake L3), P4c (harness) | Chưa làm |
| P5 (prereg, thu test, train detector, E0–E5) | Chưa làm; `prereg.md` chưa có |
| P6 (demo, dashboard, report) | Chưa làm |

Commit `7a3a35e` đã push lên nhánh, chờ owner merge.

### 16.2 Việc owner cần làm tiếp (theo `runbook_L1.md`)
1. `pip install -r requirements.txt -r innovation/requirements.txt` (có thêm timm).
2. `.env` có `INNOVATION_ENABLED=1`, `INNOVATION_RESEARCH_MODE=1`; `frontend/.env.local` có `NEXT_PUBLIC_INNOVATION=1`.
3. Đặt `method1–5_best.pth` và `liveness_best.pth` vào `checkpoints/`.
4. Chụp frame qua Lab: `owner/s1_day1` (75), `owner/s2_day2` (75, ngày khác, cùng ánh sáng), `owner/s3_dim` tuỳ chọn, `owner_enroll/s0` (15).
5. Gallery: mỗi người đúng 5 ảnh. Gồm owner, nạn nhân `VER_victim` và khoảng 50 người lớp mô phỏng, lấy từ `verification_data`.
6. In bảng màu ở tỉ lệ 100%, chụp 3 frame, fit `colormap.json`.
7. Kiểm tra parity cho method3 và method5. Phải `pass True`; dòng của method5 phải ghi `model FaceViT_ArcFace input 224`.
8. Dodging, rồi impersonation, rồi `evaluate --liveness --transfer ...`.
9. Gửi Claude `summary.json`, `attack.json`, `loss.csv` (không gửi texture hay ảnh in).

### 16.3 Việc còn mở khác
- Chốt ngưỡng giọng bằng impostor cùng kênh (người hỗ trợ, buổi 1).
- P3 xử lý trường hợp fps < 20 khi thiếu sáng.
- P2 cần VAD trước ASR, chuẩn hoá chỉ lấy chữ số.
- Kiểm tra các method có chung dữ liệu train không (ghi vào report transfer).
- Lặt vặt: ESLint setState ở Register; "hôm nay" theo UTC; nav cuộn ngang trên điện thoại; `capture.py` in tiếng Việt.

---

## 17. Phụ lục: cấu trúc code, API, config, lệnh chạy

### 17.1 Cấu trúc repo hiện tại
```
backend/                 FastAPI, SQLite, WebSocket; env.py đọc .env; face_recognizer.py (CNN + ViT, strict)
frontend/
  app/                   attendance (Check-in), register, people (+ [id]/voice), history, lab, security
  components/            TopBar, CheckLadder, WebcamCapture, FaceOverlay, AngleGuide, AttendanceTable,
                         AttendanceLog, DataState, VoiceEnrollment, FrameCapture
  lib/                   api, checks, emotion, flags, format, innovation, media (useAVRecorder), search, websocket
liveness/                anti-spoofing (module chính) + tools/
emotion/  extreme_cond_enhancer/
innovation/
  config.json config.py db.py media.py voice.py api.py requirements.txt
  eval/                  runs, metrics, latency, voice_scores, voice_crossval
  tools/                 enroll_public, audiomnist
  attacks/adv_glasses/   faces, glasses, pipeline, eot, printability, optimize, evaluate, export
tests/                   liveness/, extreme_cond_enhancer/, innovation/ (test_p0, test_voice, test_metrics,
                         test_adv_glasses), test_backend_env.py
docs/                    project_requirements.pdf, innovation/*.md
results/                 face_recognition/, liveness/, emotion/, innovation/
```

### 17.2 API innovation (chỉ có khi `INNOVATION_ENABLED=1`)
| Method | Đường dẫn | Việc |
|---|---|---|
| GET | `/api/innovation/config` | Config và ngưỡng hiện hành |
| GET | `/api/attempts` | Danh sách verification attempts |
| POST | `/api/innovation/media/probe` | Decode thử một clip (fps, audio) |
| POST | `/api/voice/enroll/start` | Cấp 3 chuỗi số enroll |
| POST | `/api/voice/enroll/submit` | Nộp bản ghi; chỉ lưu khi đạt |
| GET | `/api/voice/status`, `/api/voice/status/{id}` | Số mẫu giọng đã enroll |
| DELETE | `/api/voice/{id}` | Xoá giọng đã enroll |
| POST | `/api/innovation/frames` | Lưu frame chụp từ Lab (chỉ research mode) |

### 17.3 `innovation/config.json` (ngưỡng tạm)
- challenge: 6 chữ số, hạn 8 s, tiếng Anh, tối đa 2 lần thử lại.
- asr: `tiny.en`, `max_distance` 1.
- voice: ECAPA, 3 mẫu enroll, `min_speech_s` 1.5, `min_rms_dbfs` −40, `max_clip_fraction` 0.01, `min_snr_db` 10.
- fusion: R1.
- thresholds: face 0.5, liveness 0.5, voice 0.4 (sẽ sửa trong prereg), av 0.5.
- media: 16 kHz, `min_fps` 20.

### 17.4 Lệnh hay dùng
```bash
pip install -r requirements.txt -r innovation/requirements.txt
```
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
```bash
cd frontend && npm run dev
```
```bash
python -m innovation.eval.latency
```
```bash
python -m innovation.eval.voice_scores --employee <ID> --genuine <thư_mục_clip> --impostors data/public/audiomnist/raw
```
```bash
python -m innovation.attacks.adv_glasses.printability chart
```
```bash
python -m innovation.attacks.adv_glasses.printability fit data/innovation/frames/chart/s1/0001.jpg
```
```bash
python -m innovation.attacks.adv_glasses.pipeline --frames data/innovation/frames/owner/s1_day1 --checkpoint checkpoints/method5_best.pth
```
```bash
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --victim VER_victim --mode impersonate --color-map data/innovation/chart/colormap.json --steps 1500 --batch 8
```
```bash
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<run> --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth
```
```bash
python -m innovation.attacks.adv_glasses.export --run results/innovation/<run> --ipd-mm <IPD>
```
