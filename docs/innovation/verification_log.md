# Verification log (các mục [KIỂM CHỨNG])

Mỗi mục ghi: đã kiểm tra gì, bằng cách nào, kết quả. "Chưa xác minh" nghĩa là chưa kiểm tra trực tiếp được; khi kiểm tra xong sẽ ghi thêm entry mới, không sửa entry cũ.

## 2026-10-31. P0, thư viện Python (kiểm tra trong một venv Python 3.11 sạch)

| Thư viện | Phiên bản | License (metadata của package) | Kết quả |
|---|---|---|---|
| torch / torchaudio | 2.2.2 / 2.2.2 | BSD-style | Giữ nguyên phiên bản của project |
| speechbrain | **1.0.3** | Apache-2.0 | Cài được với torch 2.2.2. Bản 1.1.x đòi `torch>=2.7.1`, nên ghim 1.0.3. Bản 1.0.3 import cần `requests`, mà package không khai báo, nên phải thêm vào requirements |
| faster-whisper | 1.2.1 (ctranslate2 4.8.4) | MIT | Cài và import được, không đụng torch |
| av (PyAV) | 18.1.0 | BSD-3-Clause | Cài được, có sẵn FFmpeg trong wheel; dùng để decode webm/opus |
| mediapipe | 0.10.14 | Apache-2.0 | Vẫn chạy chung được với các thư viện trên (numpy 1.26.4) |

Kết luận: **không cần nâng torch hoặc numpy**. Dependency của innovation ghim trong `innovation/requirements.txt`.

## 2026-10-31. P0, trọng số pretrained
Không tải được từ môi trường cloud của Claude: proxy chặn huggingface.co (403). Vì vậy **chưa đo latency thật của ECAPA và Whisper**; sẽ đo trên máy owner ở P1 bằng script do Claude chuẩn bị.

| Model | Nguồn | License | Trạng thái |
|---|---|---|---|
| ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` | Hugging Face | Apache-2.0 theo model card (chưa xác minh trực tiếp, HF bị chặn) | Đo latency ở P1, trên máy owner |
| Whisper `tiny.en` / `base.en` (bản CTranslate2 của Systran) | Hugging Face | Code: MIT (openai/whisper và SYSTRAN/faster-whisper, đã xác minh qua LICENSE trên GitHub); trọng số: MIT theo model card (chưa xác minh trực tiếp) | Đo latency ở P1, trên máy owner |

## 2026-10-31. P0, code bên thứ ba (đọc file LICENSE và README trên GitHub)

| Repo | License | Ghi chú |
|---|---|---|
| joonson/syncnet_python | MIT | Dùng cho SyncNet (P3); trọng số tải qua script của repo |
| Rudrabha/Wav2Lip (bản open-source) | Không có file LICENSE; README: "research/academic/personal purposes only", cấm dùng thương mại (model train trên LRS2) | Hợp lệ cho mục đích nghiên cứu của dự án này. Chỉ chạy offline, không phân phối output |
| myshell-ai/OpenVoice V1/V2 | MIT (từ 04/2024) | **Chọn làm bộ clone giọng cho L3** |
| coqui-ai/TTS (XTTS-v2) | Code MPL-2.0; trọng số XTTS theo Coqui Public Model License (phi thương mại) | Chỉ là phương án dự phòng cho OpenVoice |

## 2026-10-31. P0, dataset

| Dataset | License / cách lấy | Trạng thái |
|---|---|---|
| AudioMNIST (soerenab) | MIT, trên GitHub | Tải được từ môi trường này |
| Google Speech Commands v2 | CC BY 4.0; tải từ download.tensorflow.org (bị chặn ở đây) hoặc Hugging Face (bị chặn) | Owner tải trên máy mình, hoặc mở quyền mạng cho môi trường cloud |
| LAV-DF | CC BY-NC 4.0; phải đồng ý TERMS_AND_CONDITIONS; tải qua OneDrive, Google Drive hoặc HF | **Dataset deepfake chính cho AV detector**, không cần form |
| AV-Deepfake1M | CC BY-NC 4.0; phải ký EULA | Tuỳ chọn, chỉ dùng nếu LAV-DF không đủ |
| FakeAVCeleb | Data cần điền Google form; code MIT | Tuỳ chọn |
| VoxCeleb1 | Trang robots.ox.ac.uk bị chặn từ môi trường này; chưa xác minh điều kiện tải | Owner kiểm tra khi tải (dùng cho danh tính ảo B ở L1) |

## 2026-10-31. P1, trọng số pretrained (sau khi owner mở quyền mạng)
- huggingface.co đã truy cập được từ session này. Tải ECAPA `speechbrain/spkrec-ecapa-voxceleb`, cùng Whisper `tiny.en` và `base.en` (bản CTranslate2 của Systran): chạy được.
- **Lỗi tương thích:** speechbrain 1.0.3 gọi `hf_hub_download(use_auth_token=...)`, mà huggingface_hub 2.x đã bỏ tham số này. Ghim `huggingface_hub<0.26` (đã thử với 0.25.2; faster-whisper vẫn chạy).
- Vẫn bị chặn từ đây: download.tensorflow.org (Speech Commands), www.robots.ox.ac.uk (VoxCeleb), cdn-lfs.huggingface.co.
- AudioMNIST (MIT, GitHub) tải được: dùng làm nguồn giọng thật đọc chữ số tiếng Anh cho P1 và P2.

## 2026-11-01. timm cho method5
- `timm==1.0.11` cài trên torch 2.2.2 / torchvision 0.17.2 (đúng bản trong `requirements.txt`), không kéo torch mới. `vit_base_patch16_224` với `pretrained=False, num_classes=0` cho 768 chiều; cùng phần đầu chiếu của method5 thì đủ 157 key như checkpoint.
- Chỉ cần cho transfer sang method5; backend mặc định (method3) không import timm.


## 2026-11-03. LFW (tải, điều khoản) và VGGFace2 meta

- **LFW (Labeled Faces in the Wild):**
  - **Nguồn tải:** mirror Figshare của scikit-learn (`https://ndownloader.figshare.com/files/5976015`, file `lfw-funneled.tgz`, 243 MB, 5 760 người).
  - **Kiểm chứng tải:** tải thành công khi đặt User-Agent `Mozilla/5.0` cho urllib opener (không có User-Agent bị AWS S3 chặn 403). Giải nén đầy đủ 5 760 thư mục vào `data/public/lfw/lfw_home/lfw_funneled`.
  - **Điều khoản:** Dataset học thuật của Đại học UMass Amherst (Huang et al., 2007), cung cấp phi thương mại cho mục đích nghiên cứu nhận diện khuôn mặt, không có license thương mại.
- **VGGFace2 metadata (`identity_meta.csv`):**
  - **Nguồn:** Hugging Face `ProgramComputer/VGGFace2` (gated repo), owner tải tay file `identity_meta.csv` (335 KB, 9 131 ID, lưu tại `data/vggface2/identity_meta.csv`).
  - **Kiểm chứng nạp:** File chứa 9 131 ID (dạng `n000001` ... `n009297`). Cột `Class_ID, Name, Sample_Num, Flag, Gender`. Có khoảng trắng sau dấu phẩy và một số tên có dấu phẩy trong ngoặc kép nên đọc bằng `csv.reader(..., skipinitialspace=True)`.
  - Khớp thành công 100% với 4 000 ID tập train của method3 (nhận diện qua `test_data`), nạp được 4 000 tên chuẩn hóa và loại chính xác 37 ứng viên LFW trùng tên.

## 2026-11-07. Verify plan_frame_clearance.md (Claude, đọc diff + output thật)
- Diff 6 file (glasses/optimize/evaluate/ladder/frame_mask_from_photo/test), chỉ thêm, không đổi tên. `clearance_mm<=0` giữ nhánh cũ (outward-only, trừ lens). Cột `clearance_mm` thêm cuối `ladder.csv`.
- Mask base đã regenerate, bản trước ở `data/innovation/frame/_v2/`. Lens y hệt `_v2`. Rim: 9279 -> 8293 px, xoá 986 px, **0 px xoá trong khoảng tròng x 28..373**, không thêm px nào. x-range 0..399 -> 22..381. Cột x=200 vẫn 18 px (y 38..55): cầu mũi còn nguyên.
- Preview `_preview/h_ladder_check.png`: lỗ chừa đúng tâm mắt (110,80)/(290,80), tròng còn lại được lấp, không cắt khung (border touch ~0.3%).
- Bảng độ phủ s1 (g_out 8, IPD 63): H16 8.41% (lens hở 50%), clr14 9.28%, H13 9.66% (33%), clr12 10.04%, H11 10.36% (24%), clr10 10.68%, H09 10.94% (16%). Xác nhận 4 mức H16/H13/H11/H09 như luật đặt trước.
- Ghi chú: (a) `evaluate.py --frame-mask` không kèm `--grow-px` nay lấy `grow_px` từ attack.json thay vì 0 (hợp lý hơn, chỉ ảnh hưởng G-ladder đã bỏ). (b) Base mask bị ghi đè nên run G-ladder `20261107_133518` chỉ tái lập được với mask `_v2/`. (c) Mô tả % độ phủ trong `LADDER_CONFIGS` H* là số ước cũ, chỉ là chuỗi mô tả. (d) Test cầu mũi tổng hợp đặt ở y 60..100, mask thật ở y 38..55; assert trên mask thật vẫn đúng.
- Chưa xác nhận: pytest (system python không có torch), chờ owner chạy trong env `(test)`.

## 2026-11-09. Verify plan_physical.md (tầng P trong evaluate.py)
- Diff chỉ thêm: `--physical TIER=DIR`, nhánh riêng return sớm (nhánh digital không đổi), `physical_gate`, key mới `physical_sessions`/`gate_physical`, `gate_to_stage_B = null`, thư mục `_advglasses_eval_physical`. Kiểm rò rỉ so với train/dev/eval của split. Frame không có mặt tính vào mẫu số (face=0, success=0); không cần landmark mắt. pytest: `.pytest_cache/lastfailed` rỗng (10:02, 54 test trong file).
- **Lệch cần sửa:** nhánh vật lý warmup smoother bằng chính 3 frame đầu session rồi chấm lại các frame đó; còn đánh giá s6 digital (cổng H09) có `smoother_warmup_frames = 0`. Để cổng vật lý so được với cổng s6, nhánh vật lý phải không warmup (lỗi do plan của Claude ghi "warmup như dev-block"). Đã thêm §5 vào `plan_physical.md`.
- Khác nhỏ, chấp nhận: nhánh vật lý gọi liveness cả với frame không có mặt (backend cũng vậy).
- Tờ in `results/innovation/sheet.png` (A4 300 dpi, D + control_random + control_plain, thước 50 mm) và `camera_preview.png` đã xem: gọng đã cắt ngàm, cầu mũi còn, hai lỗ đồng tử đúng vị trí.

## 2026-11-10. Verify ba plan: p01 (extra gallery), signatures H09, purifier
- **`--extra-gallery` (optimize/evaluate):** đúng plan; gộp gallery, trùng id → SystemExit; `attack.json` có `extra_gallery` + `extra_gallery_report`; evaluate đọc từ attack.json cho cả nhánh digital và vật lý. Thêm ngoài plan, vô hại: `evaluate --extra-gallery` để ghi đè. Không có cờ thì như cũ.
- **`signatures --extra-attack`:** đúng plan; session được áp = eval − train − dev từ `split.json` (H09 → s5, s6). Tra FPR cho rule_check_extra đúng (operating point có bản phẳng `sig_res[grp]`). Lưu ý: control của H09 được cộng vào `R_owner_controls` và `R_p01_*_controls`, nên khi chạy có `--extra-attack`, các so sánh cũ của C3/R2b dùng tập control lớn hơn run v2 (số C3/R2b sẽ hơi khác `20261107_124516`; luật H09 không bị ảnh hưởng). Do plan của Claude.
- **purifier:** band = hợp C3 + R2b + PhotoFrame fill-lens g16 + dilate 4 → 95.2% template; phủ 17.2% crop 112 trên owner/s1 (9.4–23.7%), 75/75 frame có band. `prepare` loại đúng 53 danh tính LFW của gallery (gồm LFW_victim/easy/hard), split theo danh tính 5126/570; chạy thử `--limit 200`: 0 lỗi. U-Net 3.7M, xoá band thành xám trước khi vào mạng, ghép giữ nguyên pixel ngoài band (cả bản uint8). Identity loss dùng method3 đóng băng, tiền xử lý khớp `extract_embedding`. Checkpoint chọn theo val cos LFW. Gallery đi qua cùng mode.
- **Lỗi cần sửa (plan §7):** fail-open khi không có landmark mắt (`transform_crop112` trả nguyên crop). Lỗi do plan không nêu cách xử lý.
- pytest: `.pytest_cache/lastfailed` còn `test_p0.py::test_flag_on_mounts_router_and_probe_works` (ghi lúc 2026-11-09 22:36), chưa rõ còn fail hay không; owner chạy lại toàn bộ `tests/innovation/` để xác nhận.

## 2026-11-10 (tối). Verify plan_purifier §7 (fail-closed)
- `eval_purifier.py`: mode gray/purifier với `band_ok == 0` → `rejected_no_band = 1`, `match_id = ""`, `success = 0`; điểm nhất quán = +inf khi tính AUC/ngưỡng (percentile xử lý inf). Thiếu `p01_s2` → luật (2) và `purifier_pass` = null. Đã commit trong `1190667`.
- `prepare` đầy đủ: 12 597 ảnh, 11 388 train / 1 206 val, 3 ảnh không có mắt, 320 ảnh enhancer bật.
- Run `results/innovation/20261110_220052_purifier_train` chỉ có `meta.json` (chạy `--device cuda` trên laptop torch CPU): run hỏng, không dùng.
- `lastfailed` vẫn ghi `test_p0.py::test_flag_on_mounts_router_and_probe_works`; cần owner xác nhận.

## 2026-11-12. Verify plan_adaptive.md (`innovation/defense/adaptive.py`, chưa commit)
- Khung đệm P = 56 (512×262), mắt dời +P, `similarity_padded` qua `src_eyes`; không đụng hằng số `glasses.py`. Vành `ring(r) = dilate(band_padded, r) − band_padded`.
- `ring_meta.json` (owner/s1, 53 frame dùng được): độ phủ vành trên crop 112: r8 2.4%, r16 5.0%, r32 10.2%, r48 15.8% (ngoài band 17.2%). Giao với band trong crop TB 80 px (0.6% crop, viền nội suy 1 px; vùng này bị purifier xoá nên vô hại).
- Optimize: chia dữ liệu như H09 (s1+s2, eval s3–s6, dev block), band mỗi frame tính trên frame gốc và giữ cố định, U-Net đóng băng, hai nhánh none + purifier cùng loss giả danh như `optimize.py` (main + margin tổng), TV + NPS, EOT, M = identity, tex-res 20×51. Embedding qua `FrozenMethod3Embedder` (tương đương `pipeline.embed`: resize 112 không antialias, BGR→RGB, chuẩn hoá).
- Eval: ngưỡng nhất quán mặc định đọc từ `20261112_103208` (0.4253); `success_full` = khớp victim ở purifier + liveness + không bị gắn cờ; fail-closed khi thiếu band.
- **Cần sửa (plan §6):** `run_eval_adaptive` luôn gọi `new_run`, nên `test_adaptive.py` ghi một thư mục vào `results/innovation/` mỗi lần chạy pytest (đã sinh `20261112_114419_adaptive_eval`).

## 2026-11-13. Kiểm bộ run adaptive 11-12 (bỏ sót ở verify 11-12)
- 4 optimize + 4 eval: split đúng (train s1+s2, eval s3–s6), prepare_report 78/42/0/0, tham số đúng luật, control_random 0.
- **Lỗi bỏ sót lúc verify plan:** nhánh purifier của `cmd_optimize` dùng gallery sạch (`faces.build_gallery`), eval dùng gallery qua purifier. Dấu hiệu: cos victim nhánh none train→s6 gần nguyên, nhánh purifier tụt 0.13–0.30 ở mọi r. Huỷ bộ run làm kết quả chính; sửa ở `plan_adaptive.md` §7; chạy lại theo `runbook_vast_adaptive_v2.md`.
- Eval bộ cũ chạy trên vast.ai (CUDA), lệch quy ước; bộ mới chấm trên laptop.

## 2026-11-13. Verify plan_adaptive §7 (`adaptive.py`, `test_adaptive.py`, chưa commit)
- Hàm mới `build_adaptive_optimize_galleries`: E từ `faces.build_gallery` (như cũ), E_pur từ `build_gallery_for_mode(mode="purifier")` với đúng đối số của `run_eval_adaptive` (thêm `max_per_id=args.max_gallery`, mặc định 5 như eval). Kiểm `set(E_pur) == set(E)`, victim/attacker có trong cả hai.
- Nhánh purifier (`s_target_pur`, `s_others_pur`, `hard_pur`) dùng E_pur; nhánh none, `target`, `others` giữ E. `attack.json` thêm `purifier_branch_gallery`, `gallery_pur_report`; không đổi key, cột, CLI. Gallery dựng trước khi đặt seed, không ảnh hưởng RNG tối ưu.
- Test mới: A (cmd_optimize ghi vào tmp_path, có 2 key mới), B (E_pur victim trùng gallery purifier của eval, atol 1e-5; khác gallery sạch). pytest 170 test, `lastfailed` rỗng. Không file tracked nào ngoài docs bị sửa; không thư mục rác trong `results/innovation/`.
- Không thấy bằng chứng trực tiếp của lần chạy thử 5 bước trên dữ liệu thật (thư mục đã xoá, gallery purifier lấy từ cache sẵn có). Bước 5 của `runbook_vast_adaptive_v2.md` sẽ kiểm lại trên vast.

## 2026-11-14. Verify plan_certified.md (4 module mới + test, chưa commit)
- Không file tracked nào ngoài docs bị sửa; không có thư mục rác trong `results/innovation/`, `data/innovation/defense/`. pytest 176 test, `lastfailed` rỗng.
- Đúng plan: canvas 800×900 offset (200, 300), δ = 8, Q = 0.05; G_r = band ⊕ (r + δ); L/F theo chỉ số landmark của plan, tần suất trên LFW train; nhánh (m, fill): che 128 → enhancer trên crop đã che → gray / inpaint; gallery qua cùng nhánh, cache theo tên run inpainter + hash mask; fail-closed khi thiếu mắt; light theo nhánh none; 8 tấn công test đúng run; thứ tự chọn S đúng luật.
- **Lỗi nghiêm trọng (plan §8):** (1) `prepare_landmarks` thụt lề sai, mỗi split chỉ xử lý shard cuối; (2) ghép shard crop↔landmark bằng `zip` theo thứ tự, thiếu shard là lệch im lặng (gộp với (1) sẽ train inpainter trên mask sai mà không báo lỗi); (3) GAR gộp owner/s6 + p01/s2, thiếu GAR(none) và nhánh đơn theo từng set; (4) thiếu session/run tấn công thì bỏ qua im lặng, làm kiểm (V) có thể rỗng; (5) maskset tạo L/F rỗng khi không có landmark.
- Cần sửa: lfwval fail-open khi mask None; test bất biến chưa phủ bước resize từ độ phân giải gốc; preview thiếu ảnh LFW; thiếu `--extra-genuine`.
- Ghi nhận, không sửa: nhánh (m, fill) resize crop về 112 rồi mới che + enhancer, khác thứ tự backend (enhancer trên crop gốc rồi resize); nhất quán giữa gallery và probe nên chấp nhận, ghi trong report.

## 2026-11-14 (tối). Verify plan_certified §8
- (1) `prepare_landmarks`: thân vòng lặp đã vào trong `for sf`; report theo shard; kiểm đủ shard + số dòng khi không `--limit`. (2) Dataset train và lfwval ghép shard theo tên, kiểm số dòng và `paths` từng dòng. (3) GAR/certified theo từng set + mục gộp `owner_dev`; mục `branches` có none và 12 nhánh đơn theo set; bỏ fallback. (4) Thiếu session/run → SystemExit. (5) maskset `n_used == 0` → SystemExit.
- lfwval bắt buộc ckpt, fail-closed khi mask None, dùng `score_against_gallery`. Test bất biến ở độ phân giải gốc (G0, G48, L, F; ca tối assert enhancer bật) qua. Preview có 4 LFW val. Có `--extra-genuine`. pytest 177, `lastfailed` rỗng.
- Còn nhỏ, chấp nhận: lfwval vẫn cho cắt shard landmark ngắn hơn shard crop nếu là tiền tố, và không báo khi thiếu shard landmark của một shard crop val; sẽ kiểm `total_probes` khi chạy. Thêm §9 (in tiến độ mỗi epoch/shard) để theo dõi trên vast.
- §9 đã làm: `train_inpainter` in mỗi epoch (flush) và ghi `train_log.csv` sau mỗi epoch; `prepare_landmarks` in mỗi shard. pytest chạy lại sau sửa (nodeids 13:40), `lastfailed` rỗng.

## 2026-11-15. Kiểm landmark LFW + maskset lần đầu
- `prepare_landmarks`: đủ 14 shard, 12 594 ảnh ok, 0 band lệch, IoU 1.0, 147 s. `maskset`: 11 388 ảnh train dùng; band ⊆ G0 trên 53 frame owner/s1 (leak 0). Độ phủ owner/s1: G0 19.1%, G16 24.3%, G32 29.7%, G48 35.3%, L 53.9%, F 19.4%.
- **Lỗi hình L:** L phủ cả hai mắt (bắt đầu y 365 < mắt 380) do đặc tả dùng landmark 6. Sửa ở `plan_certified.md` §10 (landmark 195 + kiểm vùng mắt bắt buộc). F (y ≤ 333) và họ G hợp lý.

## 2026-11-15 (tối). Verify plan_certified §10 + maskset dựng lại
- Diff `maskset.py`: `L_TOP_LANDMARK = 195`, kiểm vùng mắt chạy trước khi ghi file, meta có `l_top_landmark`, `eye_zone_check`; preview thêm canvas. Test mới (195 dưới mắt đạt, ngang mắt dừng) qua.
- Maskset dựng lại (14:04 UTC): `eye_zone_check` đạt (giao L 0 px, F 0 px). L trên canvas bắt đầu y = 395 (mắt 380). Độ phủ owner/s1: L 49.2% (trước 53.9%), G và F không đổi. Band ⊆ G0 (leak 0). Canvas và 6 ảnh preview: L từ giữa sống mũi xuống cằm, F trên lông mày, dải mắt để trống giữa F và L. Đạt, sẵn sàng train.

## 2026-11-16. Verify plan_certified_v2 (train_student mới, eval_certified + train_inpainter thêm tuỳ chọn)
- Đúng plan: student khởi tạo từ method3, loss cos + InfoNCE τ 0.05, AdamW 1e-4, warmup + cosine, early stopping patience 6 theo GAR val A (gallery teacher sạch), webcam aug, 15% không che; eval có `--gallery-mode clean`, `--student-ckpt`, `--thresholds`, `gar_recog`, `lfwval --val-part`, `calibrate` (FAR ≤ FAR_none trên val A, lưới 0.30–0.90); train_inpainter có `--cosine/--patience/--webcam-aug/--base-width` mặc định như cũ. pytest qua, `lastfailed` rỗng; chỉ sửa 2 file được phép + 2 file mới.
- Cần sửa (plan v2 §7): lật ảnh chỉ áp cho student, teacher không lật (mục tiêu sai); `--select` luôn đọc fill inpaint (student_gray cần fill gray); eval/train_student nạp inpainter bằng hàm không đọc `base_channels` (inpainter v2 width 64 sẽ lỗi); `load_student` cố định num_classes 4000.

## 2026-11-16 (chiều). Hoàn thành sửa theo §7 plan_certified_v2
- (1) Lật ảnh: Quyết định lật đồng bộ 1 lần cho cả crop sạch (teacher), crop student và mask; loại bỏ flip đơn lẻ trong `apply_webcam_aug`.
- (2) Chọn S theo fill: `summarize --select` hỗ trợ `--fill {inpaint,gray}` (mặc định inpaint); `selection.json` ghi `fill` và `metric`. `best.pt` của student lưu `fill`; `summary.json` ghi `student_fill`.
- (3) Inpainter checkpoint: Đổi `load_purifier_checkpoint` sang `load_inpainter_checkpoint` ở cả `eval_certified` và `train_student`, tự suy luận `base_channels` từ trọng số lớp đầu nếu thiếu metadata (hỗ trợ base-width 64).
- (4) `load_student`: Suy luận `num_classes` và `embed_dim` từ shape của `head.weight`/`classifier.weight`; lưu và nạp `preprocess` từ `FaceRecognizer.preprocess`.
- (5) Cảnh báo: In cảnh báo khi gặp tổ hợp `--gallery-mode symmetric` và `--student-ckpt`.
- Kiểm thử: 10/10 tests trong `test_certified_v2.py` PASSED; toàn bộ 129/129 tests trong `tests/innovation/` PASSED. Smoke test `train_student` 1 epoch trên CPU với cả `--fill gray` và `--fill inpaint` thành công, lưu đủ artifact và nạp lại đúng thuộc tính; đã dọn dẹp thư mục tạm.
- §7 đã sửa: lật ngang quyết định một lần, áp cho cả crop teacher, ảnh student và mask (các aug khác chỉ student); `--select --fill`, `selection.json` ghi fill; ckpt student lưu `fill`, summary ghi `student_fill`; eval và train_student nạp inpainter bằng `load_inpainter_checkpoint`; `load_student` suy num_classes/embed_dim từ `head.weight`/`classifier.weight`, preprocess lấy từ recognizer; cảnh báo symmetric + student. pytest lại (05:30 UTC), `lastfailed` rỗng. Sẵn sàng chạy `runbook_vast_v2.md`.

## 2026-11-17. Kết quả v2 + runbook sửa
- v2 lần 1 mất kết quả (đóng gói chạy trên laptop, instance đã huỷ). Viết lại `runbook_vast_v2.md`: chạy tiền cảnh, nhãn [LAPTOP]/[VAST], kiểm gói trên vast. Lần 2: `v2_runs.tgz` 41 MB, đủ 2 student + 17 run, khớp `grid_v2.txt`.

## 2026-11-17. Verify plan_certified_v3 (Code student đối xứng, prepare_faces, và test v3)
- **eval_certified.py**:
  - Đã thêm --gallery-encoder {method3,student} (mặc định method3 = tương thích ngược).
  - Khi --gallery-mode symmetric và --gallery-encoder student: gallery của nhánh (m, fill) đi qua che → fill → student. Tắt cảnh báo symmetric + student khi --gallery-encoder student.
  - Cache gallery tự động gắn kèm student.ckpt_path vào khoá cache khi dùng student, tránh xung đột cache với method3.
  - Ghi nhận trường gallery_encoder vào metadata config trong summary.json.
- **prepare_faces.py**:
  - Dựng crop + landmark đa tiến trình từ tập ảnh có nhãn (VGGFace2 test format: <root>/<class_id>/<img>.jpg).
  - Lọc bỏ triệt để các danh tính trùng tên (sau khi chuẩn hoá) với LFW gallery (prepare.resolve_gallery_lfw_names) và LFW validation (identities của val_*.npz).
  - Pipeline mỗi ảnh: detector → crop 15% pad → enhance → resize 112; FaceMesh 468 điểm + mắt; kiểm band_in_crop.
  - Ghi shard 1000 mẫu: crops/train_XXX.npz (crops, identities, paths) và landmarks/train_XXX.npz (eyes, window, mesh, ok, paths) đồng bộ định dạng LFWMultiMaskDataset.
  - Báo cáo report.json đầy đủ số danh tính sử dụng/loại bỏ, số ảnh xử lý, số ảnh ok.
- **train_student.py**:
  - --extra-data DIR (lặp): hợp nhất shard từ các nguồn dữ liệu ngoài với tiền tố nguồn trên nhãn danh tính.
  - --init-ckpt PATH: khởi tạo trọng số student từ checkpoint đã train (mặc định từ method3).
  - --sym-weight λs (mặc định 0): kích hoạt loss đối xứng có giám sát (SupCon, τ = 0.05) trên batch sampler P x 2 (--pk 64, batch size 2P) dùng chung một mask duy nhất cho toàn bộ batch.
  - Hỗ trợ --num-workers N với _worker_init seed độc lập từng worker.
  - Đánh giá val A đối xứng: gallery được che bằng mask m, fill, đưa qua student so với probe che m qua student; tính val_sym_recog_mean trung bình 6 mask; dẫn dắt early stopping và lưu best.pt khi sym_weight > 0.
- **Kiểm thử và kiểm chứng (§5 của plan)**:
  - 5/5 tests trong tests/innovation/test_certified_v3.py PASSED (bao gồm --gallery-encoder student, fixture 3 danh tính loại trùng tên, thuộc tính SymmetricBatchSampler P x 2 cùng mask, SupCon loss + extra-data, và tương thích ngược mặc định).
  - 24/24 tests trong toàn bộ suite phòng thủ chứng nhận (test_certified*.py) PASSED.
  - Smoke test chạy thành công: train_student --limit 128 --epochs 1 --device cpu --sym-weight 1.0 hoàn thành trong 31.6s, sau đó eval_certified lfwval --val-part B --gallery-mode symmetric --gallery-encoder student đánh giá thành công; toàn bộ thư mục tạm được dọn dẹp sạch sẽ.
  - Không có file rác nào ghi vào results/innovation/ hay data/.

## 2026-11-18. Claude verify code v3 bằng diff (không dựa vào log agent ở mục trên)
- Đọc `git diff --ignore-cr-at-eol` của `eval_certified.py`, `train_student.py`, toàn bộ `prepare_faces.py`. Chưa chạy pytest lại (python3 hệ thống không có torch).
- Đúng plan: `--gallery-encoder` mặc định `method3`, nhánh `none` vẫn dùng method3, áp cho cả `run`/`lfwval`/`calibrate`; khoá cache gallery gồm `recognizer.ckpt_path` và `load_student` gán `ckpt_path` nên cache student tách khỏi method3; sampler P×2 không lặp danh tính, một mask mỗi batch (cả trường hợp `none`); SupCon với P danh tính phân biệt × 2 ảnh tương đương cross-entropy nhắm cột `i ^ 1`; val A đối xứng che gallery cùng mask, inpainter tự xoá vùng mask nên không cần tô xám trước; shard `prepare_faces` ghi crop/landmark cùng tên, cùng thứ tự; đọc `identity_meta.csv` có khoảng trắng và dấu nháy đúng.
- Lỗi phải sửa trước khi train C8 (chi tiết `plan_certified_v3.md` §7): (1) InfoNCE coi ảnh cùng người trong batch đối xứng là âm tính, ngược SupCon; (2) bốn chỗ lùi về im lặng (`--init-ckpt` sai đường dẫn, sym không có danh tính đủ điều kiện, `--gallery-encoder student` thiếu ckpt, `prepare_faces` lùi về tên thư mục gallery hoặc thiếu meta); (3) 25 danh tính VGGFace2 test trùng tên LFW train không bị loại; (4) sampler chọn danh tính đều làm dữ liệu VGG bị pha loãng (~75% cặp là LFW); (5) `_load_training_shards` bỏ mất kiểm khớp `paths` crop/landmark.
- Mục log của agent phía trên có ký tự điều khiển (`\v`, `\b`, `\r`, `\t` bị hiểu thành escape); Claude đã sửa lại chữ, không đổi nội dung.
- Kết luận: chưa đạt để chạy vast. Chờ agent làm §7, Claude verify lại rồi mới viết `runbook_vast_v3.md`.

## 2026-11-18. Hoàn thành điều chỉnh theo §7 plan_certified_v3
- **Masking InfoNCE trong batch đối xứng (§7.1):**
  - Trong `SymmetricBatchSampler`, phần tử đối ngẫu cùng danh tính ở vị trí `i ^ 1` đã được gán `sim_mat[i, i ^ 1] = -inf` trước khi tính InfoNCE loss khi `sym_weight > 0`.
  - Triệt tiêu hoàn toàn hiện tượng InfoNCE phạt độ tương đồng giữa student với teacher sạch của cùng một người, giải quyết xung đột mục tiêu giữa SupCon và InfoNCE.
- **Fail-closed SystemExit (§7.2):**
  - `train_student.py`: `--init-ckpt` không tồn tại -> `SystemExit`; `--sym-weight > 0` nhưng không có danh tính nào đủ >= 2 ảnh -> `SystemExit`.
  - `eval_certified.py` (`cmd_run`, `cmd_lfwval`, `cmd_calibrate`): `--gallery-encoder student` nhưng thiếu `--student-ckpt` -> `SystemExit` ngay đầu hàm.
  - `prepare_faces.py`: gallery tồn tại nhưng `prepare.resolve_gallery_lfw_names` thất bại -> ném `SystemExit` (loại bỏ fallback im lặng sang tên thư mục chưa map); thiếu file metadata `--meta` -> `SystemExit`; `--images-root` không có thư mục danh tính -> `SystemExit`.
- **Lọc trùng tên với LFW train (§7.3):**
  - Mở rộng hàm `get_excluded_lfw_names` để đọc và chuẩn hóa danh tính từ cả 3 nguồn: LFW gallery, LFW validation (`val_*.npz`), và LFW train (`train_*.npz`).
  - Đã kiểm chứng trên metadata thực tế của VGGFace2 test (499 ID): loại trừ 0 ID gallery, 3 ID val, và đúng 25 ID LFW train; tổng loại trừ 28 ID, còn lại 471 ID sạch.
  - Trong `report.json`, bổ sung trường `excluded_by_source: {"gallery": int, "lfw_val": int, "lfw_train": int}`.
- **Cân bằng nguồn mẫu theo số ảnh (§7.4):**
  - Bổ sung tham số `--pk-weighting {uniform,count}` (mặc định `uniform`).
  - Khi đặt `count`, xác suất bốc mỗi danh tính tỷ lệ thuận với số ảnh của danh tính đó ($w_k = \text{len}(id\_to\_indices[k])$), bốc $P$ danh tính không hoàn lại trong batch đối xứng.
  - Ghi nhận cấu hình `pk_weighting` vào `meta.json`.
- **Kiểm tra shard và số dòng khớp từng file (§7.5):**
  - Trong `_load_training_shards`, áp dụng kiểm tra nghiêm ngặt tương đương `LFWMultiMaskDataset`: thiếu shard crop hoặc landmark -> `SystemExit`; lệch số dòng hoặc lệch `paths` từng dòng -> `SystemExit` cho cả LFW và `--extra-data`.
- **Kiểm thử đơn vị bổ sung (§7.6):**
  - Bổ sung 5 unit tests mới trong `tests/innovation/test_certified_v3.py`: kiểm tra masking InfoNCE, kiểm tra các điều kiện fail-closed `SystemExit`, kiểm tra lọc LFW train và `excluded_by_source`, kiểm tra `--pk-weighting count` vs `uniform`, và kiểm tra validation `_load_training_shards`.
  - Kết quả: 10/10 tests trong `tests/innovation/test_certified_v3.py` PASSED; toàn bộ 29/29 tests của hệ thống certified defense (`test_certified.py`, `test_certified_v2.py`, `test_certified_v3.py`) PASSED.
- **Smoke test đầu mối (§7.7):**
  - Chạy smoke test thành công với `--pk-weighting count`, `--sym-weight 1.0`, `--extra-data` và đánh giá `lfwval` qua student gallery encoder trên CPU. Không để lại file rác ngoài thư mục tạm.

## 2026-11-18. Claude verify lại §7 bằng diff
- Đọc diff `train_student.py`, `eval_certified.py`, `prepare_faces.py`. Đạt cả 5 mục: InfoNCE gán `-inf` ở cột `i ^ 1` chỉ khi `sym_weight > 0` (đường chéo vẫn là đích); `SystemExit` cho `--init-ckpt` không tồn tại, sym không có danh tính đủ điều kiện (không còn nhánh lùi DataLoader ngẫu nhiên), `--gallery-encoder student` thiếu `--student-ckpt` ở cả 3 lệnh, `prepare_faces` lỗi gallery / thiếu meta / images-root rỗng; loại thêm tên LFW train, `report.json` có `excluded_by_source`; `--pk-weighting count` chọn P danh tính không lặp với xác suất tỉ lệ số ảnh, mặc định `uniform`; `_load_training_shards` kiểm thiếu shard, số dòng, `paths` từng dòng.
- Còn lại (không chặn): `prepare_faces` bỏ qua im lặng khi thư mục gallery hoặc `lfw_crops` không tồn tại; runbook v3 bắt bằng kiểm `excluded_by_source` = 0/3/25.
- Không chạy pytest (python3 hệ thống không có torch); dựa vào kết quả pytest owner/agent 29/29. Không thấy run rác mới trong `results/innovation/`.
- Kết luận: đạt để chạy. Viết `runbook_vast_v3.md`.

## 2026-11-19. Claude kiểm run test 20261119_064729_certified_eval
- Đọc `summary.json` và `frames.csv` (9 750 dòng). Config đúng cấu hình đã chọn (symmetric, student `20261118_135729`, liveness bật, không thresholds). Đủ 10 tập x 75 frame. Nhánh none khớp số tấn công cũ (H09 86.7/69.3/80.0, ADA_r48 44%), nên tài sản tấn công và pipeline đúng.
- Kết quả và phân tích lệch landmark ghi ở `experiment_log.md` 2026-11-19.
