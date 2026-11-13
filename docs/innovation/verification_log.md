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
