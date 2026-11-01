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

