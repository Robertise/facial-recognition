# Face Anti-Spoofing (Liveness Detection) — MultiFTNet from Scratch

Mô-đun nhận diện khuôn mặt giả mạo (Anti-Spoofing / Liveness Detection) được huấn luyện từ đầu (**from scratch**, không sử dụng bất kỳ trọng số pretrained nào) trên tập dữ liệu **CelebA-Spoof** (Hugging Face).

Kiến trúc sử dụng **MultiFTNet** (backbone `MiniFASNetV2SE` + nhánh giám sát phụ trợ Fourier Transform `FTGenerator`), với đầu vào $80 \times 80$, phân loại nhị phân `live` (Real) vs `spoof` (Fake).

---

## 1. Chuẩn bị & Huấn luyện trên vast.ai (GPU)

### 1.1 Cấu hình instance đề xuất
- **GPU:** 1x RTX 3090 / 4090 (VRAM $\ge 16$ GB).
- **Disk:** $\ge 50$ GB (lưu cache Parquet và file `.npz`).
- **OS/Image:** Ubuntu, Python 3.10 – 3.12, PyTorch $\ge 2.0$.

### 1.2 Cài đặt thư viện
```bash
pip install huggingface_hub "pyarrow<18" mediapipe==0.10.14 "protobuf>=4.25.3,<5" opencv-python-headless scikit-learn matplotlib tqdm jupyter
```

### 1.3 Tiền xử lý dữ liệu (Tải shard HF & Calibrate crop)
```bash
# Tải 12 shard train (~6 GB) + 10 shard test (~5 GB), calibrate crop và lưu vào ~/fas_data
python utils/prepare_fas_data.py --out-dir ~/fas_data --n-train-shards 12
```
Quá trình này sẽ:
1. Tải và đọc Parquet trực tiếp theo streaming batch.
2. Tự động xác minh schema và ánh xạ nhãn ($0 \leftrightarrow \text{live}$, $1 \leftrightarrow \text{spoof}$).
3. Chạy MediaPipe Face Detection trên 2,000 ảnh thật để **calibrate crop scale & shift** (khớp tỷ lệ crop giữa lúc train và webcam inference).
4. Lưu các tệp: `~/fas_data/train.npz`, `~/fas_data/val.npz`, `~/fas_data/test.npz`, `~/fas_data/meta.json`, `calib_hist.png`, `samples.png`.

### 1.4 Huấn luyện mô hình
Chạy notebook không cần mở giao diện đồ họa:
```bash
jupyter nbconvert --to notebook --execute training/training_liveness.ipynb --output training_liveness_run.ipynb
```
Notebook sẽ:
1. Huấn luyện MultiFTNet 30 epochs (AdamW, Cosine Annealing, loss cân bằng $\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{FT}}$).
2. Tự động chọn ngưỡng tối ưu trên tập Validation theo tiêu chí cực tiểu hóa ACER ($\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$).
3. Đánh giá trên toàn bộ tập Test, vẽ biểu đồ ROC, ma trận nhầm lẫn và lưu kết quả tại `results/`.
4. Xuất trọng số `checkpoints/liveness_best.pth` và cấu hình `checkpoints/liveness_config.json`.

---

## 2. Tích hợp Checkpoint về máy Local (Windows)

Sau khi huấn luyện trên vast.ai hoàn tất, tải 2 tệp về máy và đặt vào thư mục `checkpoints/`:
- `checkpoints/liveness_best.pth`
- `checkpoints/liveness_config.json`

Cấu trúc file `liveness_config.json`:
```json
{
  "input_size": 80,
  "ft_size": 10,
  "class_names": ["live", "spoof"],
  "live_index": 0,
  "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "color": "RGB"},
  "crop": {"scale_w": 1.6, "scale_h": 1.6, "shift_x": 0.0, "shift_y": 0.0},
  "threshold": 0.50,
  "model": {"arch": "MultiFTNet/MiniFASNetV2SE", "num_classes": 2, "embedding_size": 128, "conv6_kernel": [5, 5]}
}
```

### 2.1 Kiến trúc hỗ trợ & Cách chọn Checkpoint

Hệ thống hỗ trợ 2 kiến trúc liveness qua hàm `build_model(config["model"])`:

1. **`MultiFTNet/MiniFASNetV2SE` (Run 2 - From Scratch):**
   - Huấn luyện từ đầu không dùng trọng số pretrained.
   - Trường cấu hình `model`:
     ```json
     "model": {
       "arch": "MultiFTNet/MiniFASNetV2SE",
       "num_classes": 2,
       "embedding_size": 128,
       "conv6_kernel": [5, 5]
     }
     ```

2. **`MobileNetV3Large-FT` (Run 3 - Fine-Tune):**
   - Transfer learning: fine-tune toàn bộ trên backbone MobileNetV3-Large pretrained ImageNet kết hợp nhánh phụ Fourier Transform (`FTGenerator`, 40 kênh tại feature map stride 8).
   - Trường cấu hình `model`:
     ```json
     "model": {
       "arch": "MobileNetV3Large-FT",
       "num_classes": 2,
       "pretrained_init": "torchvision IMAGENET1K_V1"
     }
     ```

**Cách chọn checkpoint:**
`LivenessDetector` tự động đọc trường `model.arch` trong `liveness_config.json` để dựng đúng kiến trúc model tương ứng và load trọng số. Để đổi checkpoint (giữa Run 2 và Run 3), chỉ cần đặt cặp file `liveness_best.pth` và `liveness_config.json` tương ứng vào `checkpoints/` (hoặc cấu hình đường dẫn qua biến môi trường `LIVENESS_CHECKPOINT` và `LIVENESS_CONFIG`).

### 2.2 Checkpoint mặc định hiện tại (từ 2026-10-25)

| Vị trí | Model | Dùng cho |
| :--- | :--- | :--- |
| `checkpoints/liveness_best.pth` + `liveness_config.json` | **Run 3** (`MobileNetV3Large-FT`), có `smoothing` + `gate` | **Demo / hệ thống** (mặc định, không cần biến môi trường) |
| `checkpoints/liveness_run3/` | Run 3 (bản gốc, cùng config với root) | Tham chiếu |
| `checkpoints/liveness_run2/` | Run 2 (from scratch), threshold chọn trên val (0.8141) | Bản gốc của report |
| `checkpoints/liveness_run2_live/` | Run 2, threshold calib live 0.10 | **Report** (from scratch) + baseline cho `utils/liveness_offline_eval.py` |

Thư mục `checkpoints/` bị gitignore — file `.pth`/`.json` phải chia sẻ thủ công.

### 2.3 Làm mượt theo thời gian + gate khoảng cách

Khi config có `smoothing.enabled = true`, backend (`backend/main.py`) tạo một `LivenessSmoother` (`liveness/temporal.py`) cho mỗi kết nối WebSocket và quyết định theo trạng thái đã làm mượt thay vì từng frame:

```json
"smoothing": {"enabled": true, "window_s": 1.5, "t_on": 0.20, "t_off": 0.05, "min_frames": 3, "reset_s": 1.0, "jump_ratio": 0.5},
"gate": {"enabled": true, "min_ratio": 0.22, "max_ratio": 0.70}
```

- Trung bình `p_live` trong cửa sổ `window_s` giây; chuyển sang `Real` khi trung bình ≥ `t_on`, sang `Fake` khi < `t_off` (hysteresis).
- Nhãn trả về: `Real`, `Fake`, `Checking` (chưa đủ `min_frames`), `Lại gần hơn` / `Lùi ra xa` (bề rộng mặt ngoài [`min_ratio`, `max_ratio`] × bề rộng khung hình).
- Emotion / recognition chỉ chạy khi trạng thái là `Real`. `liveness.score` = trung bình đã làm mượt; `liveness.raw_score` = điểm của frame hiện tại.
- Config không có key `smoothing` (ví dụ run 2) → hành vi cũ, quyết định từng frame.
- Số liệu, cấu hình bị loại và limitation (ngược sáng, ảnh tĩnh ở một số góc): xem `plans/liveness_stability_decisions.md` (ngoài repo).

---

## 3. Biến môi trường (Environment Variables)

Mô-đun `LivenessDetector` hỗ trợ các biến môi trường sau:

| Biến môi trường | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `LIVENESS_CHECKPOINT` | `checkpoints/liveness_best.pth` | Đường dẫn tệp trọng số PyTorch `.pth` |
| `LIVENESS_CONFIG` | Cùng thư mục với checkpoint (`liveness_config.json`) | Đường dẫn tệp cấu hình JSON |
| `LIVENESS_THRESHOLD` | Giá trị trong `liveness_config.json` | Ghi đè ngưỡng quyết định Real/Fake ($[0.0, 1.0]$) |
| `FAS_DATA_DIR` | `~/fas_data` | Thư mục chứa dữ liệu npz khi huấn luyện |
| `FAS_SMOKE` | Không đặt (hoặc `0`) | Đặt `1` để chạy smoke test nhanh trên CPU (1 epoch, $\le 512$ mẫu) |

---

## 4. Kiểm tra & Chạy thử nghiệm

### Kiểm tra ảnh tĩnh:
```bash
python test_liveness_photo.py --save
```

### Kiểm tra thời gian thực với Webcam:
```bash
python test_liveness_webcam.py --cam 0
```
(Nếu muốn ghi đè ngưỡng trực tiếp từ dòng lệnh: `python test_liveness_webcam.py --cam 0 --threshold 0.45`).

### Khởi động toàn bộ hệ thống Điểm danh (Backend + Frontend):
```bash
# Windows PowerShell:
python -m uvicorn backend.main:app --reload --port 8000
# Tab khác:
cd frontend && npm run dev
```
Truy cập: `http://localhost:3000/attendance`
