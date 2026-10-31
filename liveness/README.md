# Face Anti-Spoofing (Liveness Detection)

Mô-đun phát hiện khuôn mặt giả mạo (ảnh in, ảnh/video trên màn hình điện thoại) cho hệ thống điểm danh.
Model được huấn luyện trên tập **CelebA-Spoof** (Hugging Face), đầu vào $80 \times 80$, phân loại nhị phân `live` (Real) vs `spoof` (Fake).

## Cấu trúc

| File | Vai trò |
| :--- | :--- |
| `liveness_detector.py` | `LivenessDetector`: MediaPipe face detection → crop đã calibrate → model → `p_live` |
| `fas_model.py` | Kiến trúc: `MultiFTNet` (MiniFASNetV2SE + nhánh Fourier `FTGenerator`) và `MobileNetV3FAS`; `build_model(config["model"])` |
| `fas_common.py` | Nguồn duy nhất cho preprocessing (resize, normalize), crop theo bbox, Fourier target, đọc/validate `liveness_config.json` |
| `temporal.py` | `LivenessSmoother`: làm mượt theo thời gian + hysteresis; `make_smoother(config)` |
| `gates.py` | `face_size_status`: gate khoảng cách theo tỷ lệ bề rộng mặt / khung hình |
| `phone_detector.py` | `face_in_phone` + `PhoneDetector` (YOLOv8, chỉ dùng cho đánh giá offline) |
| `tools/` | Script chuẩn bị dữ liệu, thu thập, đánh giá và demo (xem mục 4) |

Test: `tests/liveness/`. Kết quả đánh giá: `results/liveness/`.

---

## 1. Kiến trúc hỗ trợ

`LivenessDetector` đọc trường `model.arch` trong `liveness_config.json` để dựng đúng kiến trúc rồi load trọng số:

1. **`MultiFTNet/MiniFASNetV2SE` (Run 2, from scratch):** huấn luyện từ đầu, không dùng trọng số pretrained.
   ```json
   "model": {"arch": "MultiFTNet/MiniFASNetV2SE", "num_classes": 2, "embedding_size": 128, "conv6_kernel": [5, 5]}
   ```
2. **`MobileNetV3Large-FT` (Run 3, fine-tune):** transfer learning trên backbone MobileNetV3-Large (ImageNet) kết hợp nhánh phụ Fourier (`FTGenerator`, 40 kênh tại feature map stride 8).
   ```json
   "model": {"arch": "MobileNetV3Large-FT", "num_classes": 2, "pretrained_init": "torchvision IMAGENET1K_V1"}
   ```

Cả hai dùng loss $\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{FT}}$; ngưỡng chọn trên tập validation theo tiêu chí cực tiểu ACER ($\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$).

---

## 2. Checkpoint & cấu hình

Thư mục `checkpoints/` bị gitignore — file `.pth`/`.json` phải chia sẻ thủ công. Mặc định detector đọc:
- `checkpoints/liveness_best.pth`
- `checkpoints/liveness_config.json`

Để đổi giữa Run 2 và Run 3, chỉ cần đặt cặp file tương ứng vào `checkpoints/` (hoặc trỏ qua biến môi trường, xem mục 3).

| Vị trí | Model | Dùng cho |
| :--- | :--- | :--- |
| `checkpoints/liveness_best.pth` + `liveness_config.json` | **Run 3** (`MobileNetV3Large-FT`), có `smoothing` + `gate` | **Demo / hệ thống** (mặc định) |
| `checkpoints/liveness_run3/` | Run 3 (bản gốc, cùng config với root) | Tham chiếu |
| `checkpoints/liveness_run2/` | Run 2 (from scratch), threshold chọn trên val (0.8141) | Bản gốc của report |
| `checkpoints/liveness_run2_live/` | Run 2, threshold calib live 0.10 | Report (from scratch) + baseline cho `tools/offline_eval.py` |

Ví dụ `liveness_config.json`:
```json
{
  "input_size": 80,
  "ft_size": 10,
  "class_names": ["live", "spoof"],
  "live_index": 0,
  "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "color": "RGB"},
  "crop": {"scale_w": 1.6, "scale_h": 1.6, "shift_x": 0.0, "shift_y": 0.0},
  "threshold": 0.50,
  "model": {"arch": "MultiFTNet/MiniFASNetV2SE", "num_classes": 2, "embedding_size": 128, "conv6_kernel": [5, 5]},
  "smoothing": {"enabled": true, "window_s": 1.5, "t_on": 0.20, "t_off": 0.05, "min_frames": 3, "reset_s": 1.0, "jump_ratio": 0.5},
  "gate": {"enabled": true, "min_ratio": 0.22, "max_ratio": 0.70}
}
```
Các key bắt buộc: `input_size`, `ft_size`, `class_names`, `live_index`, `crop`, `threshold`, `model`. `smoothing` và `gate` là tuỳ chọn.

### 2.1 Làm mượt theo thời gian + gate khoảng cách

Khi config có `smoothing.enabled = true`, backend (`backend/main.py`) tạo một `LivenessSmoother` cho mỗi kết nối WebSocket và quyết định theo trạng thái đã làm mượt thay vì từng frame:

- Trung bình `p_live` trong cửa sổ `window_s` giây; chuyển sang `Real` khi trung bình ≥ `t_on`, sang `Fake` khi < `t_off` (hysteresis).
- Trạng thái reset khi mất mặt quá `reset_s` giây hoặc tâm bbox nhảy quá `jump_ratio` × bề rộng mặt trước đó.
- Nhãn trả về: `Real`, `Fake`, `Checking` (chưa đủ `min_frames`), `Move closer` / `Move back` (bề rộng mặt ngoài [`min_ratio`, `max_ratio`] × bề rộng khung hình).
- Emotion / recognition chỉ chạy khi trạng thái là `Real`. `liveness.score` = trung bình đã làm mượt; `liveness.raw_score` = điểm của frame hiện tại.
- Config không có key `smoothing` (ví dụ Run 2) → quyết định từng frame như cũ.

---

## 3. Biến môi trường

| Biến | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `LIVENESS_CHECKPOINT` | `checkpoints/liveness_best.pth` | Đường dẫn trọng số `.pth` |
| `LIVENESS_CONFIG` | `liveness_config.json` cùng thư mục với checkpoint | Đường dẫn file cấu hình |
| `LIVENESS_THRESHOLD` | Giá trị `threshold` trong config | Ghi đè ngưỡng Real/Fake ($[0, 1]$) |

---

## 4. Tools

Chạy từ thư mục gốc của repo (các đường dẫn mặc định như `results/`, `data/`, `checkpoints/` là tương đối so với thư mục gốc). Dependency bổ sung cho tool nằm trong `requirements-dev.txt`.

| Lệnh | Mục đích |
| :--- | :--- |
| `python -m liveness.tools.demo_photo [--save] [--img path]` | Liveness + emotion trên ảnh tĩnh trong `liveness/tools/samples/` |
| `python -m liveness.tools.demo_webcam [--cam 0] [--threshold 0.45]` | Liveness + emotion real-time qua webcam (Q/ESC thoát, S lưu frame) |
| `python -m liveness.tools.prepare_fas_data --out-dir ~/fas_data --n-train-shards 12` | Tải CelebA-Spoof từ Hugging Face, calibrate crop MediaPipe, xuất `train/val/test.npz` + `meta.json` |
| `python -m liveness.tools.capture --person <id>` | Thu thập dữ liệu webcam theo 5 kịch bản (real still/motion, spoof photo close/far, spoof video) vào `data/liveness_capture/` |
| `python -m liveness.tools.live_eval [--checkpoint X.pth]` | Calibrate ngưỡng + đánh giá live 20 trial real / 20 trial spoof → `results/liveness/live/` |
| `python -m liveness.tools.offline_eval --sessions data/liveness_capture --models run3=checkpoints/liveness_best.pth` | Đánh giá offline các policy làm mượt/gate trên dữ liệu đã thu → `results/liveness/offline/` |
| `python -m liveness.tools.fft_probe` | Phân tích đặc trưng tần số cao / Moiré (2D-FFT) trên `frames.csv` của offline eval |

---

## 5. Kiểm thử

```bash
python -m pytest tests/liveness
```
Các test dùng mock, không cần checkpoint hay webcam.
