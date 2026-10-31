# Face Attendance — Facial Recognition with Emotion & Liveness

Hệ thống điểm danh bằng khuôn mặt qua webcam (COS30082 Applied Machine Learning). Mỗi frame được xử lý theo thứ tự:

1. **Anti-spoofing / liveness** (`liveness/`): phát hiện mặt bằng MediaPipe, phân loại Real / Fake, làm mượt theo thời gian.
2. Chỉ khi mặt là **Real** mới chạy tiếp **emotion** (`emotion/`) và **face recognition** (`backend/face_recognizer.py`, embedding 512 chiều + cosine similarity).
3. Nhận diện khớp thì điểm danh được ghi vào SQLite (`attendance.db`). Người chưa đăng ký có thể đăng ký 5 góc mặt ở trang `/register`.

Đề bài: [`docs/project_requirements.pdf`](docs/project_requirements.pdf).

## Cấu trúc

```
.
├── backend/                 # FastAPI: WebSocket /ws/attendance, /ws/register + REST API, SQLite
├── frontend/                # Next.js UI (trang /attendance, /register)
├── liveness/                # Face anti-spoofing (module chính) — xem liveness/README.md
│   └── tools/               # chuẩn bị dữ liệu, thu thập, đánh giá, demo
├── emotion/                 # Nhận diện cảm xúc (EmotionCNN, 7 lớp)
├── extreme_cond_enhancer/   # Tăng sáng/giảm sáng crop mặt trước emotion + recognition (tuỳ chọn)
├── tests/                   # pytest: tests/liveness, tests/extreme_cond_enhancer
├── results/                 # Kết quả đánh giá: face_recognition/, liveness/, emotion/
├── docs/                    # Đề bài
├── requirements.txt         # Dependency runtime
└── requirements-dev.txt     # + pytest và dependency cho liveness/tools
```

`checkpoints/`, `data/`, `dataset/` và `attendance.db` bị gitignore và phải chia sẻ riêng.

## Checkpoint cần có

| File | Dùng bởi | Ghi đè bằng biến môi trường |
| :--- | :--- | :--- |
| `checkpoints/liveness_best.pth` + `checkpoints/liveness_config.json` | `LivenessDetector` (bắt buộc) | `LIVENESS_CHECKPOINT`, `LIVENESS_CONFIG`, `LIVENESS_THRESHOLD` |
| `checkpoints/emotion/emotion_best.pt` (hoặc `.pth`) | `EmotionDetector` | `EMOTION_CHECKPOINT` |
| `checkpoints/method3_best.pth` | `FaceRecognizer` (FaceCNN + ArcFace) | `CHECKPOINT_PATH`, `RECOGNITION_THRESHOLD` (mặc định 0.5) |

Thiếu checkpoint face recognition thì backend vẫn chạy nhưng dùng trọng số ngẫu nhiên (có cảnh báo). Thiếu checkpoint liveness thì backend không xử lý được frame.

## Cài đặt & chạy

Yêu cầu Python 3.10–3.12 và Node.js 20+.

```bash
pip install -r requirements.txt
```

```bash
cd frontend && npm install
```

Chạy backend (từ thư mục gốc repo):

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Chạy frontend (terminal khác):

```bash
cd frontend && npm run dev
```

Mở `http://localhost:3000`.

Biến tuỳ chọn: `TORCH_THREADS` giới hạn số thread CPU của torch. Trên laptop Windows có P/E-core, đặt `TORCH_THREADS=1` có thể nhanh hơn.

## Kiểm thử

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

Các demo và tool đánh giá liveness được mô tả trong [`liveness/README.md`](liveness/README.md).
