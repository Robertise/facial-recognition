# Runbook: train inpainter nhiều mask trên vast.ai (plan_certified), 2026-11-14

Máy vast.ai **mới, trống**. Chỉ phần **train** chạy trên vast. Landmark, dựng mask và mọi eval chạy trên laptop.

## 0. Điều kiện
1. Agent xong `plan_certified.md` §8 + §9, Claude đã verify. Owner commit, gửi Claude hash.
2. Trên laptop đã chạy xong (Claude đã kiểm):
   - `python -m innovation.defense.prepare_landmarks` → `data/innovation/defense/lfw_landmarks/` có 12 shard `train_000..011.npz`, 2 shard `val_000..001.npz`, `report.json`.
   - `python -m innovation.defense.maskset build` → `data/innovation/defense/maskset/` có `G0.png G16.png G32.png G48.png L.png F.png`, `region_*.png`, `maskset_meta.json`, `maskset_preview.png`. **Claude xem preview và meta trước khi train.**

## 1. Thuê máy
GPU VRAM ≥ 8 GB, RAM ≥ 16 GB, disk ≥ 20 GB. Template PyTorch CUDA 12.x, Python 3.10–3.12. Bật SSH (và Jupyter nếu muốn upload bằng trình duyệt).

## 2. Đóng gói trên laptop (cmd hoặc PowerShell, không cần env)
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
tar --exclude=__pycache__ -czf inpainter_pkg.tgz backend liveness emotion extreme_cond_enhancer innovation checkpoints/method3_best.pth data/innovation/defense/lfw_crops data/innovation/defense/lfw_landmarks data/innovation/defense/maskset
tar -tzf inpainter_pkg.tgz | findstr /i "train_inpainter.py method3_best.pth lfw_crops/train_011.npz lfw_crops/val_001.npz lfw_landmarks/train_011.npz lfw_landmarks/val_001.npz maskset/F.png maskset/G48.png"
```
Lệnh kiểm phải in đủ 8 dòng: `innovation/defense/train_inpainter.py`, `checkpoints/method3_best.pth`, `data/innovation/defense/lfw_crops/train_011.npz`, `.../lfw_crops/val_001.npz`, `.../lfw_landmarks/train_011.npz`, `.../lfw_landmarks/val_001.npz`, `.../maskset/F.png`, `.../maskset/G48.png`. Gói khoảng 450–500 MB (lfw_crops chiếm phần lớn).

| Thứ | Đường dẫn trên vast | Vì sao |
|---|---|---|
| backend, liveness, emotion, enhancer | `/workspace/code-inno/{backend,liveness,emotion,extreme_cond_enhancer}/` | `FaceRecognizer`; `faces.py` (qua `maskset.py`) import `backend.processing`, file này import liveness, emotion |
| code innovation | `/workspace/code-inno/innovation/` | `train_inpainter.py`, `maskset.py`, `purifier.py` |
| method3 | `/workspace/code-inno/checkpoints/method3_best.pth` | identity loss |
| crop LFW (14 shard) | `/workspace/code-inno/data/innovation/defense/lfw_crops/` | ảnh train/val |
| landmark LFW (14 shard) | `/workspace/code-inno/data/innovation/defense/lfw_landmarks/` | mắt + window để render mask |
| tập mask | `/workspace/code-inno/data/innovation/defense/maskset/` | 6 mask canvas |

Không cần: frame owner/p01, gallery, ảnh LFW gốc, kết quả cũ.

## 3. Đưa gói lên `/workspace/`
Cách A: Jupyter của instance → thư mục `/workspace` → Upload `inpainter_pkg.tgz`.
Cách B: lấy cổng và IP từ lệnh SSH ở nút Connect của instance (dạng `ssh -p 41234 root@203.0.113.7`, số ví dụ là giả), rồi trên laptop tại thư mục `code-inno`:
```
scp -P 41234 inpainter_pkg.tgz root@203.0.113.7:/workspace/
```

## 4. Trên vast: giải nén, cài, kiểm
```
cd /workspace && mkdir -p code-inno && tar -xzf inpainter_pkg.tgz -C code-inno && cd /workspace/code-inno
apt-get update && apt-get install -y libgl1 libglib2.0-0
pip install "mediapipe==0.10.14" "protobuf>=4.25.3,<5" "numpy==1.26.4" "opencv-contrib-python==4.11.0.86"
python -c "import torch, mediapipe, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), mediapipe.__version__, numpy.__version__, cv2.__version__)"
python -c "import innovation.defense.train_inpainter; print('import ok')"
python - <<'PY'
from pathlib import Path
import numpy as np
c = sorted(p.name for p in Path("data/innovation/defense/lfw_crops").glob("*.npz"))
l = sorted(p.name for p in Path("data/innovation/defense/lfw_landmarks").glob("*.npz"))
print("crops", len(c), "landmarks", len(l), "same names:", c == l)
print("ok images:", sum(int(np.load("data/innovation/defense/lfw_landmarks/" + n)["ok"].sum()) for n in l))
print("masks:", sorted(p.name for p in Path("data/innovation/defense/maskset").glob("*.png") if not p.name.startswith(("region_", "maskset"))))
PY
```
Phải ra: `True`, `0.10.14`, `1.26.4`, `4.11.0`, `import ok`; `crops 14 landmarks 14 same names: True`; số ok khớp `report.json` trên laptop; `masks: ['F.png', 'G0.png', 'G16.png', 'G32.png', 'G48.png', 'L.png']`.

## 5. Chạy thử
```
cd /workspace/code-inno
python -m innovation.defense.train_inpainter --limit 256 --epochs 1 --device cuda --output-dir /workspace/tmp_inp
cat /workspace/tmp_inp/summary.json
rm -rf /workspace/tmp_inp
```
Phải có `summary.json` với `val_cos_per_mask` đủ 6 mask. Lỗi thì dừng, báo Claude.

## 6. Chạy thật
```
cd /workspace/code-inno
nohup python -m innovation.defense.train_inpainter --device cuda > inpainter_log.txt 2>&1 &
tail -f inpainter_log.txt
```
Mỗi epoch in một dòng. `Ctrl+C` chỉ thoát `tail`. Xong khi log in `Inpainter training completed: results/innovation/<ts>_inpainter_train`.

## 7. Kiểm và mang về
```
cd /workspace/code-inno
ls results/innovation/*_inpainter_train
cat results/innovation/*_inpainter_train/summary.json
tar -czf inpainter_run.tgz results/innovation/*_inpainter_train inpainter_log.txt
```
Thư mục phải có `best.pt`, `train_log.csv` (30 dòng), `meta.json`, `summary.json`. Tải `inpainter_run.tgz` về (Jupyter Download, hoặc trên laptop tại `code-inno`: `scp -P 41234 root@203.0.113.7:/workspace/code-inno/inpainter_run.tgz .`), rồi tại gốc `code-inno`:
```
tar -xzf inpainter_run.tgz
dir /b results\innovation | findstr inpainter_train
```
Thấy thư mục rồi mới huỷ instance. Gửi Claude tên thư mục; Claude kiểm rồi đưa lệnh eval trên laptop (`lfwval`, `run --split dev`, chọn S, `run --split test`).
