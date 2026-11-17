# Runbook: chạy tối ưu tấn công thích ứng trên vast.ai (2026-11-12)

> **Lỗi thời về cách chạy (2026-11-17):** các lệnh `nohup ... &` / `tail -f` trong file này không dùng nữa. Trên vast mọi lệnh chạy tiền cảnh, xem quy tắc ở đầu `runbook_vast_v2.md`.

> **Đã thay bằng `runbook_vast_adaptive_v2.md` (2026-11-13).** Bản này là v1, dùng cho bộ run 11-12 đã huỷ (xem `experiment_log.md` 2026-11-13).

Chỉ phần **optimize** chạy trên vast.ai. Phần **eval** chạy trên laptop (cùng môi trường với mọi eval khác: liveness, enhancer, MediaPipe).

## 0. Không giả định máy cũ còn gì
Instance train purifier (nếu chưa huỷ) chỉ có: torch, `innovation/` + `backend/` **bản cũ**, `checkpoints/method3_best.pth`, `lfw_crops`. Thiếu: `liveness/`, `emotion/`, `extreme_cond_enhancer/`, `adaptive.py` mới, frame owner, gallery, band/ring, checkpoint purifier, MediaPipe. → Làm theo đủ các bước dưới, kể cả khi dùng lại instance cũ (bước 3 ghi đè code).

## 1. Thuê máy
- GPU: RTX 3060 / 3070 / 3080 / A4000 trở lên, VRAM ≥ 8 GB. RAM ≥ 16 GB. Ổ đĩa ≥ 20 GB.
- Image: PyTorch có CUDA 12.x, **Python 3.10–3.12** (mediapipe 0.10.14 không có bản cho 3.13).

## 2. Đóng gói trên laptop (từ thư mục gốc `code-inno`, env `(test)`)
Commit code trước (để biết run dùng commit nào), rồi:
```
tar --exclude=__pycache__ -czf adaptive_pkg.tgz backend liveness emotion extreme_cond_enhancer innovation checkpoints/method3_best.pth data/innovation/frames/owner data/innovation/gallery data/innovation/defense/band_template.png data/innovation/defense/band_meta.json data/innovation/defense/ring_meta.json data/innovation/defense/ring_r8.png data/innovation/defense/ring_r16.png data/innovation/defense/ring_r32.png data/innovation/defense/ring_r48.png results/innovation/20261112_004047_purifier_train/best.pt
```
Vì sao từng thứ:
| Mục | Lý do |
|---|---|
| `backend/`, `liveness/`, `emotion/` | `faces.py` import `backend.processing`, file này import liveness, emotion, head_pose ngay khi nạp (không chạy liveness nhưng phải import được) |
| `extreme_cond_enhancer/` | Thiếu thì enhancer **tắt im lặng** → không loại được frame thiếu sáng → dữ liệu train khác laptop |
| `innovation/` | code optimize (bản mới có `adaptive.py`) |
| `checkpoints/method3_best.pth` | recognizer |
| `data/innovation/frames/owner` (cả s1–s6) | optimize kiểm các session eval s3–s6 phải tồn tại; chỉ s1, s2 dùng để tối ưu |
| `data/innovation/gallery` | gallery 1:N (owner, LFW_victim, ...) |
| band + ring | vùng gọng và vành |
| `.../20261112_004047_purifier_train/best.pt` | purifier seed 0 (mặc định của adaptive) |
Không cần: `lfw_crops` (406 MB), các checkpoint khác, `attendance.db`.

## 3. Trên vast.ai
```
cd /workspace && mkdir -p code-inno && tar -xzf adaptive_pkg.tgz -C code-inno && cd code-inno
apt-get update && apt-get install -y libgl1 libglib2.0-0
pip install "mediapipe==0.10.14" "protobuf>=4.25.3,<5" "numpy==1.26.4" "opencv-contrib-python==4.11.0.86"
python -c "import torch, mediapipe, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), mediapipe.__version__, numpy.__version__, cv2.__version__)"
```
Phải in `True` và mediapipe `0.10.14`. Nếu torch báo lỗi với numpy 1.26.4: báo Claude, không tự đổi bản.

## 4. Chạy thử 5 bước (kiểm môi trường)
```
python -m innovation.defense.adaptive optimize --ring-px 8 --steps 5 --device cuda
python -c "import json,glob;d=sorted(glob.glob('results/innovation/*_adaptive_optimize'))[-1];a=json.load(open(d+'/attack.json'));print(d,a['prepare_report'])"
```
**Phải ra `used 78`, `enhancer_applied 42`, `no_face 0`, `no_mesh 0`** (đúng như run H09 trên laptop). Khác → dừng, báo Claude. Đúng → xoá thư mục chạy thử vừa in ra (`rm -rf <thư mục đó>`), để khỏi lẫn với run thật.

## 5. Chạy thật (4 độ rộng, nối tiếp, không chết khi mất kết nối)
```
nohup bash -c 'for r in 8 16 32 48; do python -m innovation.defense.adaptive optimize --ring-px $r --device cuda; done' > adaptive_log.txt 2>&1 &
tail -f adaptive_log.txt
```
Ước lượng (chưa đo): 5–15 phút mỗi r. Xem cột `seconds` trong log sau vài trăm bước để biết thực tế.

## 6. Kiểm và mang về
```
python -c "import json,glob;[print(d,json.load(open(d+'/attack.json')).get('ring_px')) for d in sorted(glob.glob('results/innovation/*_adaptive_optimize'))]"
tar -czf adaptive_runs.tgz results/innovation/*_adaptive_optimize adaptive_log.txt
```
Phải đúng 4 thư mục, ring_px 8, 16, 32, 48. Tải `adaptive_runs.tgz` về, giải nén **tại thư mục gốc `code-inno`** trên laptop (để thành `results/innovation/<run>`). Xác nhận đủ 4 thư mục trên laptop rồi mới huỷ instance.

## 7. Eval trên laptop
Sau khi agent xong `plan_adaptive.md` §6. Gửi Claude tên 4 thư mục; Claude kiểm `split.json` + `attack.json` rồi đưa đúng 4 lệnh eval.
