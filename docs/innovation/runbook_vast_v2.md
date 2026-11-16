# Runbook v2 trên vast.ai: train student + lưới cấu hình dev (plan_certified_v2), 2026-11-16

Máy vast.ai **mới, trống**. Train student và chạy toàn bộ lưới dev **trên vast** (dev không gate liveness nên chạy được trên GPU). **Không** đưa frame test (owner/s6, p01) lên vast; test chạy một lần trên laptop sau khi Claude chọn cấu hình.

## 0. Điều kiện
Agent xong `plan_certified_v2.md` §7, Claude đã verify; owner commit, gửi hash.

## 1. Thuê máy
GPU VRAM ≥ 8 GB, RAM ≥ 16 GB, disk ≥ 25 GB, template PyTorch CUDA 12.x, Python 3.10–3.12, bật SSH.

## 2. Đóng gói trên laptop (cmd/PowerShell)
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
tar --exclude=__pycache__ -czf v2_pkg.tgz backend liveness emotion extreme_cond_enhancer innovation checkpoints/method3_best.pth data/innovation/defense data/innovation/gallery data/innovation/gallery_people data/innovation/frames/owner/s3_vary data/innovation/frames/owner/s4_vary data/innovation/frames/owner/s5_vary results/innovation/20261115_144634_inpainter_train/best.pt
tar -tzf v2_pkg.tgz | findstr /i "train_student.py eval_certified.py method3_best.pth lfw_crops/val_001.npz lfw_landmarks/val_001.npz maskset/L.png gallery_people/p01 s5_vary/0074.jpg 20261115_144634_inpainter_train/best.pt"
certutil -hashfile v2_pkg.tgz SHA256
```
Lệnh `findstr` phải in đủ các mục: `innovation/defense/train_student.py`, `innovation/defense/eval_certified.py`, `checkpoints/method3_best.pth`, `.../lfw_crops/val_001.npz`, `.../lfw_landmarks/val_001.npz`, `.../maskset/L.png`, ít nhất một dòng `data/innovation/gallery_people/p01/...`, `.../s5_vary/0074.jpg`, `results/innovation/20261115_144634_inpainter_train/best.pt`. Ghi lại chuỗi SHA256. Không được thấy `s6_vary` hay `frames/p01`.

| Thứ | Trên vast (`/workspace/code-inno/...`) | Dùng cho |
|---|---|---|
| backend, liveness, emotion, extreme_cond_enhancer, innovation | thư mục cùng tên | code (liveness chỉ cần import được) |
| `checkpoints/method3_best.pth` | như bên trái | teacher, nhánh none |
| `data/innovation/defense/` (lfw_crops, lfw_landmarks, maskset, band) | như bên trái | train student, lfwval, calibrate, render mask |
| `data/innovation/gallery`, `data/innovation/gallery_people` | như bên trái | gallery dev |
| `data/innovation/frames/owner/s3_vary, s4_vary, s5_vary` | như bên trái | dev owner |
| inpainter v1 `results/innovation/20261115_144634_inpainter_train/best.pt` | như bên trái | fill inpaint |

## 3. Đưa lên vast (scp, không dùng Jupyter cho file lớn)
Lấy cổng và IP từ lệnh SSH của instance (dạng `ssh -p 41234 root@203.0.113.7`, số ví dụ là giả). Trên laptop tại `code-inno`:
```
scp -P 41234 v2_pkg.tgz root@203.0.113.7:/workspace/
```

## 4. Trên vast: kiểm, giải nén, cài
```
sha256sum /workspace/v2_pkg.tgz
```
Phải **trùng** chuỗi `certutil`. Trùng thì:
```
cd /workspace && rm -rf code-inno && mkdir -p code-inno && tar -xzf v2_pkg.tgz -C code-inno && cd /workspace/code-inno
apt-get update && apt-get install -y libgl1 libglib2.0-0
pip install "mediapipe==0.10.14" "protobuf>=4.25.3,<5" "numpy==1.26.4" "opencv-contrib-python==4.11.0.86"
python -c "import torch, mediapipe, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), mediapipe.__version__, numpy.__version__, cv2.__version__)"
python -c "import innovation.defense.train_student, innovation.defense.eval_certified; print('import ok')"
python -c "import json;m=json.load(open('data/innovation/defense/maskset/maskset_meta.json'));print(m['landmarks']['l_top_landmark'], m['eye_zone_check']['pass'])"
ls data/innovation/frames/owner
```
Phải ra `True`, `0.10.14`, `1.26.4`, `4.11.0`, `import ok`, `195 True`, và `ls` chỉ có `s3_vary s4_vary s5_vary`.

## 5. Chạy thử
```
cd /workspace/code-inno
python -m innovation.defense.train_student --fill gray --limit 128 --epochs 1 --device cuda --output-dir /workspace/tmp_st && cat /workspace/tmp_st/summary.json && rm -rf /workspace/tmp_st
```

## 6. Train hai student (nối tiếp, chạy nền)
```
cd /workspace/code-inno
nohup bash -c 'python -m innovation.defense.train_student --fill gray --device cuda; python -m innovation.defense.train_student --fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --device cuda' > student_log.txt 2>&1 &
tail -f student_log.txt
```
Xong khi log có hai dòng `Student training completed: ...`.

## 7. Lưới cấu hình dev (chạy nền, tự ghi bảng tên run)
Tạo file `grid_v2.sh` rồi chạy:
```
cd /workspace/code-inno
cat > grid_v2.sh <<'SH'
set -e
INP=results/innovation/20261115_144634_inpainter_train/best.pt
latest() { ls -dt results/innovation/*_$1 | head -1; }
st_by_fill() { for d in $(ls -dt results/innovation/*_student_train); do python -c "import json,sys;sys.exit(0 if json.load(open('$d/meta.json'))['args']['fill']=='$1' else 1)" && { echo $d; return; }; done; }
SG=$(st_by_fill gray); SI=$(st_by_fill inpaint)
echo "SG=$SG" > grid_v2.txt; echo "SI=$SI" >> grid_v2.txt
ev() { name=$1; shift; python -m innovation.defense.eval_certified lfwval --val-part B --inpainter-ckpt $INP --device cuda "$@"; echo "$name lfwB $(latest certified_lfwval)" >> grid_v2.txt; python -m innovation.defense.eval_certified run --split dev --inpainter-ckpt $INP --device cuda "$@"; echo "$name dev $(latest certified_eval)" >> grid_v2.txt; }
cal() { name=$1; shift; python -m innovation.defense.eval_certified calibrate --inpainter-ckpt $INP --device cuda "$@"; echo "$name calib $(latest certified_calib)" >> grid_v2.txt; }
ev C0 --gallery-mode symmetric
ev C1 --gallery-mode clean
cal C2 --gallery-mode clean;                      ev C2 --gallery-mode clean --thresholds $(latest certified_calib)/thresholds.json
ev C3 --gallery-mode clean --student-ckpt $SG/best.pt
ev C4 --gallery-mode clean --student-ckpt $SI/best.pt
cal C5g --gallery-mode clean --student-ckpt $SG/best.pt; ev C5g --gallery-mode clean --student-ckpt $SG/best.pt --thresholds $(latest certified_calib)/thresholds.json
cal C5i --gallery-mode clean --student-ckpt $SI/best.pt; ev C5i --gallery-mode clean --student-ckpt $SI/best.pt --thresholds $(latest certified_calib)/thresholds.json
echo DONE >> grid_v2.txt
SH
nohup bash grid_v2.sh > grid_log.txt 2>&1 &
tail -f grid_v2.txt
```
(C0 = cấu hình v1 đo lại bằng `gar_recog` trên LFW B để so; C5g/C5i tính cả hai, Claude áp luật chọn đúng một theo C3/C4.) Xong khi `grid_v2.txt` có dòng `DONE`. Lỗi thì xem `grid_log.txt`, gửi Claude.

## 8. Mang về
```
cd /workspace/code-inno
tar -czf v2_runs.tgz results/innovation/*_student_train results/innovation/*_certified_lfwval results/innovation/*_certified_eval results/innovation/*_certified_calib student_log.txt grid_v2.txt grid_log.txt
sha256sum v2_runs.tgz
```
Trên laptop tại `code-inno`: `scp -P 41234 root@203.0.113.7:/workspace/code-inno/v2_runs.tgz .`, kiểm `certutil -hashfile v2_runs.tgz SHA256` trùng, rồi `tar -xzf v2_runs.tgz`. Gửi Claude nội dung `grid_v2.txt`. Claude áp luật chọn cấu hình + S rồi đưa lệnh test (laptop, `--liveness`, một lần).
