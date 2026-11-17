# Runbook v2 trên vast.ai: train student + lưới cấu hình dev (plan_certified_v2), sửa 2026-11-17

**Quy tắc bắt buộc:** mọi lệnh trên vast chạy **tiền cảnh** (không `nohup`, không `&` cuối lệnh, không `tail -f`). Mỗi khối lệnh chạy xong sẽ tự trả về dấu nhắc `$`, rồi mới sang khối sau. Log giữ bằng `| tee <file>`. Giữ cửa sổ SSH mở trong lúc chạy.

Mỗi khối ghi rõ **[LAPTOP]** hay **[VAST]**. Lần chạy trước mất kết quả vì lệnh đóng gói kết quả bị chạy trên laptop.

Máy vast.ai **mới, trống**. Không đưa frame test (owner/s6, p01) lên vast.

## 0. Điều kiện
Code commit `820ddcf` (đã có). Gói `v2_pkg.tgz` (525 665 702 byte, 2026-11-17) đã có sẵn trên laptop tại gốc `code-inno`, **không cần đóng gói lại** nếu code không đổi sau `820ddcf`.

**[LAPTOP]** kiểm gói còn khớp code hiện tại:
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
git status --short innovation
certutil -hashfile v2_pkg.tgz SHA256
```
`git status --short innovation` phải không in gì (code không đổi). Ghi lại chuỗi SHA256.

## 1. Thuê máy
GPU VRAM ≥ 8 GB, RAM ≥ 16 GB, disk ≥ 25 GB, template PyTorch CUDA 12.x, Python 3.10–3.12, bật SSH.

## 2. Tải gói lên
**[LAPTOP]** (cổng và IP lấy từ lệnh SSH của instance, dạng `ssh -p 41234 root@203.0.113.7`; số ví dụ là giả):
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
scp -P 41234 v2_pkg.tgz root@203.0.113.7:/workspace/
```

## 3. Kiểm, giải nén, cài
**[VAST]**
```
sha256sum /workspace/v2_pkg.tgz
```
Phải trùng chuỗi `certutil`. Trùng thì:

**[VAST]**
```
cd /workspace && rm -rf code-inno && mkdir -p code-inno && tar -xzf v2_pkg.tgz -C code-inno && cd /workspace/code-inno && ls
```
**[VAST]**
```
apt-get update && apt-get install -y libgl1 libglib2.0-0
```
**[VAST]**
```
pip install "mediapipe==0.10.14" "protobuf>=4.25.3,<5" "numpy==1.26.4" "opencv-contrib-python==4.11.0.86"
```
**[VAST]**
```
cd /workspace/code-inno
python -c "import torch, mediapipe, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), mediapipe.__version__, numpy.__version__, cv2.__version__)"
python -c "import innovation.defense.train_student, innovation.defense.eval_certified; print('import ok')"
python -c "import json;m=json.load(open('data/innovation/defense/maskset/maskset_meta.json'));print(m['landmarks']['l_top_landmark'], m['eye_zone_check']['pass'])"
ls data/innovation/frames/owner
```
Phải ra `True`, `0.10.14`, `1.26.4`, `4.11.0`, `import ok`, `195 True`; `ls` chỉ có `s3_vary s4_vary s5_vary`.

## 4. Chạy thử (vài chục giây)
**[VAST]**
```
cd /workspace/code-inno
python -m innovation.defense.train_student --fill gray --limit 128 --epochs 1 --device cuda --output-dir /workspace/tmp_st
cat /workspace/tmp_st/summary.json
rm -rf /workspace/tmp_st
```

## 5. Train student gray (tiền cảnh, ~15 phút)
**[VAST]**
```
cd /workspace/code-inno
python -m innovation.defense.train_student --fill gray --device cuda 2>&1 | tee student_gray_log.txt
```
Chờ tới dòng `Student training completed: results/innovation/<ts>_student_train` và dấu nhắc `$` hiện lại.

## 6. Train student inpaint (tiền cảnh, ~15 phút)
**[VAST]**
```
cd /workspace/code-inno
python -m innovation.defense.train_student --fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --device cuda 2>&1 | tee student_inpaint_log.txt
```
Chờ dòng `Student training completed: ...` và dấu nhắc `$`.

## 7. Lưới cấu hình dev (tiền cảnh, ~30 phút)
**[VAST]** tạo script (lệnh này chỉ ghi file, xong ngay):
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
```
**[VAST]** chạy (tiền cảnh):
```
cd /workspace/code-inno
bash grid_v2.sh 2>&1 | tee grid_log.txt
cat grid_v2.txt
```
Xong khi dấu nhắc `$` hiện lại và `cat grid_v2.txt` có dòng cuối `DONE` (19 dòng run + 2 dòng SG/SI). Có lỗi thì script dừng (do `set -e`): gửi Claude 30 dòng cuối `grid_log.txt`.

## 8. Đóng gói kết quả — **chạy trên VAST, không phải laptop**
**[VAST]**
```
cd /workspace/code-inno
ls -d results/innovation/*_student_train results/innovation/*_certified_*
tar -czf v2_runs.tgz results/innovation/*_student_train results/innovation/*_certified_lfwval results/innovation/*_certified_eval results/innovation/*_certified_calib student_gray_log.txt student_inpaint_log.txt grid_v2.txt grid_log.txt
ls -l v2_runs.tgz
sha256sum v2_runs.tgz
```
`ls -d` phải liệt kê 2 thư mục `*_student_train` và 17 thư mục `*_certified_*`. Gói phải nặng **vài chục MB** (mỗi `best.pt` của student cỡ 20 MB); nếu chỉ vài trăm KB là sai, dừng lại.

## 9. Mang về
**[LAPTOP]**
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
del v2_runs.tgz
scp -P 41234 root@203.0.113.7:/workspace/code-inno/v2_runs.tgz .
certutil -hashfile v2_runs.tgz SHA256
tar -xzf v2_runs.tgz
dir /b results\innovation | findstr _student_train
```
(`del` xoá gói sai từ lần trước; nếu báo không có file thì bỏ qua.) SHA256 phải trùng `sha256sum` trên vast; `dir` phải in 2 thư mục student. **Chỉ huỷ instance sau khi bước này đạt.** Gửi Claude nội dung `grid_v2.txt` mới.
