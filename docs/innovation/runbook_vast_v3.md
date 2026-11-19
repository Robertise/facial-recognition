# Runbook v3 trên vast.ai: dựng VGGFace2 test, train student C8, lưới dev C7/C8/C9 (plan_certified_v3), 2026-11-18

**Quy tắc bắt buộc:** mọi lệnh trên vast chạy **tiền cảnh** (không `nohup`, không `&` cuối lệnh, không `tail -f`). Mỗi khối chạy xong tự trả dấu nhắc `$` rồi mới sang khối sau. Log giữ bằng `2>&1 | tee <file>`. Giữ cửa sổ SSH mở suốt lúc chạy.

Mỗi khối ghi rõ **[LAPTOP]** (cmd Windows) hay **[VAST]**. Máy vast **mới, trống**: không giả định có sẵn gì. Không đưa frame test (owner/s6, p01) lên vast.

Thời gian ước tính: tải gói + VGGFace2 ~15 phút, `prepare_faces` ~10–20 phút, train C8 ~15–25 phút, lưới dev ~20 phút.

## 0. Đóng gói trên laptop
Code v3 nằm ở working tree (đã verify 18/11, chưa commit). Nếu muốn có hash để ghi log, commit trước bước này (owner tự làm) rồi ghi hash; `meta.json` trên vast luôn có `git_commit` rỗng.

**[LAPTOP]**
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
dir data\innovation\frames\owner
dir results\innovation\20261117_123128_student_train
dir data\vggface2\identity_meta.csv
```
Phải thấy `s3_vary s4_vary s5_vary` (và các session khác, sẽ không đóng gói), thư mục student có `best.pt meta.json summary.json train_log.csv`, và file `identity_meta.csv`.

**[LAPTOP]** đóng gói (giống `v2_pkg.tgz` cộng `identity_meta.csv` và student v2; khoảng 560 MB, chạy 1–3 phút):
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
tar -czf v3_pkg.tgz --exclude=__pycache__ backend emotion extreme_cond_enhancer innovation liveness checkpoints/method3_best.pth data/innovation/defense data/innovation/gallery data/innovation/gallery_people data/innovation/frames/owner/s3_vary data/innovation/frames/owner/s4_vary data/innovation/frames/owner/s5_vary data/vggface2/identity_meta.csv results/innovation/20261115_144634_inpainter_train/best.pt results/innovation/20261117_123128_student_train
dir v3_pkg.tgz
certutil -hashfile v3_pkg.tgz SHA256
```
Ghi lại kích thước và chuỗi SHA256. Kích thước phải cỡ 550–600 MB; dưới 500 MB là thiếu thư mục, dừng lại.

## 1. Thuê máy
GPU VRAM ≥ 8 GB, RAM ≥ 32 GB, **≥ 16 vCPU** (`prepare_faces` và render mask chạy CPU), disk ≥ 40 GB, template PyTorch CUDA 12.x, Python 3.10–3.12, bật SSH. Cổng và IP lấy từ lệnh SSH của instance, dạng `ssh -p 41234 root@203.0.113.7` (số trong runbook là ví dụ giả, thay bằng số thật).

## 2. Tải gói lên
**[LAPTOP]**
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
scp -P 41234 v3_pkg.tgz root@203.0.113.7:/workspace/
```

## 3. Kiểm, giải nén, cài
**[VAST]**
```
sha256sum /workspace/v3_pkg.tgz
```
Phải trùng chuỗi `certutil`. Không trùng thì `scp` lại.

**[VAST]**
```
cd /workspace && rm -rf code-inno && mkdir -p code-inno && tar -xzf v3_pkg.tgz -C code-inno && cd /workspace/code-inno && ls
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
python -c "import innovation.defense.train_student, innovation.defense.eval_certified, innovation.defense.prepare_faces; print('import ok')"
python -c "import json;m=json.load(open('data/innovation/defense/maskset/maskset_meta.json'));print(m['landmarks']['l_top_landmark'], m['eye_zone_check']['pass'])"
ls data/innovation/frames/owner
ls results/innovation
grep -c "pk-weighting" innovation/defense/train_student.py
nproc
df -h /workspace
```
Phải ra: `True`, `0.10.14`, `1.26.4`, `4.11.0`; `import ok`; `195 True`; `ls` frames chỉ có `s3_vary s4_vary s5_vary`; `results/innovation` có `20261115_144634_inpainter_train` và `20261117_123128_student_train`; `grep` ra số ≥ 1 (đúng code v3 đã sửa); `df` còn ≥ 30 GB trống. Ghi lại `nproc`.

## 4. Tải VGGFace2 test (2.03 GB) từ Hugging Face
Dataset `ProgramComputer/VGGFace2` bị gated: tài khoản owner đã có quyền. Cần access token (huggingface.co → Settings → Access Tokens, loại Read).

**[VAST]** nhập token (gõ/dán token rồi Enter, không hiện chữ):
```
read -rs HF_TOKEN; export HF_TOKEN; echo ${#HF_TOKEN}
```
Phải in một số > 30 (độ dài token).

**[VAST]** tải (tiền cảnh, vài phút, có thanh tiến độ):
```
mkdir -p /workspace/vgg && cd /workspace/vgg
curl -L --fail -H "Authorization: Bearer $HF_TOKEN" -o vggface2_test.tar.gz https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_test.tar.gz
ls -l vggface2_test.tar.gz
```
Kích thước phải ~2 030 000 000 byte. Nếu chỉ vài KB: file là trang lỗi (token sai hoặc chưa chấp nhận điều khoản dataset); xem `head -c 300 vggface2_test.tar.gz`, sửa token rồi tải lại.

**[VAST]** giải nén (tiền cảnh, vài phút) và tìm thư mục chứa danh tính:
```
cd /workspace/vgg && tar -xzf vggface2_test.tar.gz && find /workspace/vgg -maxdepth 3 -type d -name n000001
```
`find` in ra đường dẫn thư mục `n000001` (một danh tính test). Thư mục **cha** của nó là `IMAGES_ROOT`. Thường là `/workspace/vgg/test`. Đặt biến theo đúng đường dẫn in ra (ví dụ nếu in `/workspace/vgg/test/n000001`):
```
export IMAGES_ROOT=/workspace/vgg/test
ls $IMAGES_ROOT | wc -l
ls $IMAGES_ROOT/n000001 | head -3
ls $IMAGES_ROOT/n000001 | wc -l
```
Phải ra **500** thư mục; ảnh dạng `0001_01.jpg`; `n000001` có vài trăm ảnh. Khác thì dừng, gửi Claude kết quả `find` và `ls`.

**[VAST]** xoá file nén để lấy chỗ:
```
rm /workspace/vgg/vggface2_test.tar.gz && df -h /workspace
```

## 5. Dựng crop + landmark VGGFace2 (tiền cảnh, ~10–20 phút)
**[VAST]** (nếu mở SSH mới thì `export IMAGES_ROOT=...` lại như §4):
```
cd /workspace/code-inno
rm -rf data/innovation/defense/vgg_test
python -m innovation.defense.prepare_faces --images-root $IMAGES_ROOT --meta data/vggface2/identity_meta.csv --per-id 60 --out-dir data/innovation/defense/vgg_test --workers 8 2>&1 | tee prepare_faces_log.txt
```
Nếu `nproc` ≥ 24 có thể dùng `--workers 16`. Chờ dòng `[prepare_faces] Finished: ...` và dấu nhắc `$`.

**[VAST]** kiểm:
```
cd /workspace/code-inno
python -c "import json;r=json.load(open('data/innovation/defense/vgg_test/report.json'));print(r['identities_used'], r['identities_excluded'], r['excluded_by_source'], r['total_images'], r['total_ok'], len(r['shards']))"
ls data/innovation/defense/vgg_test/crops | wc -l
ls data/innovation/defense/vgg_test/landmarks | wc -l
```
Kỳ vọng: `identities_used` = **472** (500 danh tính test trừ 28 trùng tên: 0 gallery, 3 LFW val, 25 LFW train); `excluded_by_source` và `identities_excluded` đếm trên **toàn bộ** `identity_meta.csv` (cả tập train VGG) nên lớn hơn, lần chạy 18/11 ra 530 với gallery 8 / lfw_val 51 / lfw_train 471; `total_images` khoảng 28 000 (≤ 60 × số danh tính); số shard crop = số shard landmark = `len(shards)`. **Dừng và gửi Claude** nếu: `identities_used` khác 472, `total_ok / total_images` < 0.70, hoặc số shard hai bên khác nhau.

## 6. Chạy thử (khoảng 1 phút)
**[VAST]**
```
cd /workspace/code-inno
python -m innovation.defense.train_student --fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --init-ckpt results/innovation/20261117_123128_student_train/best.pt --extra-data data/innovation/defense/vgg_test --sym-weight 1.0 --pk 16 --pk-weighting count --limit 512 --epochs 1 --num-workers 4 --device cuda --output-dir /workspace/tmp_st
cat /workspace/tmp_st/summary.json
python -c "import json;a=json.load(open('/workspace/tmp_st/meta.json'));print(a['init_ckpt'],a['sym_weight'],a['pk_weighting'],a['extra_data'])"
rm -rf /workspace/tmp_st
```
Phải in dòng `[StudentEmbedder] Initialized student weights from: ...20261117_123128...`, `summary.json` có `best_val_sym_recog_mean`, và dòng cuối in đúng init/1.0/count/vgg_test. Lỗi thì gửi Claude 30 dòng cuối.

## 7. Train student C8 (tiền cảnh, ~15–25 phút)
**[VAST]**
```
cd /workspace/code-inno
python -m innovation.defense.train_student --fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --init-ckpt results/innovation/20261117_123128_student_train/best.pt --extra-data data/innovation/defense/vgg_test --sym-weight 1.0 --pk 64 --pk-weighting count --epochs 15 --patience 4 --num-workers 8 --device cuda 2>&1 | tee student_c8_log.txt
```
Chờ dòng `Student training completed: results/innovation/<ts>_student_train` và dấu nhắc `$`. Mỗi epoch in `sym_recog=...`; nếu `supcon` là `nan` hoặc epoch 1 lâu hơn 5 phút thì gửi Claude log.

## 8. Lưới dev C7/C8/C9 (tiền cảnh, ~20 phút)
**[VAST]** tạo script (chỉ ghi file, xong ngay):
```
cd /workspace/code-inno
cat > grid_v3.sh <<'SH'
set -e
INP=results/innovation/20261115_144634_inpainter_train/best.pt
S7=results/innovation/20261117_123128_student_train
S8=$(ls -dt results/innovation/2026111[89]_*_student_train | head -1)
test -f $S8/best.pt
latest() { ls -dt results/innovation/*_$1 | head -1; }
echo "S7=$S7" > grid_v3.txt; echo "S8=$S8" >> grid_v3.txt
ev() { name=$1; shift; python -m innovation.defense.eval_certified lfwval --val-part B --inpainter-ckpt $INP --device cuda "$@"; echo "$name lfwB $(latest certified_lfwval)" >> grid_v3.txt; python -m innovation.defense.eval_certified run --split dev --inpainter-ckpt $INP --device cuda "$@"; echo "$name dev $(latest certified_eval)" >> grid_v3.txt; }
cal() { name=$1; shift; python -m innovation.defense.eval_certified calibrate --inpainter-ckpt $INP --device cuda "$@"; echo "$name calib $(latest certified_calib)" >> grid_v3.txt; }
ev C7 --gallery-mode symmetric --gallery-encoder student --student-ckpt $S7/best.pt
ev C8 --gallery-mode symmetric --gallery-encoder student --student-ckpt $S8/best.pt
cal C9 --gallery-mode symmetric --gallery-encoder student --student-ckpt $S8/best.pt; ev C9 --gallery-mode symmetric --gallery-encoder student --student-ckpt $S8/best.pt --thresholds $(latest certified_calib)/thresholds.json
echo DONE >> grid_v3.txt
SH
```
**[VAST]** chạy:
```
cd /workspace/code-inno
bash grid_v3.sh 2>&1 | tee grid_v3_log.txt
cat grid_v3.txt
```
Xong khi dấu nhắc `$` hiện lại và `grid_v3.txt` có 2 dòng S7/S8, 7 dòng run (C7 lfwB/dev, C8 lfwB/dev, C9 calib/lfwB/dev) và dòng cuối `DONE`. Có lỗi thì script dừng (`set -e`): gửi Claude 30 dòng cuối `grid_v3_log.txt`.

## 9. Đóng gói kết quả — **chạy trên VAST, không phải laptop**
**[VAST]**
```
cd /workspace/code-inno
ls -d results/innovation/2026111[89]_*
cp data/innovation/defense/vgg_test/report.json vgg_test_report.json
tar -czf v3_runs.tgz results/innovation/2026111[89]_* student_c8_log.txt grid_v3.txt grid_v3_log.txt prepare_faces_log.txt vgg_test_report.json
ls -l v3_runs.tgz
sha256sum v3_runs.tgz
```
`ls -d` phải liệt kê đúng **8** thư mục: 1 `*_student_train`, 3 `*_certified_lfwval`, 3 `*_certified_eval`, 1 `*_certified_calib`. Gói phải **vài chục MB** (student `best.pt` cỡ 20 MB); chỉ vài trăm KB là sai, dừng lại. Không mang shard VGG về.

## 10. Mang về
**[LAPTOP]**
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
scp -P 41234 root@203.0.113.7:/workspace/code-inno/v3_runs.tgz .
certutil -hashfile v3_runs.tgz SHA256
tar -xzf v3_runs.tgz
dir /b results\innovation | findstr 2026111
type grid_v3.txt
```
SHA256 phải trùng `sha256sum` trên vast; `dir` phải thấy 8 thư mục mới. **Chỉ huỷ instance sau khi bước này đạt.** Gửi Claude nội dung `grid_v3.txt`. Không commit `v3_pkg.tgz`/`v3_runs.tgz`.
