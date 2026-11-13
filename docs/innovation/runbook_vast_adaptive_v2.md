# Runbook v2: chạy lại tấn công thích ứng trên vast.ai (sau `plan_adaptive.md` §7), 2026-11-13

Thay `runbook_vast_adaptive.md` (v1). Máy vast.ai là **máy mới, trống hoàn toàn**: không có code, data, checkpoint hay thư viện nào ngoài image PyTorch. Chỉ **optimize** chạy trên vast. **Eval chạy trên laptop.** Luật: `experiment_log.md` mục 2026-11-13.

---

## 0. Điều kiện trước khi thuê máy
1. Agent làm xong `plan_adaptive.md` §7, Claude đã verify diff + test + chạy thử 5 bước trên CPU.
2. Owner commit code + docs, gửi Claude hash commit. Trên vast không có `.git`, nên `meta.json` sẽ để `git_commit` rỗng; Claude ghi hash tay vào `experiment_log.md`.
3. 8 thư mục adaptive cũ (`20261112_1204..` đến `20261112_1210..`) **giữ nguyên trên laptop**, không đưa lên vast.

## 1. Thuê máy
- GPU bất kỳ có VRAM ≥ 8 GB. RAM ≥ 16 GB. Disk ≥ 20 GB.
- Image: template PyTorch của vast.ai (CUDA 12.x), **Python 3.10–3.12** (mediapipe 0.10.14 không có bản cho 3.13). Lần trước chạy tốt với Python 3.12.15, torch 2.14.1+cu132.
- Bật SSH. Bật Jupyter nếu muốn upload/download bằng trình duyệt.

## 2. Đóng gói trên laptop
Mở Anaconda Prompt hoặc PowerShell (không cần env), chạy:
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
tar --exclude=__pycache__ -czf adaptive_pkg_v2.tgz backend liveness emotion extreme_cond_enhancer innovation checkpoints/method3_best.pth data/innovation/frames/owner data/innovation/gallery data/innovation/defense/band_template.png data/innovation/defense/band_meta.json data/innovation/defense/ring_meta.json data/innovation/defense/ring_r8.png data/innovation/defense/ring_r16.png data/innovation/defense/ring_r32.png data/innovation/defense/ring_r48.png results/innovation/20261112_004047_purifier_train/best.pt
```
Kiểm gói (2 lệnh, chạy được trong cả cmd và PowerShell):
```
tar -tzf adaptive_pkg_v2.tgz | findstr /i "defense/adaptive.py checkpoints/method3_best.pth 20261112_004047_purifier_train/best.pt ring_r48.png band_template.png s6_vary/0000.jpg"
tar -xOzf adaptive_pkg_v2.tgz innovation/defense/adaptive.py | findstr /c:"purifier_branch_gallery"
```
- Lệnh 1 phải in đủ 6 đường dẫn: `innovation/defense/adaptive.py`, `checkpoints/method3_best.pth`, `results/innovation/20261112_004047_purifier_train/best.pt`, `data/innovation/defense/ring_r48.png`, `data/innovation/defense/band_template.png`, `data/innovation/frames/owner/s6_vary/0000.jpg`.
- Lệnh 2 phải in ít nhất 1 dòng (chứng tỏ gói chứa `adaptive.py` **đã sửa** theo §7). Không in gì thì dừng: đang đóng gói code cũ.
- Gói khoảng 75 MB (frame owner 34 MB, method3 21 MB, purifier 15 MB, gallery 6.5 MB, code ~3 MB).

Trong gói có gì, nằm đâu trên vast sau khi giải nén vào `/workspace/code-inno`:

| Thứ | Đường dẫn trên vast | Vì sao cần |
|---|---|---|
| backend (recognizer, processing) | `/workspace/code-inno/backend/` | `FaceRecognizer`, `faces.py` import `backend.processing` |
| liveness, emotion | `/workspace/code-inno/liveness/`, `/workspace/code-inno/emotion/` | `backend.processing` import khi nạp (optimize không chạy liveness) |
| enhancer | `/workspace/code-inno/extreme_cond_enhancer/` | thiếu thì enhancer tắt im lặng, không loại được 42 frame tối, dữ liệu train sai |
| code innovation (có `defense/adaptive.py` bản §7) | `/workspace/code-inno/innovation/` | optimize |
| recognizer method3 | `/workspace/code-inno/checkpoints/method3_best.pth` | mô hình bị tấn công |
| frame owner s1–s6 | `/workspace/code-inno/data/innovation/frames/owner/{s1_day1,s2_day2,s3_vary,s4_vary,s5_vary,s6_vary}/` | train s1+s2; s3–s6 phải tồn tại để tách split |
| gallery (54 danh tính, có `owner`, `LFW_victim`) | `/workspace/code-inno/data/innovation/gallery/` | gallery sạch (nhánh none) và gallery qua purifier (nhánh purifier) |
| band + vành | `/workspace/code-inno/data/innovation/defense/{band_template.png,band_meta.json,ring_meta.json,ring_r8.png,ring_r16.png,ring_r32.png,ring_r48.png}` | vùng xoá và vành tấn công |
| purifier seed 0 | `/workspace/code-inno/results/innovation/20261112_004047_purifier_train/best.pt` | mặc định của `--purifier-ckpt` |

Không cần: `lfw_crops`, `tests/`, checkpoint khác, `attendance.db`, các thư mục kết quả khác.

## 3. Đưa gói lên vast
Cách A (trình duyệt): mở Jupyter của instance, vào thư mục `/workspace`, bấm Upload, chọn `adaptive_pkg_v2.tgz`.

Cách B (scp, từ laptop, tại thư mục `code-inno`): trên trang vast.ai, bấm nút SSH/Connect của instance, sẽ thấy lệnh dạng `ssh -p 41234 root@203.0.113.7`. Lấy **số cổng sau `-p`** và **IP sau `root@`** của chính instance đó (hai số ví dụ trên là giả), rồi chạy:
```
scp -P 41234 adaptive_pkg_v2.tgz root@203.0.113.7:/workspace/
```
(thay `41234` và `203.0.113.7` bằng cổng và IP thật của instance; đây là hai giá trị duy nhất chỉ có sau khi thuê máy).

## 4. Trên vast: giải nén, cài thư viện, kiểm môi trường
SSH vào máy (hoặc mở Terminal trong Jupyter), chạy lần lượt:
```
cd /workspace
mkdir -p code-inno
tar -xzf adaptive_pkg_v2.tgz -C code-inno
cd /workspace/code-inno
apt-get update && apt-get install -y libgl1 libglib2.0-0
pip install "mediapipe==0.10.14" "protobuf>=4.25.3,<5" "numpy==1.26.4" "opencv-contrib-python==4.11.0.86"
python -c "import torch, mediapipe, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), mediapipe.__version__, numpy.__version__, cv2.__version__)"
```
Phải in `True`, `0.10.14`, `1.26.4`, `4.11.0`. Torch báo lỗi với numpy 1.26.4 thì dừng, báo Claude, không tự đổi bản.

Kiểm đủ file (chạy từ `/workspace/code-inno`):
```
python - <<'PY'
from pathlib import Path
need = [
    "backend", "liveness", "emotion", "extreme_cond_enhancer", "innovation/defense/adaptive.py",
    "checkpoints/method3_best.pth",
    "data/innovation/gallery/owner", "data/innovation/gallery/LFW_victim",
    "data/innovation/defense/band_template.png", "data/innovation/defense/band_meta.json", "data/innovation/defense/ring_meta.json",
    "data/innovation/defense/ring_r8.png", "data/innovation/defense/ring_r16.png", "data/innovation/defense/ring_r32.png", "data/innovation/defense/ring_r48.png",
    "results/innovation/20261112_004047_purifier_train/best.pt",
] + [f"data/innovation/frames/owner/{s}" for s in ["s1_day1", "s2_day2", "s3_vary", "s4_vary", "s5_vary", "s6_vary"]]
miss = [p for p in need if not Path(p).exists()]
print("MISSING:", miss if miss else "none")
print("frames per session:", {s.name: len(list(s.glob("*.jpg"))) for s in sorted(Path("data/innovation/frames/owner").iterdir()) if s.is_dir()})
print("gallery ids:", len([p for p in Path("data/innovation/gallery").iterdir() if p.is_dir() and not p.name.startswith((".", "_"))]))
print("fix present:", "purifier_branch_gallery" in Path("innovation/defense/adaptive.py").read_text(encoding="utf-8"))
PY
python -c "import innovation.defense.adaptive; print('import ok')"
```
Phải ra: `MISSING: none`; mỗi session s1–s6 có 75 frame; `gallery ids: 54`; `fix present: True`; `import ok`. Sai chỗ nào thì dừng, báo Claude.

## 5. Chạy thử 5 bước (kiểm dữ liệu)
```
cd /workspace/code-inno
python -m innovation.defense.adaptive optimize --ring-px 8 --steps 5 --device cuda
python -c "import json,glob;d=sorted(glob.glob('results/innovation/*_adaptive_optimize'))[-1];a=json.load(open(d+'/attack.json'));g=a['gallery_pur_report'];print(d);print(a['prepare_report']);print(a['purifier_branch_gallery'], g['LFW_victim'], g['owner'])"
```
Phải ra:
- `prepare_report`: `{'no_face': 0, 'no_mesh': 0, 'enhancer_applied': 42, 'used': 78}` (giống H09 trên laptop).
- `purifier` và `used` = 5 cho cả `LFW_victim` lẫn `owner`.

Khác thì dừng, báo Claude. Đúng thì xoá run thử (máy mới nên đây là thư mục adaptive duy nhất), giữ `_cache` để run thật dùng lại gallery:
```
rm -rf results/innovation/*_adaptive_optimize
ls results/innovation
```
`ls` chỉ còn `20261112_004047_purifier_train` và `_cache`.

## 6. Chạy thật (4 độ rộng nối tiếp, mất SSH vẫn chạy)
```
cd /workspace/code-inno
nohup bash -c 'for r in 8 16 32 48; do python -m innovation.defense.adaptive optimize --ring-px $r --device cuda; done' > adaptive_log_v2.txt 2>&1 &
tail -f adaptive_log_v2.txt
```
`Ctrl+C` chỉ thoát `tail`, không dừng job. Lần trước mỗi r mất khoảng 80 giây tối ưu, nên tổng khoảng 10 phút. Xong khi lệnh sau in `4`:
```
grep -c "Optimization finished" /workspace/code-inno/adaptive_log_v2.txt
```

## 7. Kiểm và đóng gói kết quả (trên vast)
```
cd /workspace/code-inno
python -c "import json,glob;[print(d, a['ring_px'], a['purifier_branch_gallery'], a['prepare_report']) for d in sorted(glob.glob('results/innovation/*_adaptive_optimize')) for a in [json.load(open(d+'/attack.json'))]]"
tar -czf adaptive_runs_v2.tgz results/innovation/*_adaptive_optimize adaptive_log_v2.txt
ls -la adaptive_runs_v2.tgz
```
Phải đúng 4 dòng, `ring_px` lần lượt 8, 16, 32, 48, cột thứ ba `purifier`, `prepare_report` như bước 5. Mỗi thư mục chứa `attack.json`, `loss.csv`, `mask.png`, `meta.json`, `split.json`, `texture.pt`, `texture_print.png`.

## 8. Mang về laptop
Cách A: trong Jupyter, vào `/workspace/code-inno`, chuột phải `adaptive_runs_v2.tgz` → Download, lưu vào `C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno\`.

Cách B (scp, cùng cổng và IP ở bước 3), trên laptop:
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
scp -P 41234 root@203.0.113.7:/workspace/code-inno/adaptive_runs_v2.tgz .
```
Giải nén **tại gốc `code-inno`** (để thành `results\innovation\<run>` và `adaptive_log_v2.txt` ở gốc):
```
cd "C:\Swinburne University\COS30082 - Applied Machine Learning\PROJECT\code-inno"
tar -xzf adaptive_runs_v2.tgz
dir /b results\innovation | findstr adaptive_optimize
```
(PowerShell: thay dòng cuối bằng `Get-ChildItem results\innovation -Name | Select-String adaptive_optimize`.) Phải thấy **8** thư mục `_adaptive_optimize`: 4 cũ `20261112_120403`, `20261112_120550`, `20261112_120730`, `20261112_120911` và 4 mới. Đủ rồi mới huỷ instance.

## 9. Báo Claude, rồi eval trên laptop
Gửi Claude hash commit và tên 4 thư mục mới. Claude kiểm `split.json`, `attack.json`, `loss.csv`, rồi đưa đúng 4 lệnh eval với tên thật. Eval chạy trên laptop, env `(test)`, từ gốc `code-inno`, thêm `--device cpu`. **Không chạy eval trên vast.**
