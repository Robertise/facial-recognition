# Runbook L1, giai đoạn A (kiểm chứng digital)

Các bước chạy giai đoạn A trong [`plan_L1.md`](plan_L1.md). Mọi lệnh chạy từ thư mục gốc repo, giống nhau trên Windows (cmd, PowerShell) và macOS/Linux. Sau mỗi bước có ghi file cần gửi lại cho Claude.

## 0. Chuẩn bị

1. Pull nhánh mới, rồi cài dependency:

```bash
pip install -r requirements.txt -r innovation/requirements.txt
```

2. `.env` ở thư mục gốc repo (bật cả research mode để chụp frame):

```
INNOVATION_ENABLED=1
INNOVATION_RESEARCH_MODE=1
```

3. `frontend/.env.local` có dòng `NEXT_PUBLIC_INNOVATION=1`. Khởi động lại backend và `npm run dev`.
4. Đặt checkpoint thật vào `checkpoints/`: `method3_best.pth` (bắt buộc); `method1_best.pth`, `method2_best.pth`, `method4_best.pth`, `method5_best.pth` (cho transfer); bộ liveness như khi chạy app.

## 1. Chụp frame của owner (kẻ tấn công)

Vào **Lab → Frame capture**. Đeo kính như thường ngày, ngồi như lúc điểm danh, xoay đầu chậm (trái, phải, lên, xuống một chút) trong lúc chụp.

| Person | Session | Frames | Ghi chú |
|---|---|---|---|
| `owner` | `s1_bright` | 75 | Đủ sáng |
| `owner` | `s2_dim` (hoặc `s2_day2`) | 75 | Ánh sáng khác hoặc ngày khác. **Đây là tập đánh giá** (session cuối cùng theo tên) |
| `owner_enroll` | `s0` | 15 | Ảnh để đưa owner vào gallery, không dùng trong tấn công |

Frame được lưu ở `data/innovation/frames/<person>/<session>/`. Folder `data/` không được commit.

## 2. Gallery (những người "đã enroll")

Tạo `data/innovation/gallery/`, **mỗi người một folder**, mỗi folder 5–10 ảnh có mặt (ảnh toàn khung cũng được, code tự phát hiện mặt):

```
data/innovation/gallery/
  owner/            <- copy 8–10 ảnh từ data/innovation/frames/owner_enroll/s0/
  VOX_id10270/      <- nạn nhân ảo (5–10 ảnh mặt)
  CLS_0001/ ...     <- khoảng 50 người của lớp mô phỏng
```

- **Nạn nhân ảo và lớp mô phỏng:** lấy từ VoxCeleb1 test, hoặc từ `verification_data` của dataset môn học.
- **Kiểm tra trùng người:** nạn nhân ảo **không được nằm trong tập train của method3**. Cách kiểm tra: nếu tên hoặc ID người đó có trong `classification_data/train_data` thì đổi người khác. Ghi lựa chọn vào `experiment_log.md`.
- Tên folder chính là ID xuất hiện trong kết quả.

## 3. Bảng màu (ánh xạ in sang camera)

1. Tạo bảng màu:

```bash
python -m innovation.attacks.adv_glasses.printability chart
```

2. In `data/innovation/chart/chart.png` **ở tỉ lệ 100%** (tắt "fit to page") trên đúng loại giấy sẽ in gọng.
3. Giơ tờ in trước webcam, dưới ánh sáng sẽ dùng khi thử. Chụp 3 frame bằng Lab: person `chart`, session `s1`.
4. Fit ánh xạ màu:

```bash
python -m innovation.attacks.adv_glasses.printability fit data/innovation/frames/chart/s1/0001.jpg
```

Lệnh này in ra sai số `fit_mae_8bit` và ghi `data/innovation/chart/colormap.json`. Nếu báo không thấy 4 marker ArUco ở góc, chụp lại cho tờ in nằm trọn trong khung hình.

## 4. Kiểm tra pipeline (bắt buộc trước khi tối ưu)

```bash
python -m innovation.attacks.adv_glasses.pipeline --frames data/innovation/frames/owner/s1_bright
```

Phải ra `pass True` (cosine ≥ 0.999). **Gửi Claude dòng kết quả này.**

## 5. Dodging trước (bài kiểm tra dễ, chạy được trên CPU)

```bash
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --mode dodge --color-map data/innovation/chart/colormap.json --steps 300 --batch 6
```

```bash
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<thư_mục_vừa_tạo> --liveness
```

## 6. Impersonation (kết quả E1a)

Chạy trên GPU nếu có. Trên CPU vẫn được, nhưng 1500 bước sẽ lâu.

```bash
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --victim VOX_id10270 --mode impersonate --color-map data/innovation/chart/colormap.json --steps 1500 --batch 8
```

```bash
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<thư_mục_vừa_tạo> --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth
```

Dòng `gate to stage B` cho biết có qua cổng sang giai đoạn B không (tầng D ≥ 80% và control ≤ 20%).

## 7. Ablation độ phủ (nếu kịp)

Lặp lại bước 6 với ba cấu hình gọng:

| Gọng | Tham số | Độ phủ đo thử |
|---|---|---|
| Thường | `--thickness 14 --rim-scale 1.0` | ~4% |
| Dày | `--thickness 35 --rim-scale 1.1` | ~8% (mặc định) |
| Ngoại cỡ | `--thickness 55 --rim-scale 1.2` | ~12% |

Độ phủ đúng của từng lần chạy nằm trong `attack.json` → `coverage_train`.

## 8. In ra (chỉ khi cần thử vật lý)

```bash
python -m innovation.attacks.adv_glasses.export --run results/innovation/<run tối ưu> --ipd-mm <IPD của người đeo>
```

In ở tỉ lệ 100%, kiểm tra thước 50 mm trên tờ in, cắt theo viền xám, dán lên gọng nhựa.

## Gửi lại cho Claude

Gửi `summary.json` (của evaluate), cùng `attack.json` và `loss.csv` (của optimize). **Không gửi** `texture_print.png`, `texture.pt` hay ảnh in; các file này đã bị `.gitignore` chặn.
