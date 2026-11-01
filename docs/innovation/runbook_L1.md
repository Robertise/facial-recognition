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
4. Đặt checkpoint thật vào `checkpoints/`: `method3_best.pth` (bắt buộc); `method1_best.pth`, `method2_best.pth`, `method4_best.pth`, `method5_best.pth` (cho transfer); `liveness_best.pth` như khi chạy app.
   - method5 là **ViT-B/16** (224×224, chuẩn hoá ImageNet), cần `timm` (đã có trong `innovation/requirements.txt`). Các method khác là `FaceCNN` (112×112, [−1, 1]).
   - Mọi script L1 load checkpoint ở **chế độ strict**: thiếu file, hoặc có key thiếu/thừa so với class, thì dừng ngay với thông báo lỗi (không chạy tiếp bằng trọng số ngẫu nhiên). Gặp lỗi này thì gửi nguyên thông báo cho Claude.

## 1. Chụp frame của owner (kẻ tấn công)

Vào **Lab → Frame capture**. Đeo kính như thường ngày, ngồi như lúc điểm danh, xoay đầu chậm (trái, phải, lên, xuống một chút) trong lúc chụp.

| Person | Session | Frames | Ghi chú |
|---|---|---|---|
| `owner` | `s1_day1` | 75 | Ánh sáng bình thường |
| `owner` | `s2_day2` | 75 | **Ngày khác, cùng ánh sáng bình thường** như `s1_day1`. **Đây là tập đánh giá** (session cuối cùng theo tên) |
| `owner` | `s3_dim` (tuỳ chọn) | 40 | Phòng tối. Chỉ dùng cho tầng báo cáo riêng `@dim`. **Nếu có session này thì lệnh optimize phải thêm `--eval-sessions s2_day2 s3_dim`**, nếu không code sẽ coi `s2_day2` là tập tối ưu |
| `owner_enroll` | `s0` | 15 | Ảnh để chọn 5 góc cho gallery, không dùng trong tấn công |

Vì sao tập đánh giá phải cùng ánh sáng: khi tối, lighting enhancer biến đổi crop, nên kết quả lẫn hai nguyên nhân (gọng kính và enhancer). Frame nào enhancer có tác động đều được `evaluate` tách sang tầng `…@dim`, không tính vào tỉ lệ chính và không tính vào cổng sang giai đoạn B.

Frame được lưu ở `data/innovation/frames/<person>/<session>/`. Folder `data/` không được commit.

## 2. Gallery (những người "đã enroll")

Tạo `data/innovation/gallery/`, **mỗi người một folder, đúng 5 ảnh có mặt**, giống DB sau khi Register (5 góc: front, left, right, up, down). Ảnh toàn khung cũng được, code tự phát hiện mặt. Code chỉ lấy 5 ảnh đầu theo tên file; người nào có ít hơn 5 ảnh dùng được sẽ bị in cảnh báo và ghi vào `attack.json` → `gallery_report.uneven`.

```
data/innovation/gallery/
  owner/            <- 5 ảnh từ data/innovation/frames/owner_enroll/s0/: 1_front 2_left 3_right 4_up 5_down
  VER_victim/       <- nạn nhân ảo (5 ảnh mặt)
  VER_0001/ ...     <- khoảng 50 người của lớp mô phỏng, mỗi người 5 ảnh
```

Vì sao đúng 5: số ảnh mỗi người phải bằng nhau và bằng DB thật, để điểm tối ưu (max mềm theo ảnh) và điểm `find_match` (max theo góc) so sánh công bằng giữa các người.

- **Nạn nhân ảo và lớp mô phỏng: lấy trực tiếp từ `verification_data`** của dataset môn học. Lý do: ID trong `classification_data/train_data` đã ẩn danh, nên không so tên được để biết một người có nằm trong tập train hay không; còn `verification_data` là ảnh của người ngoài tập train (dùng để đánh giá verification), nên không trùng với tập train của method3.
  - Nếu ảnh trong `verification_data` không nhóm theo người, chọn theo các cặp "same" trong `verification_pairs_val.txt` để có đủ 5 ảnh cùng một người. Không đủ 5 thì lấy người khác.
  - VoxCeleb1 chỉ là phương án dự phòng; khi đó ghi rõ là chưa kiểm tra trùng với tập train.
- Ghi lựa chọn (ID nạn nhân, cách lấy 5 ảnh) vào `experiment_log.md`.
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
python -m innovation.attacks.adv_glasses.pipeline --frames data/innovation/frames/owner/s1_day1
```

Phải ra `pass True` (cosine ≥ 0.999). Chạy thêm một lần với `--checkpoint checkpoints/method5_best.pth`: dòng kết quả phải ghi `model FaceViT_ArcFace input 224`. **Gửi Claude cả hai dòng.**

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
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --victim VER_victim --mode impersonate --color-map data/innovation/chart/colormap.json --steps 1500 --batch 8
```

```bash
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<thư_mục_vừa_tạo> --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth
```

Dòng `gate to stage B` cho biết có qua cổng sang giai đoạn B không (tầng D ≥ 80% và control ≤ 20%, tính trên frame ánh sáng bình thường). Cổng chỉ có kết luận khi chạy với `--liveness`; thiếu `--liveness` thì `pass` là `null`. Liveness đi qua đúng `_liveness_decision` của backend, nên frame quá xa hoặc quá gần bị chặn bằng "Move closer" / "Move back" như khi điểm danh.

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
