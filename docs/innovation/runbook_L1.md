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

Tạo `data/innovation/gallery/` bằng tool `innovation.tools.build_gallery`. Gallery gồm 3 loại người:
1. `owner/`: kẻ tấn công (đã có sẵn: 5 ảnh chụp từ `data/innovation/frames/owner_enroll/s0/`).
2. `LFW_victim/`: nạn nhân ảo chính, lấy từ LFW, có độ tương đồng baseline với owner gần trung vị nhất (và tuỳ chọn `LFW_victim_easy`, `LFW_victim_hard`).
3. `LFW_0001/` ... `LFW_0050/`: lớp mô phỏng (50 người từ LFW), để bài toán nhận diện là 1:N thực tế.

Mỗi người đúng 5 ảnh có mặt (giống 5 góc trong DB khi Register). Các ứng viên đã được sàng lọc không trùng với tập train của method3 (qua đối chiếu tên VGGFace2 nếu có meta và hiệu chỉnh tâm ArcFace trên miền 64×64).

```
data/innovation/gallery/
  owner/            <- 5 ảnh từ data/innovation/frames/owner_enroll/s0/: 1_front 2_left 3_right 4_up 5_down
  LFW_victim/       <- nạn nhân ảo chính (5 ảnh LFW)
  LFW_victim_easy/  <- (tuỳ chọn) nạn nhân gần p90
  LFW_victim_hard/  <- (tuỳ chọn) nạn nhân gần p10
  LFW_0001/ ...     <- 50 người của lớp mô phỏng, mỗi người 5 ảnh LFW
```

Lệnh dựng gallery:
```bash
python -m innovation.tools.build_gallery \
    --gallery data/innovation/gallery \
    --owner-frames data/innovation/frames/owner/s1_day1 \
    --train-ids data/dataset_original/classification_data/train_data \
    --members-calib data/dataset_original/classification_data/test_data \
    --pairs data/dataset_original/verification_pairs_val.txt \
    --n-class 50 --seed 0 --extra-victims
```
Tool tự động tải và giải nén LFW nếu chưa có, không bao giờ đụng vào `owner/`, và xuất báo cáo `_build_report.json`, danh sách ứng viên `_candidates.csv`. Dùng thêm `--replace` nếu cần ghi đè các folder `LFW_*` cũ.

## 3. Bảng màu (ánh xạ in sang camera)

> **Lưu ý quy trình:** Giai đoạn A (kiểm chứng digital) chạy không cần `--color-map` (khi bỏ qua cờ này, code tự đặt $M = \text{identity}$ và ghi vào log `attack.json` với `color_map_calibrated: false`). Bảng màu chỉ bắt buộc in và fit ánh xạ trước giai đoạn B (thử nghiệm vật lý).

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
# Giai đoạn A: có thể bỏ qua --color-map (mặc định M = identity)
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --mode dodge --steps 300 --batch 6
```

```bash
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<thư_mục_vừa_tạo> --liveness
```

## 6. Impersonation (kết quả E1a)

Chạy trên GPU nếu có. Trên CPU vẫn được, nhưng 1500 bước sẽ lâu.

```bash
# Giai đoạn A: chạy không có --color-map (M = identity, ghi vào log); lưới mặc định --tex-res 15 40
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery --attacker-id owner --victim LFW_victim --mode impersonate --steps 1500 --batch 8
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

## 8. In ra (chuẩn bị cho giai đoạn B)

Để tiện một lần ra tiệm in đủ mọi thứ, dùng chế độ `--sheet` để in nhiều gọng trên một tờ A4 kèm controls và thước chuẩn 50 mm:

```bash
# In một run kèm control_random và control_plain
python -m innovation.attacks.adv_glasses.export --sheet results/innovation/<run_impersonate> --controls --ipd-mm <IPD của người đeo>

# Hoặc in nhiều run cùng lúc trên 1 tờ A4
python -m innovation.attacks.adv_glasses.export --sheet results/innovation/<run1> results/innovation/<run2> --controls --ipd-mm <IPD của người đeo>

# Xuất riêng lẻ từng run (cũ)
python -m innovation.attacks.adv_glasses.export --run results/innovation/<run tối ưu> --ipd-mm <IPD của người đeo>
```

Tiện ích cũng tự động xuất ảnh preview "camera thấy" tại `results/<run>/camera_preview.png` (render gọng lên frame thật của owner ở kích thước webcam từ IPD rồi phóng 4x) để xem trước mức độ rõ nét.

**Dặn kỹ khi ra tiệm in:**
1. **Tỉ lệ 100%:** Chọn "Actual Size" (100%), tuyệt đối **tắt "fit to page" / "shrink to fit"**. Sau khi in xong, dùng thước kẻ thật đo vạch 50 mm trên tờ in: phải chính xác đúng 50 mm.
2. **Tắt tự chỉnh màu:** Yêu cầu tiệm in **tắt toàn bộ tính năng tự động nâng cấp ảnh/chỉnh màu** (disable auto color enhancement / photo enhance / Vivid).
3. **Giấy mờ (matte paper):** Bắt buộc in trên giấy mờ hoặc bìa mờ, **tuyệt đối không dùng giấy bóng (glossy)** để tránh hiện tượng lóa sáng phản chiếu vào webcam làm hỏng nhận diện và liveness.
4. **Ghi lại thiết bị:** Ghi lại tên model máy in và loại giấy tiệm sử dụng.
5. **Cùng máy in, cùng giấy:** Bảng màu (`chart.png`) và kính thật **bắt buộc phải in trên cùng một máy in và cùng một loại giấy**.

## Gửi lại cho Claude

Gửi `summary.json` (của evaluate), cùng `attack.json` và `loss.csv` (của optimize). **Không gửi** `texture_print.png`, `texture.pt` hay ảnh in; các file này đã bị `.gitignore` chặn.
