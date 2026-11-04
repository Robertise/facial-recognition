# Plan: dựng gallery cho L1 từ LFW

Ngày lập: 2026-11-03. Thuộc [`plan_L1.md`](plan_L1.md) §2 và §8 (giai đoạn A), thay cho mục 2 của [`runbook_L1.md`](runbook_L1.md).

**Quy tắc cho agent:**
- Viết code trên working tree. **Không commit, không push.** Owner verify rồi tự commit.
- Không đổi tên CLI args, key JSON hay hàm đã có. Chỉ thêm mới.
- Dùng lại code có sẵn (`innovation/attacks/adv_glasses/faces.py`: `FaceDetector`, `crop`, `enhance`, `list_images`, `GALLERY_PER_ID`; `backend.face_recognizer.FaceRecognizer`). Không viết lại đường xử lý mặt.
- Việc ngoài phạm vi plan này: ghi lại, không tự sửa.

## 1. Vì sao phải đổi nguồn

Claude đã kiểm tra dataset của team trên máy owner (`data/dataset_original`, bộ HW2P2 của CMU 11-785):

- `verification_pairs_val.txt`: 8,805 cặp, 9,355 ảnh. Ghép các cặp "same" thì **mọi nhóm đều đúng 2 ảnh** (4,315 nhóm 2 ảnh, 725 ảnh lẻ). Không người nào có đủ 5 ảnh.
- Ảnh `verification_data` là **64×64**, crop sát mặt. MediaPipe khó phát hiện mặt trên ảnh này, và crop nới 15% của backend sẽ khác hẳn ảnh webcam.
- Folder trong `classification_data/*` có tên dạng `n000003`. Đây là **ID gốc của VGGFace2, không ẩn danh**, nên có thể đối chiếu tên người (§4.1).

Vì vậy runbook §2 (lấy nạn nhân và lớp mô phỏng từ `verification_data`) không làm được. Owner chốt dùng **LFW** (2026-11-03).

## 2. Mục tiêu

Tool `innovation/tools/build_gallery.py` tạo ra trong `data/innovation/gallery/`:

- `LFW_victim/`: nạn nhân ảo chính, 5 ảnh.
- (tùy chọn, cờ `--extra-victims`) `LFW_victim_easy/`, `LFW_victim_hard/`.
- `LFW_0001/` … `LFW_0050/`: lớp mô phỏng, mỗi người 5 ảnh.
- **Không đụng** vào `owner/` (đã có 5 ảnh theo 5 góc).

Kèm báo cáo `data/innovation/gallery/_build_report.json` và `_candidates.csv` (§6).

Ràng buộc với mọi người trong gallery:
1. Đúng 5 ảnh có mặt, phát hiện bằng đúng detector của backend.
2. Không nằm trong tập train của method3 (§4).
3. Mỗi folder chỉ chứa một người (§3.2).

## 3. Nguồn ảnh: LFW

### 3.1 Tải
- Dùng `sklearn.datasets.fetch_lfw_people(data_home="data/public/lfw", funneled=True, download_if_missing=True)` (scikit-learn đã có trong `requirements.txt`). Hàm này tải và giải nén ảnh JPG gốc 250×250 theo folder từng người (`.../lfw_home/lfw_funneled/<Ten_Nguoi>/<Ten_Nguoi>_0001.jpg`).
- Tool **chỉ đọc các file JPG đó**, không dùng mảng đã thu nhỏ mà hàm trả về.
- Thêm cờ `--lfw-root` để trỏ thẳng vào một folder LFW có sẵn (nếu owner tải tay).
- **[KIỂM CHỨNG]** Link tải chạy được từ máy owner; điều khoản sử dụng của LFW (dataset học thuật, không có license chính thức). Ghi kết quả vào `verification_log.md`.
- `data/public/` đã nằm ngoài git (kiểm tra `.gitignore`; nếu chưa thì thêm dòng ignore, đây là thay đổi duy nhất được phép ở `.gitignore`).

### 3.2 Ảnh hợp lệ cho một người
LFW thường có nhiều người trong một ảnh. Backend chọn mặt có confidence cao nhất, nên có thể chọn nhầm người. Một ảnh chỉ hợp lệ khi:
- `FaceDetector()` (`model_selection=0`, như backend) trả về bbox, **và**
- tâm bbox nằm trong vùng giữa ảnh: lệch khỏi tâm ảnh không quá 20% cạnh ảnh (LFW căn người chính vào giữa). Ngưỡng này là tham số `--center-tol 0.20`.

Với mỗi người: duyệt ảnh theo tên file, lấy **5 ảnh hợp lệ đầu tiên**. Người có ít hơn 5 ảnh hợp lệ thì bỏ. Embedding tính đúng như `faces.build_gallery`: `crop` → `enhance` → `recognizer.extract_embedding` (checkpoint method3, strict).

## 4. Kiểm tra "chưa thấy lúc train"

Hai lớp, chạy cả hai, ghi kết quả từng người vào `_candidates.csv`.

### 4.1 Đối chiếu tên (chính xác, nếu có meta)
- Danh sách ID train = tên các folder trong `data/dataset_original/classification_data/train_data` (dạng `n0xxxxx`).
- Cần file meta của VGGFace2 ánh xạ ID sang tên (`identity_meta.csv`, cột `Class_ID, Name, ...`). Cờ `--vggface2-meta <path>`.
  - **[KIỂM CHỨNG]** Agent tìm nguồn công khai của file này (bản phát hành của VGGFace2 hoặc bản sao trong repo chính thức). Ghi nguồn vào `verification_log.md`. **Không bịa ánh xạ.** Không tìm được nguồn đáng tin thì bỏ lớp này và ghi rõ là đã bỏ.
- Chuẩn hóa tên trước khi so: chữ thường, bỏ dấu, thay `_`, `-`, khoảng trắng bằng một khoảng trắng, bỏ dấu câu. Người LFW trùng tên với một ID train thì **loại** (`excluded_reason = name_in_train`).

### 4.2 Kiểm tra bằng tâm ArcFace (thực nghiệm)
method3 có ArcFace head (`head.weight`, 4000×512). Sau khi chuẩn hóa L2, mỗi hàng là "tâm" của một người trong tập train.

- **Điểm của một người:** với mỗi ảnh trong 5 ảnh, lấy top-1 class và cosine với tâm đó. Tính `agree` = số ảnh có cùng top-1 class phổ biến nhất (từ 1 đến 5), và `med_top1` = trung vị cosine top-1.
- **Hiệu chỉnh ngưỡng trên miền 64×64** (miền mà model đã train):
  - *Member* (người model đã thấy): 200 ID ngẫu nhiên (seed 0) trong `classification_data/test_data`, mỗi ID 5 ảnh, embed thẳng ảnh 64×64 (resize như `_preprocess_face`, **không** detect). Lưu ý: chỉ dùng `test_data`, không dùng ảnh trong `train_data`.
  - *Non-member*: 200 nhóm 2 ảnh "same" ngẫu nhiên trong `verification_pairs_val.txt`, embed cùng cách.
  - Báo cáo phân bố `med_top1` của hai nhóm và AUC phân tách giữa chúng.
- **Luật loại (đặt trước):** loại một ứng viên LFW nếu `agree ≥ 3` **và** `med_top1 ≥ T`, với `T` = phân vị 5% của `med_top1` trong nhóm member (tức giữ lại tối đa khoảng 5% member nếu áp lên chính miền 64×64). Ghi `T` và AUC vào báo cáo.
- **Giới hạn phải ghi trong báo cáo:** ngưỡng được hiệu chỉnh trên ảnh 64×64, còn ảnh LFW đi qua crop của backend, nên phân bố có thể lệch. Đây là heuristic, không phải chứng minh. Lớp 4.1 (nếu có) mới là kiểm tra chính xác.

## 5. Chọn nạn nhân và lớp mô phỏng (luật đặt trước)

Chỉ xét các ứng viên đủ 5 ảnh hợp lệ và không bị loại ở §4.

- **Điểm baseline của ứng viên c:** `s(c)` = trung bình, trên các frame của owner trong `--owner-frames` (mặc định `data/innovation/frames/owner/s1_day1`), của max cosine giữa embedding frame (không đeo kính, qua đường backend, bỏ frame bị enhancer tác động) và 5 embedding của c.
  - **Chỉ dùng s1_day1.** Tool phải từ chối nếu `--owner-frames` trỏ vào `s2_day2` hoặc `s3_dim` (tập đánh giá), báo lỗi rõ ràng.
- **Nạn nhân chính `LFW_victim`:** ứng viên có `s(c)` gần **trung vị** nhất. Lý do: không chọn người quá giống owner (tấn công dễ bất thường) hay quá khác (khó bất thường).
- **(Tùy chọn) `LFW_victim_easy` / `LFW_victim_hard`:** ứng viên gần phân vị 90% và 10%.
- **Lớp mô phỏng `LFW_0001…LFW_0050`:** chọn ngẫu nhiên 50 người (seed `--seed 0`) từ các ứng viên còn lại, không gồm các nạn nhân.
- Không lọc lớp mô phỏng theo `s(c)`. Nếu baseline owner bị nhận nhầm thành ai đó (`s(c) ≥ 0.5` và cao hơn điểm của chính owner) thì ghi lại; đó là số liệu E0, không phải lý do loại người.
- Cần tối thiểu 53 ứng viên hợp lệ (50 + 3 nạn nhân). Thiếu thì dừng và báo số lượng, không tự nới luật.

## 6. Đầu ra

- `gallery/<id>/1.jpg … 5.jpg`: bản copy ảnh gốc LFW (không crop, không sửa), đúng thứ tự đã chọn.
- `gallery/_build_report.json`: tham số chạy, git commit, đường dẫn nguồn, số người LFW đã xét, số người đủ 5 ảnh, số bị loại theo từng lý do, `T`, AUC member/non-member, có dùng §4.1 không (kèm nguồn meta), các giá trị `s(c)` của nạn nhân và phân vị tương ứng, ánh xạ `id → tên LFW`.
- `gallery/_candidates.csv`: mỗi dòng một ứng viên LFW. Cột: `lfw_name, n_images, n_valid, name_in_train, agree, med_top1, s_baseline, role, excluded_reason`.
- Các file `_*.json/_*.csv` nằm trong `data/` (ngoài git). Tên người LFW không đưa vào `results/` hay docs.
- `faces.build_gallery` phải bỏ qua các file bắt đầu bằng `_` (hiện nó chỉ đọc folder con, nên kiểm tra lại là đủ).

CLI dự kiến:
```bash
python -m innovation.tools.build_gallery \
    --gallery data/innovation/gallery \
    --owner-frames data/innovation/frames/owner/s1_day1 \
    --train-ids data/dataset_original/classification_data/train_data \
    --members-calib data/dataset_original/classification_data/test_data \
    --pairs data/dataset_original/verification_pairs_val.txt \
    --n-class 50 --seed 0 [--extra-victims] [--vggface2-meta <path>] [--lfw-root <path>]
```
Mặc định không ghi đè folder đã có; ghi đè chỉ khi có `--replace` (và không bao giờ đụng `owner/`).

## 7. Test (`tests/innovation/test_build_gallery.py`)
- Chuẩn hóa tên: `"José_Ángel Martín"` khớp `"jose angel martin"`.
- Lọc ảnh nhiều người: bbox lệch tâm > 20% bị loại.
- Luật chọn trên điểm giả: nạn nhân là trung vị; easy/hard đúng phân vị; lớp mô phỏng không chứa nạn nhân; cùng seed cho cùng kết quả.
- Từ chối `--owner-frames` trỏ vào `s2_day2` / `s3_dim`.
- Không đụng `owner/`; không ghi đè khi thiếu `--replace`.
- Thiếu ứng viên (< 53) thì dừng.

## 8. Cập nhật tài liệu
- `runbook_L1.md` §2: thay bằng lệnh tải LFW và lệnh `build_gallery`, giải thích ngắn 3 loại người trong gallery. Lệnh optimize ở §6 đổi `--victim VER_victim` thành `--victim LFW_victim`.
- `plan_L1.md` §8: nguồn nạn nhân và lớp mô phỏng là LFW.
- `decisions.md`, mục 2026-11-03: `verification_data` không dùng được (số liệu ở §1), chọn LFW, ID train là VGGFace2 nên đối chiếu được tên, luật chọn nạn nhân theo trung vị.
- `verification_log.md`: LFW (tải, điều khoản), nguồn `identity_meta.csv` (hoặc ghi là không tìm được).
- `experiment_log.md`: một dòng cho lần chạy `build_gallery` (số liệu từ báo cáo).

## 9. Nghiệm thu (Claude kiểm tra bằng diff và file thật)
1. `pytest tests/innovation` pass, gồm test mới.
2. Chạy thật trên máy owner: `_build_report.json` có đủ trường ở §6; có ≥ 53 ứng viên hợp lệ; `T` và AUC member/non-member được ghi.
3. Owner mở `LFW_victim/` và 5 folder lớp mô phỏng ngẫu nhiên, xác nhận mỗi folder chỉ có một người.
4. Chạy lại `pipeline --frames data/innovation/frames/owner/s1_day1` vẫn `pass True` (không ảnh hưởng gì, nhưng xác nhận môi trường còn đúng).

Sau đó mới chạy bước dodging và impersonation của runbook.
