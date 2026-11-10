# Plan: kẻ tấn công thứ hai — p01 giả danh owner (digital, gọng H09)

Viết bởi Claude, 2026-11-09. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục / cấu hình có sẵn. Chỉ THÊM.** Luật: `experiment_log.md` mục 2026-11-09 "Luật đặt trước: p01 giả danh owner".

## Bối cảnh
p01 chỉ có gallery riêng ở `data/innovation/gallery_people/p01/` (không được thêm vào `data/innovation/gallery/`). Kịch bản điểm danh thật là p01 cũng đã enroll, nên gallery 1:N của thí nghiệm này = gallery chính + `gallery_people/p01`.

## 1. `optimize.py`: `--extra-gallery`
- Thêm `--extra-gallery DIR [DIR ...]` (mặc định không có). Mỗi DIR có cấu trúc như gallery chính (`DIR/<id>/*.jpg`). Embed mỗi DIR bằng `faces.build_gallery` (cache riêng theo root như hiện tại), gộp vào gallery chính; **trùng id thì SystemExit** (giống `signatures.merge_galleries`).
- Kiểm `--attacker-id` / `--victim` sau khi gộp.
- Ghi `extra_gallery` (list đường dẫn) vào `attack.json` và `meta.json`; ghi `gallery_report` của từng DIR vào key mới `extra_gallery_report`. Không có cờ thì `attack.json` không có key `extra_gallery` hoặc là `[]`, mọi hành vi như cũ.

## 2. `evaluate.py`: đọc `extra_gallery`
- Cả nhánh digital và nhánh `--physical`: nếu `attack.json` có `extra_gallery` không rỗng, gộp y như optimize (cho method3 và mọi model `--transfer`). Không có thì như cũ.
- `summary.json`: thêm key `extra_gallery` (list, có thể rỗng).

## 3. Test
- Không có `--extra-gallery`: gallery và `attack.json` y như trước (so key).
- Có `--extra-gallery`: id mới xuất hiện trong E; trùng id → SystemExit; evaluate đọc lại đúng danh sách.

## 4. Chạy
Agent: `pytest tests/innovation/`; in `git diff --stat`. **Không chạy optimize thật.**

Owner chạy sau khi Claude kiểm contact sheet p01_attack:
```
python -m innovation.attacks.adv_glasses.evaluate --help   # xác nhận cờ có sẵn
# Bước A (chỉ baseline, trước mọi đánh giá tấn công): xem experiment_log mục p01
python -m innovation.attacks.adv_glasses.optimize --attacker-frames data/innovation/frames/p01_attack --gallery data/innovation/gallery --extra-gallery data/innovation/gallery_people --attacker-id p01 --victim owner --mode impersonate --checkpoint checkpoints/method3_best.pth --frame-mask data/innovation/frame/owner_frame_mask.png --frame-lens data/innovation/frame/owner_frame_lens.png --grow-px 8 --clearance-mm 9 --ipd-mm 63 --tex-res 15 40 --steps 3000 --lr 0.03 --seed 0 --eval-sessions a3_vary a4_vary --dev-frac 0.2 --dev-mode block
```
Sau optimize: gửi Claude kiểm `split.json` (train chỉ a1_day + a2_day). Rồi:
```
# Bước A (chỉ baseline, không lộ tấn công)
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<run p01> --eval-session a3_vary --tiers baseline --liveness
# Bước B (một lần, trên session đạt bước A; thay a3_vary bằng a4_vary nếu a3 trượt)
python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<run p01> --eval-session a3_vary --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth
```
