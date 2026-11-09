# Plan: tầng P (vật lý) trong `evaluate.py`

Viết bởi Claude, 2026-11-08. Dành cho coding agent. **Không commit. Không đổi tên CLI arg / key JSON / cột CSV / thư mục / tầng có sẵn. Chỉ THÊM.** Luật: `experiment_log.md` mục 2026-11-08 "thử vật lý H09".

## 1. CLI mới
- `--physical TIER=DIR [TIER=DIR ...]`, ví dụ `P=data/innovation/frames/owner_physical/p_adv P_control_random=.../p_rand P_control_plain=.../p_plain`. Bắt buộc có `--run` (lấy victim, gallery, checkpoint, mode, attacker từ `attack.json`).
- Khi có `--physical`: **chỉ chấm các tầng vật lý**, bỏ qua mọi tầng digital (baseline/D/control/EOT) và không đọc `split.json` để chọn frame. Không dán gì lên frame: frame thô → `jpeg(.., 95)` → luồng y hệt hiện tại (detector, liveness smoother theo session, enhancer, recognizer, find_match).
- Kiểm chống rò rỉ: mọi frame vật lý không được nằm trong train/dev/eval của `split.json` (kiểm như `--eval-session`); thư mục không tồn tại hoặc rỗng thì `SystemExit`.

## 2. Hành vi
- **Không bỏ frame vì không tìm được mắt** (đeo gọng giấy có thể làm landmark mắt hỏng). Frame không có mặt: `face=0`, `success=0`, vẫn tính vào mẫu số.
- Liveness: một smoother cho mỗi tầng vật lý (mỗi session một socket), warmup bằng các frame đầu chính session đó như logic dev-block hiện tại; ghi `smoother_warmup_frames`.
- Transfer: các model `--transfer` chấm tầng `P` (không cần chấm control).
- `frames.csv`: cùng cột như cũ; cột `tier` = tên tầng vật lý, `session` = tên thư mục.
- `summary.json`: cùng key như cũ; `results["method3/P"]`, `results["method3/P_control_random"]`, ... tách `@dim` như cũ. Thêm key mới `physical_sessions` (map tầng → thư mục) và `gate_physical` = `{P_rate, control_max, liveness_gated, pass}`, pass = P ≥ 0.80 và max(P_control_random, P_control_plain) ≤ 0.20, null khi không có `--liveness`. Giữ key `gate_to_stage_B` = null trong chế độ này.
- Thư mục kết quả: `results/innovation/<ts>_advglasses_eval_physical/`.

## 3. Test
- Chế độ không có `--physical` cho kết quả y hệt trước (so `frames.csv` trên fixture nhỏ có sẵn trong test).
- `--physical` với frame không có mắt vẫn được tính (face=1 nếu detector thấy mặt) và frame không mặt tính là thất bại.
- Rò rỉ: thư mục vật lý trùng frame trong split → `SystemExit`.
- `gate_physical` đúng trên dữ liệu giả (P 0.85 / control 0.1 → pass; P 0.85 / control 0.3 → fail).

## 4. Chạy
Agent: `pytest tests/innovation/`; in `git diff --stat`. **Không chạy evaluate thật.** Owner chạy sau khi quay:
`python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<run H09 color-map> --liveness --physical P=data/innovation/frames/owner_physical/p_adv P_control_random=data/innovation/frames/owner_physical/p_rand P_control_plain=data/innovation/frames/owner_physical/p_plain --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth`

## 5. Sửa sau verify (2026-11-09)
- Nhánh `--physical`: **bỏ warmup smoother** (giống `--eval-session`, nơi `smoother_warmup_frames = 0`). Smoother tạo mới cho mỗi tầng, chạy từ frame đầu; `smoother_warmup_frames = 0` trong summary. Không đổi gì khác.
- Test: summary vật lý có `smoother_warmup_frames == 0`; frame đầu mỗi tầng chỉ xuất hiện một lần trong `frames.csv`.
