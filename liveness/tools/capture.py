# -*- coding: utf-8 -*-
"""
liveness/tools/capture.py
==========================
Guided webcam capture tool for Face Anti-Spoofing data collection.

Records raw PNG frames across 5 standardized scenarios with console-first key control,
countdown timers, pause trimming, and structured manifest/metadata generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standardized scenarios (fixed IDs and durations)
SCENARIOS = [
    {
        "id": "real_still",
        "label": "real",
        "duration": 20.0,
        "prompt": "Ngồi bình thường trước camera, nhìn thẳng, giữ yên",
    },
    {
        "id": "real_motion",
        "label": "real",
        "duration": 25.0,
        "prompt": "Nói chuyện, há miệng to, cười, quay trái/phải, gật đầu",
    },
    {
        "id": "spoof_photo_close",
        "label": "spoof",
        "duration": 20.0,
        "prompt": "Mở ẢNH mặt bạn trên điện thoại, đưa sát camera cho che hết viền",
    },
    {
        "id": "spoof_photo_far",
        "label": "spoof",
        "duration": 20.0,
        "prompt": "Vẫn ảnh đó, lùi điện thoại ra xa để thấy rõ viền điện thoại",
    },
    {
        "id": "spoof_video",
        "label": "spoof",
        "duration": 20.0,
        "prompt": "Phát VIDEO mặt bạn trên điện thoại, khoảng cách tự nhiên",
    },
]

SCENARIO_BY_ID = {s["id"]: s for s in SCENARIOS}


# ── Non-blocking console key polling ──────────────────────────────────────────

def poll_console_key() -> Optional[str]:
    """Poll console for key press without blocking. Supports Windows and POSIX."""
    if sys.platform == "win32":
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    return "enter"
                if ch == " ":
                    return "space"
                if ch in ("\x03", "\x1b"):  # Ctrl+C or ESC
                    return "q"
                return ch.lower()
        except Exception:
            pass
    else:
        try:
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return "enter"
                if ch == " ":
                    return "space"
                if ch in ("\x03", "\x1b"):
                    return "q"
                return ch.lower()
        except Exception:
            pass
    return None


def poll_all_keys(has_window: bool) -> Optional[str]:
    """Poll both console input and OpenCV window input."""
    k_con = poll_console_key()
    if k_con:
        return k_con

    if has_window:
        try:
            k_cv = cv2.waitKey(1) & 0xFF
            if k_cv != 255:
                if k_cv in (13, 10):
                    return "enter"
                if k_cv == 32:
                    return "space"
                if k_cv in (ord("r"), ord("R")):
                    return "r"
                if k_cv in (ord("s"), ord("S")):
                    return "s"
                if k_cv in (ord("q"), ord("Q"), 27):
                    return "q"
                return chr(k_cv).lower()
        except Exception:
            pass
    return None


# ── CLI & Argument Parsing ───────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Guided webcam capture for FAS data")
    parser.add_argument("--person", type=str, required=True,
                        help="Participant ID (letters, numbers, underscore only: [a-z0-9_])")
    parser.add_argument("--cam", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                        help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Capture height (default: 480)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Target capture frame rate (default: 5.0)")
    parser.add_argument("--out-dir", type=str, default="data/liveness_capture",
                        help="Base output directory (default: data/liveness_capture)")
    parser.add_argument("--scenarios", type=str, default="",
                        help="Comma-separated scenario IDs to run (default: all scenarios in order)")
    parser.add_argument("--no-window", action="store_true", default=False,
                        help="Run in headless console mode without graphical display")
    return parser.parse_args()


# ── Main Guided Capture Loop ──────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate person ID
    if not re.fullmatch(r"[a-z0-9_]+", args.person):
        print(f"[ERROR] Invalid person ID '{args.person}'. Must match regex [a-z0-9_]")
        sys.exit(1)

    # 1. Privacy Consent
    print("=" * 72)
    print("  HƯỚNG DẪN THU THẬP DỮ LIỆU WEBCAM CHO LIVENESS / ANTI-SPOOFING")
    print("=" * 72)
    print("Dữ liệu khuôn mặt lưu cục bộ, chỉ dùng cho project, không đưa lên git. Đồng ý? (y/n): ", end="", flush=True)
    consent = sys.stdin.readline().strip().lower()
    if consent != "y":
        print("[THOÁT] Không nhận được sự đồng ý. Chương trình dừng.")
        sys.exit(0)

    # Print console control reminder
    print("-" * 72)
    print("  ĐIỀU KHIỂN BÀN PHÍM:")
    print("  *** CLICK VÀO CỬA SỔ CONSOLE (KHÔNG PHẢI CỬA SỔ VIDEO) TRƯỚC KHI BẤM PHÍM ***")
    print("  - [Enter] : Bắt đầu kịch bản hiện tại (đếm ngược 5s rồi ghi)")
    print("  - [Space] : TẠM DỪNG / TIẾP TỤC (khi dừng: tự động bỏ 1.0s cuối; bấm lại: đếm ngược 3s)")
    print("  - [r]     : Quay lại kịch bản hiện tại từ đầu (xoá frame trong session này)")
    print("  - [s]     : Bỏ qua kịch bản hiện tại")
    print("  - [q]     : Thoát chương trình (giữ lại những gì đã hoàn thành)")
    print("-" * 72)

    # Select scenarios
    if args.scenarios.strip():
        req_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        for rid in req_ids:
            if rid not in SCENARIO_BY_ID:
                print(f"[ERROR] Unknown scenario '{rid}'. Available: {list(SCENARIO_BY_ID.keys())}")
                sys.exit(1)
        selected_scenarios = [SCENARIO_BY_ID[rid] for rid in req_ids]
    else:
        selected_scenarios = list(SCENARIOS)

    # 2. Camera Initialization
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở camera index {args.cam}!")
        print("  Gợi ý: Kiểm tra xem có ứng dụng khác (trình duyệt, backend, Zoom, Teams) đang chiếm webcam không.")
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"[ERROR] Camera index {args.cam} đã mở nhưng không đọc được frame nào!")
        print("  Gợi ý: Tắt backend hoặc trình duyệt đang sử dụng webcam rồi chạy lại.")
        cap.release()
        sys.exit(1)

    actual_h, actual_w = test_frame.shape[:2]
    print(f"[INFO] Camera {args.cam} sẵn sàng: {actual_w}x{actual_h} | Target FPS: {args.fps}")

    # 3. MediaPipe Face Detection for Real-time Visual Guidance
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    # 4. Session Directory Setup
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / args.person / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    meta_path = out_dir / "meta.json"

    # Initialize manifest file with header
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow(["person", "session", "scenario", "label", "frame_idx", "t", "file"])
    manifest_file.flush()

    # Meta data tracking
    completed_scenarios: List[str] = []
    skipped_scenarios: List[str] = []
    frame_counts: Dict[str, int] = {}
    pause_counts: Dict[str, int] = {}

    win_name = "Liveness Capture Guide"
    has_window = not args.no_window
    if has_window:
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    print(f"\n[Session] Bắt đầu phiên thu thập: {out_dir}")
    sample_interval = 1.0 / args.fps

    try:
        for sc_idx, sc_info in enumerate(selected_scenarios, 1):
            sc_id = sc_info["id"]
            sc_label = sc_info["label"]
            sc_duration = float(sc_info["duration"])
            sc_prompt = sc_info["prompt"]

            sc_dir = out_dir / sc_id
            sc_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 72)
            print(f"  [{sc_idx}/{len(selected_scenarios)}] KỊCH BẢN: {sc_id.upper()} ({sc_duration:.0f}s, nhãn: {sc_label})")
            print(f"  HƯỚNG DẪN: {sc_prompt}")
            if sc_label == "spoof":
                print("  *** CHÚ Ý: MẶT THẬT KHÔNG ĐƯỢC LỌT VÀO KHUNG HÌNH! ***")
            print("  Bấm [Enter] trong console để bắt đầu đếm ngược 5s...")
            print("=" * 72)

            # Scenario state machine:
            # "READY" -> "COUNTDOWN_5" -> "RECORDING" <-> "PAUSED" -> "COUNTDOWN_3" -> "RECORDING" -> "DONE"
            substate = "READY"
            countdown_start = 0.0
            countdown_len = 5.0
            rec_start_wall = 0.0
            rec_elapsed = 0.0
            last_sample_t = -1.0
            pause_count = 0

            # Buffered frames in memory: list of {"frame": bgr, "t": t_rec}
            buffered_frames: List[Dict[str, Any]] = []

            last_console_log = time.time()
            last_face_seen = time.time()
            current_bbox: Optional[List[int]] = None

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()

                # MediaPipe face detection for guidance
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_fd.process(rgb)
                if res.detections:
                    last_face_seen = now
                    best = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
                    bb = best.location_data.relative_bounding_box
                    x = max(0, int(bb.xmin * actual_w))
                    y = max(0, int(bb.ymin * actual_h))
                    w = int(bb.width * actual_w)
                    h = int(bb.height * actual_h)
                    current_bbox = [x, y, w, h]
                else:
                    current_bbox = None

                face_missing = (now - last_face_seen) > 1.0

                # Poll keys
                key = poll_all_keys(has_window)

                if key == "q":
                    print("\n[QUIT] Người dùng yêu cầu thoát.")
                    return

                elif key == "s":
                    print(f"\n[SKIP] Bỏ qua kịch bản {sc_id}.")
                    skipped_scenarios.append(sc_id)
                    frame_counts[sc_id] = 0
                    pause_counts[sc_id] = pause_count
                    break

                elif key == "r":
                    print(f"\n[RESET] Làm lại kịch bản {sc_id} từ đầu.")
                    substate = "READY"
                    buffered_frames.clear()
                    rec_elapsed = 0.0
                    last_sample_t = -1.0
                    pause_count = 0
                    # Clean up directory files if any were written
                    for f in sc_dir.glob("*.png"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    continue

                elif key == "enter" and substate == "READY":
                    substate = "COUNTDOWN_5"
                    countdown_len = 5.0
                    countdown_start = now
                    print("  [5s] Bắt đầu đếm ngược 5 giây...")

                elif key == "space" and substate == "RECORDING":
                    # Pause: drop the last 1.0s of frames
                    pause_count += 1
                    t_last = buffered_frames[-1]["t"] if buffered_frames else rec_elapsed
                    cutoff_t = max(0.0, t_last - 1.0)
                    buffered_frames = [f for f in buffered_frames if f["t"] <= cutoff_t]
                    rec_elapsed = cutoff_t
                    last_sample_t = cutoff_t
                    substate = "PAUSED"
                    print(f"\n  [PAUSE] Tạm dừng. Đã cắt bỏ 1.0s cuối (còn {rec_elapsed:.1f}s / {sc_duration:.1f}s). Bấm Space để tiếp tục.")

                elif key == "space" and substate == "PAUSED":
                    substate = "COUNTDOWN_3"
                    countdown_len = 3.0
                    countdown_start = now
                    print("  [3s] Chuẩn bị tiếp tục sau 3 giây...")

                # State handling
                if substate in ("COUNTDOWN_5", "COUNTDOWN_3"):
                    elapsed_cd = now - countdown_start
                    cd_remaining = int(math.ceil(countdown_len - elapsed_cd))
                    if elapsed_cd >= countdown_len:
                        substate = "RECORDING"
                        rec_start_wall = time.time()
                        print(f"  [REC] Bắt đầu ghi hình kịch bản {sc_id}...")

                elif substate == "RECORDING":
                    dt = now - rec_start_wall
                    rec_start_wall = now
                    rec_elapsed += dt

                    # Check sample tick
                    if (rec_elapsed - last_sample_t) >= sample_interval:
                        buffered_frames.append({
                            "frame": frame.copy(),
                            "t": rec_elapsed,
                        })
                        last_sample_t = rec_elapsed

                    # Check scenario completion
                    if rec_elapsed >= sc_duration:
                        substate = "DONE"
                        print(f"\n  [DONE] Hoàn tất ghi {len(buffered_frames)} frames cho {sc_id}. Đang lưu vào đĩa...")

                        # Write buffered frames to disk
                        for idx, item in enumerate(buffered_frames):
                            fn = f"{idx:05d}.png"
                            fp = sc_dir / fn
                            cv2.imwrite(str(fp), item["frame"])
                            rel_file = f"{args.person}/{session_id}/{sc_id}/{fn}"
                            manifest_writer.writerow([
                                args.person,
                                session_id,
                                sc_id,
                                sc_label,
                                idx,
                                f"{item['t']:.3f}",
                                rel_file,
                            ])
                        manifest_file.flush()

                        completed_scenarios.append(sc_id)
                        frame_counts[sc_id] = len(buffered_frames)
                        pause_counts[sc_id] = pause_count
                        print(f"  [SAVED] Đã lưu {len(buffered_frames)} frame vào {sc_dir.name}/")
                        break

                # Periodic console progress update (every 2.0s)
                if now - last_console_log >= 2.0:
                    last_console_log = now
                    w_info = f"face_w={current_bbox[2]}px" if current_bbox else "NO FACE"
                    if substate == "RECORDING":
                        rem = max(0.0, sc_duration - rec_elapsed)
                        print(f"  [Console] REC {sc_id}: còn {rem:.1f}s | Saved: {len(buffered_frames)} | {w_info}")
                    elif substate == "PAUSED":
                        print(f"  [Console] PAUSED {sc_id}: đã có {rec_elapsed:.1f}s / {sc_duration:.1f}s | {w_info}")
                    elif substate.startswith("COUNTDOWN"):
                        rem_cd = max(0.0, countdown_len - (now - countdown_start))
                        print(f"  [Console] Chuẩn bị: {rem_cd:.1f}s | {w_info}")
                    if face_missing and substate == "RECORDING":
                        print("  [CẢNH BÁO] Không thấy mặt > 1.0s!")

                # Visual HUD rendering
                if has_window:
                    display = frame.copy()

                    # Draw face box if detected
                    if current_bbox:
                        bx, by, bw, bh = current_bbox
                        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 220, 0), 2)
                        cv2.putText(display, f"face_w: {bw}px", (bx, max(20, by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

                    # Top Banner
                    cv2.rectangle(display, (0, 0), (actual_w, 100), (20, 20, 20), -1)

                    # Line 1: Scenario & Status
                    if substate == "READY":
                        status_str = "READY (Nhan Enter de bat dau)"
                        status_color = (200, 200, 200)
                    elif substate.startswith("COUNTDOWN"):
                        rem_cd = max(0, int(math.ceil(countdown_len - (now - countdown_start))))
                        status_str = f"COUNTDOWN {rem_cd}s"
                        status_color = (0, 255, 255)
                    elif substate == "RECORDING":
                        rem_s = max(0.0, sc_duration - rec_elapsed)
                        status_str = f"REC con {rem_s:.1f}s ({len(buffered_frames)} frames)"
                        status_color = (0, 255, 0)
                    elif substate == "PAUSED":
                        status_str = f"PAUSED ({rec_elapsed:.1f}s/{sc_duration:.0f}s - Nhan Space de tiep tuc)"
                        status_color = (0, 165, 255)
                    else:
                        status_str = substate
                        status_color = (255, 255, 255)

                    cv2.putText(display, f"[{sc_idx}/{len(selected_scenarios)}] {sc_id.upper()} | {status_str}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

                    # Line 2: Prompt
                    cv2.putText(display, sc_prompt[:70], (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                    # Line 3: Reminder or warning
                    if sc_label == "spoof":
                        cv2.putText(display, "NHAC NHO: Mat that KHONG duoc lot vao khung hinh!",
                                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2)
                    elif face_missing:
                        cv2.putText(display, "CANH BAO: Khong thay mat (> 1.0s)!",
                                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)
                    else:
                        cv2.putText(display, "Phim: Enter=Bat dau | Space=Dung/Tiep | r=Reset | s=Skip | q=Thoat",
                                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

                    cv2.imshow(win_name, display)

    finally:
        cap.release()
        manifest_file.close()
        mp_fd.close()
        if has_window:
            cv2.destroyAllWindows()

        # Write meta.json
        meta_dict = {
            "person": args.person,
            "session": session_id,
            "cam": args.cam,
            "actual_resolution": [actual_w, actual_h],
            "target_fps": args.fps,
            "opencv_version": cv2.__version__,
            "mediapipe_version": getattr(mp, "__version__", "unknown"),
            "platform": platform.platform(),
            "scenarios_completed": completed_scenarios,
            "scenarios_skipped": skipped_scenarios,
            "frame_counts": frame_counts,
            "pause_counts": pause_counts,
        }

        with open(meta_path, "w", encoding="utf-8") as jf:
            json.dump(meta_dict, jf, indent=2)

        print("\n" + "=" * 72)
        print("  PHIÊN THU THẬP KẾT THÚC")
        print(f"  Thư mục: {out_dir}")
        print(f"  Kịch bản hoàn tất: {completed_scenarios}")
        print(f"  Kịch bản bỏ qua : {skipped_scenarios}")
        print(f"  Tổng frames     : {sum(frame_counts.values())}")
        print(f"  Manifest        : {manifest_path}")
        print(f"  Meta JSON       : {meta_path}")
        print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
