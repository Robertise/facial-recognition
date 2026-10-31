#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/liveness_live_eval.py
===========================
Live evaluation & calibration script for Face Anti-Spoofing (Liveness) models.
Supports:
1. Calibration: 1 minute live + 1 minute spoof presentation attack -> optimal threshold.
2. Evaluation: 20 trials real + 20 trials spoof (duration --trial-seconds per trial).
   Majority vote per trial, requires >= 18/20 on both sides to PASS.
3. Headless offline test mode via --source <video> --no-display --max-frames <N>.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
#  PURE FUNCTIONS (Modular, Unit-Testable)
# ═════════════════════════════════════════════════════════════════════════════

def select_threshold(
    p: list[float] | np.ndarray,
    y: list[str | int] | np.ndarray,
) -> float:
    """
    Select threshold t that minimizes Balanced Error Rate: (FAR + FRR) / 2
    based on recorded calibration frames.

    FAR = fraction of spoof frames with p >= t
    FRR = fraction of real frames with p < t
    Tie-breaking: pick larger t (strict security preference).
    Requires >= 100 frames for real and >= 100 frames for spoof.

    Args:
        p: Array-like of confidence / p_live values in [0.0, 1.0].
        y: Array-like of labels ('real'/0 vs 'spoof'/1).

    Returns:
        Optimal threshold float.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y)

    if len(p_arr) != len(y_arr):
        raise ValueError(f"Length mismatch: len(p)={len(p_arr)}, len(y)={len(y_arr)}")

    is_spoof = (y_arr == "spoof") | (y_arr == 1) | (y_arr == "1")
    is_real = (y_arr == "real") | (y_arr == 0) | (y_arr == "0")

    n_spoof = int(np.sum(is_spoof))
    n_real = int(np.sum(is_real))

    if n_real < 100 or n_spoof < 100:
        raise ValueError(
            f"Insufficient calibration data: real={n_real}, spoof={n_spoof} (need >= 100 each)"
        )

    p_spoof = p_arr[is_spoof]
    p_real = p_arr[is_real]

    unique_p = np.unique(p_arr)
    best_t = 0.5
    min_ber = float("inf")

    for t in unique_p:
        t_val = float(t)
        far = float(np.mean(p_spoof >= t_val))
        frr = float(np.mean(p_real < t_val))
        ber = (far + frr) / 2.0

        if ber < min_ber:
            min_ber = ber
            best_t = t_val
        elif ber == min_ber and t_val > best_t:
            best_t = t_val

    return float(best_t)


def trial_decision(
    ps: list[float] | np.ndarray,
    threshold: float,
) -> tuple[Optional[str], int, int]:
    """
    Compute majority-vote verdict for an evaluation trial lasting `trial-seconds`.

    Args:
        ps: List of confidence (p_live) values for frames where a face was detected.
        threshold: Decision threshold for Real vs Fake.

    Returns:
        (decision, real_votes, spoof_votes)
        where decision is 'real', 'spoof', or None (if no face was detected in trial).
    """
    if len(ps) == 0:
        return None, 0, 0

    real_votes = sum(1 for p in ps if p >= threshold)
    spoof_votes = len(ps) - real_votes

    if real_votes > spoof_votes:
        decision = "real"
    elif spoof_votes > real_votes:
        decision = "spoof"
    else:
        # Tie in votes: Real if mean(ps) >= threshold else spoof
        decision = "real" if (float(np.mean(ps)) >= threshold) else "spoof"

    return decision, real_votes, spoof_votes


def summarize(
    checkpoint: str,
    arch: str,
    threshold: Optional[float],
    threshold_source: str,
    calib_records: list[dict[str, Any]],
    eval_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize calibration and evaluation metrics into a standardized dictionary.
    Status rule:
        PASS: both real and spoof have >= 18 correct AND exactly/at least 20 trials each.
        FAIL: both reached >= 20 trials but failed to score >= 18 on either.
        INCOMPLETE: either real or spoof trials < 20.
    """
    p_real = [r["p_live"] for r in calib_records if r.get("label") == "real"]
    p_spoof = [r["p_live"] for r in calib_records if r.get("label") == "spoof"]

    def _stats(arr_vals: list[float]) -> dict[str, Optional[float]]:
        if not arr_vals:
            return {"p10": None, "p50": None, "p90": None}
        a = np.asarray(arr_vals, dtype=float)
        return {
            "p10": round(float(np.percentile(a, 10)), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p90": round(float(np.percentile(a, 90)), 4),
        }

    real_trials = [t for t in eval_trials if t.get("target_label") == "real"]
    spoof_trials = [t for t in eval_trials if t.get("target_label") == "spoof"]

    real_correct = sum(1 for t in real_trials if t.get("is_correct") is True)
    real_total = len(real_trials)
    spoof_correct = sum(1 for t in spoof_trials if t.get("is_correct") is True)
    spoof_total = len(spoof_trials)

    if real_total >= 20 and spoof_total >= 20:
        status = "PASS" if (real_correct >= 18 and spoof_correct >= 18) else "FAIL"
    else:
        status = "INCOMPLETE"

    summary = {
        "checkpoint": str(checkpoint),
        "arch": str(arch),
        "threshold": round(float(threshold), 4) if threshold is not None else None,
        "threshold_source": str(threshold_source),
        "calibration": {
            "real_frames": len(p_real),
            "spoof_frames": len(p_spoof),
            "real_p_live": _stats(p_real),
            "spoof_p_live": _stats(p_spoof),
        },
        "evaluation": {
            "real": {"correct": real_correct, "total": real_total},
            "spoof": {"correct": spoof_correct, "total": spoof_total},
            "status": status,
        },
    }
    return summary


def print_summary_table(summary: dict[str, Any]) -> None:
    """Print readable summary table to console."""
    calib = summary["calibration"]
    eval_res = summary["evaluation"]
    thresh_val = summary["threshold"]
    thresh_str = (
        f"{thresh_val:.4f} (source: {summary['threshold_source']})"
        if thresh_val is not None
        else "None"
    )

    print("\n" + "=" * 70)
    print("  LIVENESS LIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Model Architecture : {summary['arch']}")
    print(f"  Checkpoint         : {summary['checkpoint']}")
    print(f"  Threshold          : {thresh_str}")
    print("-" * 70)
    print("  CALIBRATION:")
    r_stats = calib["real_p_live"]
    s_stats = calib["spoof_p_live"]
    r_stat_str = (
        f"(p10={r_stats['p10']}, p50={r_stats['p50']}, p90={r_stats['p90']})"
        if r_stats["p10"] is not None
        else ""
    )
    s_stat_str = (
        f"(p10={s_stats['p10']}, p50={s_stats['p50']}, p90={s_stats['p90']})"
        if s_stats["p10"] is not None
        else ""
    )
    print(f"    Real Frames      : {calib['real_frames']} {r_stat_str}")
    print(f"    Spoof Frames     : {calib['spoof_frames']} {s_stat_str}")
    print("-" * 70)
    print("  EVALUATION:")
    r_cr = eval_res["real"]["correct"]
    r_tot = eval_res["real"]["total"]
    s_cr = eval_res["spoof"]["correct"]
    s_tot = eval_res["spoof"]["total"]
    r_pct = (r_cr / r_tot * 100) if r_tot > 0 else 0.0
    s_pct = (s_cr / s_tot * 100) if s_tot > 0 else 0.0
    print(f"    Real Trials      : {r_cr} / {r_tot} ({r_pct:.1f}%)")
    print(f"    Spoof Trials     : {s_cr} / {s_tot} ({s_pct:.1f}%)")
    print(f"    Overall Status   : {eval_res['status']}")
    print("=" * 70 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI PARSER & MAIN WORKFLOW
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Liveness Live Webcam Calibration & Evaluation Script"
    )
    parser.add_argument(
        "--cam", type=int, default=0, help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to video file for offline testing (overrides --cam)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint .pth (sets LIVENESS_CHECKPOINT)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="run2",
        help="Tag prefix for output log filenames (default: run2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override, skipping calibration step",
    )
    parser.add_argument(
        "--trial-seconds",
        type=float,
        default=3.0,
        help="Duration of each evaluation trial in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/liveness_live",
        help="Output directory for CSV and JSON logs",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Target frame width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Target frame height (default: 480)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Run in headless mode without graphical display (for testing)",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["real", "spoof"],
        default="real",
        help="Fixed label for headless mode (default: real)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process before exiting (headless mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Checkpoint & Environment Setup
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        os.environ["LIVENESS_CHECKPOINT"] = str(ckpt_path)
        config_path = ckpt_path.parent / "liveness_config.json"
        if config_path.is_file():
            os.environ["LIVENESS_CONFIG"] = str(config_path)

    # Lazy import LivenessDetector after environment variables are set
    from liveness.liveness_detector import LivenessDetector

    # Instantiate detector (without passing threshold)
    detector = LivenessDetector()
    arch = detector.config.get("model", {}).get("arch", "Unknown")
    ckpt_used = str(detector.ckpt_path.resolve())

    print(f"[LiveEval] Initialized LivenessDetector with arch: {arch}")
    print(f"[LiveEval] Using checkpoint: {ckpt_used}")

    # 2. Threshold State
    if args.threshold is not None:
        current_threshold = float(args.threshold)
        threshold_source = "arg"
        print(f"[LiveEval] Threshold overridden via CLI: {current_threshold:.4f}")
    else:
        current_threshold = None
        threshold_source = "none"

    # 3. Output Logging Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.tag}_{timestamp_str}"
    csv_path = out_dir / f"{base_name}.csv"
    json_path = out_dir / f"{base_name}.json"

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["t", "phase", "label", "trial_id", "p_live", "face_w", "face_h", "threshold"]
    )
    csv_file.flush()
    print(f"[LiveEval] Logging frame data to: {csv_path}")

    # 4. Open Video Source
    if args.source:
        cap = cv2.VideoCapture(args.source)
        is_webcam = False
        print(f"[LiveEval] Reading from video file: {args.source}")
    else:
        cap = cv2.VideoCapture(args.cam)
        is_webcam = True
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"[LiveEval] Reading from webcam index: {args.cam}")

    if not cap.isOpened():
        csv_file.close()
        raise RuntimeError(
            f"Cannot open video source: {args.source if args.source else args.cam}"
        )

    # 5. Runtime Tracking State
    calib_records: list[dict[str, Any]] = []
    eval_trials: list[dict[str, Any]] = []

    # Modes: "IDLE", "CALIB_REAL", "CALIB_SPOOF", "EVAL"
    mode = "IDLE"
    eval_target_label: Optional[str] = None
    eval_trial_id = 0
    eval_start_time = 0.0
    eval_trial_ps: list[float] = []

    status_msg = ""
    status_msg_expire = 0.0

    t_start = time.time()
    frame_idx = 0

    try:
        # ── HEADLESS MODE ─────────────────────────────────────────────────────
        if args.no_display:
            print(
                f"[LiveEval] Running headless (label={args.label}, max_frames={args.max_frames})..."
            )
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Check liveness
                is_real, conf, label_det, bbox = detector.check(frame)
                p_live = float(conf)

                if bbox is not None:
                    face_w = int(bbox[2])
                    face_h = int(bbox[3])
                    calib_records.append({"label": args.label, "p_live": p_live})
                    thresh_str = (
                        round(current_threshold, 4)
                        if current_threshold is not None
                        else ""
                    )
                    csv_writer.writerow(
                        [
                            round(time.time() - t_start, 3),
                            "calib",
                            args.label,
                            0,
                            round(p_live, 4),
                            face_w,
                            face_h,
                            thresh_str,
                        ]
                    )
                    csv_file.flush()

                if args.max_frames is not None and frame_idx >= args.max_frames:
                    print(f"[LiveEval] Reached max frames ({args.max_frames}). Exiting.")
                    break

        # ── INTERACTIVE GUI MODE ──────────────────────────────────────────────
        else:
            win_name = "Liveness Live Evaluation & Calibration"
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            print("[LiveEval] Controls:")
            print("  [r] Toggle Calib REAL    [p] Toggle Calib SPOOF")
            print("  [c] Compute Threshold    [1] Eval REAL trial    [2] Eval SPOOF trial")
            print("  [q] / [ESC] Quit and save summary")

            while True:
                ret, frame = cap.read()
                if not ret:
                    if not is_webcam:
                        print("[LiveEval] End of video source reached.")
                        break
                    time.sleep(0.01)
                    continue

                frame_idx += 1
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))

                # Check liveness
                is_real, conf, label_det, bbox = detector.check(frame)
                p_live = float(conf)

                thresh_str = (
                    round(current_threshold, 4)
                    if current_threshold is not None
                    else ""
                )

                # Process current frame according to state
                if bbox is not None:
                    face_w = int(bbox[2])
                    face_h = int(bbox[3])

                    if mode == "CALIB_REAL":
                        calib_records.append({"label": "real", "p_live": p_live})
                        csv_writer.writerow(
                            [
                                round(time.time() - t_start, 3),
                                "calib",
                                "real",
                                0,
                                round(p_live, 4),
                                face_w,
                                face_h,
                                thresh_str,
                            ]
                        )
                        csv_file.flush()

                    elif mode == "CALIB_SPOOF":
                        calib_records.append({"label": "spoof", "p_live": p_live})
                        csv_writer.writerow(
                            [
                                round(time.time() - t_start, 3),
                                "calib",
                                "spoof",
                                0,
                                round(p_live, 4),
                                face_w,
                                face_h,
                                thresh_str,
                            ]
                        )
                        csv_file.flush()

                    elif mode == "EVAL":
                        eval_trial_ps.append(p_live)
                        csv_writer.writerow(
                            [
                                round(time.time() - t_start, 3),
                                "eval",
                                eval_target_label,
                                eval_trial_id,
                                round(p_live, 4),
                                face_w,
                                face_h,
                                thresh_str,
                            ]
                        )
                        csv_file.flush()

                # Handle EVAL trial timer
                if mode == "EVAL":
                    elapsed = time.time() - eval_start_time
                    if elapsed >= args.trial_seconds:
                        assert current_threshold is not None
                        decision, r_v, s_v = trial_decision(
                            eval_trial_ps, current_threshold
                        )
                        is_correct = (
                            (decision == eval_target_label)
                            if decision is not None
                            else False
                        )

                        eval_trials.append(
                            {
                                "trial_id": eval_trial_id,
                                "target_label": eval_target_label,
                                "decision": decision,
                                "is_correct": is_correct,
                                "frame_count": len(eval_trial_ps),
                                "real_votes": r_v,
                                "spoof_votes": s_v,
                            }
                        )

                        sym = "PASS" if is_correct else ("NO FACE" if decision is None else "FAIL")
                        status_msg = (
                            f"Trial #{eval_trial_id} [{eval_target_label}]: {sym} "
                            f"(Verdict={decision}, {r_v} Real / {s_v} Spoof)"
                        )
                        status_msg_expire = time.time() + 4.0
                        mode = "IDLE"
                        eval_target_label = None

                # ── Draw GUI Elements ─────────────────────────────────────────
                # 1. Bounding box & face label
                if bbox is not None:
                    bx, by, bw, bh = bbox
                    active_t = (
                        current_threshold if current_threshold is not None else 0.5
                    )
                    is_pred_real = p_live >= active_t
                    box_color = (0, 220, 0) if is_pred_real else (0, 0, 220)
                    cv2.rectangle(
                        frame, (bx, by), (bx + bw, by + bh), box_color, 2
                    )
                    cv2.putText(
                        frame,
                        f"p_live: {p_live:.3f}",
                        (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                    )

                # 2. Semi-transparent top HUD banner
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (0, 0), (frame.shape[1], 115), (20, 20, 20), -1
                )
                cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

                # Line 1: Status bar
                t_display = (
                    f"{current_threshold:.4f} ({threshold_source})"
                    if current_threshold is not None
                    else "Not Set"
                )
                hud_line1 = (
                    f"p_live: {p_live:.3f} | Thresh: {t_display} | Mode: {mode}"
                )
                cv2.putText(
                    frame,
                    hud_line1,
                    (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                )

                # Line 2: Counts & Trial Countdown
                n_c_real = sum(1 for r in calib_records if r["label"] == "real")
                n_c_spoof = sum(1 for r in calib_records if r["label"] == "spoof")

                if mode == "EVAL":
                    rem = max(0.0, args.trial_seconds - (time.time() - eval_start_time))
                    hud_line2 = (
                        f"EVAL [{eval_target_label.upper()}]: {rem:.1f}s left "
                        f"(Trial #{eval_trial_id}, {len(eval_trial_ps)} frames)"
                    )
                    color_l2 = (0, 255, 255)
                else:
                    hud_line2 = f"Calib Frames: Real={n_c_real} | Spoof={n_c_spoof}"
                    color_l2 = (200, 200, 200)

                cv2.putText(
                    frame,
                    hud_line2,
                    (12, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color_l2,
                    1,
                )

                # Line 3: Evaluation Score
                r_cr = sum(
                    1
                    for t in eval_trials
                    if t["target_label"] == "real" and t["is_correct"]
                )
                r_tot = sum(1 for t in eval_trials if t["target_label"] == "real")
                s_cr = sum(
                    1
                    for t in eval_trials
                    if t["target_label"] == "spoof" and t["is_correct"]
                )
                s_tot = sum(1 for t in eval_trials if t["target_label"] == "spoof")

                hud_line3 = (
                    f"Trials Completed: Real={r_cr}/{r_tot} (Target >=18/20) | "
                    f"Spoof={s_cr}/{s_tot} (Target >=18/20)"
                )
                cv2.putText(
                    frame,
                    hud_line3,
                    (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (120, 240, 120),
                    1,
                )

                # Line 4: Controls
                hud_line4 = "[r] Calib Real  [p] Calib Spoof  [c] Thresh  [1] Eval Real  [2] Eval Spoof  [q] Quit"
                cv2.putText(
                    frame,
                    hud_line4,
                    (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (180, 180, 180),
                    1,
                )

                # Bottom notification banner (if active)
                if status_msg and time.time() < status_msg_expire:
                    cv2.putText(
                        frame,
                        status_msg,
                        (12, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow(win_name, frame)

                # ── Key Handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break

                elif key == ord("r"):
                    if mode == "EVAL":
                        status_msg = "Cannot start calibration during evaluation trial!"
                        status_msg_expire = time.time() + 2.5
                    elif mode == "CALIB_REAL":
                        mode = "IDLE"
                        status_msg = "Calibration [REAL] Paused."
                        status_msg_expire = time.time() + 2.0
                    else:
                        mode = "CALIB_REAL"
                        status_msg = "Recording Calibration [REAL]..."
                        status_msg_expire = time.time() + 2.0

                elif key == ord("p"):
                    if mode == "EVAL":
                        status_msg = "Cannot start calibration during evaluation trial!"
                        status_msg_expire = time.time() + 2.5
                    elif mode == "CALIB_SPOOF":
                        mode = "IDLE"
                        status_msg = "Calibration [SPOOF] Paused."
                        status_msg_expire = time.time() + 2.0
                    else:
                        mode = "CALIB_SPOOF"
                        status_msg = "Recording Calibration [SPOOF]..."
                        status_msg_expire = time.time() + 2.0

                elif key == ord("c"):
                    if mode == "EVAL":
                        status_msg = "Cannot calibrate during evaluation trial!"
                        status_msg_expire = time.time() + 2.5
                    else:
                        try:
                            p_calib = [r["p_live"] for r in calib_records]
                            y_calib = [r["label"] for r in calib_records]
                            opt_t = select_threshold(p_calib, y_calib)
                            current_threshold = opt_t
                            threshold_source = "calib"
                            status_msg = f"Optimal Threshold Calibrated: {opt_t:.4f}"
                            status_msg_expire = time.time() + 4.0
                            print(f"[LiveEval] {status_msg}")
                        except ValueError as err:
                            status_msg = str(err)
                            status_msg_expire = time.time() + 4.0
                            print(f"[LiveEval] WARN: {status_msg}")

                elif key == ord("1"):
                    if current_threshold is None:
                        status_msg = "Cannot start eval: Please calibrate (c) or pass --threshold first!"
                        status_msg_expire = time.time() + 4.0
                    elif mode == "EVAL":
                        status_msg = "An evaluation trial is already in progress!"
                        status_msg_expire = time.time() + 2.0
                    else:
                        mode = "EVAL"
                        eval_target_label = "real"
                        eval_trial_id += 1
                        eval_start_time = time.time()
                        eval_trial_ps = []
                        status_msg = f"Started Evaluation Trial #{eval_trial_id} [REAL]..."
                        status_msg_expire = time.time() + 2.0

                elif key == ord("2"):
                    if current_threshold is None:
                        status_msg = "Cannot start eval: Please calibrate (c) or pass --threshold first!"
                        status_msg_expire = time.time() + 4.0
                    elif mode == "EVAL":
                        status_msg = "An evaluation trial is already in progress!"
                        status_msg_expire = time.time() + 2.0
                    else:
                        mode = "EVAL"
                        eval_target_label = "spoof"
                        eval_trial_id += 1
                        eval_start_time = time.time()
                        eval_trial_ps = []
                        status_msg = f"Started Evaluation Trial #{eval_trial_id} [SPOOF]..."
                        status_msg_expire = time.time() + 2.0

    finally:
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

    # 6. Generate and Save Summary JSON
    summary_data = summarize(
        checkpoint=ckpt_used,
        arch=arch,
        threshold=current_threshold,
        threshold_source=threshold_source,
        calib_records=calib_records,
        eval_trials=eval_trials,
    )

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_data, jf, indent=2)

    print(f"\n[LiveEval] Saved summary JSON to: {json_path}")
    print_summary_table(summary_data)


if __name__ == "__main__":
    main()
