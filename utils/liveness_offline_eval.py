# -*- coding: utf-8 -*-
"""
utils/liveness_offline_eval.py
==============================
Offline evaluation tool for Face Anti-Spoofing stability policies.

Step 1: Frame-level evaluation cache (frames.csv) with JPEG simulation, MediaPipe detection,
        multi-model liveness inference, and YOLO cell phone detection.
Step 2: Policy grid evaluation (policies.csv, summary.json) with window-level scoring,
        flips/min measurement, and deterministic decision rule arbitration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np
import torch

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Limit threads for CPU inference timing parity
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liveness.temporal import LivenessSmoother
from liveness.gates import face_size_status
from liveness.phone_detector import PhoneDetector, face_in_phone
from liveness.fas_common import crop_face


# ── Policy Grid Definitions ───────────────────────────────────────────────────

P1_WINDOW_S = [0.6, 1.0, 1.5]
P1_THRESH_PAIRS = [(0.10, 0.05), (0.20, 0.05), (0.30, 0.10), (0.50, 0.20)]
P2_MIN_RATIOS = [0.15, 0.18, 0.20, 0.22]
P2_MAX_RATIO = 0.70

POLICY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
MODEL_ORDER = {"run2": 0, "run3": 1}


def build_policy_configs(has_phone: bool) -> List[Dict[str, Any]]:
    """Generate all policy configurations according to specification."""
    configs: List[Dict[str, Any]] = []

    # P0: Frame-level threshold
    configs.append({
        "policy": "P0",
        "params": {},
        "param_str": "none",
    })

    # P1: LivenessSmoother
    for w in P1_WINDOW_S:
        for ton, toff in P1_THRESH_PAIRS:
            configs.append({
                "policy": "P1",
                "params": {"window_s": w, "t_on": ton, "t_off": toff},
                "param_str": f"w={w}_on={ton}_off={toff}",
            })

    # P2: P1 + face size gate
    for w in P1_WINDOW_S:
        for ton, toff in P1_THRESH_PAIRS:
            for mr in P2_MIN_RATIOS:
                configs.append({
                    "policy": "P2",
                    "params": {
                        "window_s": w,
                        "t_on": ton,
                        "t_off": toff,
                        "min_ratio": mr,
                        "max_ratio": P2_MAX_RATIO,
                    },
                    "param_str": f"w={w}_on={ton}_off={toff}_minr={mr}_maxr={P2_MAX_RATIO}",
                })

    # P3: P2 + phone detector
    if has_phone:
        for w in P1_WINDOW_S:
            for ton, toff in P1_THRESH_PAIRS:
                for mr in P2_MIN_RATIOS:
                    configs.append({
                        "policy": "P3",
                        "params": {
                            "window_s": w,
                            "t_on": ton,
                            "t_off": toff,
                            "min_ratio": mr,
                            "max_ratio": P2_MAX_RATIO,
                        },
                        "param_str": f"w={w}_on={ton}_off={toff}_minr={mr}_maxr={P2_MAX_RATIO}",
                    })

    return configs


# ── Decision Rule Evaluation (Pure Function) ──────────────────────────────────

def evaluate_decision_rules(
    policies_data: List[Dict[str, Any]],
    required_scenarios: Sequence[str] = (
        "real_still", "real_motion", "spoof_photo_close", "spoof_photo_far", "spoof_video"
    ),
    min_windows: int = 10,
) -> Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply Section 3 decision rules:
    - PASS if for all persons: real_still and real_motion >= 0.95; every spoof_* >= 0.90.
    - If any required scenario has < min_windows: marked INSUFFICIENT and cannot pass.
    - Official configuration selection priority:
      (1) Simpler policy (P0 < P1 < P2 < P3)
      (2) run2 before run3
      (3) Maximum margin = min(real_accept - 0.95, spoof_reject - 0.90) across all person x scenario
      (4) Tie-break: smaller window_s
    - If no config passes: verdict is NO_PASS (or INSUFFICIENT).

    Returns:
        (verdict, chosen_config, ranked_configs)
    """
    # Group policies_data by (model, policy, param_str)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in policies_data:
        key = (row["model"], row["policy"], row["param_str"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    candidates: List[Dict[str, Any]] = []

    for (model, policy, param_str), rows in grouped.items():
        is_insufficient = False
        margins = []
        person_scenario_scores: Dict[Tuple[str, str], float] = {}

        # Collect scores per person and scenario
        for r in rows:
            person = r["person"]
            scenario = r["scenario"]
            n_win = int(r["n_windows"])
            score = float(r["score"])

            if n_win < min_windows:
                is_insufficient = True

            person_scenario_scores[(person, scenario)] = score
            if r["label"] == "real":
                margins.append(score - 0.95)
            else:
                margins.append(score - 0.90)

        # Check if all required scenarios are present
        persons = sorted(list(set(r["person"] for r in rows)))
        for p in persons:
            for sc in required_scenarios:
                if (p, sc) not in person_scenario_scores:
                    is_insufficient = True

        overall_margin = min(margins) if margins else -1.0
        avg_flips = (
            sum(float(r["flips_per_min"]) for r in rows) / len(rows)
            if rows
            else 0.0
        )

        # Extract window_s for tie breaking
        params = rows[0].get("params", {})
        window_s = float(params.get("window_s", 0.0))

        is_pass = (not is_insufficient) and (overall_margin >= 0.0)

        candidates.append({
            "model": model,
            "policy": policy,
            "params": params,
            "param_str": param_str,
            "margin": overall_margin,
            "is_pass": is_pass,
            "is_insufficient": is_insufficient,
            "window_s": window_s,
            "avg_flips_per_min": avg_flips,
            "rows": rows,
        })

    # Sort key:
    # PASS candidates first:
    # (1) policy order: P0 (0), P1 (1), P2 (2), P3 (3)
    # (2) model order: run2 (0) before run3 (1)
    # (3) -margin (larger margin first)
    # (4) window_s (smaller window_s first)
    def sort_key(c):
        p_order = POLICY_ORDER.get(c["policy"], 99)
        m_order = MODEL_ORDER.get(c["model"], 99)
        pass_prio = 0 if c["is_pass"] else 1
        return (pass_prio, p_order, m_order, -c["margin"], c["window_s"])

    candidates.sort(key=sort_key)

    pass_candidates = [c for c in candidates if c["is_pass"]]

    if pass_candidates:
        verdict = "PASS"
        chosen = pass_candidates[0]
    else:
        # If any candidate was insufficient on data, report INSUFFICIENT or NO_PASS
        any_insufficient = any(c["is_insufficient"] for c in candidates)
        verdict = "INSUFFICIENT" if any_insufficient else "NO_PASS"
        # Reference best margin candidate
        chosen = max(candidates, key=lambda c: c["margin"]) if candidates else None

    return verdict, chosen, candidates


# ── Step 1: Feature Extraction & Cache Building ───────────────────────────────

def run_feature_extraction(
    session_dirs: List[Path],
    models_cfg: Dict[str, str],
    jpeg_quality: int,
    out_csv_path: Path,
    no_phone: bool,
) -> Path:
    """Run MediaPipe, Liveness models, and YOLO to produce frames.csv cache."""
    from liveness.liveness_detector import LivenessDetector

    print("\n[Step 1] Initializing Liveness Detectors and MediaPipe...")

    # Shared MediaPipe detector
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    detectors: Dict[str, LivenessDetector] = {}
    for name, ckpt_str in models_cfg.items():
        ckpt_p = Path(ckpt_str).resolve()
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_p}")
        cfg_p = ckpt_p.parent / "liveness_config.json"
        if not cfg_p.is_file():
            raise FileNotFoundError(f"Liveness config not found: {cfg_p}")

        os.environ["LIVENESS_CHECKPOINT"] = str(ckpt_p)
        os.environ["LIVENESS_CONFIG"] = str(cfg_p)
        detector = LivenessDetector(device="cpu")
        detectors[name] = detector
        print(f"  Loaded model '{name}': {ckpt_p.name} (arch={detector.config.get('model', {}).get('arch')})")

    # Optional YOLO Phone detector
    phone_detector: Optional[PhoneDetector] = None
    if not no_phone:
        print("  Initializing YOLOv8 Phone Detector (class 67)...")
        phone_detector = PhoneDetector(weights="checkpoints/yolov8n.pt", imgsz=416, conf=0.25, device="cpu")

    # Find all manifest files
    manifest_files: List[Path] = []
    for s_path in session_dirs:
        if s_path.is_file() and s_path.name == "manifest.csv":
            manifest_files.append(s_path)
        elif s_path.is_dir():
            direct_m = s_path / "manifest.csv"
            if direct_m.is_file():
                manifest_files.append(direct_m)
            else:
                manifest_files.extend(list(s_path.rglob("manifest.csv")))

    manifest_files = sorted(list(set(manifest_files)))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest.csv found in provided session directories: {session_dirs}")

    print(f"  Found {len(manifest_files)} manifest(s) to process.")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_out)

    model_names = list(models_cfg.keys())
    p_headers = [f"p_{m}" for m in model_names]
    ms_headers = [f"ms_{m}" for m in model_names]

    header = [
        "person", "session", "scenario", "label", "frame_idx", "t", "frame_w",
        "face_x", "face_y", "face_w", "face_h",
    ] + p_headers + [
        "n_phone", "phone_conf_max", "face_in_phone",
    ] + ms_headers + ["ms_phone"]

    writer.writerow(header)
    f_out.flush()

    total_frames = 0
    t_start = time.time()

    for m_file in manifest_files:
        session_root = m_file.parent
        print(f"\n  Processing manifest: {m_file}")

        with open(m_file, "r", encoding="utf-8") as jf:
            reader = csv.DictReader(jf)
            for row in reader:
                total_frames += 1
                person = row["person"]
                session = row["session"]
                scenario = row["scenario"]
                label = row["label"]
                frame_idx = row["frame_idx"]
                t_val = float(row["t"])
                rel_file = row["file"]

                # Resolve image file
                cand_paths = [
                    session_root / rel_file,
                    session_root / Path(rel_file).name,
                    session_root / scenario / Path(rel_file).name,
                    Path(rel_file),
                ]
                img_path = None
                for cp in cand_paths:
                    if cp.is_file():
                        img_path = cp
                        break

                if img_path is None:
                    print(f"    [WARN] Image file not found: {rel_file} (skipping)")
                    continue

                raw_bgr = cv2.imread(str(img_path))
                if raw_bgr is None or raw_bgr.size == 0:
                    continue

                # Simulate frontend JPEG compression
                _, buf = cv2.imencode(".jpg", raw_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                frame_jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                h, w = frame_jpeg.shape[:2]

                # MediaPipe detection
                rgb = cv2.cvtColor(frame_jpeg, cv2.COLOR_BGR2RGB)
                res = mp_fd.process(rgb)
                face_bbox: Optional[List[int]] = None

                if res.detections:
                    best = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
                    bb = best.location_data.relative_bounding_box
                    bx = max(0, int(bb.xmin * w))
                    by = max(0, int(bb.ymin * h))
                    bw = int(bb.width * w)
                    bh = int(bb.height * h)
                    if bw > 0 and bh > 0:
                        face_bbox = [bx, by, bw, bh]

                # Evaluate models
                p_vals: Dict[str, str] = {}
                ms_vals: Dict[str, float] = {}

                for m_name in model_names:
                    det = detectors[m_name]
                    if face_bbox is not None:
                        t0 = time.perf_counter()
                        patch = crop_face(frame_jpeg, face_bbox, det.crop_cfg)
                        if patch is not None and patch.size > 0:
                            _, conf, _ = det.check_face(patch)
                            p_vals[m_name] = f"{conf:.4f}"
                        else:
                            p_vals[m_name] = "0.0000"
                        ms_vals[m_name] = (time.perf_counter() - t0) * 1000.0
                    else:
                        p_vals[m_name] = ""
                        ms_vals[m_name] = 0.0

                # YOLO Phone detection
                n_phone = 0
                phone_conf_max = 0.0
                f_in_phone = False
                ms_phone = 0.0

                if phone_detector is not None:
                    t0 = time.perf_counter()
                    phone_boxes = phone_detector.detect(frame_jpeg)
                    ms_phone = (time.perf_counter() - t0) * 1000.0
                    n_phone = len(phone_boxes)
                    if phone_boxes:
                        phone_conf_max = max(p[4] for p in phone_boxes)
                    if face_bbox is not None:
                        f_in_phone = face_in_phone(face_bbox, phone_boxes, min_overlap=0.8)

                # Write record
                fx = face_bbox[0] if face_bbox else ""
                fy = face_bbox[1] if face_bbox else ""
                fw = face_bbox[2] if face_bbox else ""
                fh = face_bbox[3] if face_bbox else ""

                row_out = [
                    person, session, scenario, label, frame_idx, f"{t_val:.3f}", w,
                    fx, fy, fw, fh,
                ] + [p_vals[m] for m in model_names] + [
                    n_phone, f"{phone_conf_max:.3f}", 1 if f_in_phone else 0,
                ] + [f"{ms_vals[m]:.2f}" for m in model_names] + [f"{ms_phone:.2f}"]

                writer.writerow(row_out)

        f_out.flush()

    f_out.close()
    mp_fd.close()
    elapsed = time.time() - t_start
    print(f"\n[Step 1] Finished extraction of {total_frames} frames in {elapsed:.1f}s. Saved to: {out_csv_path}")
    return out_csv_path


# ── Step 2: Policy Simulation & Offline Scoring ───────────────────────────────

def run_policy_evaluation(
    frames_csv_path: Path,
    models_cfg: Dict[str, str],
    has_phone: bool,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Execute all policies across cached frame sequences and output policies.csv + summary.json."""
    from liveness.fas_common import load_config

    print("\n[Step 2] Reading cached frame records...")
    with open(frames_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_frames = list(reader)

    if not all_frames:
        raise ValueError(f"No frame records found in {frames_csv_path}")

    model_names = list(models_cfg.keys())
    model_thresholds: Dict[str, float] = {}
    for m_name, ckpt_str in models_cfg.items():
        cfg_p = Path(ckpt_str).parent / "liveness_config.json"
        cfg_data = load_config(cfg_p)
        model_thresholds[m_name] = float(cfg_data.get("threshold", 0.10))

    # Calculate median inference latencies
    median_ms_model: Dict[str, float] = {}
    for m in model_names:
        latencies = [
            float(r[f"ms_{m}"]) for r in all_frames if r.get(f"ms_{m}") and float(r[f"ms_{m}"]) > 0
        ]
        median_ms_model[m] = round(float(np.median(latencies)), 2) if latencies else 0.0

    phone_latencies = [
        float(r["ms_phone"]) for r in all_frames if r.get("ms_phone") and float(r["ms_phone"]) > 0
    ]
    median_ms_phone = round(float(np.median(phone_latencies)), 2) if phone_latencies else 0.0

    # Group sequences by (person, session, scenario)
    sequences: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in all_frames:
        key = (r["person"], r["session"], r["scenario"])
        if key not in sequences:
            sequences[key] = []
        sequences[key].append(r)

    # Sort each sequence by t and drop initial 1.0s warm-up
    filtered_sequences: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for key, seq in sequences.items():
        seq.sort(key=lambda item: float(item["t"]))
        filtered = [item for item in seq if float(item["t"]) >= 1.0]
        if filtered:
            filtered_sequences[key] = filtered

    policy_configs = build_policy_configs(has_phone=has_phone)
    print(f"  Evaluating {len(policy_configs)} policy configurations across {len(filtered_sequences)} sequence(s)...")

    policies_rows: List[Dict[str, Any]] = []

    for m_name in model_names:
        p_thresh = model_thresholds[m_name]

        for p_cfg in policy_configs:
            pol_code = p_cfg["policy"]
            p_params = p_cfg["params"]
            param_str = p_cfg["param_str"]

            for (person, session, scenario), seq in filtered_sequences.items():
                label = seq[0]["label"]
                frame_w = float(seq[0]["frame_w"])

                # Instantiate smoother for sequence
                if pol_code in ("P1", "P2", "P3"):
                    smoother = LivenessSmoother(
                        window_s=p_params["window_s"],
                        t_on=p_params["t_on"],
                        t_off=p_params["t_off"],
                    )
                else:
                    smoother = None

                annotated_frames: List[Dict[str, Any]] = []

                for frame in seq:
                    t_val = float(frame["t"])
                    p_raw_str = frame.get(f"p_{m_name}", "")
                    has_face = p_raw_str != ""
                    p_val = float(p_raw_str) if has_face else None

                    bbox = (
                        [int(frame["face_x"]), int(frame["face_y"]), int(frame["face_w"]), int(frame["face_h"])]
                        if has_face and frame["face_w"] != ""
                        else None
                    )
                    f_in_phone = bool(int(frame.get("face_in_phone", 0)))

                    display_state = "Fake"

                    if pol_code == "P0":
                        if has_face and p_val is not None and p_val >= p_thresh:
                            display_state = "Real"
                        else:
                            display_state = "Fake"

                    elif pol_code == "P1":
                        assert smoother is not None
                        st, _ = smoother.update(t_val, p_val, bbox)
                        display_state = st

                    elif pol_code == "P2":
                        assert smoother is not None
                        size_st = face_size_status(bbox, frame_w, p_params["min_ratio"], p_params["max_ratio"])
                        if size_st == "ok":
                            st, _ = smoother.update(t_val, p_val, bbox)
                            display_state = st
                        else:
                            display_state = "Adjust"

                    elif pol_code == "P3":
                        assert smoother is not None
                        effective_p = 0.0 if f_in_phone else p_val
                        size_st = face_size_status(bbox, frame_w, p_params["min_ratio"], p_params["max_ratio"])
                        if size_st == "ok":
                            st, _ = smoother.update(t_val, effective_p, bbox)
                            display_state = st
                        else:
                            display_state = "Adjust"

                    annotated_frames.append({
                        "t": t_val,
                        "display_state": display_state,
                    })

                # Window evaluation: 1.0s non-overlapping intervals by floor(t)
                windows: Dict[int, List[str]] = {}
                for af in annotated_frames:
                    w_idx = int(math.floor(af["t"]))
                    if w_idx not in windows:
                        windows[w_idx] = []
                    windows[w_idx].append(af["display_state"])

                # Window verdict: Real if >= 50% frames in window are Real
                n_windows = len(windows)
                correct_windows = 0

                for w_states in windows.values():
                    real_count = sum(1 for s in w_states if s == "Real")
                    window_real = (real_count / len(w_states)) >= 0.5
                    if label == "real" and window_real:
                        correct_windows += 1
                    elif label == "spoof" and not window_real:
                        correct_windows += 1

                score = (correct_windows / n_windows) if n_windows > 0 else 0.0
                score_name = "real_accept" if label == "real" else "spoof_reject"

                # Calculate flips per minute
                flips = 0
                for i in range(1, len(annotated_frames)):
                    prev_is_real = (annotated_frames[i - 1]["display_state"] == "Real")
                    curr_is_real = (annotated_frames[i]["display_state"] == "Real")
                    if prev_is_real != curr_is_real:
                        flips += 1

                duration_min = (
                    (annotated_frames[-1]["t"] - annotated_frames[0]["t"]) / 60.0
                    if len(annotated_frames) > 1
                    else 0.0
                )
                flips_per_min = (flips / duration_min) if duration_min > 0 else 0.0

                policies_rows.append({
                    "model": m_name,
                    "policy": pol_code,
                    "params": p_params,
                    "param_str": param_str,
                    "person": person,
                    "session": session,
                    "scenario": scenario,
                    "label": label,
                    "score_name": score_name,
                    "score": round(score, 4),
                    "n_windows": n_windows,
                    "flips_per_min": round(flips_per_min, 2),
                })

    # Save policies.csv
    policies_csv_path = out_dir / "policies.csv"
    with open(policies_csv_path, "w", newline="", encoding="utf-8") as pf:
        fieldnames = [
            "model", "policy", "param_str", "person", "session", "scenario",
            "label", "score_name", "score", "n_windows", "flips_per_min",
        ]
        writer = csv.DictWriter(pf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(policies_rows)
    print(f"  Saved policy evaluations to: {policies_csv_path}")

    # Evaluate decision rules
    verdict, chosen, ranked_candidates = evaluate_decision_rules(policies_rows)

    # Format summary table
    chosen_summary = None
    if chosen is not None:
        chosen_summary = {
            "model": chosen["model"],
            "policy": chosen["policy"],
            "param_str": chosen["param_str"],
            "params": chosen["params"],
            "margin": round(chosen["margin"], 4),
            "is_pass": chosen["is_pass"],
            "is_insufficient": chosen["is_insufficient"],
            "avg_flips_per_min": round(chosen["avg_flips_per_min"], 2),
        }

    # Build per-person per-scenario table for the selected/best candidate
    chosen_table: List[Dict[str, Any]] = []
    if chosen is not None:
        for r in chosen["rows"]:
            chosen_table.append({
                "person": r["person"],
                "scenario": r["scenario"],
                "label": r["label"],
                "score_name": r["score_name"],
                "score": r["score"],
                "n_windows": r["n_windows"],
                "flips_per_min": r["flips_per_min"],
            })

    summary_dict = {
        "verdict": verdict,
        "chosen_config": chosen_summary,
        "table": chosen_table,
        "median_ms_model": median_ms_model,
        "median_ms_phone": median_ms_phone,
    }

    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_dict, jf, indent=2)
    print(f"  Saved summary to: {summary_json_path}")

    # Print summary table to console
    print("\n" + "=" * 88)
    print("  LIVENESS OFFLINE STABILITY EVALUATION — TOP CONFIGURATIONS")
    print("=" * 88)
    print(f"  Overall Verdict: {verdict}")
    print("-" * 88)
    header_fmt = f"{'Model':<6} | {'Pol':<3} | {'real_still':<10} | {'real_motion':<11} | {'close':<7} | {'far':<7} | {'video':<7} | {'Margin':<8} | {'Flips/min':<9} | Params"
    print(header_fmt)
    print("-" * 88)

    # Display top 5 candidates
    sc_keys = ["real_still", "real_motion", "spoof_photo_close", "spoof_photo_far", "spoof_video"]
    for cand in ranked_candidates[:5]:
        row_map: Dict[str, List[float]] = {k: [] for k in sc_keys}
        for r in cand["rows"]:
            sc = r["scenario"]
            if sc in row_map:
                row_map[sc].append(float(r["score"]))

        # Min score across persons for each scenario
        min_scores = {
            k: (min(row_map[k]) if row_map[k] else 0.0) for k in sc_keys
        }

        m_str = f"{cand['margin']:+.3f}"
        flips_str = f"{cand['avg_flips_per_min']:.1f}"
        pass_marker = " [PASS]" if cand["is_pass"] else (" [INSUFF]" if cand["is_insufficient"] else "")

        print(
            f"{cand['model']:<6} | {cand['policy']:<3} | "
            f"{min_scores['real_still']*100:>8.1f}% | "
            f"{min_scores['real_motion']*100:>9.1f}% | "
            f"{min_scores['spoof_photo_close']*100:>5.1f}% | "
            f"{min_scores['spoof_photo_far']*100:>5.1f}% | "
            f"{min_scores['spoof_video']*100:>5.1f}% | "
            f"{m_str:>8} | {flips_str:>9} | {cand['param_str']}{pass_marker}"
        )
    print("=" * 88 + "\n")

    return policies_csv_path, summary_json_path


# ── CLI & Main Entrypoint ─────────────────────────────────────────────────────

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Offline stability evaluator for FAS")
    parser.add_argument("--sessions", nargs="+", default=[],
                        help="Session directory paths or parent directories containing manifest.csv")
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "run2=checkpoints/liveness_run2_live/liveness_best_scratch.pth",
            "run3=checkpoints/liveness_run3/liveness_best_finetune.pth",
        ],
        help="Model definitions in format name=path/to/checkpoint.pth"
    )
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG simulation quality (default: 95)")
    parser.add_argument("--tag", type=str, default="eval",
                        help="Output subfolder tag (default: eval)")
    parser.add_argument("--out-dir", type=str, default="results/liveness_offline",
                        help="Output directory (default: results/liveness_offline)")
    parser.add_argument("--no-phone", action="store_true", default=False,
                        help="Disable YOLO cell phone detection")
    parser.add_argument("--reuse-cache", type=str, default=None,
                        help="Reuse existing frames.csv and skip Step 1")
    return parser.parse_args()


def main():
    args = parse_eval_args()

    # Parse models
    models_cfg: Dict[str, str] = {}
    for item in args.models:
        if "=" in item:
            name, p = item.split("=", 1)
            models_cfg[name.strip()] = p.strip()
        else:
            models_cfg[Path(item).stem] = item.strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_cache:
        cache_path = Path(args.reuse_cache)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        print(f"[OfflineEval] Reusing existing cache: {cache_path}")
    else:
        if not args.sessions:
            print("[ERROR] Please provide --sessions or --reuse-cache!")
            sys.exit(1)
        session_dirs = [Path(s).resolve() for s in args.sessions]
        cache_path = out_dir / "frames.csv"
        run_feature_extraction(
            session_dirs=session_dirs,
            models_cfg=models_cfg,
            jpeg_quality=args.jpeg_quality,
            out_csv_path=cache_path,
            no_phone=args.no_phone,
        )

    run_policy_evaluation(
        frames_csv_path=cache_path,
        models_cfg=models_cfg,
        has_phone=(not args.no_phone),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
