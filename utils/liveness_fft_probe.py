# -*- coding: utf-8 -*-
"""
utils/liveness_fft_probe.py
===========================
Offline 2D-FFT Moiré and High-Frequency Feature Probe for Face Anti-Spoofing.

Analyzes native-resolution patches across 3 regions (face, wide, ring) and 3 YCrCb
channels to measure high-frequency energy ratio ('hf') and narrow spectral peaks ('peak').
Computes AUC on fit set, selects 98th-percentile real threshold, and benchmarks detection
rates across real, spoof, backlight, and corner-leak scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Cache for Hann windows and frequency masks ────────────────────────────────
_GRID_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def get_grid_and_masks(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (hann_2d, mask_num, mask_den, mask_peak) for square patch of size n x n.
    Frequencies r are normalized in [0, 0.5] cycles/pixel (radial distance from DC).
    """
    if n not in _GRID_CACHE:
        hann_1d = np.hanning(n)
        hann_2d = np.outer(hann_1d, hann_1d)

        freq = np.fft.fftshift(np.fft.fftfreq(n, d=1.0))
        fx, fy = np.meshgrid(freq, freq)
        r = np.sqrt(fx**2 + fy**2)

        mask_num = (r >= 0.25) & (r <= 0.5)
        mask_den = (r >= 0.05) & (r <= 0.5)
        mask_peak = (r >= 0.10) & (r <= 0.5)

        _GRID_CACHE[n] = (hann_2d, mask_num, mask_den, mask_peak)
    return _GRID_CACHE[n]


def compute_patch_fft_features(patch: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Computes (hf, peak) features on a 2D square image patch:
    - hf: sum |F|^2 in r in [0.25, 0.5] / sum |F|^2 in r in [0.05, 0.5]
    - peak: 99.9th percentile of [logmag - blur(logmag, 15x15)] in r in [0.10, 0.5]

    Returns (None, None) if patch is invalid or side < 64.
    """
    if patch is None or patch.ndim != 2:
        return None, None
    n = patch.shape[0]
    if n < 64 or patch.shape[1] != n:
        return None, None

    hann_2d, mask_num, mask_den, mask_peak = get_grid_and_masks(n)

    p = patch.astype(np.float64)
    p = p - np.mean(p)
    p = p * hann_2d

    f_coeff = np.fft.fft2(p)
    f_shift = np.fft.fftshift(f_coeff)
    mag = np.abs(f_shift)
    mag2 = mag**2
    logmag = np.log1p(mag)

    # 1. hf feature
    den = float(np.sum(mag2[mask_den]))
    if den > 0.0:
        hf = float(np.sum(mag2[mask_num])) / den
    else:
        hf = 0.0

    # 2. peak feature
    blurred = cv2.blur(logmag.astype(np.float32), (15, 15))
    diff = logmag - blurred
    peak_vals = diff[mask_peak]
    if len(peak_vals) > 0:
        peak = float(np.percentile(peak_vals, 99.9))
    else:
        peak = 0.0

    return hf, peak


def extract_center_square(region: np.ndarray, max_size: int = 256, min_size: int = 64) -> Optional[np.ndarray]:
    """
    Extracts largest square patch in center of region, capped at max_size.
    Returns None if side < min_size.
    """
    h, w = region.shape[:2]
    side = min(h, w)
    if side < min_size:
        return None
    if side > max_size:
        side = max_size

    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return region[y1 : y1 + side, x1 : x1 + side]


def extract_three_regions(
    img_ycrcb: np.ndarray,
    fx: float,
    fy: float,
    fw: float,
    fh: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extracts 3 regions for each YCrCb channel:
    - face: exact face bbox.
    - wide: concentric bbox, 2.5x face dimensions.
    - ring: wide region with face area filled with mean of surround to avoid step edges.

    Returns: (face_ch_dict, wide_ch_dict, ring_ch_dict)
    """
    h_img, w_img = img_ycrcb.shape[:2]

    # Face bbox clipped
    fx1 = max(0, int(round(fx)))
    fy1 = max(0, int(round(fy)))
    fx2 = min(w_img, int(round(fx + fw)))
    fy2 = min(h_img, int(round(fy + fh)))

    # Wide bbox: concentric 2.5x
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0
    ww = fw * 2.5
    wh = fh * 2.5

    wx1 = max(0, int(round(cx - ww / 2.0)))
    wy1 = max(0, int(round(cy - wh / 2.0)))
    wx2 = min(w_img, int(round(cx + ww / 2.0)))
    wy2 = min(h_img, int(round(cy + wh / 2.0)))

    # Relative coordinates of face inside wide
    rf_y1 = max(0, fy1 - wy1)
    rf_y2 = max(0, fy2 - wy1)
    rf_x1 = max(0, fx1 - wx1)
    rf_x2 = max(0, fx2 - wx1)

    mask_face = np.zeros((wy2 - wy1, wx2 - wx1), dtype=bool)
    mask_face[rf_y1:rf_y2, rf_x1:rf_x2] = True

    channel_indices = {"Y": 0, "Cr": 1, "Cb": 2}
    face_dict = {}
    wide_dict = {}
    ring_dict = {}

    for ch_name, ch_idx in channel_indices.items():
        ch_img = img_ycrcb[:, :, ch_idx]
        face_dict[ch_name] = ch_img[fy1:fy2, fx1:fx2]

        wide_ch = ch_img[wy1:wy2, wx1:wx2].copy()
        wide_dict[ch_name] = wide_ch

        ring_ch = wide_ch.copy()
        if np.any(~mask_face):
            fill_val = float(np.mean(wide_ch[~mask_face]))
        else:
            fill_val = float(np.mean(wide_ch))
        ring_ch[mask_face] = fill_val
        ring_dict[ch_name] = ring_ch

    return face_dict, wide_dict, ring_dict


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes ROC-AUC using Mann-Whitney U rank statistic with tie handling.
    y_true: 1 for spoof (positive), 0 for real (negative).
    """
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 == 0 or n0 == 0:
        return 0.5

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    unique_vals, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            indices = np.where(inverse == i)[0]
            ranks[indices] = np.mean(ranks[indices])

    r1 = np.sum(ranks[y_true == 1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


# ── Feature extraction pipeline ───────────────────────────────────────────────

FEATURE_NAMES = [
    "face_Y_hf", "face_Y_peak",
    "face_Cr_hf", "face_Cr_peak",
    "face_Cb_hf", "face_Cb_peak",
    "wide_Y_hf", "wide_Y_peak",
    "wide_Cr_hf", "wide_Cr_peak",
    "wide_Cb_hf", "wide_Cb_peak",
    "ring_Y_hf", "ring_Y_peak",
    "ring_Cr_hf", "ring_Cr_peak",
    "ring_Cb_hf", "ring_Cb_peak",
]


def process_dataset(
    frames_csv_path: str,
    data_root: str,
    jpeg_quality: int,
    out_dir: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts 18 FFT features for all valid rows in frames.csv and saves fft_features.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_out_path = os.path.join(out_dir, "fft_features.csv")

    rows_to_process: List[Dict[str, Any]] = []
    with open(frames_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fw_str = r.get("face_w", "").strip()
            if fw_str and fw_str.lower() != "nan" and float(fw_str) > 0:
                rows_to_process.append(r)

    total_rows = len(rows_to_process)
    print(f"[FFT Probe] Loaded {total_rows} frames with valid face_w from {frames_csv_path}")

    fieldnames = [
        "person", "session", "scenario", "label", "frame_idx", "face_w",
        *FEATURE_NAMES,
        "ms",
    ]

    all_records: List[Dict[str, Any]] = []

    with open(csv_out_path, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        t_start_all = time.perf_counter()

        for idx, row in enumerate(rows_to_process, 1):
            t0 = time.perf_counter()

            person = row["person"]
            session = row["session"]
            scenario = row["scenario"]
            label = row["label"]
            frame_idx = int(float(row["frame_idx"]))
            face_w = float(row["face_w"])
            face_x = float(row["face_x"])
            face_y = float(row["face_y"])
            face_h = float(row["face_h"])

            # 1. Read PNG
            img_rel_path = os.path.join(data_root, person, session, scenario, f"{frame_idx:05d}.png")
            img_bgr = cv2.imread(img_rel_path)
            if img_bgr is None:
                print(f"[Warning] Frame missing: {img_rel_path}, skipping.")
                continue

            # 2. Simulate frontend JPEG q95
            if jpeg_quality is not None and jpeg_quality > 0:
                ok, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if ok:
                    img_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

            # 3. Convert to YCrCb
            img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

            # 4. Extract 3 regions
            face_dict, wide_dict, ring_dict = extract_three_regions(
                img_ycrcb, face_x, face_y, face_w, face_h
            )

            regions_map = {
                "face": face_dict,
                "wide": wide_dict,
                "ring": ring_dict,
            }

            feat_vals: Dict[str, Any] = {}

            # 5. Compute features for each region x channel
            for reg_name in ["face", "wide", "ring"]:
                ch_dict = regions_map[reg_name]
                for ch_name in ["Y", "Cr", "Cb"]:
                    reg_ch = ch_dict[ch_name]
                    square_patch = extract_center_square(reg_ch, max_size=256, min_size=64)
                    if square_patch is not None:
                        hf, peak = compute_patch_fft_features(square_patch)
                    else:
                        hf, peak = None, None

                    feat_vals[f"{reg_name}_{ch_name}_hf"] = round(hf, 6) if hf is not None else ""
                    feat_vals[f"{reg_name}_{ch_name}_peak"] = round(peak, 6) if peak is not None else ""

            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            record: Dict[str, Any] = {
                "person": person,
                "session": session,
                "scenario": scenario,
                "label": label,
                "frame_idx": frame_idx,
                "face_w": round(face_w, 2),
                **feat_vals,
                "ms": dt_ms,
            }

            writer.writerow(record)
            all_records.append(record)

            if idx % 200 == 0 or idx == total_rows:
                elapsed = time.perf_counter() - t_start_all
                fps = idx / elapsed if elapsed > 0 else 0
                print(f"[FFT Probe] Processed {idx}/{total_rows} frames ({fps:.1f} fps)")

    print(f"[FFT Probe] Features written to {csv_out_path}")
    return all_records, csv_out_path


# ── Analysis & Report ─────────────────────────────────────────────────────────

FIT_SESSIONS = ["20261025_120633", "20261025_123329"]
TEST_SESSIONS = ["20261025_131122", "20261025_165159"]

SCENARIO_DISPLAY_NAMES = [
    # Fit session 1
    ("20261025_120633", "real_still", "Fit1_st"),
    ("20261025_120633", "real_motion", "Fit1_mo"),
    ("20261025_120633", "spoof_photo_close", "Fit1_cl"),
    ("20261025_120633", "spoof_photo_far", "Fit1_far"),
    ("20261025_120633", "spoof_video", "Fit1_vid"),
    # Fit session 2
    ("20261025_123329", "real_still", "Fit2_st"),
    ("20261025_123329", "real_motion", "Fit2_mo"),
    ("20261025_123329", "spoof_photo_close", "Fit2_cl"),
    ("20261025_123329", "spoof_photo_far", "Fit2_far"),
    ("20261025_123329", "spoof_video", "Fit2_vid"),
    # Test session 1 (backlight)
    ("20261025_131122", "real_still", "Tst1_st"),
    ("20261025_131122", "real_motion", "Tst1_mo"),
    ("20261025_131122", "spoof_photo_close", "Tst1_cl"),
    ("20261025_131122", "spoof_photo_far", "Tst1_far"),
    ("20261025_131122", "spoof_video", "Tst1_vid"),
    # Test session 2 (corner leak angle)
    ("20261025_165159", "spoof_photo_close", "Tst2_cl"),
]


def analyze_features(records: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    """
    Performs ROC-AUC ranking on fit set, sets 98th percentile real threshold,
    and evaluates moiré detection rates across scenarios.
    """
    # Convert records to arrays
    n = len(records)
    sessions = np.array([r["session"] for r in records])
    scenarios = np.array([r["scenario"] for r in records])
    labels = np.array([r["label"] for r in records])
    y_spoof = (labels == "spoof").astype(int)

    fit_mask = np.isin(sessions, FIT_SESSIONS)
    test_mask = np.isin(sessions, TEST_SESSIONS)

    features_analysis = []

    for feat in FEATURE_NAMES:
        # Extract numeric values
        raw_vals = np.array([
            float(r[feat]) if r[feat] != "" else np.nan
            for r in records
        ], dtype=np.float64)

        # Fit set evaluation
        fit_valid = fit_mask & ~np.isnan(raw_vals)
        y_fit = y_spoof[fit_valid]
        s_fit = raw_vals[fit_valid]

        raw_auc = compute_auc(y_fit, s_fit)

        if raw_auc < 0.5:
            inverted = True
            fit_auc = 1.0 - raw_auc
            aligned_vals = -raw_vals
        else:
            inverted = False
            fit_auc = raw_auc
            aligned_vals = raw_vals

        # Threshold = 98th percentile of real frames in fit set
        real_fit_mask = fit_mask & (labels == "real") & ~np.isnan(aligned_vals)
        real_fit_scores = aligned_vals[real_fit_mask]
        if len(real_fit_scores) > 0:
            threshold = float(np.percentile(real_fit_scores, 98.0))
        else:
            threshold = 0.0

        # Flagged mask: aligned_vals > threshold
        flagged_mask = (aligned_vals > threshold) & ~np.isnan(aligned_vals)

        # Calculate % flagged by session x scenario
        scenario_pcts: Dict[str, Dict[str, float]] = {}
        for sess, scen, short_name in SCENARIO_DISPLAY_NAMES:
            if sess not in scenario_pcts:
                scenario_pcts[sess] = {}
            sub_mask = (sessions == sess) & (scenarios == scen) & ~np.isnan(aligned_vals)
            n_sub = int(np.sum(sub_mask))
            if n_sub > 0:
                pct = float(np.mean(flagged_mask[sub_mask]) * 100.0)
            else:
                pct = 0.0
            scenario_pcts[sess][scen] = round(pct, 1)

        # Aggregated stats
        fit_real_pct = float(np.mean(flagged_mask[fit_mask & (labels == "real")]) * 100.0)
        fit_spoof_pct = float(np.mean(flagged_mask[fit_mask & (labels == "spoof")]) * 100.0)

        t1_real = (sessions == "20261025_131122") & (labels == "real")
        t1_spoof = (sessions == "20261025_131122") & (labels == "spoof")
        t2_spoof = (sessions == "20261025_165159") & (labels == "spoof")

        test_real_pct = float(np.mean(flagged_mask[t1_real]) * 100.0) if np.any(t1_real) else 0.0
        test_backlight_pct = float(np.mean(flagged_mask[t1_spoof]) * 100.0) if np.any(t1_spoof) else 0.0
        test_corner_pct = float(np.mean(flagged_mask[t2_spoof]) * 100.0) if np.any(t2_spoof) else 0.0

        features_analysis.append({
            "feature": feat,
            "fit_auc": round(fit_auc, 4),
            "raw_auc": round(raw_auc, 4),
            "inverted": inverted,
            "threshold": round(threshold, 6),
            "raw_threshold": round(-threshold, 6) if inverted else round(threshold, 6),
            "flag_condition": f"(-{feat}) > {threshold:.4f}" if inverted else f"{feat} > {threshold:.4f}",
            "scenario_flag_pct": scenario_pcts,
            "summary_flag_pct": {
                "fit_real": round(fit_real_pct, 1),
                "fit_spoof": round(fit_spoof_pct, 1),
                "test_real_backlight": round(test_real_pct, 1),
                "test_spoof_backlight": round(test_backlight_pct, 1),
                "test_spoof_corner_leak": round(test_corner_pct, 1),
            },
        })

    # Sort by fit_auc descending
    features_analysis.sort(key=lambda x: x["fit_auc"], reverse=True)
    top6 = features_analysis[:6]

    # Save summary JSON
    summary_data = {
        "generated_at": datetime.now().isoformat(),
        "fit_sessions": FIT_SESSIONS,
        "test_sessions": TEST_SESSIONS,
        "top6_features": top6,
        "all_18_features": features_analysis,
    }

    summary_json_path = os.path.join(out_dir, "fft_summary.json")
    with open(summary_json_path, mode="w", encoding="utf-8") as f_json:
        json.dump(summary_data, f_json, indent=2)

    print(f"[FFT Probe] Summary JSON written to {summary_json_path}")

    # ── Console Table Output ──────────────────────────────────────────────────
    print("\n" + "=" * 132)
    print("TOP 6 FFT MOIRÉ FEATURES (Ranked by Fit AUC on sessions 120633 & 123329)")
    print("Threshold = 98th percentile of Real frames on Fit set (False Alarm Rate ≈ 2.0%)")
    print("=" * 132)

    # Header row
    hdr_left = f"{'Rank':<4} {'Feature':<16} {'AUC':<6} {'Inv':<4} {'Thresh':<9}"
    scen_headers = [f"{short:>8}" for _, _, short in SCENARIO_DISPLAY_NAMES]
    print(hdr_left + " | " + " ".join(scen_headers))
    print("-" * 132)

    for rank_idx, item in enumerate(top6, 1):
        feat = item["feature"]
        auc_str = f"{item['fit_auc']:.4f}"
        inv_str = "YES" if item["inverted"] else "NO"
        th_str = f"{item['threshold']:>8.4f}"

        pct_vals = []
        for sess, scen, _ in SCENARIO_DISPLAY_NAMES:
            val = item["scenario_flag_pct"].get(sess, {}).get(scen, 0.0)
            pct_vals.append(f"{val:>7.1f}%")

        row_str = f"{rank_idx:<4} {feat:<16} {auc_str:<6} {inv_str:<4} {th_str:<9} | " + " ".join(pct_vals)
        print(row_str)

    print("=" * 132)

    # Aggregated Summary Table
    print("\n" + "=" * 90)
    print("AGGREGATED DETECTION RATES (% Flagged by Scenario Group)")
    print("=" * 90)
    agg_hdr = f"{'Rank':<4} {'Feature':<16} {'Fit AUC':<8} {'Fit Real':<10} {'Fit Spoof':<11} {'Test Real':<11} {'Test Sp(BL)':<13} {'Test Sp(Corner)':<15}"
    print(agg_hdr)
    print("-" * 90)
    for rank_idx, item in enumerate(top6, 1):
        s = item["summary_flag_pct"]
        print(
            f"{rank_idx:<4} {item['feature']:<16} {item['fit_auc']:<8.4f} "
            f"{s['fit_real']:>8.1f}%  {s['fit_spoof']:>9.1f}%  {s['test_real_backlight']:>9.1f}%  "
            f"{s['test_spoof_backlight']:>11.1f}%  {s['test_spoof_corner_leak']:>13.1f}%"
        )
    print("=" * 90 + "\n")

    return summary_data


# ── Main CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offline 2D-FFT Moiré and High-Frequency Feature Probe for Face Anti-Spoofing"
    )
    parser.add_argument(
        "--frames-csv",
        type=str,
        default="results/liveness_offline/widecrop_20261025_172845/frames.csv",
        help="Path to frames.csv containing face bbox and labels",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/liveness_capture",
        help="Root directory of captured frames",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for simulating frontend compression (default: 95)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: results/liveness_offline/fft_<YYYYmmdd_HHMMSS>/)",
    )

    args = parser.parse_args()

    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("results", "liveness_offline", f"fft_{timestamp}")
    else:
        out_dir = args.out

    print(f"[FFT Probe] Initializing with:")
    print(f"  --frames-csv   : {args.frames_csv}")
    print(f"  --data-root    : {args.data_root}")
    print(f"  --jpeg-quality : {args.jpeg_quality}")
    print(f"  --out          : {out_dir}")

    records, _ = process_dataset(
        frames_csv_path=args.frames_csv,
        data_root=args.data_root,
        jpeg_quality=args.jpeg_quality,
        out_dir=out_dir,
    )

    analyze_features(records, out_dir=out_dir)


if __name__ == "__main__":
    main()
