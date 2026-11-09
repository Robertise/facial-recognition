"""
frame_mask_from_photo.py – Extract eyeglass frame mask from photo of actual glasses,
register into template space (400x150) according to physical millimeter scale (IPD and px_per_cm),
and export artifacts for PhotoFrame adversarial glasses attack.

Outputs saved to --out-dir (default: data/innovation/frame/):
  - owner_frame_mask.png: binary template mask (400x150, 0/255)
  - owner_frame_lens.png: binary lens opening mask (400x150, 0/255)
  - owner_frame_meta.json: calibration metadata, dimensions, template coverage, clip ratio
  - owner_frame_overlay.png: visual inspection overlay comparing original & template

Usage:
  python -m innovation.tools.frame_mask_from_photo --image <path> --px-per-cm <float> --ipd-mm <float> [options]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TEMPLATE_W, TEMPLATE_H = 400, 150
EYES_TEMPLATE = np.array([[110.0, 80.0], [290.0, 80.0]], dtype=np.float32)


def segment_glasses_photo(
    img: np.ndarray,
    dark_threshold: int | None = None,
    rim_max_l: float | None = 70.0,
    px_per_cm: float = 152.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Segment eyeglasses photo into rim mask, lens mask, silhouette, and centroids.

    Returns:
      rim_mask: (H, W) uint8 binary (255 for rim, 0 elsewhere)
      lens_mask: (H, W) uint8 binary (255 for the two lenses, 0 elsewhere)
      silhouette: (H, W) uint8 binary (rim | lens)
      (centroid_left, centroid_right): (cx, cy) coordinates in image pixels
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_chan = lab[:, :, 0]
    px_per_mm = px_per_cm / 10.0

    # 1. Dark rim segmentation
    if dark_threshold is not None:
        _, dark_bin = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
        print(f"      [segmentation] Using dark_threshold = {dark_threshold} (grayscale override)")
    else:
        actual_rim_max_l = float(rim_max_l) if rim_max_l is not None else 70.0
        dark_bin = (l_chan < actual_rim_max_l).astype(np.uint8) * 255
        print(f"      [segmentation] Using rim_max_l = {actual_rim_max_l:.1f} (L-channel threshold)")

    # Clean dark rim with morphology (kernel ~0.5 mm)
    k_sz = max(3, int(round(0.5 * px_per_mm)))
    if k_sz % 2 == 0:
        k_sz += 1
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    rim_clean = cv2.morphologyEx(dark_bin, cv2.MORPH_CLOSE, k_morph)
    rim_clean = cv2.morphologyEx(rim_clean, cv2.MORPH_OPEN, k_morph)

    # 2. Extract largest connected component as glasses frame
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rim_clean)
    if num_labels <= 1:
        raise ValueError("Could not find any foreground object in the photo.")

    max_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    rim_clean = (labels == max_idx).astype(np.uint8) * 255

    contours_rim, _ = cv2.findContours(rim_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rim:
        raise ValueError("Could not find outer contour of glasses frame.")

    glasses_contour = max(contours_rim, key=cv2.contourArea)
    filled_frame = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_frame, [glasses_contour], -1, 255, -1)

    # Restrict rim strictly to inside the glasses contour (drops ruler, external marks)
    rim_clean = cv2.bitwise_and(rim_clean, filled_frame)

    # 3. Lens segmentation
    # Strategy A: Blue tint reflection (anti-reflective coating: HSV hue 85..135, sat >= 20, val >= 40)
    blue_mask = (
        (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 135) &
        (hsv[:, :, 1] >= 20) &
        (hsv[:, :, 2] >= 40)
    ).astype(np.uint8) * 255
    blue_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Strategy B: Enclosed holes inside the glasses contour
    holes = cv2.bitwise_and(filled_frame, cv2.bitwise_not(rim_clean))
    holes_clean = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Check holes connected components
    num_holes, labels_holes, stats_holes, centroids_holes = cv2.connectedComponentsWithStats(holes_clean)
    large_holes = [(i, stats_holes[i, cv2.CC_STAT_AREA]) for i in range(1, num_holes) if stats_holes[i, cv2.CC_STAT_AREA] > 500]
    large_holes.sort(key=lambda x: x[1], reverse=True)

    # Check blue mask connected components inside filled_frame
    blue_inside = cv2.bitwise_and(blue_clean, filled_frame)
    num_blue, labels_blue, stats_blue, centroids_blue = cv2.connectedComponentsWithStats(blue_inside)
    large_blue = [(i, stats_blue[i, cv2.CC_STAT_AREA]) for i in range(1, num_blue) if stats_blue[i, cv2.CC_STAT_AREA] > 500]
    large_blue.sort(key=lambda x: x[1], reverse=True)

    lens_mask = np.zeros((h, w), dtype=np.uint8)
    lens_centroids = []

    if len(large_holes) >= 2:
        comp1, comp2 = large_holes[0][0], large_holes[1][0]
        lens_mask[(labels_holes == comp1) | (labels_holes == comp2)] = 255
        c1 = centroids_holes[comp1]
        c2 = centroids_holes[comp2]
        lens_centroids = [c1, c2]

        # Fill small holes inside rim (all non-lens holes)
        for i in range(1, num_holes):
            if i != comp1 and i != comp2:
                rim_clean[labels_holes == i] = 255
    elif len(large_blue) >= 2:
        comp1, comp2 = large_blue[0][0], large_blue[1][0]
        lens_mask[(labels_blue == comp1) | (labels_blue == comp2)] = 255
        c1 = centroids_blue[comp1]
        c2 = centroids_blue[comp2]
        lens_centroids = [c1, c2]
    else:
        cnts_holes, _ = cv2.findContours(holes_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_holes = sorted(cnts_holes, key=cv2.contourArea, reverse=True)
        if len(cnts_holes) < 2:
            raise ValueError(f"Could not reliably detect 2 lens openings (found {len(cnts_holes)}). Please check photo contrast.")
        cv2.drawContours(lens_mask, cnts_holes[:2], -1, 255, -1)
        for c in cnts_holes[:2]:
            M = cv2.moments(c)
            lens_centroids.append(np.array([M["m10"] / (M["m00"] + 1e-6), M["m01"] / (M["m00"] + 1e-6)]))

    # Sort centroids left to right
    if lens_centroids[0][0] > lens_centroids[1][0]:
        c_left, c_right = lens_centroids[1], lens_centroids[0]
    else:
        c_left, c_right = lens_centroids[0], lens_centroids[1]

    # Smooth rim outline with Gaussian blur (sigma ≈ 0.3 mm)
    sigma = 0.3 * px_per_mm
    k_g = max(3, int(round(sigma * 4)))
    if k_g % 2 == 0:
        k_g += 1
    blurred = cv2.GaussianBlur(rim_clean.astype(np.float32), (k_g, k_g), sigma)
    rim_smooth = (blurred > 127).astype(np.uint8) * 255

    # Combine silhouette = rim | lens, strictly preserving 100% lens opening
    rim_mask = cv2.bitwise_and(rim_smooth, cv2.bitwise_not(lens_mask))
    silhouette = cv2.bitwise_or(rim_mask, lens_mask)

    return rim_mask, lens_mask, silhouette, (np.array(c_left, dtype=np.float32), np.array(c_right, dtype=np.float32))


def compute_template_affine_mm(
    c_left: np.ndarray,
    c_right: np.ndarray,
    lens_mask: np.ndarray,
    silhouette: np.ndarray,
    ipd_mm: float,
    px_per_cm: float,
    pupil_y_frac: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute 2x3 affine matrix registering glasses into 400x150 template according to physical mm scale.

    - scale = (180 / ipd_mm) / (px_per_cm / 10)
    - rotation: line connecting centroids becomes horizontal
    - x translation: midpoint of centroids -> x = 200
    - y translation: point at pupil_y_frac from top edge of lens bounding box -> y = 80
    """
    px_per_mm = px_per_cm / 10.0
    scale = (180.0 / ipd_mm) / px_per_mm

    # Vector between lens centroids
    src_v = c_right - c_left
    src_dist = float(np.linalg.norm(src_v))
    src_angle = float(np.arctan2(src_v[1], src_v[0]))
    delta_angle = -src_angle

    cos_t, sin_t = np.cos(delta_angle), np.sin(delta_angle)
    R_rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    # Horizontal: midpoint of lens centroids -> x = 200
    p_mid = (c_left + c_right) / 2.0
    mid_rot = R_rot @ p_mid
    t_x = 200.0 - scale * mid_rot[0]

    # Vertical: rotated lens bounding boxes
    cnts_lens, _ = cv2.findContours(lens_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lens = sorted(cnts_lens, key=cv2.contourArea, reverse=True)[:2]
    if len(cnts_lens) < 2:
        raise ValueError("Need at least 2 lens components to determine pupil height")

    cnts_rot = [(R_rot @ c.reshape(-1, 2).T).T for c in cnts_lens]
    # Sort left to right
    cnts_rot.sort(key=lambda pts: pts[:, 0].mean())

    y_min_L, y_max_L = float(cnts_rot[0][:, 1].min()), float(cnts_rot[0][:, 1].max())
    y_min_R, y_max_R = float(cnts_rot[1][:, 1].min()), float(cnts_rot[1][:, 1].max())

    y_top = (y_min_L + y_min_R) / 2.0
    h_box = ((y_max_L - y_min_L) + (y_max_R - y_min_R)) / 2.0
    y_pupil_rot = y_top + pupil_y_frac * h_box
    t_y = 80.0 - scale * y_pupil_rot

    R = scale * R_rot
    M = np.hstack([R, np.array([[t_x], [t_y]])]).astype(np.float32)

    # Metrics
    lens_center_dist_mm = src_dist / px_per_mm

    # Lens centroids in template
    cL_tpl = M @ np.array([c_left[0], c_left[1], 1.0])
    cR_tpl = M @ np.array([c_right[0], c_right[1], 1.0])

    # Template eye is (110, 80)
    dx_tpl = 110.0 - float(cL_tpl[0])
    dy_tpl = 80.0 - float(cL_tpl[1])
    dx_mm = dx_tpl * (ipd_mm / 180.0)
    dy_mm = dy_tpl * (ipd_mm / 180.0)
    eye_offset_mm = float(np.sqrt(dx_mm**2 + dy_mm**2))

    # Silhouette clip ratio
    ys, xs = np.where(silhouette > 127)
    if len(xs) > 0:
        pts = np.vstack([xs, ys, np.ones_like(xs)])
        tpl_pts = M @ pts
        inside = (
            (tpl_pts[0] >= 0) & (tpl_pts[0] < TEMPLATE_W) &
            (tpl_pts[1] >= 0) & (tpl_pts[1] < TEMPLATE_H)
        )
        clip_ratio = float(1.0 - inside.mean())
    else:
        clip_ratio = 0.0

    meta = {
        "scale": scale,
        "rotation_deg": float(np.rad2deg(delta_angle)),
        "t_x": float(t_x),
        "t_y": float(t_y),
        "lens_center_dist_mm": round(lens_center_dist_mm, 2),
        "eye_offset_mm": round(eye_offset_mm, 2),
        "eye_offset_dx_mm": round(dx_mm, 2),
        "eye_offset_dy_mm": round(dy_mm, 2),
        "silhouette_clip_ratio": round(clip_ratio, 5),
        "silhouette_clip_pct": round(clip_ratio * 100.0, 2),
        "pupil_y_frac": pupil_y_frac,
        "lens_centroids_template": {
            "left": [round(float(cL_tpl[0]), 2), round(float(cL_tpl[1]), 2)],
            "right": [round(float(cR_tpl[0]), 2), round(float(cR_tpl[1]), 2)],
        },
        "lens_box_rot": {
            "left_y_range": [round(y_min_L, 1), round(y_max_L, 1)],
            "right_y_range": [round(y_min_R, 1), round(y_max_R, 1)],
            "y_top_mean": round(y_top, 1),
            "h_box_mean": round(h_box, 1),
        },
    }
    return M, meta


def build_overlay(
    img: np.ndarray,
    img_warped: np.ndarray,
    mask_template: np.ndarray,
    lens_template: np.ndarray,
    M_affine: np.ndarray,
    c_left: np.ndarray,
    c_right: np.ndarray,
) -> np.ndarray:
    """Create a composite image showing both original photo alignment and template mask."""
    h_orig, w_orig = img.shape[:2]

    # 1. Overlay on original image (inverse warp template mask)
    M3 = np.vstack([M_affine, [0, 0, 1]]).astype(np.float64)
    M_inv = np.linalg.inv(M3)[:2].astype(np.float32)
    mask_orig = cv2.warpAffine(mask_template, M_inv, (w_orig, h_orig), flags=cv2.INTER_NEAREST)
    lens_orig = cv2.warpAffine(lens_template, M_inv, (w_orig, h_orig), flags=cv2.INTER_NEAREST)

    orig_vis = img.copy()
    # Green tint on rim
    green_layer = np.zeros_like(img)
    green_layer[:, :] = (0, 220, 0)
    rim_idx = mask_orig > 127
    orig_vis[rim_idx] = cv2.addWeighted(orig_vis[rim_idx], 0.5, green_layer[rim_idx], 0.5, 0)

    # Cyan tint on lens
    cyan_layer = np.zeros_like(img)
    cyan_layer[:, :] = (255, 200, 0)
    lens_idx = lens_orig > 127
    orig_vis[lens_idx] = cv2.addWeighted(orig_vis[lens_idx], 0.7, cyan_layer[lens_idx], 0.3, 0)

    # Mark centroids
    for cx, cy in (c_left, c_right):
        pt = (int(round(cx)), int(round(cy)))
        cv2.drawMarker(orig_vis, pt, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

    # Scale original image down if too large (e.g. width 800)
    target_w = 800
    scale_down = target_w / float(w_orig)
    orig_resized = cv2.resize(orig_vis, (target_w, int(h_orig * scale_down)), interpolation=cv2.INTER_AREA)

    # 2. Template view (400x150 upscaled 2x to 800x300 for inspection)
    tpl_h, tpl_w = TEMPLATE_H * 2, TEMPLATE_W * 2
    tpl_img = cv2.resize(img_warped, (tpl_w, tpl_h), interpolation=cv2.INTER_LINEAR)
    tpl_mask = cv2.resize(mask_template, (tpl_w, tpl_h), interpolation=cv2.INTER_NEAREST)
    tpl_lens = cv2.resize(lens_template, (tpl_w, tpl_h), interpolation=cv2.INTER_NEAREST)

    tpl_vis = tpl_img.copy()
    tpl_green = np.zeros_like(tpl_vis)
    tpl_green[:, :] = (0, 220, 0)
    tpl_vis[tpl_mask > 127] = cv2.addWeighted(tpl_vis[tpl_mask > 127], 0.5, tpl_green[tpl_mask > 127], 0.5, 0)

    # Target eye crosses at (110*2, 80*2) and (290*2, 80*2)
    for cx, cy in EYES_TEMPLATE:
        pt = (int(round(cx * 2)), int(round(cy * 2)))
        cv2.drawMarker(tpl_vis, pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(tpl_vis, pt, 6, (0, 0, 255), 1)

    # Also draw registered lens centroids in template space (cyan cross)
    cL_tpl = M_affine @ np.array([c_left[0], c_left[1], 1.0])
    cR_tpl = M_affine @ np.array([c_right[0], c_right[1], 1.0])
    for c_t in (cL_tpl, cR_tpl):
        pt = (int(round(c_t[0] * 2)), int(round(c_t[1] * 2)))
        cv2.drawMarker(tpl_vis, pt, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 1)

    # Add text labels
    cv2.putText(orig_resized, "Original Photo with Detected Rim & Lens", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(tpl_vis, "Registered Template (400x150, Eyes=(110,80)/(290,80), Red=Eye, Cyan=LensCenter)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Stack vertically
    canvas = np.vstack([orig_resized, tpl_vis])
    return canvas


def cut_hinge_nubs_template(
    sil_warped: np.ndarray,
    lens_warped: np.ndarray,
    keep_hinges: bool = False,
    side_thicken: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cut outer hinge nubs in 400x150 template space without touching the nose bridge.

    Only columns strictly outside the horizontal span of the lenses (x < left lens edge,
    x > right lens edge) are subjected to vertical opening (1x51 kernel), preserving
    the ~18 px nose bridge around x ≈ 200. Optionally thickens the outer 45 px bands
    horizontally by `side_thicken` px.
    """
    xs_before = np.where(sil_warped > 127)[1]
    x_range_before = [int(xs_before.min()), int(xs_before.max())] if len(xs_before) > 0 else [0, 0]

    sil_cut = sil_warped.copy()
    xs_lens = np.where(lens_warped > 127)[1]
    if not keep_hinges and len(xs_lens) > 0:
        x_left_lens = int(xs_lens.min())
        x_right_lens = int(xs_lens.max())
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51))
        if x_left_lens > 0:
            sil_cut[:, :x_left_lens] = cv2.morphologyEx(sil_cut[:, :x_left_lens], cv2.MORPH_OPEN, k_vert)
        if x_right_lens + 1 < TEMPLATE_W:
            sil_cut[:, x_right_lens + 1:] = cv2.morphologyEx(sil_cut[:, x_right_lens + 1:], cv2.MORPH_OPEN, k_vert)
    else:
        x_left_lens = int(xs_lens.min()) if len(xs_lens) > 0 else 0
        x_right_lens = int(xs_lens.max()) if len(xs_lens) > 0 else TEMPLATE_W - 1

    if side_thicken > 0:
        k_side = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * int(side_thicken) + 1, 1))
        dil_h = cv2.dilate(sil_cut, k_side)
        band_w = 45
        sil_cut[:, :band_w] = np.maximum(sil_cut[:, :band_w], dil_h[:, :band_w])
        sil_cut[:, TEMPLATE_W - band_w:] = np.maximum(sil_cut[:, TEMPLATE_W - band_w:], dil_h[:, TEMPLATE_W - band_w:])

    mask_template = cv2.bitwise_and(sil_cut, cv2.bitwise_not(lens_warped))
    xs_after = np.where(sil_cut > 127)[1]
    x_range_after = [int(xs_after.min()), int(xs_after.max())] if len(xs_after) > 0 else [0, 0]
    bridge_x200_px = int((mask_template[:, TEMPLATE_W // 2] > 127).sum())

    hinge_info = {
        "keep_hinges": bool(keep_hinges),
        "side_thicken_px": int(side_thicken),
        "lens_x_span": [x_left_lens, x_right_lens],
        "x_range_before": x_range_before,
        "x_range_after": x_range_after,
        "bridge_x200_rim_px": bridge_x200_px,
    }
    return sil_cut, mask_template, hinge_info


def compute_template_border_touch_pct(
    mask_path: Path | str | np.ndarray,
    lens_path: Path | str | np.ndarray | None,
    grow_px: float,
    clearance_mm: float = 0.0,
    ipd_mm: float = 63.0,
) -> float:
    """Compute percentage of the dilated frame mask that touches/clips at the 400x150 template border."""
    from innovation.attacks.adv_glasses import glasses

    pf = glasses.PhotoFrame(mask_path, lens_path, grow_px=grow_px, clearance_mm=clearance_mm, ipd_mm=ipd_mm)
    base = pf._load_image(pf.mask_path)
    lens = pf._load_image(pf.lens_path)
    sil = cv2.bitwise_or((base > 127).astype(np.uint8) * 255, (lens > 127).astype(np.uint8) * 255)

    pad = max(16, int(round(grow_px)) + 8)
    sil_pad = cv2.copyMakeBorder(sil, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    r = int(round(grow_px))
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        grown_pad = cv2.dilate(sil_pad, k)
    else:
        grown_pad = sil_pad

    if clearance_mm <= 0:
        lens_pad = cv2.copyMakeBorder((lens > 127).astype(np.uint8) * 255, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        mask_pad = cv2.bitwise_and(grown_pad, cv2.bitwise_not(lens_pad))
    else:
        mask_pad = grown_pad.copy()
        r_clr = max(1, int(round(clearance_mm * (180.0 / ipd_mm))))
        for cx, cy in glasses.EYES:
            cv2.circle(mask_pad, (int(round(cx)) + pad, int(round(cy)) + pad), r_clr, 0, -1)

    total_px = float((mask_pad > 127).sum())
    if total_px <= 0:
        return 0.0
    inside_px = float((mask_pad[pad:pad + TEMPLATE_H, pad:pad + TEMPLATE_W] > 127).sum())
    # Also include 1-px border touch if not overflowing pad
    border_mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = True
    inside_arr = mask_pad[pad:pad + TEMPLATE_H, pad:pad + TEMPLATE_W] > 127
    on_border_or_outside = (total_px - inside_px) + float((inside_arr & border_mask).sum())
    return round(100.0 * on_border_or_outside / total_px, 2)


def measure_clearance_table(
    mask_path: Path | str,
    lens_path: Path | str,
    attacker_frames: Path | str = "data/innovation/frames/owner/s1_day1",
    clearance_candidates: tuple[float, ...] = (16.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0),
    g_out: float = 8.0,
    ipd_mm: float = 63.0,
) -> list[dict[str, Any]]:
    """Measure coverage across s1 frames for H-ladder (fill-lens + pupil clearance)."""
    import torch
    from innovation.attacks.adv_glasses import faces, glasses

    s1_dir = Path(attacker_frames)
    if not s1_dir.exists():
        s1_dirs = list(Path(attacker_frames).parent.glob("*s1*"))
        if s1_dirs:
            s1_dir = s1_dirs[0]
        else:
            raise FileNotFoundError(f"Frames directory not found: {attacker_frames}")

    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()
    frame_files = faces.list_images(s1_dir)
    items = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        bbox = detector.detect(img)
        if bbox is None:
            continue
        _, meta = faces.enhance(faces.crop(img, bbox))
        if meta and meta.get("applied"):
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            continue
        win = faces.crop_window(bbox, img.shape)
        items.append({"A": glasses.similarity(eyes), "window": win})

    if not items:
        raise RuntimeError(f"No usable normal-light frames found in {s1_dir}")

    lens_raw = cv2.imread(str(lens_path), cv2.IMREAD_GRAYSCALE)
    if lens_raw is None:
        raise ValueError(f"Could not read lens mask: {lens_path}")
    lens_pixels = (lens_raw > 127)
    total_lens_pixels = int(lens_pixels.sum())

    g_out_mm = round(g_out * (ipd_mm / 180.0), 2)
    results = []
    print(f"\n{'='*105}")
    print(f"MEASURING H-LADDER (FILL-LENS + PUPIL CLEARANCE) ON {len(items)} USABLE NORMAL-LIGHT s1 FRAMES")
    print(f"Template: {mask_path} | Fixed g_out = {g_out:.1f} px (+{g_out_mm:.2f} mm), ipd_mm = {ipd_mm:.1f}")
    print(f"{'='*105}")
    print(f" {'clr_mm':>6} | {'clr_px':>6} | {'g_out':>5} | {'Mean Cov':>9} | {'Min Cov':>9} | {'Max Cov':>9} | {'Lens Open':>10} | {'Border %':>8} | {'Config':<10}")
    print(f"{'-'*105}")

    for clr_mm in clearance_candidates:
        shape = glasses.PhotoFrame(mask_path, lens_path, grow_px=g_out, clearance_mm=clr_mm, ipd_mm=ipd_mm)
        tpl_mask_arr = shape.mask()
        tpl_mask_t = torch.from_numpy(tpl_mask_arr)[None]
        covs = [glasses.coverage(tpl_mask_t, it["A"], it["window"]) for it in items]
        mean_cov = float(np.mean(covs))
        min_cov = float(np.min(covs))
        max_cov = float(np.max(covs))

        if total_lens_pixels > 0:
            encroached = float(tpl_mask_arr[lens_pixels].mean())
            lens_open_pct = 100.0 * max(0.0, 1.0 - encroached)
        else:
            lens_open_pct = 0.0

        clr_px = round(clr_mm * (180.0 / ipd_mm), 1)
        border_pct = compute_template_border_touch_pct(mask_path, lens_path, grow_px=g_out, clearance_mm=clr_mm, ipd_mm=ipd_mm)

        cfg_tag = f"H{int(round(clr_mm)):02d}" if int(round(clr_mm)) in (16, 13, 11, 9) else ""
        results.append({
            "clearance_mm": float(clr_mm),
            "clearance_px": clr_px,
            "grow_px": float(g_out),
            "grow_mm": g_out_mm,
            "mean_coverage": mean_cov,
            "min_coverage": min_cov,
            "max_coverage": max_cov,
            "lens_open_pct": round(lens_open_pct, 2),
            "border_touch_pct": border_pct,
            "config": cfg_tag,
        })
        print(f" {clr_mm:6.1f} | {clr_px:6.1f} | {g_out:5.1f} | {mean_cov*100:8.2f}% | {min_cov*100:8.2f}% | {max_cov*100:8.2f}% | {lens_open_pct:9.1f}% | {border_pct:7.2f}% | {cfg_tag:<10}")

    print(f"{'='*105}\n")
    return results


def measure_coverage_table(
    mask_path: Path | str,
    lens_path: Path | str,
    attacker_frames: Path | str = "data/innovation/frames/owner/s1_day1",
    grow_candidates: tuple[float, ...] | None = None,
    ipd_mm: float = 63.0,
    silhouette: np.ndarray | None = None,
    M_affine: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Measure coverage across s1 frames, grow_mm, and template clipping."""
    import torch
    from innovation.attacks.adv_glasses import faces, glasses

    if grow_candidates is None:
        grow_candidates = tuple(float(g) for g in range(0, 65, 4))

    s1_dir = Path(attacker_frames)
    if not s1_dir.exists():
        s1_dirs = list(Path(attacker_frames).parent.glob("*s1*"))
        if s1_dirs:
            s1_dir = s1_dirs[0]
        else:
            raise FileNotFoundError(f"Frames directory not found: {attacker_frames}")

    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()
    frame_files = faces.list_images(s1_dir)
    items = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        bbox = detector.detect(img)
        if bbox is None:
            continue
        _, meta = faces.enhance(faces.crop(img, bbox))
        if meta and meta.get("applied"):
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            continue
        win = faces.crop_window(bbox, img.shape)
        items.append({"A": glasses.similarity(eyes), "window": win})

    if not items:
        raise RuntimeError(f"No usable normal-light frames found in {s1_dir}")

    # Load lens mask to check opening invariant
    lens_raw = cv2.imread(str(lens_path), cv2.IMREAD_GRAYSCALE)
    if lens_raw is None:
        raise ValueError(f"Could not read lens mask: {lens_path}")
    lens_pixels = (lens_raw > 127)
    total_lens_pixels = int(lens_pixels.sum())

    # Pre-compute padded canvas for clip calculation if silhouette & M_affine provided
    sil_pad = None
    PAD = 120
    if silhouette is not None and M_affine is not None:
        W_pad, H_pad = TEMPLATE_W + 2 * PAD, TEMPLATE_H + 2 * PAD
        M_pad = M_affine.copy()
        M_pad[0, 2] += PAD
        M_pad[1, 2] += PAD
        sil_pad = cv2.warpAffine(silhouette, M_pad, (W_pad, H_pad), flags=cv2.INTER_NEAREST)

    results = []
    print(f"\n{'='*95}")
    print(f"MEASURING COVERAGE ON {len(items)} USABLE NORMAL-LIGHT s1 FRAMES")
    print(f"Template: {mask_path}")
    print(f"{'='*95}")
    print(f" {'grow_px':>7} | {'grow_mm':>7} | {'Mean Cov':>9} | {'Min Cov':>9} | {'Max Cov':>9} | {'Lens Open':>10} | {'Clip %':>7} | {'Note':<20}")
    print(f"{'-'*95}")

    for g in grow_candidates:
        shape = glasses.PhotoFrame(mask_path, lens_path, grow_px=g)
        tpl_mask_arr = shape.mask()
        tpl_mask_t = torch.from_numpy(tpl_mask_arr)[None]
        covs = [glasses.coverage(tpl_mask_t, it["A"], it["window"]) for it in items]
        mean_cov = float(np.mean(covs))
        min_cov = float(np.min(covs))
        max_cov = float(np.max(covs))

        # Check lens opening remaining
        if total_lens_pixels > 0:
            encroached = float(tpl_mask_arr[lens_pixels].mean())
            lens_open_pct = 100.0 * max(0.0, 1.0 - encroached)
        else:
            lens_open_pct = 100.0

        grow_mm = round(g * (ipd_mm / 180.0), 2)

        # Silhouette clipping outside template 400x150
        clip_pct = 0.0
        if sil_pad is not None:
            if g > 0:
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * round(g) + 1), int(2 * round(g) + 1)))
                grown_sil = cv2.dilate(sil_pad, k_dil)
            else:
                grown_sil = sil_pad
            tot_px = float((grown_sil > 127).sum())
            in_px = float((grown_sil[PAD:PAD + TEMPLATE_H, PAD:PAD + TEMPLATE_W] > 127).sum())
            clip_pct = round(100.0 * (1.0 - in_px / tot_px), 2) if tot_px > 0 else 0.0

        note = ""
        if abs(mean_cov - 0.110) < 0.008:
            note = "~11% target"
        elif abs(mean_cov - 0.150) < 0.008:
            note = "~15% target"

        results.append({
            "grow_px": g,
            "grow_mm": grow_mm,
            "mean_coverage": mean_cov,
            "min_coverage": min_cov,
            "max_coverage": max_cov,
            "lens_open_pct": lens_open_pct,
            "clip_pct": clip_pct,
            "note": note,
        })

        print(f" {g:7.1f} | {grow_mm:7.2f} | {mean_cov*100:8.2f}% | {min_cov*100:8.2f}% | {max_cov*100:8.2f}% | {lens_open_pct:9.1f}% | {clip_pct:6.2f}% | {note:<20}")

    print(f"{'='*95}")

    best_11 = min(results, key=lambda r: abs(r["mean_coverage"] - 0.110))
    best_15 = min(results, key=lambda r: abs(r["mean_coverage"] - 0.150))
    print(f"\n  [Ladder Targets] Closest to 11.0%: grow_px = {best_11['grow_px']:.1f} ({best_11['grow_mm']:.2f} mm) -> mean = {best_11['mean_coverage']*100:.2f}% (clip: {best_11['clip_pct']:.2f}%)")
    print(f"  [Ladder Targets] Closest to 15.0%: grow_px = {best_15['grow_px']:.1f} ({best_15['grow_mm']:.2f} mm) -> mean = {best_15['mean_coverage']*100:.2f}% (clip: {best_15['clip_pct']:.2f}%)\n")

    return results


def process_photo(
    image_path: Path | str,
    ipd_mm: float,
    px_per_cm: float,
    pupil_y_frac: float = 0.5,
    out_dir: Path | str = "data/innovation/frame",
    dark_threshold: int | None = None,
    rim_max_l: float | None = 70.0,
    keep_hinges: bool = False,
    side_thicken: int = 0,
    g_out: float = 8.0,
    measure_coverage: bool = False,
    attacker_frames: Path | str = "data/innovation/frames/owner/s1_day1",
) -> dict[str, Any]:
    """Run full extraction pipeline registered to mm, save artifacts, and optionally measure coverage."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Glasses photo not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Backup existing artifacts to _v1 if not yet backed up, then _v2 if _v1 exists and _v2 does not
    import shutil
    v1_dir = out_path / "_v1"
    v2_dir = out_path / "_v2"
    if not v1_dir.exists():
        v1_dir.mkdir(parents=True, exist_ok=True)
        for art in ["owner_frame_mask.png", "owner_frame_lens.png", "owner_frame_meta.json", "owner_frame_overlay.png"]:
            f_orig = out_path / art
            if f_orig.exists():
                shutil.copy2(str(f_orig), str(v1_dir / art))
        print(f"      [backup] Backed up original artifacts to {v1_dir}")
    elif not v2_dir.exists():
        v2_dir.mkdir(parents=True, exist_ok=True)
        for art in ["owner_frame_mask.png", "owner_frame_lens.png", "owner_frame_meta.json", "owner_frame_overlay.png"]:
            f_orig = out_path / art
            if f_orig.exists():
                shutil.copy2(str(f_orig), str(v2_dir / art))
        print(f"      [backup] Backed up v2 artifacts to {v2_dir}")

    print(f"\n[1/4] Segmenting eyeglasses photo: {img_path.name} ({img.shape[1]}x{img.shape[0]})...")
    rim_mask, lens_mask, silhouette, (c_left, c_right) = segment_glasses_photo(
        img, dark_threshold=dark_threshold, rim_max_l=rim_max_l, px_per_cm=px_per_cm
    )

    print(f"      Detected lens centroids in photo:")
    print(f"        Left lens:  ({c_left[0]:.1f}, {c_left[1]:.1f})")
    print(f"        Right lens: ({c_right[0]:.1f}, {c_right[1]:.1f})")

    print(f"[2/4] Registering to template space (400x150) using physical mm scale:")
    print(f"      ipd_mm = {ipd_mm:.1f} mm, px_per_cm = {px_per_cm:.1f}, pupil_y_frac = {pupil_y_frac:.2f}...")
    M_affine, transform_meta = compute_template_affine_mm(
        c_left=c_left,
        c_right=c_right,
        lens_mask=lens_mask,
        silhouette=silhouette,
        ipd_mm=ipd_mm,
        px_per_cm=px_per_cm,
        pupil_y_frac=pupil_y_frac,
    )
    print(f"      Scale factor: {transform_meta['scale']:.5f}, Rotation: {transform_meta['rotation_deg']:.2f}°")

    sil_warped = cv2.warpAffine(silhouette, M_affine, (TEMPLATE_W, TEMPLATE_H), flags=cv2.INTER_NEAREST)
    lens_warped = cv2.warpAffine(lens_mask, M_affine, (TEMPLATE_W, TEMPLATE_H), flags=cv2.INTER_NEAREST)
    sil_warped, mask_template, hinge_info = cut_hinge_nubs_template(
        sil_warped=sil_warped,
        lens_warped=lens_warped,
        keep_hinges=keep_hinges,
        side_thicken=side_thicken,
    )
    print(f"      [hinge cut] x-range before: {hinge_info['x_range_before'][0]}–{hinge_info['x_range_before'][1]} "
          f"-> after: {hinge_info['x_range_after'][0]}–{hinge_info['x_range_after'][1]} "
          f"(keep_hinges={keep_hinges}, side_thicken={side_thicken}px)")
    print(f"      [bridge check] Center column x=200 rim pixels: {hinge_info['bridge_x200_rim_px']} px")
    img_warped = cv2.warpAffine(img, M_affine, (TEMPLATE_W, TEMPLATE_H), flags=cv2.INTER_LINEAR)

    # Compute bounding boxes and measurements
    contours_lens, _ = cv2.findContours(lens_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours_lens]
    boxes.sort(key=lambda b: b[0])  # sort left to right

    meas_left = boxes[0] if len(boxes) > 0 else (0, 0, 0, 0)
    meas_right = boxes[1] if len(boxes) > 1 else (0, 0, 0, 0)
    bridge_px = float(meas_right[0] - (meas_left[0] + meas_left[2])) if len(boxes) >= 2 else 0.0

    cnt_sil, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_sil, y_sil, w_sil, h_sil = cv2.boundingRect(max(cnt_sil, key=cv2.contourArea)) if cnt_sil else (0, 0, 0, 0)

    # Base template coverage
    cov_template = float((mask_template > 127).mean())
    lens_template_px = int((lens_warped > 127).sum())

    px_per_mm = px_per_cm / 10.0

    # Build metadata dictionary
    meta: dict[str, Any] = {
        "source_image": str(img_path.resolve()),
        "image_size": [img.shape[0], img.shape[1]],
        "template_size": [TEMPLATE_H, TEMPLATE_W],
        "target_eyes": EYES_TEMPLATE.tolist(),
        "ipd_mm": ipd_mm,
        "px_per_cm": px_per_cm,
        "pupil_y_frac": pupil_y_frac,
        "rim_max_l": float(rim_max_l) if rim_max_l is not None else 70.0,
        "hinge_cut": hinge_info,
        "detected_centroids_photo": {
            "left": [float(c_left[0]), float(c_left[1])],
            "right": [float(c_right[0]), float(c_right[1])],
        },
        "transform": transform_meta,
        "coverage_base_template": cov_template,
        "lens_area_template_px": lens_template_px,
        "measurements_px": {
            "lens_left_wh": [meas_left[2], meas_left[3]],
            "lens_right_wh": [meas_right[2], meas_right[3]],
            "bridge_px": bridge_px,
            "total_width_px": w_sil,
            "total_height_px": h_sil,
        },
        "measurements_mm": {
            "lens_left_wh": [round(meas_left[2] / px_per_mm, 1), round(meas_left[3] / px_per_mm, 1)],
            "lens_right_wh": [round(meas_right[2] / px_per_mm, 1), round(meas_right[3] / px_per_mm, 1)],
            "bridge_mm": round(bridge_px / px_per_mm, 1),
            "total_width_mm": round(w_sil / px_per_mm, 1),
            "total_height_mm": round(h_sil / px_per_mm, 1),
            "reference_rough_measurements": "lens ~50x38-40mm, bridge ~18-20mm, total ~130mm",
            "previous_run_comparison_mm": {
                "lens_left_wh": [48.0, 35.7],
                "lens_right_wh": [47.3, 36.3],
                "bridge_mm": 23.2,
                "total_width_mm": 150.4,
                "total_height_mm": 45.7,
            },
        },
    }

    print(f"[3/4] Saving artifacts to {out_path}...")
    mask_file = out_path / "owner_frame_mask.png"
    lens_file = out_path / "owner_frame_lens.png"
    meta_file = out_path / "owner_frame_meta.json"
    overlay_file = out_path / "owner_frame_overlay.png"

    cv2.imwrite(str(mask_file), mask_template)
    cv2.imwrite(str(lens_file), lens_warped)
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    overlay = build_overlay(
        img, img_warped, mask_template, lens_warped, M_affine, c_left, c_right
    )
    cv2.imwrite(str(overlay_file), overlay)

    print(f"      Saved:")
    print(f"        - {mask_file.name} (rim mask template)")
    print(f"        - {lens_file.name} (lens opening template)")
    print(f"        - {meta_file.name} (calibration & metadata)")
    print(f"        - {overlay_file.name} (visual verification)")

    print(f"\n  Registration & Dimensional Analysis (ipd={ipd_mm:.1f} mm, px_per_cm={px_per_cm:.1f}):")
    print(f"    Khoang cach tam trong:          {transform_meta['lens_center_dist_mm']:.2f} mm")
    print(f"    Khoang cach mat (IPD):          {ipd_mm:.2f} mm")
    print(f"    Do lech mat so voi tam trong:   {transform_meta['eye_offset_mm']:.2f} mm (dx = {transform_meta['eye_offset_dx_mm']:+.2f} mm, dy = {transform_meta['eye_offset_dy_mm']:+.2f} mm)")
    print(f"    Ti le bong gong bi cat:         {transform_meta['silhouette_clip_pct']:.2f}% (clip ratio: {transform_meta['silhouette_clip_ratio']:.5f})")

    mm = meta["measurements_mm"]
    print(f"\n  Estimated physical dimensions (vs previous run):")
    print(f"    Left lens:   {mm['lens_left_wh'][0]} x {mm['lens_left_wh'][1]} mm  (prev: 48.0 x 35.7 mm)")
    print(f"    Right lens:  {mm['lens_right_wh'][0]} x {mm['lens_right_wh'][1]} mm  (prev: 47.3 x 36.3 mm)")
    print(f"    Bridge:      {mm['bridge_mm']} mm  (prev: 23.2 mm, diff: {round(mm['bridge_mm'] - 23.2, 1):+0.1f} mm)")
    print(f"    Total width: {mm['total_width_mm']} mm  (prev: 150.4 mm, diff: {round(mm['total_width_mm'] - 150.4, 1):+0.1f} mm)")
    print(f"    Total height:{mm['total_height_mm']} mm  (prev: 45.7 mm, diff: {round(mm['total_height_mm'] - 45.7, 1):+0.1f} mm - shadow removed)")
    print(f"    (Reference owner rough values: lens ~50x38-40 mm, bridge ~18-20 mm, total ~130 mm)")

    # 4. Coverage measurement if requested
    if measure_coverage:
        print(f"\n[4/4] Measuring coverage across s1 frames...")
        cov_results = measure_coverage_table(
            mask_path=mask_file,
            lens_path=lens_file,
            attacker_frames=attacker_frames,
            grow_candidates=tuple(float(g) for g in range(0, 65, 4)),
            ipd_mm=ipd_mm,
            silhouette=silhouette,
            M_affine=M_affine,
        )
        clr_results = measure_clearance_table(
            mask_path=mask_file,
            lens_path=lens_file,
            attacker_frames=attacker_frames,
            clearance_candidates=(16.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0),
            g_out=g_out,
            ipd_mm=ipd_mm,
        )
        meta["s1_coverage_ladder"] = cov_results
        meta["s1_clearance_ladder"] = clr_results
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"\n[4/4] Skipping s1 coverage measurement (use --measure-coverage to run).")

    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to photograph of eyeglasses")
    parser.add_argument("--px-per-cm", type=float, required=True, help="Calibration factor: pixels per centimeter from ruler")
    parser.add_argument("--ipd-mm", type=float, required=True, help="Interpupillary distance in millimeters (e.g. 63.0)")
    parser.add_argument("--pupil-y-frac", type=float, default=0.5, help="Fraction from top of lens bounding box to pupil height (default: 0.5)")
    parser.add_argument("--out-dir", default="data/innovation/frame", help="Output directory for masks and metadata")
    parser.add_argument("--dark-threshold", type=int, default=None, help="Override threshold for dark rim detection (0-255)")
    parser.add_argument("--rim-max-l", type=float, default=70.0, help="Maximum L channel threshold for dark rim segmentation (default: 70.0)")
    parser.add_argument("--keep-hinges", action="store_true", help="Do not cut outer hinge nubs in template space")
    parser.add_argument("--side-thicken", type=int, default=0, help="Horizontally dilate outer 45px bands by PX pixels after cutting hinges (default: 0)")
    parser.add_argument("--g-out", type=float, default=8.0, help="Fixed outward grow_px for clearance coverage table (default: 8.0)")
    parser.add_argument("--measure-coverage", action="store_true", help="Measure s1 frame coverage for grow_px and clearance_mm ladders")
    parser.add_argument("--attacker-frames", default="data/innovation/frames/owner/s1_day1", help="Path to s1 frames directory")

    args = parser.parse_args()
    process_photo(
        image_path=args.image,
        ipd_mm=args.ipd_mm,
        px_per_cm=args.px_per_cm,
        pupil_y_frac=args.pupil_y_frac,
        out_dir=args.out_dir,
        dark_threshold=args.dark_threshold,
        rim_max_l=args.rim_max_l,
        keep_hinges=args.keep_hinges,
        side_thicken=args.side_thicken,
        g_out=args.g_out,
        measure_coverage=args.measure_coverage,
        attacker_frames=args.attacker_frames,
    )


if __name__ == "__main__":
    main()
