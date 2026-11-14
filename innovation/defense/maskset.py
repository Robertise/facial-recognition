"""
maskset.py - Eye-anchored multi-mask set for certified face defense.

Canvas: 800 x 900 px. The 400 x 150 eyeglass template is placed at offset (200, 300),
so template eyes sit at (310, 380) and (490, 380).
Defines:
- Eye-anchored affine similarity transform to canvas coordinates.
- mask_in_window / mask_in_crop projection into 112x112 crops.
- Candidate masks:
    * Glasses family G_r, r in {0, 16, 32, 48} px (dilations of band).
    * Lower face L (mask, scarf) from LFW train statistics.
    * Forehead F (hat, headband) from LFW train statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from innovation.attacks.adv_glasses import faces, glasses
from innovation.defense.band import band_in_crop, load_band_template
from innovation.eval.runs import write_json

CANVAS_W = 800
CANVAS_H = 900
CANVAS_OFF = (200, 300)
DELTA = 8  # margin in template pixels
Q = 0.05   # frequency threshold for statistical masks

CANVAS_EYES = glasses.EYES + np.array(CANVAS_OFF, dtype=np.float32)  # [[310, 380], [490, 380]]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]
BROW_TOP = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
MASK_NAMES = ["G0", "G16", "G32", "G48", "L", "F"]

DEFAULT_MASKSET_DIR = Path("data/innovation/defense/maskset")
DEFAULT_LANDMARKS_DIR = Path("data/innovation/defense/lfw_landmarks")
DEFAULT_OWNER_S1 = Path("data/innovation/frames/owner/s1_day1")


def similarity_canvas(eyes: np.ndarray) -> np.ndarray:
    """2x3 affine matrix mapping canvas coordinates to image coordinates."""
    return glasses.similarity(eyes, src_eyes=CANVAS_EYES)


def _grid_canvas(A: np.ndarray, window: tuple[int, int, int, int], device: Any) -> torch.Tensor:
    """Sampling grid (1, h, w, 2) taking each window pixel back to canvas coordinates."""
    x0, y0, x1, y1 = window
    A3 = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    inv = np.linalg.inv(A3)[:2]
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    tx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    ty = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    gx = (2 * tx + 1) / float(CANVAS_W) - 1.0  # align_corners=False convention
    gy = (2 * ty + 1) / float(CANVAS_H) - 1.0
    return torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32)).to(device)


def mask_in_window(
    window: tuple[int, int, int, int],
    eyes: Optional[np.ndarray],
    canvas_mask: np.ndarray,
    size: int = 112,
) -> Optional[np.ndarray]:
    """Project a canvas mask (800 x 900) into a (size x size) binary mask in {0, 1}."""
    if eyes is None or window is None:
        return None
    x0, y0, x1, y1 = window
    if x1 <= x0 or y1 <= y0:
        return None

    A = similarity_canvas(eyes)
    grid = _grid_canvas(A, window, "cpu")

    mask_f32 = (canvas_mask > 127).astype(np.float32)
    mask_t = torch.from_numpy(mask_f32)[None, None]
    sampled = F.grid_sample(mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    resized = F.interpolate(sampled, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    return (resized.numpy() > 0.01).astype(np.uint8)


def mask_in_crop(
    img_shape: tuple[int, ...],
    bbox: Optional[list[int]],
    eyes: Optional[np.ndarray],
    canvas_mask: np.ndarray,
    size: int = 112,
) -> Optional[np.ndarray]:
    """Project canvas mask onto crop determined by bbox and img_shape."""
    if bbox is None or eyes is None:
        return None
    window = faces.crop_window(bbox, img_shape)
    return mask_in_window(window, eyes, canvas_mask, size=size)


def load_maskset(maskset_dir: Path = DEFAULT_MASKSET_DIR) -> dict[str, np.ndarray]:
    """Load all 6 masks from maskset_dir (returns uint8 800x900 in {0, 255})."""
    maskset_dir = Path(maskset_dir)
    masks = {}
    for name in MASK_NAMES:
        p = maskset_dir / f"{name}.png"
        if not p.exists():
            raise FileNotFoundError(f"Mask file not found: {p}")
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if arr is None or arr.shape[:2] != (CANVAS_H, CANVAS_W):
            raise ValueError(f"Invalid mask image {p}: shape {None if arr is None else arr.shape}")
        masks[name] = arr
    return masks


def build_maskset(
    landmarks_dir: Path = DEFAULT_LANDMARKS_DIR,
    crops_dir: Path = Path("data/innovation/defense/lfw_crops"),
    band_template_path: Optional[Path] = None,
    owner_s1_dir: Path = DEFAULT_OWNER_S1,
    out_dir: Path = DEFAULT_MASKSET_DIR,
    delta: int = DELTA,
    q: float = Q,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = Path(landmarks_dir)

    # 1. Build glasses family G_r, r in {0, 16, 32, 48}
    band_tmpl = load_band_template(band_template_path) if band_template_path is not None else load_band_template()
    band_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ox, oy = CANVAS_OFF
    band_canvas[oy : oy + glasses.TEMPLATE_H, ox : ox + glasses.TEMPLATE_W] = (band_tmpl > 0.5).astype(np.uint8) * 255

    regions: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    for r in (0, 16, 32, 48):
        name = f"G{r}"
        if r == 0:
            reg = band_canvas.copy()
        else:
            k_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            reg = cv2.dilate(band_canvas, k_r)
        regions[name] = reg

        r_tot = r + delta
        k_tot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_tot + 1, 2 * r_tot + 1))
        masks[name] = cv2.dilate(band_canvas, k_tot)

    # 2. Build L and F from LFW train landmarks where ok == 1
    train_shards = sorted(landmarks_dir.glob("train_*.npz"))
    freq_L = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    freq_F = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    n_used = 0
    n_used_by_shard: dict[str, int] = {}

    for ts in train_shards:
        data = np.load(ts, allow_pickle=True)
        eyes_arr = data["eyes"]
        mesh_arr = data["mesh"]
        ok_arr = data["ok"]
        n_shard = len(ok_arr)
        shard_used = 0

        for i in range(n_shard):
            if ok_arr[i] != 1:
                continue

            eyes = eyes_arr[i]
            mesh = mesh_arr[i]
            A = similarity_canvas(eyes)
            A3 = np.vstack([A, [0, 0, 1]]).astype(np.float64)
            inv_A = np.linalg.inv(A3)[:2]

            mesh_homo = np.hstack([mesh, np.ones((468, 1), dtype=np.float32)])
            mesh_c = (inv_A @ mesh_homo.T).T  # (468, 2) in canvas coordinates

            # Polygon for FACE_OVAL
            pts_oval = np.round(mesh_c[FACE_OVAL]).astype(np.int32)

            # L_i: (fill(oval) union fill(oval shifted down 27px)) intersect {y >= y(landmark 6)}
            L_i = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
            cv2.fillPoly(L_i, [pts_oval], 255)
            pts_down = pts_oval.copy()
            pts_down[:, 1] += 27
            cv2.fillPoly(L_i, [pts_down], 255)

            y_lm6 = mesh_c[6, 1]
            y_cutoff = max(0, min(CANVAS_H, int(np.floor(y_lm6))))
            L_i[:y_cutoff, :] = 0

            # F_i: {y <= min_y(BROW_TOP)} intersect {min_x(oval) <= x <= max_x(oval)}
            F_i = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
            y_min_brow = np.min(mesh_c[BROW_TOP, 1])
            x_min_oval = np.min(mesh_c[FACE_OVAL, 0])
            x_max_oval = np.max(mesh_c[FACE_OVAL, 0])

            y_top = max(0, min(CANVAS_H, int(np.ceil(y_min_brow))))
            x_left = max(0, min(CANVAS_W, int(np.floor(x_min_oval))))
            x_right = max(0, min(CANVAS_W, int(np.ceil(x_max_oval))))
            if y_top > 0 and x_right > x_left:
                F_i[0:y_top, x_left:x_right] = 255

            freq_L += (L_i > 0).astype(np.float32)
            freq_F += (F_i > 0).astype(np.float32)
            n_used += 1
            shard_used += 1

        n_used_by_shard[ts.name] = shard_used

    if n_used == 0:
        raise SystemExit(f"No valid LFW train images (ok == 1) found in {landmarks_dir}")

    freq_L /= float(n_used)
    freq_F /= float(n_used)
    reg_L = (freq_L >= q).astype(np.uint8) * 255
    reg_F = (freq_F >= q).astype(np.uint8) * 255

    regions["L"] = reg_L
    regions["F"] = reg_F

    k_delta = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * delta + 1, 2 * delta + 1))
    masks["L"] = cv2.dilate(reg_L, k_delta)
    masks["F"] = cv2.dilate(reg_F, k_delta)

    # Save PNG masks and regions
    for name in MASK_NAMES:
        cv2.imwrite(str(out_dir / f"{name}.png"), masks[name])
        cv2.imwrite(str(out_dir / f"region_{name}.png"), regions[name])

    # 3. Validation on owner/s1_day1: coverages and band_in_crop subset check
    det = faces.FaceDetector()
    eyes_fn = faces.EyeLandmarks()
    coverages: dict[str, list[float]] = {m: [] for m in MASK_NAMES}
    band_leak_pixels: list[int] = []

    if Path(owner_s1_dir).exists():
        for f in faces.list_images(Path(owner_s1_dir)):
            img = cv2.imread(str(f))
            if img is None:
                continue
            bbox = det.detect(img)
            if bbox is None:
                continue
            _, meta_enh = faces.enhance(faces.crop(img, bbox))
            if meta_enh and meta_enh.get("applied"):
                continue
            eyes = eyes_fn.eyes(img)
            if eyes is None:
                continue

            # Check coverage of each mask on 112x112 crop
            win = faces.crop_window(bbox, img.shape)
            for m in MASK_NAMES:
                m112 = mask_in_window(win, eyes, masks[m], size=112)
                cov = float(m112.mean()) if m112 is not None else 0.0
                coverages[m].append(cov)

            # Check band_in_crop subset of G0
            b112 = band_in_crop(img, bbox, eyes, band_template=band_tmpl, size=112)
            g0_112 = mask_in_window(win, eyes, masks["G0"], size=112)
            if b112 is not None and g0_112 is not None:
                # Count band pixels outside G0
                leak = int(np.sum((b112 > 0) & (g0_112 == 0)))
                band_leak_pixels.append(leak)

    cov_summary = {}
    for m in MASK_NAMES:
        c_list = coverages[m]
        cov_summary[m] = {
            "mean": round(float(np.mean(c_list)), 4) if c_list else 0.0,
            "min": round(float(np.min(c_list)), 4) if c_list else 0.0,
            "max": round(float(np.max(c_list)), 4) if c_list else 0.0,
        }

    pairwise_overlaps = {}
    for i, m1 in enumerate(MASK_NAMES):
        for m2 in MASK_NAMES[i + 1 :]:
            inter = int(np.sum((masks[m1] > 0) & (masks[m2] > 0)))
            union = int(np.sum((masks[m1] > 0) | (masks[m2] > 0)))
            pairwise_overlaps[f"{m1}_{m2}"] = {
                "intersection_px": inter,
                "iou": round(float(inter / union), 4) if union > 0 else 0.0,
            }

    meta = {
        "canvas": {"width": CANVAS_W, "height": CANVAS_H, "offset": list(CANVAS_OFF)},
        "delta": delta,
        "q": q,
        "landmarks": {"face_oval": FACE_OVAL, "brow_top": BROW_TOP, "nose_bridge": 6},
        "n_lfw_train_images_used": n_used,
        "n_lfw_train_used_by_shard": n_used_by_shard,
        "coverages_owner_s1": cov_summary,
        "pairwise_intersections": pairwise_overlaps,
        "band_subset_check": {
            "frames_checked": len(band_leak_pixels),
            "max_leak_px": int(np.max(band_leak_pixels)) if band_leak_pixels else 0,
            "pass": bool(np.max(band_leak_pixels) <= 1) if band_leak_pixels else True,
        },
    }
    write_json(out_dir / "maskset_meta.json", meta)

    # 4. Preview image: overlay 6 masks with distinct colors (4 LFW val + 2 owner/s1)
    _build_preview_image(
        masks=masks,
        out_path=out_dir / "maskset_preview.png",
        owner_s1_dir=owner_s1_dir,
        crops_dir=crops_dir,
        landmarks_dir=landmarks_dir,
    )

    return meta


def _build_preview_image(
    masks: dict[str, np.ndarray],
    out_path: Path,
    owner_s1_dir: Path,
    crops_dir: Optional[Path] = None,
    landmarks_dir: Optional[Path] = None,
) -> None:
    """Build preview image showing 6 masks overlaid with distinct colors on 4 LFW val crops + 2 owner frames."""
    colors = {
        "G0": (0, 0, 255),       # Red
        "G16": (0, 165, 255),    # Orange
        "G32": (0, 255, 255),    # Yellow
        "G48": (0, 255, 0),      # Green
        "L": (255, 0, 255),      # Magenta
        "F": (255, 255, 0),      # Cyan
    }

    sample_crops: list[np.ndarray] = []

    # 1. Get up to 4 crops from LFW val
    if crops_dir and landmarks_dir and Path(crops_dir).exists() and Path(landmarks_dir).exists():
        val_c_shards = sorted(Path(crops_dir).glob("val_*.npz"))
        for cs in val_c_shards:
            if len(sample_crops) >= 4:
                break
            ls = Path(landmarks_dir) / cs.name
            if not ls.exists():
                continue
            c_data = np.load(cs, allow_pickle=True)
            l_data = np.load(ls, allow_pickle=True)
            c_crops = c_data["crops"]
            l_eyes = l_data["eyes"]
            l_win = l_data["window"]
            l_ok = l_data["ok"]
            for i in range(len(l_ok)):
                if len(sample_crops) >= 4:
                    break
                if l_ok[i] == 1:
                    c112 = c_crops[i]
                    eyes = l_eyes[i]
                    win = tuple(l_win[i])
                    overlay = c112.copy().astype(np.float32)
                    for name in MASK_NAMES:
                        m112 = mask_in_window(win, eyes, masks[name], size=112)
                        if m112 is not None and m112.any():
                            color_arr = np.array(colors[name], dtype=np.float32)
                            idx = m112 > 0
                            overlay[idx] = overlay[idx] * 0.6 + color_arr * 0.4
                    sample_crops.append(np.clip(overlay, 0, 255).astype(np.uint8))

    while len(sample_crops) < 4:
        dummy = np.full((112, 112, 3), 128, dtype=np.uint8)
        sample_crops.append(dummy)

    # 2. Get up to 2 frames from owner/s1_day1
    det = faces.FaceDetector()
    eyes_fn = faces.EyeLandmarks()
    owner_crops: list[np.ndarray] = []

    if Path(owner_s1_dir).exists():
        for f in faces.list_images(Path(owner_s1_dir)):
            if len(owner_crops) >= 2:
                break
            img = cv2.imread(str(f))
            if img is None:
                continue
            bbox = det.detect(img)
            eyes = eyes_fn.eyes(img) if bbox else None
            if bbox and eyes is not None:
                c112 = cv2.resize(faces.crop(img, bbox), (112, 112))
                win = faces.crop_window(bbox, img.shape)
                overlay = c112.copy().astype(np.float32)
                for name in MASK_NAMES:
                    m112 = mask_in_window(win, eyes, masks[name], size=112)
                    if m112 is not None and m112.any():
                        color_arr = np.array(colors[name], dtype=np.float32)
                        idx = m112 > 0
                        overlay[idx] = overlay[idx] * 0.6 + color_arr * 0.4
                owner_crops.append(np.clip(overlay, 0, 255).astype(np.uint8))

    while len(owner_crops) < 2:
        dummy = np.full((112, 112, 3), 128, dtype=np.uint8)
        owner_crops.append(dummy)

    row = np.hstack(sample_crops + owner_crops)
    cv2.imwrite(str(out_path), row)


def main():
    ap = argparse.ArgumentParser(description="Build and evaluate the eye-anchored multi-mask set.")
    sub = ap.add_subparsers(dest="cmd")

    bld = sub.add_parser("build")
    bld.add_argument("--landmarks-dir", default=str(DEFAULT_LANDMARKS_DIR))
    bld.add_argument("--crops-dir", default=str(Path("data/innovation/defense/lfw_crops")))
    bld.add_argument("--band-template", default=None)
    bld.add_argument("--owner-s1", default=str(DEFAULT_OWNER_S1))
    bld.add_argument("--out-dir", default=str(DEFAULT_MASKSET_DIR))
    bld.add_argument("--delta", type=int, default=DELTA)
    bld.add_argument("--q", type=float, default=Q)

    args = ap.parse_args()
    if args.cmd == "build":
        meta = build_maskset(
            landmarks_dir=Path(args.landmarks_dir),
            crops_dir=Path(args.crops_dir),
            band_template_path=Path(args.band_template) if args.band_template else None,
            owner_s1_dir=Path(args.owner_s1),
            out_dir=Path(args.out_dir),
            delta=args.delta,
            q=args.q,
        )
        print(json.dumps(meta, indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
