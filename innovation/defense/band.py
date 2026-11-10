"""
band.py - Eyeglass region template ('band') construction and projection onto 112x112 face crops.

Constructs `data/innovation/defense/band_template.png` (400 x 150, matching `glasses.py` template space)
as the union of:
  - FrameShape of C3 (`20261104_140557_advglasses_impersonate`)
  - FrameShape of R2b (`20261106_132526_advglasses_impersonate`)
  - PhotoFrame owner with grow_px=16, clearance_mm=0 (fill-lens: full rim|lens silhouette dilated by 16 px)
followed by a 4 px morphological dilation.

Provides `band_in_crop(img, bbox, eyes) -> mask112` (uint8 112x112 in {0, 1}, or None when eyes/bbox missing).
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
from innovation.eval.runs import write_json

DEFAULT_C3_RUN = Path("results/innovation/20261104_140557_advglasses_impersonate")
DEFAULT_R2B_RUN = Path("results/innovation/20261106_132526_advglasses_impersonate")
DEFAULT_PHOTO_MASK = Path("data/innovation/frame/owner_frame_mask.png")
DEFAULT_PHOTO_LENS = Path("data/innovation/frame/owner_frame_lens.png")
DEFAULT_BAND_TEMPLATE = Path("data/innovation/defense/band_template.png")
DEFAULT_BAND_META = Path("data/innovation/defense/band_meta.json")
DEFAULT_OWNER_S1 = Path("data/innovation/frames/owner/s1_day1")

_CACHED_BAND_TEMPLATE: Optional[np.ndarray] = None


def _load_shape_from_run(run_dir: Path, fallback_thickness: float, fallback_rim_scale: float) -> tuple[glasses.FrameShape | glasses.PhotoFrame, dict[str, Any]]:
    attack_path = Path(run_dir) / "attack.json"
    if attack_path.exists():
        attack = json.loads(attack_path.read_text(encoding="utf-8"))
        shape = glasses.shape_from_attack(attack)
        return shape, {
            "run_dir": str(run_dir),
            "frame_shape": attack.get("frame_shape"),
            "from_attack_json": True,
        }
    shape = glasses.FrameShape(thickness=fallback_thickness, rim_scale=fallback_rim_scale)
    return shape, {
        "run_dir": str(run_dir),
        "frame_shape": {"thickness": fallback_thickness, "rim_scale": fallback_rim_scale},
        "from_attack_json": False,
    }


def build_band_template(
    c3_run: Path = DEFAULT_C3_RUN,
    r2b_run: Path = DEFAULT_R2B_RUN,
    photo_mask_path: Path = DEFAULT_PHOTO_MASK,
    photo_lens_path: Path = DEFAULT_PHOTO_LENS,
    grow_px: float = 16.0,
    clearance_mm: float = 0.0,
    dilate_px: int = 4,
    out_template: Optional[Path] = DEFAULT_BAND_TEMPLATE,
    out_meta: Optional[Path] = DEFAULT_BAND_META,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the 150x400 band template (float32 in {0.0, 1.0}) and metadata dict."""
    c3_shape, c3_info = _load_shape_from_run(c3_run, fallback_thickness=55.0, fallback_rim_scale=1.2)
    r2b_shape, r2b_info = _load_shape_from_run(r2b_run, fallback_thickness=85.0, fallback_rim_scale=1.42)

    m_c3 = (c3_shape.mask() > 0).astype(np.uint8) * 255
    m_r2b = (r2b_shape.mask() > 0).astype(np.uint8) * 255

    # PhotoFrame owner with grow_px=16, clearance_mm=0 (fill-lens: full rim|lens silhouette dilated by grow_px)
    if Path(photo_mask_path).exists():
        base_img = cv2.imread(str(photo_mask_path), cv2.IMREAD_GRAYSCALE)
        lens_img = cv2.imread(str(photo_lens_path), cv2.IMREAD_GRAYSCALE) if Path(photo_lens_path).exists() else np.zeros_like(base_img)
        if base_img is None:
            raise ValueError(f"Could not read PhotoFrame mask: {photo_mask_path}")
        if lens_img is None:
            lens_img = np.zeros_like(base_img)
        if base_img.shape[:2] != (glasses.TEMPLATE_H, glasses.TEMPLATE_W):
            base_img = cv2.resize(base_img, (glasses.TEMPLATE_W, glasses.TEMPLATE_H), interpolation=cv2.INTER_NEAREST)
        if lens_img.shape[:2] != (glasses.TEMPLATE_H, glasses.TEMPLATE_W):
            lens_img = cv2.resize(lens_img, (glasses.TEMPLATE_W, glasses.TEMPLATE_H), interpolation=cv2.INTER_NEAREST)
        silhouette = cv2.bitwise_or((base_img > 127).astype(np.uint8) * 255, (lens_img > 127).astype(np.uint8) * 255)
        pf = glasses.PhotoFrame(
            mask_path=silhouette,
            lens_path=None,
            grow_px=float(grow_px),
            clearance_mm=float(clearance_mm),
        )
        m_pf = (pf.mask() > 0).astype(np.uint8) * 255
        pf_info = {
            "mask_path": str(photo_mask_path),
            "lens_path": str(photo_lens_path),
            "grow_px": float(grow_px),
            "clearance_mm": float(clearance_mm),
            "mode": "fill-lens",
            "area_px": int((m_pf > 0).sum()),
        }
    else:
        m_pf = np.zeros((glasses.TEMPLATE_H, glasses.TEMPLATE_W), dtype=np.uint8)
        pf_info = {
            "mask_path": str(photo_mask_path),
            "lens_path": str(photo_lens_path),
            "grow_px": float(grow_px),
            "clearance_mm": float(clearance_mm),
            "mode": "missing_fallback",
            "area_px": 0,
        }

    union_u8 = cv2.bitwise_or(cv2.bitwise_or(m_c3, m_r2b), m_pf)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1))
        band_u8 = cv2.dilate(union_u8, k)
    else:
        band_u8 = union_u8

    band_float = (band_u8 > 127).astype(np.float32)
    total_px = int(glasses.TEMPLATE_H * glasses.TEMPLATE_W)
    band_px = int((band_u8 > 127).sum())

    meta: dict[str, Any] = {
        "template_width": int(glasses.TEMPLATE_W),
        "template_height": int(glasses.TEMPLATE_H),
        "total_template_px": total_px,
        "band_area_px": band_px,
        "band_area_fraction": round(float(band_px / total_px), 6),
        "dilate_px": int(dilate_px),
        "sources": {
            "C3": {**c3_info, "area_px": int((m_c3 > 0).sum())},
            "R2b": {**r2b_info, "area_px": int((m_r2b > 0).sum())},
            "PhotoFrame_owner_fill": pf_info,
        },
    }

    if out_template is not None:
        out_template = Path(out_template)
        out_template.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_template), band_u8)
    if out_meta is not None:
        write_json(Path(out_meta), meta)

    global _CACHED_BAND_TEMPLATE
    _CACHED_BAND_TEMPLATE = band_float
    return band_float, meta


def load_band_template(path: Path = DEFAULT_BAND_TEMPLATE) -> np.ndarray:
    """Return the (150, 400) float32 band template in {0.0, 1.0}, building it if needed."""
    global _CACHED_BAND_TEMPLATE
    if path == DEFAULT_BAND_TEMPLATE and _CACHED_BAND_TEMPLATE is not None:
        return _CACHED_BAND_TEMPLATE

    p = Path(path)
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape[:2] != (glasses.TEMPLATE_H, glasses.TEMPLATE_W):
                img = cv2.resize(img, (glasses.TEMPLATE_W, glasses.TEMPLATE_H), interpolation=cv2.INTER_NEAREST)
            arr = (img > 127).astype(np.float32)
            if path == DEFAULT_BAND_TEMPLATE:
                _CACHED_BAND_TEMPLATE = arr
            return arr

    arr, _ = build_band_template(out_template=p if path == DEFAULT_BAND_TEMPLATE else None,
                                 out_meta=DEFAULT_BAND_META if path == DEFAULT_BAND_TEMPLATE else None)
    return arr


def band_in_crop(
    img: np.ndarray,
    bbox: Optional[list[int]],
    eyes: Optional[np.ndarray],
    band_template: Optional[np.ndarray] = None,
    size: int = 112,
) -> Optional[np.ndarray]:
    """Warp `band_template` into `img` via `glasses.similarity(eyes)`, crop with `faces.crop_window(bbox, ...)`,
    and resize to `(size, size)` (112x112).

    Returns a binary uint8 mask of shape `(size, size)` with values in `{0, 1}`,
    or `None` if `bbox` or `eyes` is None / invalid.
    """
    if img is None or bbox is None or eyes is None:
        return None
    eyes_arr = np.asarray(eyes, dtype=np.float32)
    if eyes_arr.shape != (2, 2) or not np.all(np.isfinite(eyes_arr)):
        return None

    x0, y0, x1, y1 = faces.crop_window(bbox, img.shape)
    if x1 <= x0 or y1 <= y0:
        return None

    tmpl = band_template if band_template is not None else load_band_template()
    mask_t = torch.from_numpy(np.ascontiguousarray(tmpl, dtype=np.float32))[None]  # (1, 150, 400)

    A = glasses.similarity(eyes_arr)
    window = (x0, y0, x1, y1)
    grid = glasses._grid(A, window, device=mask_t.device)
    m_win = F.grid_sample(mask_t[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m_112 = F.interpolate(m_win, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()

    # Any pixel touched by bilinear interpolation of the dilated band template belongs to the band
    return (m_112 > 0.01).astype(np.uint8)


def measure_session_band_coverage(
    session_dir: Path = DEFAULT_OWNER_S1,
    band_template: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Measure average band coverage on 112x112 crops across frames of `session_dir`."""
    files = faces.list_images(Path(session_dir)) if Path(session_dir).exists() else []
    if not files:
        return {"session_dir": str(session_dir), "frames_total": 0, "frames_valid": 0, "mean": 0.0, "min": 0.0, "max": 0.0}

    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()
    tmpl = band_template if band_template is not None else load_band_template()

    coverages: list[float] = []
    no_face = 0
    no_eyes = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        bbox = detector.detect(img)
        if bbox is None:
            no_face += 1
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            no_eyes += 1
            continue
        m112 = band_in_crop(img, bbox, eyes, band_template=tmpl, size=112)
        if m112 is None:
            no_eyes += 1
            continue
        coverages.append(float(m112.mean()))

    return {
        "session_dir": str(session_dir),
        "frames_total": len(files),
        "frames_valid": len(coverages),
        "no_face": no_face,
        "no_eyes": no_eyes,
        "mean": round(float(np.mean(coverages)), 6) if coverages else 0.0,
        "min": round(float(np.min(coverages)), 6) if coverages else 0.0,
        "max": round(float(np.max(coverages)), 6) if coverages else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build band_template.png and report band coverage on owner/s1_day1.")
    ap.add_argument("--c3-run", default=str(DEFAULT_C3_RUN))
    ap.add_argument("--r2b-run", default=str(DEFAULT_R2B_RUN))
    ap.add_argument("--photo-mask", default=str(DEFAULT_PHOTO_MASK))
    ap.add_argument("--photo-lens", default=str(DEFAULT_PHOTO_LENS))
    ap.add_argument("--grow-px", type=float, default=16.0)
    ap.add_argument("--clearance-mm", type=float, default=0.0)
    ap.add_argument("--dilate-px", type=int, default=4)
    ap.add_argument("--session-dir", default=str(DEFAULT_OWNER_S1))
    ap.add_argument("--out-template", default=str(DEFAULT_BAND_TEMPLATE))
    ap.add_argument("--out-meta", default=str(DEFAULT_BAND_META))
    args = ap.parse_args()

    band_tmpl, meta = build_band_template(
        c3_run=Path(args.c3_run),
        r2b_run=Path(args.r2b_run),
        photo_mask_path=Path(args.photo_mask),
        photo_lens_path=Path(args.photo_lens),
        grow_px=args.grow_px,
        clearance_mm=args.clearance_mm,
        dilate_px=args.dilate_px,
        out_template=Path(args.out_template),
        out_meta=Path(args.out_meta),
    )

    cov = measure_session_band_coverage(Path(args.session_dir), band_template=band_tmpl)
    meta["owner_s1_day1_coverage"] = cov
    write_json(Path(args.out_meta), meta)

    print(
        f"Band template saved to {args.out_template} "
        f"({meta['band_area_px']}/{meta['total_template_px']} px = {meta['band_area_fraction']*100:.2f}%)"
    )
    print(
        f"Coverage on {cov['session_dir']} ({cov['frames_valid']}/{cov['frames_total']} frames): "
        f"mean={cov['mean']*100:.2f}%  min={cov['min']*100:.2f}%  max={cov['max']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
