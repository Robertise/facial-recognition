"""
adaptive.py - Adaptive attack against the purifier defense (ring around the eyeglass band).

The purifier strictly zeroes out the fixed band region around the eyes, rendering any
adversarial texture inside the band completely ineffective.
An adaptive attacker with full knowledge of the pipeline (band geometry, U-Net purifier,
face recognizer, and identity consistency detector) places the adversarial pattern on
a ring of width r px surrounding the band canvas.

Components:
1. Padded canvas (512 x 262): W' = 400 + 2*P, H' = 150 + 2*P with P = 56 px padding.
   Template eyes at (110+P, 80+P) and (290+P, 80+P).
2. Differentiable geometry & rendering on the padded canvas:
   - similarity_padded, _grid_padded, render_padded, coverage_padded, paint_padded.
3. Mask ring(r):
   - dilate(band_padded, ellipse r) AND NOT band_padded.
4. CLI commands:
   - ring: generate ring masks ring_r{r}.png, measure coverage on unenhanced s1, check disjointness.
   - optimize: joint optimization minimizing loss across both purifier and none branches.
   - eval: evaluation on held-out session across tiers (D, control_random) and modes (none, gray, purifier).

Specification: docs/innovation/plan_adaptive.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backend.face_recognizer import FaceRecognizer, _preprocess_face
from innovation.attacks.adv_glasses import eot, faces, glasses, pipeline, printability
from innovation.attacks.adv_glasses.evaluate import frame_times, jpeg, liveness_step, wilson
from innovation.attacks.adv_glasses.optimize import (
    permute_cells,
    soft_max_cos,
    split_frames,
    tv,
)
from innovation.defense.band import band_in_crop, load_band_template
from innovation.defense.eval_purifier import (
    build_gallery_for_mode,
    check_session_not_in_train_dev,
    score_against_gallery,
    transform_crop112,
)
from innovation.defense.purifier import (
    FrozenMethod3Embedder,
    PurifierUNet,
    apply_gray,
    apply_purifier,
    load_purifier_checkpoint,
)
from innovation.eval.runs import new_run, write_json

# ── Padded canvas geometry ──────────────────────────────────────────────────
TEMPLATE_W = glasses.TEMPLATE_W  # 400
TEMPLATE_H = glasses.TEMPLATE_H  # 150
PAD_PX = 56
W_PRIME = TEMPLATE_W + 2 * PAD_PX  # 512
H_PRIME = TEMPLATE_H + 2 * PAD_PX  # 262
EYES_PRIME = glasses.EYES + float(PAD_PX)  # (110+P, 80+P), (290+P, 80+P)

DEFAULT_BAND_TEMPLATE = Path("data/innovation/defense/band_template.png")
DEFAULT_DEFENSE_DIR = Path("data/innovation/defense")
DEFAULT_RING_META = DEFAULT_DEFENSE_DIR / "ring_meta.json"
DEFAULT_PURIFIER_CKPT = Path("results/innovation/20261112_004047_purifier_train/best.pt")
DEFAULT_CONSISTENCY_SUMMARY = Path("results/innovation/20261112_103208_purifier_eval/summary.json")
DEFAULT_CONSISTENCY_THRESHOLD = 0.4253
DEFAULT_CHECKPOINT = Path("checkpoints/method3_best.pth")
DEFAULT_GALLERY = Path("data/innovation/gallery")
DEFAULT_ATTACKER_FRAMES = Path("data/innovation/frames/owner")
DEFAULT_S1_DAY1 = Path("data/innovation/frames/owner/s1_day1")
DEFAULT_S6_VARY = Path("data/innovation/frames/owner/s6_vary")


def similarity_padded(dst_eyes: np.ndarray, src_eyes: np.ndarray = EYES_PRIME) -> np.ndarray:
    """2x3 affine matrix mapping padded template coordinates to frame coordinates."""
    return glasses.similarity(dst_eyes, src_eyes=src_eyes)


def _grid_padded(
    A: np.ndarray,
    window: tuple[int, int, int, int],
    device: Any,
    w_prime: int = W_PRIME,
    h_prime: int = H_PRIME,
) -> torch.Tensor:
    """Sampling grid (1, h, w, 2) taking each window pixel back to padded template coordinates."""
    x0, y0, x1, y1 = window
    A3 = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    inv = np.linalg.inv(A3)[:2]
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    tx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    ty = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    gx = (2 * tx + 1) / float(w_prime) - 1.0  # align_corners=False convention
    gy = (2 * ty + 1) / float(h_prime) - 1.0
    return torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32)).to(device)


def render_padded(
    region_bgr: torch.Tensor,
    texture_rgb: torch.Tensor,
    mask: torch.Tensor,
    A: np.ndarray,
    window: tuple[int, int, int, int],
    w_prime: int = W_PRIME,
    h_prime: int = H_PRIME,
) -> torch.Tensor:
    """Paint texture onto region_bgr using padded canvas geometry, differentiable in texture_rgb."""
    grid = _grid_padded(A, window, region_bgr.device, w_prime=w_prime, h_prime=h_prime)
    tex = texture_rgb
    if tex.shape[-2:] != (h_prime, w_prime):
        tex = F.interpolate(tex[None], size=(h_prime, w_prime), mode="bilinear", align_corners=False)[0]
    tex_bgr = tex.flip(0) * 255.0
    t = F.grid_sample(tex_bgr[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    m = F.grid_sample(mask[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    return region_bgr * (1.0 - m) + t * m


def coverage_padded(
    mask: torch.Tensor,
    A: np.ndarray,
    window: tuple[int, int, int, int],
    size: int = 112,
    w_prime: int = W_PRIME,
    h_prime: int = H_PRIME,
) -> float:
    """Share of the size x size recognition crop that the mask covers."""
    grid = _grid_padded(A, window, mask.device, w_prime=w_prime, h_prime=h_prime)
    m = F.grid_sample(mask[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    return float(m.mean())


def paint_padded(
    img: np.ndarray,
    colours_rgb: torch.Tensor,
    mask: torch.Tensor,
    A: np.ndarray,
    w_prime: int = W_PRIME,
    h_prime: int = H_PRIME,
) -> np.ndarray:
    """Composite the padded texture on the full frame (returns uint8 BGR)."""
    full = (0, 0, img.shape[1], img.shape[0])
    out = render_padded(pipeline.to_tensor(img), colours_rgb, mask, A, full, w_prime=w_prime, h_prime=h_prime)
    return out.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()


def upsample_logits_padded(
    logits: torch.Tensor,
    out_h: int = H_PRIME,
    out_w: int = W_PRIME,
) -> torch.Tensor:
    """Upsample parameter logits (e.g. 20x51) to padded template resolution using bicubic interpolation."""
    if logits.shape[-2:] == (out_h, out_w):
        return logits
    if logits.dim() == 3:
        return F.interpolate(logits[None], size=(out_h, out_w), mode="bicubic", align_corners=False)[0]
    return F.interpolate(logits, size=(out_h, out_w), mode="bicubic", align_corners=False)


# ── Ring mask generation ───────────────────────────────────────────────────

def make_band_padded(band_template: np.ndarray, pad: int = PAD_PX) -> np.ndarray:
    """Place 150x400 band_template into the center of a (150+2*pad) x (400+2*pad) canvas (returns uint8 0/255)."""
    h, w = band_template.shape[:2]
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    out[pad:pad + h, pad:pad + w] = (band_template > 0.5).astype(np.uint8) * 255
    return out


def make_ring(band_padded_u8: np.ndarray, r: int) -> np.ndarray:
    """ring(r) = dilate(band_padded, ellipse r) AND NOT band_padded. Returns uint8 {0, 255}."""
    if r <= 0:
        return np.zeros_like(band_padded_u8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(band_padded_u8, k)
    return cv2.bitwise_and(dilated, cv2.bitwise_not(band_padded_u8))


def load_or_make_ring(
    r: int,
    band_template_path: Path = DEFAULT_BAND_TEMPLATE,
    pad: int = PAD_PX,
    out_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load ring_r{r}.png if exists, otherwise generate from band_template.png and optionally save."""
    if out_dir is not None:
        target = Path(out_dir) / f"ring_r{r}.png"
        if target.exists():
            arr = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if arr is not None and arr.shape == (H_PRIME, W_PRIME):
                return arr
    tmpl = load_band_template(band_template_path)
    band_padded = make_band_padded(tmpl, pad=pad)
    ring_u8 = make_ring(band_padded, r)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"ring_r{r}.png"), ring_u8)
    return ring_u8


def check_ring_overlap_with_band(
    ring_mask_u8: np.ndarray,
    band_tmpl: np.ndarray,
    img: np.ndarray,
    bbox: list[int],
    eyes: np.ndarray,
) -> int:
    """Count overlapping pixels between warped ring and band_in_crop on a 112x112 crop."""
    band112 = band_in_crop(img, bbox, eyes, band_template=band_tmpl, size=112)
    if band112 is None:
        return 0
    win = faces.crop_window(bbox, img.shape)
    A_pad = similarity_padded(eyes)
    grid = _grid_padded(A_pad, win, device="cpu")
    mask_t = torch.from_numpy((ring_mask_u8 > 127).astype(np.float32))[None, None]
    m = F.grid_sample(mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m112 = F.interpolate(m, size=(112, 112), mode="bilinear", align_corners=False)[0, 0].numpy()
    ring112 = (m112 > 0.5).astype(np.uint8)
    return int(np.sum((ring112 > 0) & (band112 > 0)))


# ── Subcommand: ring ────────────────────────────────────────────────────────

def cmd_ring(args: argparse.Namespace) -> None:
    """Build ring masks, measure crop-112 coverage on unenhanced s1, check disjointness."""
    band_tmpl_path = Path(args.band_template)
    if not band_tmpl_path.exists():
        raise SystemExit(f"Band template not found at {band_tmpl_path}. Run 'python -m innovation.defense.band' first.")
    band_tmpl = load_band_template(band_tmpl_path)
    band_padded_u8 = make_band_padded(band_tmpl, pad=PAD_PX)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_files = faces.list_images(Path(args.attacker_frames))
    if not s1_files:
        raise SystemExit(f"No frames found in {args.attacker_frames}")

    det = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    unenhanced_frames = []
    for f in s1_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        bbox = det.detect(img)
        if bbox is None:
            continue
        _, enh_meta = faces.enhance(faces.crop(img, bbox))
        if enh_meta and enh_meta.get("applied"):
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            continue
        win = faces.crop_window(bbox, img.shape)
        A_pad = similarity_padded(eyes)
        unenhanced_frames.append({"file": str(f), "img": img, "bbox": bbox, "eyes": eyes, "win": win, "A": A_pad})

    if not unenhanced_frames:
        raise SystemExit(f"No usable unenhanced frames in {args.attacker_frames}")

    meta: dict[str, Any] = {
        "canvas": {"w": W_PRIME, "h": H_PRIME, "pad": PAD_PX, "template_w": TEMPLATE_W, "template_h": TEMPLATE_H},
        "frames_evaluated": len(unenhanced_frames),
        "source_folder": str(args.attacker_frames),
        "rings": {},
    }

    print(f"\nEvaluating ring geometries on {len(unenhanced_frames)} unenhanced frames from {args.attacker_frames}:")
    print(f"{'r (px)':<8}{'Area (px)':<12}{'Coverage Mean':<16}{'Coverage [Min, Max]':<24}{'Overlap Px (Mean)':<18}")
    print("-" * 78)

    for r in args.ring_px:
        ring_u8 = make_ring(band_padded_u8, r)
        png_path = out_dir / f"ring_r{r}.png"
        cv2.imwrite(str(png_path), ring_u8)

        ring_t = torch.from_numpy((ring_u8 > 127).astype(np.float32))[None]
        coverages = [coverage_padded(ring_t, it["A"], it["win"]) for it in unenhanced_frames]
        overlaps = [
            check_ring_overlap_with_band(ring_u8, band_tmpl, it["img"], it["bbox"], it["eyes"])
            for it in unenhanced_frames
        ]

        c_mean = float(np.mean(coverages))
        c_min = float(np.min(coverages))
        c_max = float(np.max(coverages))
        o_mean = float(np.mean(overlaps))
        o_max = int(np.max(overlaps))
        area = int(np.sum(ring_u8 > 0))

        meta["rings"][f"r{r}"] = {
            "ring_px": r,
            "approx_mm": round(r / (180.0 / 63.0), 2),
            "area_px": area,
            "png": str(png_path),
            "coverage_s1": {"mean": round(c_mean, 4), "min": round(c_min, 4), "max": round(c_max, 4)},
            "overlap_band_px_in_crop": {"mean": round(o_mean, 2), "max": o_max},
        }

        print(
            f"{r:<8}{area:<12}{c_mean * 100:.2f}%{'':<10}[{c_min * 100:.2f}%, {c_max * 100:.2f}%]{'':<6}{o_mean:.2f} px (max {o_max})"
        )

    meta_path = out_dir / "ring_meta.json"
    write_json(meta_path, meta)
    print(f"\nWrote ring masks and metadata to {meta_path}\n")


# ── Subcommand: optimize ────────────────────────────────────────────────────

def build_adaptive_optimize_galleries(
    gallery_dir: Path,
    recognizer: Any,
    detector: Any,
    eyes_finder: Any,
    band_template: np.ndarray,
    purifier_model: PurifierUNet,
    purifier_tag: str,
    device: str | torch.device,
    max_gallery: int = faces.GALLERY_PER_ID,
    cache_dir: Optional[Path] = None,
    attacker_id: Optional[str] = None,
    victim: Optional[str] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Build clean gallery E and purifier-mode gallery E_pur for adaptive optimization."""
    gallery, g_report = faces.build_gallery(
        Path(gallery_dir), recognizer, detector, max_gallery, cache_dir=cache_dir
    )
    if attacker_id and attacker_id not in gallery.embeddings:
        raise SystemExit(f"attacker '{attacker_id}' has no usable images in the gallery")
    if victim and victim not in gallery.embeddings:
        raise SystemExit(f"victim '{victim}' has no usable images in the gallery")

    gal_pur, g_pur_report = build_gallery_for_mode(
        root=Path(gallery_dir),
        recognizer=recognizer,
        detector=detector,
        eyes_finder=eyes_finder,
        mode="purifier",
        band_template=band_template,
        purifier_model=purifier_model,
        purifier_tag=purifier_tag,
        device=device,
        max_per_id=max_gallery,
        cache_dir=cache_dir,
    )

    E = {k: torch.from_numpy(v).float().to(device) for k, v in gallery.embeddings.items()}
    E_pur = {k: torch.from_numpy(v).float().to(device) for k, v in gal_pur.embeddings.items()}

    if set(E_pur) != set(E):
        diff1 = sorted(set(E) - set(E_pur))
        diff2 = sorted(set(E_pur) - set(E))
        raise SystemExit(f"E_pur and E identity sets differ: missing in E_pur={diff1}, extra in E_pur={diff2}")
    if victim and victim not in E_pur:
        raise SystemExit(f"victim '{victim}' has no usable images in purifier gallery")
    if attacker_id and attacker_id not in E_pur:
        raise SystemExit(f"attacker '{attacker_id}' has no usable images in purifier gallery")

    return E, E_pur, g_report, g_pur_report


def cmd_optimize(args: argparse.Namespace) -> None:
    """Joint optimization minimizing loss across both purifier and none branches."""
    run = new_run("adaptive_optimize")
    print(f"\n[adaptive_optimize] Run directory: {run}")

    band_tmpl_path = Path(args.band_template)
    if not band_tmpl_path.exists():
        raise SystemExit(f"Band template not found at {band_tmpl_path}")
    band_tmpl = load_band_template(band_tmpl_path)
    band_padded_u8 = make_band_padded(band_tmpl, pad=PAD_PX)

    # 1. Models and Gallery
    recognizer = FaceRecognizer(ckpt_path=args.checkpoint, strict=True)
    if hasattr(recognizer, "_model") and hasattr(recognizer._model, "parameters"):
        _orig_extract = recognizer.extract_embedding
        def _dev_aware_extract(face_bgr: np.ndarray) -> np.ndarray:
            try:
                m_dev = next(recognizer._model.parameters()).device
                tensor = _preprocess_face(face_bgr, recognizer.preprocess).to(m_dev)
                with torch.no_grad():
                    return recognizer._model.get_embedding(tensor).cpu().numpy().squeeze()
            except Exception:
                return _orig_extract(face_bgr)
        recognizer.extract_embedding = _dev_aware_extract

    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    embedder = FrozenMethod3Embedder(recognizer, device=args.device)

    purifier_ckpt = Path(args.purifier_ckpt)
    if not purifier_ckpt.exists():
        raise SystemExit(f"Purifier checkpoint not found at {purifier_ckpt}")
    purifier_model = load_purifier_checkpoint(purifier_ckpt, device=args.device)
    purifier_model.eval()

    E, E_pur, g_report, g_pur_report = build_adaptive_optimize_galleries(
        gallery_dir=Path(args.gallery),
        recognizer=recognizer,
        detector=detector,
        eyes_finder=eyes_finder,
        band_template=band_tmpl,
        purifier_model=purifier_model,
        purifier_tag=purifier_ckpt.stem,
        device=args.device,
        max_gallery=args.max_gallery,
        cache_dir=run.parent / "_cache",
        attacker_id=args.attacker_id,
        victim=args.victim,
    )
    target = args.victim
    others = [k for k in E if k != target]

    print(
        f"Purifier gallery: {len(E_pur)} identities, "
        f"{args.victim}: {g_pur_report.get(args.victim)}, "
        f"{args.attacker_id}: {g_pur_report.get(args.attacker_id)}"
    )

    # 2. Dataset split and preparation
    train_files, _, split = split_frames(
        Path(args.attacker_frames), args.eval_sessions, args.seed, args.dev_frac, args.dev_mode
    )

    items, p_report = [], {"no_face": 0, "no_mesh": 0, "enhancer_applied": 0}
    for f in train_files:
        img = cv2.imread(str(f))
        bbox = detector.detect(img) if img is not None else None
        if bbox is None:
            p_report["no_face"] += 1
            continue
        _, enh_meta = faces.enhance(faces.crop(img, bbox))
        if enh_meta and enh_meta.get("applied"):
            p_report["enhancer_applied"] += 1
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            p_report["no_mesh"] += 1
            continue
        win = faces.crop_window(bbox, img.shape)
        x0, y0, x1, y1 = win
        b112 = band_in_crop(img, bbox, eyes, band_template=band_tmpl, size=112)
        if b112 is None:
            p_report["no_mesh"] += 1
            continue
        A_pad = similarity_padded(eyes)
        items.append({
            "file": str(f),
            "window": win,
            "A": A_pad,
            "region": pipeline.to_tensor(img[y0:y1, x0:x1]),
            "band112": torch.from_numpy(b112).float(),
        })
    p_report["used"] = len(items)

    if len(items) < 4:
        raise SystemExit(f"only {len(items)} usable training frames: {p_report}")

    # 3. Mask for the specified ring-px
    ring_u8 = make_ring(band_padded_u8, args.ring_px)
    ring_mask = torch.from_numpy((ring_u8 > 127).astype(np.float32))[None].to(args.device)
    cover = [coverage_padded(ring_mask.cpu(), it["A"], it["window"]) for it in items]

    palette = torch.tensor(
        json.loads(Path(args.chart).read_text())["colours_rgb"]
        if args.chart
        else printability.chart_colours(),
        dtype=torch.float32,
        device=args.device,
    )

    # 4. Parameters and Optimizer
    tex_h, tex_w = args.tex_res[0], args.tex_res[1]
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    logits = (0.05 * torch.randn(3, tex_h, tex_w, device=args.device)).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=args.lr)

    log_rows: list[dict[str, Any]] = []
    t0 = time.time()

    print(
        f"Optimizing adaptive attack (r={args.ring_px} px, steps={args.steps}, lr={args.lr}, dev_mode={args.dev_mode})..."
    )

    for step in range(1, args.steps + 1):
        logits_full = upsample_logits_padded(logits, out_h=H_PRIME, out_w=W_PRIME)
        T = torch.sigmoid(logits_full)  # print colours in [0, 1]
        seen = T  # M = identity

        batch_idx = rng.choice(len(items), size=min(args.batch, len(items)), replace=False)
        batch = [items[i] for i in batch_idx]

        crops = []
        for it in batch:
            if args.no_eot:
                A = it["A"]
                out = render_padded(it["region"].to(args.device), seen, ring_mask, A, it["window"])
                crops.append(out)
            else:
                A = glasses.jitter(it["A"], eot.TRAIN.shift_px, eot.TRAIN.rot_deg, eot.TRAIN.scale, rng)
                out = render_padded(it["region"].to(args.device), seen, ring_mask, A, it["window"])
                crops.append(eot.photometric(out, eot.TRAIN, rng))

        # (B, 3, 112, 112) in [0, 1] BGR
        crops_112_01 = (
            torch.stack([
                F.interpolate(c[None], size=(112, 112), mode="bilinear", align_corners=False)[0]
                for c in crops
            ])
            / 255.0
        )

        # ── None branch ──────────────────────────────────────────────────────
        emb_none = embedder(crops_112_01)
        s_target_none = soft_max_cos(emb_none, E[target], args.tau)
        s_others_none = torch.stack([soft_max_cos(emb_none, E[k], args.tau) for k in others], dim=1) if others else None
        l_main_none = (1.0 - s_target_none).mean()
        l_margin_none = (
            F.relu(s_others_none - s_target_none[:, None] + args.margin).sum(1).mean()
            if others
            else torch.zeros((), device=args.device)
        )
        loss_none = l_main_none + l_margin_none

        # ── Purifier branch ──────────────────────────────────────────────────
        batch_bands = torch.stack([it["band112"] for it in batch]).to(args.device)
        purified_crops_01 = purifier_model(crops_112_01, batch_bands)
        emb_pur = embedder(purified_crops_01)
        s_target_pur = soft_max_cos(emb_pur, E_pur[target], args.tau)
        s_others_pur = torch.stack([soft_max_cos(emb_pur, E_pur[k], args.tau) for k in others], dim=1) if others else None
        l_main_pur = (1.0 - s_target_pur).mean()
        l_margin_pur = (
            F.relu(s_others_pur - s_target_pur[:, None] + args.margin).sum(1).mean()
            if others
            else torch.zeros((), device=args.device)
        )
        loss_pur = l_main_pur + l_margin_pur

        # ── Regularizers ─────────────────────────────────────────────────────
        l_tv = tv(T)
        l_nps = printability.nps(T, palette)

        total_loss = loss_pur + loss_none + args.lambda_tv * l_tv + args.lambda_nps * l_nps

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                hard_none = {k: (emb_none @ E[k].T).max(1).values for k in E}
                hard_pur = {k: (emb_pur @ E_pur[k].T).max(1).values for k in E_pur}
                best_none = torch.stack([hard_none[k] for k in sorted(E)], 1).max(1)
                best_pur = torch.stack([hard_pur[k] for k in sorted(E_pur)], 1).max(1)
                top_none_ids = [sorted(E)[i] for i in best_none.indices.tolist()]
                top_pur_ids = [sorted(E_pur)[i] for i in best_pur.indices.tolist()]
                ok_none = [(t == target and v >= 0.5) for t, v in zip(top_none_ids, best_none.values.tolist())]
                ok_pur = [(t == target and v >= 0.5) for t, v in zip(top_pur_ids, best_pur.values.tolist())]

            row = {
                "step": step,
                "loss": float(total_loss),
                "loss_pur": float(loss_pur),
                "loss_none": float(loss_none),
                "l_main_pur": float(l_main_pur),
                "l_margin_pur": float(l_margin_pur),
                "l_main_none": float(l_main_none),
                "l_margin_none": float(l_margin_none),
                "tv": float(l_tv),
                "nps": float(l_nps),
                "target_cos_pur": float(hard_pur[target].mean()),
                "target_cos_none": float(hard_none[target].mean()),
                "batch_success_pur": float(np.mean(ok_pur)),
                "batch_success_none": float(np.mean(ok_none)),
                "seconds": round(time.time() - t0, 1),
            }
            log_rows.append(row)
            print(
                " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in row.items()
                    if k in ("step", "loss", "loss_pur", "loss_none", "target_cos_pur", "target_cos_none", "batch_success_pur", "seconds")
                ),
                flush=True,
            )

    # 5. Save outputs
    with open(run / "loss.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    logits_full = upsample_logits_padded(logits, out_h=H_PRIME, out_w=W_PRIME)
    T = torch.sigmoid(logits_full).detach().cpu()
    tex_bgr = (T.flip(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(run / "texture_print.png"), tex_bgr)
    cv2.imwrite(str(run / "mask.png"), ring_u8)

    torch.save(
        {
            "logits": logits.detach().cpu(),
            "tex_res": [tex_h, tex_w],
            "ring_px": args.ring_px,
            "canvas_pad": PAD_PX,
        },
        run / "texture.pt",
    )

    write_json(run / "split.json", split)

    attack_meta = {
        "adaptive": True,
        "ring_px": int(args.ring_px),
        "canvas_pad": PAD_PX,
        "purifier_ckpt": str(args.purifier_ckpt),
        "purifier_branch_gallery": "purifier",
        "tex_res": [tex_h, tex_w],
        "mode": "impersonate",
        "attacker_id": args.attacker_id,
        "victim": args.victim,
        "target": target,
        "checkpoint": args.checkpoint,
        "model": recognizer.load_report,
        "gallery": args.gallery,
        "gallery_per_id": args.max_gallery,
        "gallery_report": g_report,
        "gallery_pur_report": g_pur_report,
        "identities": sorted(E),
        "frame_shape": {
            "frame_source": "adaptive_ring",
            "ring_px": int(args.ring_px),
            "canvas_pad": PAD_PX,
            "width": W_PRIME,
            "height": H_PRIME,
        },
        "dev_frac": args.dev_frac,
        "dev_mode": args.dev_mode,
        "no_eot": args.no_eot,
        "coverage_train": {
            "mean": float(np.mean(cover)),
            "min": float(np.min(cover)),
            "max": float(np.max(cover)),
        },
        "prepare_report": p_report,
        "final": log_rows[-1] if log_rows else {},
    }
    write_json(run / "attack.json", attack_meta)
    print(f"\nOptimization finished. Outputs saved to {run}")


# ── Subcommand: eval ────────────────────────────────────────────────────────

def run_eval_adaptive(
    run_dir: Path,
    eval_session_name: str = "s6_vary",
    attacker_frames_dir: Path = DEFAULT_ATTACKER_FRAMES,
    gallery_dir: Path = DEFAULT_GALLERY,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    purifier_ckpt_path: Optional[Path] = None,
    consistency_threshold: Optional[float] = None,
    band_template_path: Path = DEFAULT_BAND_TEMPLATE,
    liveness_enabled: bool = False,
    device: str = "cpu",
    control_seed: int = 0,
    output_dir: Optional[Path] = None,
) -> Path:
    """Core evaluation implementation for adaptive attack runs."""
    run_dir = Path(run_dir)
    attack = json.loads((run_dir / "attack.json").read_text(encoding="utf-8"))
    victim = attack["victim"]
    attacker_id = attack["attacker_id"]

    session_dir = attacker_frames_dir / eval_session_name
    if not session_dir.exists():
        raise SystemExit(f"Evaluation session directory not found: {session_dir}")

    check_session_not_in_train_dev(run_dir, session_dir)

    if output_dir is not None:
        out_run = Path(output_dir)
        out_run.mkdir(parents=True, exist_ok=True)
    else:
        out_run = new_run("adaptive_eval")
    print(f"\n[adaptive_eval] Output directory: {out_run}")

    # Resolve consistency threshold
    if consistency_threshold is None:
        if DEFAULT_CONSISTENCY_SUMMARY.exists():
            try:
                s_data = json.loads(DEFAULT_CONSISTENCY_SUMMARY.read_text(encoding="utf-8"))
                consistency_threshold = float(s_data["consistency"]["purifier"]["threshold_tpr95"])
            except Exception:
                consistency_threshold = DEFAULT_CONSISTENCY_THRESHOLD
        else:
            consistency_threshold = DEFAULT_CONSISTENCY_THRESHOLD

    # Resolve purifier checkpoint
    if purifier_ckpt_path is None:
        purifier_ckpt_path = Path(attack.get("purifier_ckpt", DEFAULT_PURIFIER_CKPT))
    if not purifier_ckpt_path.exists():
        raise SystemExit(f"Purifier checkpoint not found at {purifier_ckpt_path}")

    # Load textures
    tex_data = torch.load(run_dir / "texture.pt", map_location="cpu", weights_only=False)
    logits_raw = tex_data["logits"]
    logits_full = upsample_logits_padded(logits_raw, out_h=H_PRIME, out_w=W_PRIME)
    T_seen = torch.sigmoid(logits_full)

    shuffled_logits = permute_cells(logits_raw, seed=control_seed)
    shuffled_full = upsample_logits_padded(shuffled_logits, out_h=H_PRIME, out_w=W_PRIME)
    T_random = torch.sigmoid(shuffled_full)

    mask_img = cv2.imread(str(run_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise SystemExit(f"mask.png not found in {run_dir}")
    ring_mask_t = torch.from_numpy((mask_img > 127).astype(np.float32))[None]

    band_tmpl = load_band_template(band_template_path)

    # Models & Galleries
    recognizer = FaceRecognizer(ckpt_path=str(checkpoint_path), strict=True)
    if hasattr(recognizer, "_model") and hasattr(recognizer._model, "parameters"):
        _orig_extract = recognizer.extract_embedding
        def _dev_aware_extract(face_bgr: np.ndarray) -> np.ndarray:
            try:
                m_dev = next(recognizer._model.parameters()).device
                tensor = _preprocess_face(face_bgr, recognizer.preprocess).to(m_dev)
                with torch.no_grad():
                    return recognizer._model.get_embedding(tensor).cpu().numpy().squeeze()
            except Exception:
                return _orig_extract(face_bgr)
        recognizer.extract_embedding = _dev_aware_extract

    purifier_model = load_purifier_checkpoint(purifier_ckpt_path, device=device)
    purifier_model.eval()

    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    live = None
    if liveness_enabled:
        from liveness.liveness_detector import LivenessDetector
        live = LivenessDetector()

    eval_modes = ["none", "gray", "purifier"]
    galleries = {}
    for m in eval_modes:
        gal, _ = build_gallery_for_mode(
            root=gallery_dir,
            recognizer=recognizer,
            detector=detector,
            eyes_finder=eyes_finder,
            mode=m,
            band_template=band_tmpl,
            purifier_model=purifier_model if m == "purifier" else None,
            purifier_tag=purifier_ckpt_path.stem if m == "purifier" else "none",
            device=device,
            cache_dir=out_run.parent / "_cache",
        )
        galleries[m] = gal

    files = faces.list_images(session_dir)
    if not files:
        raise SystemExit(f"No frames in {session_dir}")
    times = frame_times(files)
    files.sort(key=lambda f: (f.parent.name, times[str(f)]))

    tiers = [
        {"name": "D", "texture": T_seen},
        {"name": "control_random", "texture": T_random},
    ]

    all_rows: list[dict[str, Any]] = []
    coverage_s6_vals: list[float] = []

    for tier in tiers:
        tier_name = tier["name"]
        tex_rgb = tier["texture"]

        smoother = None
        if live is not None:
            from liveness.temporal import make_smoother
            smoother = make_smoother(live.config)

        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            t = float(times[str(f)])

            eyes_src = eyes_finder.eyes(img)
            if eyes_src is None:
                for m in eval_modes:
                    all_rows.append({
                        "tier": tier_name,
                        "mode": m,
                        "frame": str(f),
                        "face": 0,
                        "band_ok": 0,
                        "rejected_no_band": 0,
                        "match_id": "",
                        "match_sim": "",
                        "top1_before_id": "",
                        "top1_before_sim": "",
                        "top1_after_id": "",
                        "top1_after_sim": "",
                        "top1_drop": "",
                        "target_id": victim,
                        "target_sim": "",
                        "is_real": 0 if live is not None else 1,
                        "live_label": live_label,
                        "live_score": live_score,
                        "enhancer_applied": 0,
                        "success_recog": 0,
                        "flagged": 0,
                        "success_full": 0,
                    })
                continue

            A_pad = similarity_padded(eyes_src)
            rendered = paint_padded(img, tex_rgb, ring_mask_t, A_pad)
            frame_jpg = jpeg(rendered, 95)
            bbox = detector.detect(frame_jpg)

            is_real = 1
            live_label = ""
            live_score: Any = ""
            if bbox is not None and live is not None:
                real_bool, lbl, p_score = liveness_step(live, smoother, frame_jpg, t)
                is_real = int(real_bool)
                live_label = lbl
                live_score = round(float(p_score), 4)

            if bbox is None:
                for m in eval_modes:
                    all_rows.append({
                        "tier": tier_name,
                        "mode": m,
                        "frame": str(f),
                        "face": 0,
                        "band_ok": 0,
                        "rejected_no_band": 0,
                        "match_id": "",
                        "match_sim": "",
                        "top1_before_id": "",
                        "top1_before_sim": "",
                        "top1_after_id": "",
                        "top1_after_sim": "",
                        "top1_drop": "",
                        "target_id": victim,
                        "target_sim": "",
                        "is_real": 0 if live is not None else 1,
                        "live_label": live_label,
                        "live_score": live_score,
                        "enhancer_applied": "",
                        "success_recog": 0,
                        "flagged": 0,
                        "success_full": 0,
                    })
                continue

            face_crop, enh_meta = faces.enhance(faces.crop(frame_jpg, bbox))
            enhancer_applied = int(bool(enh_meta and enh_meta.get("applied")))
            crop112 = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)

            if tier_name == "D" and enhancer_applied == 0:
                win_meas = faces.crop_window(bbox, frame_jpg.shape)
                cov = coverage_padded(ring_mask_t, A_pad, win_meas)
                coverage_s6_vals.append(cov)

            eyes_frame = eyes_finder.eyes(frame_jpg)
            band112 = band_in_crop(frame_jpg, bbox, eyes_frame, band_template=band_tmpl, size=112) if eyes_frame is not None else None
            band_ok = int(band112 is not None)

            # Mode 'none' reference
            ref_top1_id = ""
            ref_top1_sim = 0.0
            scores_by_mode: dict[str, dict[str, float]] = {}

            for m in eval_modes:
                if m in ("gray", "purifier") and band_ok == 0:
                    scores_by_mode[m] = {}
                    continue
                proc = transform_crop112(crop112, band112, mode=m, purifier_model=purifier_model, device=device)
                emb = recognizer.extract_embedding(proc)
                sc = score_against_gallery(emb, galleries[m].embeddings)
                scores_by_mode[m] = sc
                if m == "none" and sc:
                    sorted_none = sorted(sc.items(), key=lambda it: it[1], reverse=True)
                    ref_top1_id, ref_top1_sim = sorted_none[0]

            for m in eval_modes:
                if m in ("gray", "purifier") and band_ok == 0:
                    all_rows.append({
                        "tier": tier_name,
                        "mode": m,
                        "frame": str(f),
                        "face": 1,
                        "band_ok": 0,
                        "rejected_no_band": 1,
                        "match_id": "",
                        "match_sim": "",
                        "top1_before_id": ref_top1_id,
                        "top1_before_sim": round(float(ref_top1_sim), 4) if ref_top1_id else "",
                        "top1_after_id": "",
                        "top1_after_sim": "",
                        "top1_drop": "",
                        "target_id": victim,
                        "target_sim": "",
                        "is_real": int(is_real),
                        "live_label": live_label,
                        "live_score": live_score,
                        "enhancer_applied": enhancer_applied,
                        "success_recog": 0,
                        "flagged": 1,
                        "success_full": 0,
                    })
                    continue

                sc = scores_by_mode[m]
                if sc:
                    sorted_sc = sorted(sc.items(), key=lambda it: it[1], reverse=True)
                    top1_id, top1_sim = sorted_sc[0]
                    tgt_sim = sc.get(victim, 0.0)
                else:
                    top1_id, top1_sim, tgt_sim = "", 0.0, 0.0

                matched_victim = (top1_id == victim and top1_sim >= 0.5)
                succ_recog = int(matched_victim and is_real == 1)

                top1_drop_val: Any = ""
                flagged = 0
                succ_full = 0

                if m == "purifier":
                    drop = float(ref_top1_sim - sc.get(ref_top1_id, 0.0))
                    top1_drop_val = round(drop, 4)
                    flagged = int(drop >= consistency_threshold)
                    succ_full = int(succ_recog == 1 and flagged == 0)

                all_rows.append({
                    "tier": tier_name,
                    "mode": m,
                    "frame": str(f),
                    "face": 1,
                    "band_ok": band_ok,
                    "rejected_no_band": 0,
                    "match_id": top1_id if top1_sim >= 0.5 else "",
                    "match_sim": round(float(top1_sim), 4) if top1_sim >= 0.5 else "",
                    "top1_before_id": ref_top1_id,
                    "top1_before_sim": round(float(ref_top1_sim), 4) if ref_top1_id else "",
                    "top1_after_id": top1_id,
                    "top1_after_sim": round(float(top1_sim), 4),
                    "top1_drop": top1_drop_val,
                    "target_id": victim,
                    "target_sim": round(float(tgt_sim), 4),
                    "is_real": int(is_real),
                    "live_label": live_label,
                    "live_score": live_score,
                    "enhancer_applied": enhancer_applied,
                    "success_recog": succ_recog,
                    "flagged": flagged,
                    "success_full": succ_full,
                })

    # Save frames.csv
    csv_path = out_run / "frames.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Compute summary
    summary_results: dict[str, Any] = {}
    for tier in tiers:
        tier_name = tier["name"]
        for m in eval_modes:
            normal_rows = [
                r for r in all_rows
                if r["tier"] == tier_name and r["mode"] == m and r["enhancer_applied"] == 0 and r["face"] == 1
            ]
            dim_rows = [
                r for r in all_rows
                if r["tier"] == tier_name and r["mode"] == m and r["enhancer_applied"] == 1 and r["face"] == 1
            ]

            n = len(normal_rows)
            k = sum(r["success_recog"] for r in normal_rows)
            rate = round(k / n, 4) if n > 0 else 0.0
            ci_lo, ci_hi = wilson(k, n)

            entry: dict[str, Any] = {
                "rate": rate,
                "k": k,
                "n": n,
                "wilson95": [round(ci_lo, 4), round(ci_hi, 4)],
            }

            if m == "purifier":
                k_full = sum(r["success_full"] for r in normal_rows)
                rate_full = round(k_full / n, 4) if n > 0 else 0.0
                f_lo, f_hi = wilson(k_full, n)
                k_flag = sum(r["flagged"] for r in normal_rows)
                flag_rate = round(k_flag / n, 4) if n > 0 else 0.0

                entry["success_full_rate"] = rate_full
                entry["success_full_k"] = k_full
                entry["success_full_wilson95"] = [round(f_lo, 4), round(f_hi, 4)]
                entry["flag_rate"] = flag_rate
                entry["flag_k"] = k_flag

            summary_results[f"{tier_name}/{m}"] = entry

            if dim_rows:
                n_dim = len(dim_rows)
                k_dim = sum(r["success_recog"] for r in dim_rows)
                entry_dim: dict[str, Any] = {
                    "rate": round(k_dim / n_dim, 4) if n_dim > 0 else 0.0,
                    "k": k_dim,
                    "n": n_dim,
                    "wilson95": [round(x, 4) for x in wilson(k_dim, n_dim)],
                }
                if m == "purifier":
                    k_full_dim = sum(r["success_full"] for r in dim_rows)
                    entry_dim["success_full_rate"] = round(k_full_dim / n_dim, 4) if n_dim > 0 else 0.0
                    entry_dim["success_full_k"] = k_full_dim
                    entry_dim["flag_rate"] = round(sum(r["flagged"] for r in dim_rows) / n_dim, 4) if n_dim > 0 else 0.0
                summary_results[f"{tier_name}/{m}@dim"] = entry_dim

    cov_s6_summary = {
        "mean": round(float(np.mean(coverage_s6_vals)), 4) if coverage_s6_vals else 0.0,
        "min": round(float(np.min(coverage_s6_vals)), 4) if coverage_s6_vals else 0.0,
        "max": round(float(np.max(coverage_s6_vals)), 4) if coverage_s6_vals else 0.0,
    }

    summary: dict[str, Any] = {
        "run_id": out_run.name,
        "adaptive_optimize_run": str(run_dir),
        "ring_px": attack.get("ring_px"),
        "ring_coverage_s6": cov_s6_summary,
        "eval_session": eval_session_name,
        "liveness_enabled": bool(liveness_enabled),
        "consistency_threshold": consistency_threshold,
        "purifier_ckpt": str(purifier_ckpt_path),
        "checkpoint": str(checkpoint_path),
        "victim": victim,
        "attacker_id": attacker_id,
        "results": summary_results,
    }

    write_json(out_run / "summary.json", summary)

    # Print summary
    print(f"\nEvaluation Summary on {eval_session_name} (ring r={attack.get('ring_px')} px):")
    print(f"Ring coverage on {eval_session_name}: {cov_s6_summary['mean'] * 100:.2f}% [{cov_s6_summary['min'] * 100:.2f}%, {cov_s6_summary['max'] * 100:.2f}%]")
    print(f"Consistency Threshold: {consistency_threshold:.4f}\n")
    print(f"{'Tier / Mode':<28}{'Recog Rate':<16}{'Success Full (Unflagged)':<28}{'Flag Rate':<12}")
    print("-" * 84)
    for k_name, res in summary_results.items():
        if "@dim" in k_name:
            continue
        recog_str = f"{res['rate'] * 100:.1f}% ({res['k']}/{res['n']})"
        full_str = (
            f"{res['success_full_rate'] * 100:.1f}% ({res['success_full_k']}/{res['n']})"
            if "success_full_rate" in res
            else "-"
        )
        flag_str = f"{res['flag_rate'] * 100:.1f}%" if "flag_rate" in res else "-"
        print(f"{k_name:<28}{recog_str:<16}{full_str:<28}{flag_str:<12}")

    print(f"\nDetailed frames saved to {csv_path}")
    print(f"Summary saved to {out_run / 'summary.json'}\n")
    return out_run


def cmd_eval(args: argparse.Namespace) -> None:
    run_eval_adaptive(
        run_dir=Path(args.run),
        eval_session_name=args.eval_session,
        attacker_frames_dir=Path(args.attacker_frames),
        gallery_dir=Path(args.gallery),
        checkpoint_path=Path(args.checkpoint),
        purifier_ckpt_path=Path(args.purifier_ckpt) if args.purifier_ckpt else None,
        consistency_threshold=args.consistency_threshold,
        band_template_path=Path(args.band_template),
        liveness_enabled=args.liveness,
        device=args.device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


# ── Main Entrypoint ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. ring
    p_ring = sub.add_parser("ring", help="Build ring masks, measure crop 112 coverage, check disjointness")
    p_ring.add_argument("--ring-px", type=int, nargs="+", default=[8, 16, 32, 48], help="Ring widths r in px")
    p_ring.add_argument("--band-template", default=str(DEFAULT_BAND_TEMPLATE), help="Path to 400x150 band_template.png")
    p_ring.add_argument("--out-dir", default=str(DEFAULT_DEFENSE_DIR), help="Output directory for ring masks and meta")
    p_ring.add_argument("--attacker-frames", default=str(DEFAULT_S1_DAY1), help="Folder with frames to measure coverage")

    # 2. optimize
    p_opt = sub.add_parser("optimize", help="Jointly optimize adaptive texture on the ring")
    p_opt.add_argument("--ring-px", type=int, required=True, help="Ring width r in px (e.g. 8, 16, 32, 48)")
    p_opt.add_argument("--purifier-ckpt", default=str(DEFAULT_PURIFIER_CKPT), help="Path to trained purifier best.pt")
    p_opt.add_argument("--attacker-frames", default=str(DEFAULT_ATTACKER_FRAMES), help="Directory of attacker frames")
    p_opt.add_argument("--gallery", default=str(DEFAULT_GALLERY), help="Gallery directory")
    p_opt.add_argument("--attacker-id", default="owner", help="Attacker folder name in gallery")
    p_opt.add_argument("--victim", default="LFW_victim", help="Victim folder name in gallery")
    p_opt.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Face recognizer checkpoint")
    p_opt.add_argument("--band-template", default=str(DEFAULT_BAND_TEMPLATE), help="Path to band_template.png")
    p_opt.add_argument("--chart", help="Path to chart.json (optional, defaults to standard palette)")
    p_opt.add_argument("--steps", type=int, default=3000, help="Optimization steps")
    p_opt.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    p_opt.add_argument("--batch", type=int, default=8, help="Batch size")
    p_opt.add_argument("--tau", type=float, default=0.05, help="Softmax temperature")
    p_opt.add_argument("--margin", type=float, default=0.05, help="Loss margin")
    p_opt.add_argument("--lambda-tv", type=float, default=2.5, help="Weight for TV regularizer")
    p_opt.add_argument("--lambda-nps", type=float, default=0.5, help="Weight for NPS regularizer")
    p_opt.add_argument("--tex-res", type=int, nargs=2, default=[20, 51], metavar=("H", "W"), help="Parameter grid")
    p_opt.add_argument("--eval-sessions", nargs="*", default=["s3_vary", "s4_vary", "s5_vary", "s6_vary"], help="Held-out eval sessions")
    p_opt.add_argument("--dev-frac", type=float, default=0.2, help="Dev fraction")
    p_opt.add_argument("--dev-mode", choices=("random", "block"), default="block", help="Dev split mode")
    p_opt.add_argument("--no-eot", action="store_true", help="Disable EOT during optimization")
    p_opt.add_argument("--max-gallery", type=int, default=faces.GALLERY_PER_ID, help="Images per identity in gallery")
    p_opt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    p_opt.add_argument("--seed", type=int, default=0, help="Random seed")
    p_opt.add_argument("--log-every", type=int, default=25, help="Log progress every N steps")

    # 3. eval
    p_eval = sub.add_parser("eval", help="Evaluate an adaptive attack run on held-out session")
    p_eval.add_argument("--run", required=True, help="Path to adaptive optimize run directory")
    p_eval.add_argument("--eval-session", default="s6_vary", help="Session folder name in attacker frames")
    p_eval.add_argument("--attacker-frames", default=str(DEFAULT_ATTACKER_FRAMES), help="Directory of attacker frames")
    p_eval.add_argument("--gallery", default=str(DEFAULT_GALLERY), help="Gallery directory")
    p_eval.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Face recognizer checkpoint")
    p_eval.add_argument("--purifier-ckpt", help="Purifier checkpoint (defaults to run's attack.json or best.pt)")
    p_eval.add_argument("--consistency-threshold", type=float, help="Override consistency detector threshold")
    p_eval.add_argument("--band-template", default=str(DEFAULT_BAND_TEMPLATE), help="Path to band_template.png")
    p_eval.add_argument("--liveness", action="store_true", help="Enable liveness evaluation")
    p_eval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    p_eval.add_argument("--output-dir", help="Explicit output directory (default: new run in results/innovation/)")

    args = parser.parse_args()
    if args.cmd == "ring":
        cmd_ring(args)
    elif args.cmd == "optimize":
        cmd_optimize(args)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
