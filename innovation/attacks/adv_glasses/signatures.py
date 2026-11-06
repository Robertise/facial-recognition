"""
signatures.py – Feasibility measurement of defensive signatures S1/S2/S3 (zero-training).

Checks whether adversarial glasses artifacts leave measurable signatures that separate
attacked frames from genuine face frames on method3 (FaceCNN ArcFace) and the 1:N gallery:
  S1 (Residual identity):
    - s1_second: score of top-2
    - s1_gap_inv: -(score top-1 - score top-2)
    - s1_ratio: score top-2 / score top-1
  S2 (Concentrated evidence):
    - 8 horizontal stripes of 14 px across 112x112 face crop, occluded with neutral value (0.0 in [-1, 1])
    - s2_maxdrop: max_k (orig_score - occluded_score_k) / orig_score
    - s2_flips: number of stripes where top-1 changes id or drops below 0.5
    - s2_conc: max_k drop_k / sum_k max(drop_k, 0)
    - s2_drop_0 ... s2_drop_7: full 8-drop vector
  S3 (Temporal instability):
    - 1.5s sliding window across continuous session frames
    - s3_top1_switch: ratio of consecutive frame pairs with different top-1 id
    - s3_score_std: standard deviation of score of current top-1 id across window

Outputs results to results/innovation/<ts>_signatures/ with frames.csv, summary.json,
and hist_<signature>.png plots.

See docs/innovation/plan_signatures.md for the full specification.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backend.face_recognizer import FaceRecognizer
from innovation.attacks.adv_glasses import eot, faces, glasses, pipeline, printability
from innovation.attacks.adv_glasses.evaluate import frame_times, jpeg, paint, unseen_frame
from innovation.attacks.adv_glasses.optimize import control_random_texture, upsample_logits
from innovation.eval.metrics import auc
from innovation.eval.runs import new_run, write_json


SIGNATURE_SCALARS = [
    "s1_second",
    "s1_gap_inv",
    "s1_ratio",
    "s2_maxdrop",
    "s2_flips",
    "s2_conc",
    "s3_top1_switch",
    "s3_score_std",
]


# ==============================================================================
# S1, S2, S3 Signature Computation Functions
# ==============================================================================

def score_identities(emb: np.ndarray, gallery_embs: dict[str, np.ndarray]) -> dict[str, float]:
    """Max cosine similarity of query embedding with gallery images for each identity."""
    scores: dict[str, float] = {}
    for person_id, embs in gallery_embs.items():
        sims = embs @ emb
        scores[person_id] = float(np.max(sims))
    return scores


def get_top_k(scores: dict[str, float], k: int = 2) -> list[tuple[str, float]]:
    """Returns top k (identity, score) pairs sorted descending by score."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


def compute_s1(top1_score: float, top2_score: float) -> dict[str, float]:
    """Compute S1 residual identity signatures."""
    s1_second = float(top2_score)
    s1_gap_inv = float(-(top1_score - top2_score))
    s1_ratio = float(top2_score / top1_score) if top1_score > 0 else 0.0
    return {
        "s1_second": s1_second,
        "s1_gap_inv": s1_gap_inv,
        "s1_ratio": s1_ratio,
    }


def occlude_stripes(img_112: np.ndarray, num_stripes: int = 8, fill_val: int = 128) -> list[np.ndarray]:
    """
    Split a 112x112 face crop into `num_stripes` horizontal stripes of height (112 // num_stripes) px,
    and return `num_stripes` versions of the image, each with one stripe occluded with `fill_val`.
    For CNN_PREPROCESS (mean 0.5, std 0.5 on [0, 1]), fill_val 128 (or 127.5) maps to ~0.0 in [-1, 1].
    """
    h, w = img_112.shape[:2]
    stripe_h = h // num_stripes
    occ_list = []
    for k in range(num_stripes):
        occ = img_112.copy()
        occ[k * stripe_h : (k + 1) * stripe_h, :] = fill_val
        occ_list.append(occ)
    return occ_list


def compute_s2(
    face_bgr: np.ndarray,
    orig_top1_id: str,
    orig_top1_score: float,
    recognizer: Any,
    gallery_embs: dict[str, np.ndarray],
    num_stripes: int = 8,
) -> dict[str, Any]:
    """
    Divide crop 112x112 into 8 horizontal stripes (14 px each).
    Replace each stripe with neutral value (0.0 after normalization in [-1, 1]).
    Extract embedding and compute score of original top-1 id:
      - s2_maxdrop: max_k (orig_score - occluded_score_k) / orig_score
      - s2_flips: number of stripes where top-1 changes id or drops below 0.5
      - s2_conc: max_k drop_k / sum_k max(drop_k, 0)
      - s2_drop_0 ... s2_drop_7: the 8 drop values
    """
    resized = cv2.resize(face_bgr, (112, 112))
    stripe_h = 112 // num_stripes

    # Fast batch path for PyTorch models
    if hasattr(recognizer, "_model") and hasattr(recognizer, "preprocess") and hasattr(recognizer._model, "get_embedding"):
        from backend.face_recognizer import _preprocess_face
        tensor = _preprocess_face(resized, recognizer.preprocess)  # (1, 3, 112, 112)
        batch = tensor.repeat(num_stripes, 1, 1, 1)
        for k in range(num_stripes):
            batch[k, :, k * stripe_h : (k + 1) * stripe_h, :] = 0.0
        with torch.no_grad():
            embs = recognizer._model.get_embedding(batch).cpu().numpy()
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
    else:
        # Fallback for mock recognizers without _model
        occ_images = occlude_stripes(resized, num_stripes=num_stripes, fill_val=128)
        embs = np.stack([recognizer.extract_embedding(img) for img in occ_images])

    drops: list[float] = []
    flips: int = 0
    for k in range(num_stripes):
        emb_k = embs[k]
        scores_k = score_identities(emb_k, gallery_embs)
        score_che_k = scores_k.get(orig_top1_id, 0.0)
        drop_k = float(orig_top1_score - score_che_k)
        drops.append(drop_k)

        top1_occ_id, top1_occ_score = get_top_k(scores_k, 1)[0]
        if top1_occ_id != orig_top1_id or top1_occ_score < 0.5:
            flips += 1

    max_drop = max(drops) if drops else 0.0
    s2_maxdrop = (max_drop / orig_top1_score) if orig_top1_score > 0 else 0.0
    sum_pos_drop = sum(max(d, 0.0) for d in drops)
    s2_conc = (max_drop / sum_pos_drop) if sum_pos_drop > 0 else 0.0

    res: dict[str, Any] = {
        "s2_maxdrop": float(s2_maxdrop),
        "s2_flips": int(flips),
        "s2_conc": float(s2_conc),
    }
    for k in range(num_stripes):
        res[f"s2_drop_{k}"] = float(drops[k])
    return res


def compute_s3_signatures_for_window(
    history: list[dict],
    curr_top1_id: str,
) -> tuple[float, float]:
    """
    Inside the 1.5s sliding window:
    - s3_top1_switch: ratio of consecutive frame pairs with different top-1 id
    - s3_score_std: std of the score of the current frame's top-1 id across window
    """
    if len(history) < 2:
        return 0.0, 0.0

    diff_count = sum(1 for a, b in zip(history[:-1], history[1:]) if a["top1_id"] != b["top1_id"])
    switch_rate = float(diff_count / (len(history) - 1))

    scores_for_top1 = [h["scores"].get(curr_top1_id, 0.0) for h in history]
    score_std = float(np.std(scores_for_top1, ddof=0))
    return switch_rate, score_std


def make_leave_one_out_gallery(
    gallery_embs: dict[str, np.ndarray],
    target_id: str,
    query_idx: int = 0,
) -> dict[str, np.ndarray]:
    """
    Returns a copy of gallery_embs where target_id has the embedding at query_idx removed.
    """
    loo = dict(gallery_embs)
    target_embs = gallery_embs[target_id]
    if len(target_embs) <= 1:
        raise ValueError(f"Cannot perform leave-one-out on {target_id} with <= 1 embeddings")
    loo[target_id] = np.delete(target_embs, query_idx, axis=0)
    return loo


# ==============================================================================
# Metric & Evaluation Utilities
# ==============================================================================

def bootstrap_auc_ci(
    pos: np.ndarray,
    neg: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[list[float]]]:
    """Compute point AUC and 95% bootstrap confidence interval."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return None, None

    point_auc = auc(pos, neg)
    rng = np.random.default_rng(seed)
    n_p, n_n = len(pos), len(neg)
    boot_aucs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        p_samp = pos[rng.integers(0, n_p, size=n_p)]
        n_samp = neg[rng.integers(0, n_n, size=n_n)]
        boot_aucs[b] = auc(p_samp, n_samp)

    lo = float(np.percentile(boot_aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return round(float(point_auc), 4), [round(lo, 4), round(hi, 4)]


def plot_signature_histograms(
    frames: list[dict],
    signatures: list[str],
    out_dir: Path,
) -> None:
    """
    Generates hist_<signature>.png for each scalar signature comparing
    main attacks (A_C3 pooled, A_R2b pooled) vs pooled negatives.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neg_frames = [r for r in frames if r["label"] == 0]
    c3_frames = [r for r in frames if r["set"] in ("A_C3", "A_C3_EOT")]
    r2b_frames = [r for r in frames if r["set"] in ("A_R2b", "A_R2b_EOT")]

    for sig in signatures:
        neg_vals = [float(r[sig]) for r in neg_frames if r[sig] != "" and r[sig] is not None]
        c3_vals = [float(r[sig]) for r in c3_frames if r[sig] != "" and r[sig] is not None]
        r2b_vals = [float(r[sig]) for r in r2b_frames if r[sig] != "" and r[sig] is not None]

        if not neg_vals and not c3_vals and not r2b_vals:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        all_vals = neg_vals + c3_vals + r2b_vals
        vmin, vmax = min(all_vals), max(all_vals)
        bins = np.linspace(vmin, vmax, 30) if vmin < vmax else 10

        if neg_vals:
            ax.hist(neg_vals, bins=bins, alpha=0.5, label=f"Real / Negatives (N={len(neg_vals)})", color="forestgreen", density=True)
        if c3_vals:
            ax.hist(c3_vals, bins=bins, alpha=0.5, label=f"A_C3 + EOT (N={len(c3_vals)})", color="crimson", density=True)
        if r2b_vals:
            ax.hist(r2b_vals, bins=bins, alpha=0.5, label=f"A_R2b + EOT (N={len(r2b_vals)})", color="royalblue", density=True)

        ax.set_title(f"Signature Distribution: {sig}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{sig} (higher = more suspect)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        plot_path = out_dir / f"hist_{sig}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# ==============================================================================
# Pipeline & Data Collection
# ==============================================================================

def load_texture_assets(run_dir: Path, control_seed: int = 0) -> dict[str, Any]:
    """Load mask, colormap and colors for D and controls from an attack run."""
    attack = json.loads((run_dir / "attack.json").read_text(encoding="utf-8"))
    shape = glasses.FrameShape(**attack["frame_shape"])
    mask = torch.from_numpy(shape.mask())[None]
    cmap = printability.ColorMap.load(run_dir / "colormap.json")
    tex_data = torch.load(run_dir / "texture.pt", map_location="cpu")
    logits_raw = tex_data["logits"]
    logits_full = upsample_logits(logits_raw)
    T = torch.sigmoid(logits_full)
    seen = cmap(T)
    T_random = control_random_texture(logits_raw, seed=control_seed)
    seen_random = cmap(T_random)
    flat = seen.reshape(3, -1)
    plain = flat.mean(1, keepdim=True).expand_as(flat).reshape(seen.shape).clone()
    return {
        "attack": attack,
        "mask": mask,
        "seen": seen,
        "seen_random": seen_random,
        "plain": plain,
        "victim": attack.get("victim"),
    }


def process_continuous_session(
    session_dir: Path,
    detector: faces.FaceDetector,
    eyes_finder: faces.EyeLandmarks,
    recognizer: FaceRecognizer,
    gallery_embs: dict[str, np.ndarray],
    rng: np.random.Generator,
    c3_assets: dict[str, Any],
    r2b_assets: Optional[dict[str, Any]],
    skipped_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Process one continuous session (e.g. s2_day2, s5_vary, s6_vary).
    Renders attacks, baselines and controls, runs S1/S2/S3, and keeps valid frames.
    """
    files = faces.list_images(session_dir)
    if not files:
        return []

    times = frame_times(files)
    files.sort(key=lambda f: (f.parent.name, times[str(f)]))

    # Pre-extract landmarks and eyes once per frame
    prepared = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            continue
        A = glasses.similarity(eyes)
        A_unseen = glasses.jitter(A, eot.UNSEEN.shift_px, eot.UNSEEN.rot_deg, eot.UNSEEN.scale, rng)
        bbox0 = detector.detect(img)
        side = float(max(bbox0[2], bbox0[3]) * (1 + 2 * faces.CROP_PAD)) if bbox0 is not None else float(max(img.shape[:2]))
        seed = int(rng.integers(1 << 30))
        prepared.append((f, img, A, A_unseen, side, seed, times[str(f)]))

    if not prepared:
        return []

    # Define streams to evaluate in this session
    streams: list[dict[str, Any]] = [
        {"set": "R_owner", "tier": "baseline", "assets": None, "label": 0, "target_id": "owner"},
        {"set": "R_owner_EOT", "tier": "baseline_EOT", "assets": None, "label": 0, "target_id": "owner"},
        {"set": "A_C3", "tier": "D", "assets": c3_assets, "label": 1, "target_id": c3_assets["victim"]},
        {"set": "A_C3_EOT", "tier": "D_EOT", "assets": c3_assets, "label": 1, "target_id": c3_assets["victim"]},
        {"set": "R_owner_controls", "tier": "control_random_C3", "assets": c3_assets, "label": 0, "target_id": "owner"},
        {"set": "R_owner_controls", "tier": "control_plain_C3", "assets": c3_assets, "label": 0, "target_id": "owner"},
    ]
    if r2b_assets is not None:
        streams.extend([
            {"set": "A_R2b", "tier": "D", "assets": r2b_assets, "label": 1, "target_id": r2b_assets["victim"]},
            {"set": "A_R2b_EOT", "tier": "D_EOT", "assets": r2b_assets, "label": 1, "target_id": r2b_assets["victim"]},
            {"set": "R_owner_controls", "tier": "control_random_R2b", "assets": r2b_assets, "label": 0, "target_id": "owner"},
            {"set": "R_owner_controls", "tier": "control_plain_R2b", "assets": r2b_assets, "label": 0, "target_id": "owner"},
        ])

    session_rows: list[dict[str, Any]] = []

    for stream in streams:
        set_name = stream["set"]
        tier = stream["tier"]
        assets = stream["assets"]
        label = stream["label"]
        target_id = stream["target_id"]

        stream_history: list[dict[str, Any]] = []
        stream_candidates: list[dict[str, Any]] = []

        for f, img, A, A_unseen, side, seed, t in prepared:
            # Render frame
            if tier == "baseline":
                frame = img
            elif tier == "baseline_EOT":
                frame = unseen_frame(img, np.random.default_rng(seed), side)
            elif tier == "D":
                frame = paint(img, assets["seen"], assets["mask"], A)
            elif tier == "D_EOT":
                frame = unseen_frame(paint(img, assets["seen"], assets["mask"], A_unseen), np.random.default_rng(seed), side)
            elif tier.startswith("control_random"):
                frame = paint(img, assets["seen_random"], assets["mask"], A)
            elif tier.startswith("control_plain"):
                frame = paint(img, assets["plain"], assets["mask"], A)
            else:
                frame = img

            frame = jpeg(frame, 95)
            bbox = detector.detect(frame)
            if bbox is None:
                continue

            face, meta = faces.enhance(faces.crop(frame, bbox))
            if meta and meta.get("applied"):
                skipped_counts["dim_frames"] += 1
                continue

            emb = recognizer.extract_embedding(face)
            scores = score_identities(emb, gallery_embs)
            (top1_id, top1_score), (top2_id, top2_score) = get_top_k(scores, 2)

            # Update stream history for S3
            stream_history.append({"t": t, "top1_id": top1_id, "scores": scores})

            # Check criteria
            if label == 1:
                # Attack frame must match victim with score >= 0.5
                if top1_id != target_id or top1_score < 0.5:
                    skipped_counts["attack_failed"] += 1
                    continue
            else:
                # Negative frame must match owner with score >= 0.5
                if top1_id != target_id or top1_score < 0.5:
                    skipped_counts["negative_failed"] += 1
                    continue

            # Compute S1 and S2
            s1 = compute_s1(top1_score, top2_score)
            s2 = compute_s2(face, top1_id, top1_score, recognizer, gallery_embs)

            # Window for S3 at time t
            cutoff = t - 1.5
            window = [h for h in stream_history if h["t"] >= cutoff]
            s3_switch, s3_std = compute_s3_signatures_for_window(window, top1_id)

            cand = {
                "set": set_name,
                "source": session_dir.name,
                "frame": f"{tier}_{f.name}",
                "t": round(float(t), 3),
                "label": int(label),
                "top1_id": top1_id,
                "top1_score": round(float(top1_score), 4),
                "top2_id": top2_id,
                "top2_score": round(float(top2_score), 4),
                **{k: round(float(v), 4) for k, v in s1.items()},
                **{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in s2.items()},
                "s3_top1_switch": round(float(s3_switch), 4),
                "s3_score_std": round(float(s3_std), 4),
            }
            stream_candidates.append(cand)

        session_rows.extend(stream_candidates)

    return session_rows


def process_lfw_leave_one_out(
    gallery_dir: Path,
    detector: faces.FaceDetector,
    recognizer: FaceRecognizer,
    full_gallery_embs: dict[str, np.ndarray],
    skipped_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Process R_lfw: for each LFW identity in gallery, leave 1 image out as query,
    gallery retains the other 4 embeddings. Top-1 must match that person with cos >= 0.5.
    """
    rows: list[dict[str, Any]] = []

    for person_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        p_name = person_dir.name
        if p_name == "owner":
            continue
        if p_name not in full_gallery_embs or len(full_gallery_embs[p_name]) <= 1:
            continue

        images = faces.list_images(person_dir)
        if not images:
            continue

        # Find first valid face to use as LOO query
        found = False
        for q_idx, q_file in enumerate(images[:5]):
            img = cv2.imread(str(q_file))
            if img is None:
                continue
            bbox = detector.detect(img)
            if bbox is None:
                continue
            face, meta = faces.enhance(faces.crop(img, bbox))
            if meta and meta.get("applied") or face.size == 0:
                skipped_counts["dim_frames"] += 1
                continue

            # Query index in full_gallery_embs
            emb = recognizer.extract_embedding(face)

            # LOO gallery with 1 embedding removed
            loo_embs = make_leave_one_out_gallery(full_gallery_embs, p_name, query_idx=min(q_idx, len(full_gallery_embs[p_name]) - 1))
            scores = score_identities(emb, loo_embs)
            (top1_id, top1_score), (top2_id, top2_score) = get_top_k(scores, 2)

            if top1_id != p_name or top1_score < 0.5:
                skipped_counts["negative_failed"] += 1
                continue

            s1 = compute_s1(top1_score, top2_score)
            s2 = compute_s2(face, top1_id, top1_score, recognizer, loo_embs)

            rows.append({
                "set": "R_lfw",
                "source": p_name,
                "frame": q_file.name,
                "t": "",
                "label": 0,
                "top1_id": top1_id,
                "top1_score": round(float(top1_score), 4),
                "top2_id": top2_id,
                "top2_score": round(float(top2_score), 4),
                **{k: round(float(v), 4) for k, v in s1.items()},
                **{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in s2.items()},
                "s3_top1_switch": "",
                "s3_score_std": "",
            })
            found = True
            break

        if not found:
            skipped_counts["negative_failed"] += 1

    return rows


# ==============================================================================
# Main Orchestration
# ==============================================================================

def run_signatures_study(
    c3_run: Path,
    r2b_run: Path,
    frames_dir: Path,
    gallery_dir: Path,
    checkpoint: Path,
    seed: int = 0,
    control_seed: int = 0,
    n_boot: int = 2000,
    output_dir: Optional[Path] = None,
) -> Path:
    """Main execution of signatures feasibility evaluation."""
    run_dir = output_dir or new_run("signatures", {
        "c3_run": str(c3_run),
        "r2b_run": str(r2b_run),
        "frames_dir": str(frames_dir),
        "gallery_dir": str(gallery_dir),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "control_seed": control_seed,
        "bootstrap_iterations": n_boot,
    })

    rng = np.random.default_rng(seed)
    recognizer = FaceRecognizer(threshold=0.5, ckpt_path=str(checkpoint), strict=True)
    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    gallery, _ = faces.build_gallery(gallery_dir, recognizer, detector, max_per_id=faces.GALLERY_PER_ID, cache_dir=Path("results/innovation/_cache"))
    gallery_embs = gallery.embeddings

    c3_assets = load_texture_assets(c3_run, control_seed=control_seed)
    r2b_assets = load_texture_assets(r2b_run, control_seed=control_seed)

    skipped_counts = {"dim_frames": 0, "attack_failed": 0, "negative_failed": 0}
    all_rows: list[dict[str, Any]] = []

    # 1. Process s2_day2: only C3 applies (C3 trained on s1)
    s2_dir = frames_dir / "s2_day2"
    if s2_dir.exists():
        rows_s2 = process_continuous_session(
            s2_dir, detector, eyes_finder, recognizer, gallery_embs, rng,
            c3_assets=c3_assets, r2b_assets=None, skipped_counts=skipped_counts,
        )
        all_rows.extend(rows_s2)

    # 2. Process s5_vary: both C3 and R2b apply (R2b trained on s1 + s2)
    s5_dir = frames_dir / "s5_vary"
    if s5_dir.exists():
        rows_s5 = process_continuous_session(
            s5_dir, detector, eyes_finder, recognizer, gallery_embs, rng,
            c3_assets=c3_assets, r2b_assets=r2b_assets, skipped_counts=skipped_counts,
        )
        all_rows.extend(rows_s5)

    # 3. Process s6_vary: both C3 and R2b apply
    s6_dir = frames_dir / "s6_vary"
    if s6_dir.exists():
        rows_s6 = process_continuous_session(
            s6_dir, detector, eyes_finder, recognizer, gallery_embs, rng,
            c3_assets=c3_assets, r2b_assets=r2b_assets, skipped_counts=skipped_counts,
        )
        all_rows.extend(rows_s6)

    # 4. Process R_lfw
    rows_lfw = process_lfw_leave_one_out(
        gallery_dir, detector, recognizer, gallery_embs, skipped_counts,
    )
    all_rows.extend(rows_lfw)

    if not all_rows:
        raise SystemExit("Error: No valid frames were extracted for signatures study.")

    # Write frames.csv
    csv_fields = [
        "set", "source", "frame", "t", "label", "top1_id", "top1_score", "top2_id", "top2_score",
        "s1_second", "s1_gap_inv", "s1_ratio",
        "s2_maxdrop", "s2_flips", "s2_conc",
        "s2_drop_0", "s2_drop_1", "s2_drop_2", "s2_drop_3",
        "s2_drop_4", "s2_drop_5", "s2_drop_6", "s2_drop_7",
        "s3_top1_switch", "s3_score_std",
    ]
    with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # Define comparisons for summary.json
    comparisons_def: dict[str, tuple[list[str], list[str]]] = {
        # Main comparisons:
        "A_C3_vs_pooled_negatives": (
            ["A_C3", "A_C3_EOT"],
            ["R_owner", "R_owner_EOT", "R_owner_controls", "R_lfw"],
        ),
        "A_R2b_vs_pooled_negatives": (
            ["A_R2b", "A_R2b_EOT"],
            ["R_owner", "R_owner_EOT", "R_owner_controls", "R_lfw"],
        ),
        # Secondary comparisons:
        "A_C3_vs_R_owner_controls": (
            ["A_C3", "A_C3_EOT"],
            ["R_owner_controls"],
        ),
        "A_C3_vs_R_lfw": (
            ["A_C3", "A_C3_EOT"],
            ["R_lfw"],
        ),
        "A_R2b_vs_R_owner_controls": (
            ["A_R2b", "A_R2b_EOT"],
            ["R_owner_controls"],
        ),
        "A_R2b_vs_R_lfw": (
            ["A_R2b", "A_R2b_EOT"],
            ["R_lfw"],
        ),
    }

    eval_results: dict[str, dict[str, Any]] = {}
    for comp_name, (pos_sets, neg_sets) in comparisons_def.items():
        comp_res: dict[str, Any] = {}
        pos_subset = [r for r in all_rows if r["set"] in pos_sets]
        neg_subset = [r for r in all_rows if r["set"] in neg_sets]

        for sig in SIGNATURE_SCALARS:
            pos_vals = [float(r[sig]) for r in pos_subset if r[sig] != "" and r[sig] is not None]
            neg_vals = [float(r[sig]) for r in neg_subset if r[sig] != "" and r[sig] is not None]
            pt_auc, ci = bootstrap_auc_ci(np.array(pos_vals), np.array(neg_vals), n_boot=n_boot, seed=seed)
            comp_res[sig] = {
                "auc": pt_auc,
                "ci95": ci,
                "n_pos": len(pos_vals),
                "n_neg": len(neg_vals),
            }
        eval_results[comp_name] = comp_res

    # Counts by set
    set_counts: dict[str, int] = {}
    for r in all_rows:
        set_counts[r["set"]] = set_counts.get(r["set"], 0) + 1

    summary = {
        "run_dir": str(run_dir),
        "c3_run": str(c3_run),
        "r2b_run": str(r2b_run),
        "comparisons": eval_results,
        "counts": set_counts,
        "skipped": skipped_counts,
    }
    write_json(run_dir / "summary.json", summary)

    # Plot histograms
    plot_signature_histograms(all_rows, SIGNATURE_SCALARS, run_dir)

    # Print AUC table
    print("\n" + "=" * 88)
    print("SIGNATURE FEASIBILITY EVALUATION RESULTS (S1 / S2 / S3)")
    print("=" * 88)
    print(f"{'Comparison':<32} {'Signature':<16} {'AUC':<8} {'95% CI':<18} {'N_pos':<7} {'N_neg':<7}")
    print("-" * 88)
    for comp_name, sig_dict in eval_results.items():
        for sig, stats in sig_dict.items():
            auc_str = f"{stats['auc']:.4f}" if stats['auc'] is not None else "N/A"
            ci_str = f"[{stats['ci95'][0]:.4f}, {stats['ci95'][1]:.4f}]" if stats['ci95'] is not None else "N/A"
            print(f"{comp_name:<32} {sig:<16} {auc_str:<8} {ci_str:<18} {stats['n_pos']:<7} {stats['n_neg']:<7}")
        print("-" * 88)

    print(f"Artifacts written to: {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate defensive signatures S1/S2/S3 (zero-training).")
    ap.add_argument("--c3-run", default="results/innovation/20261104_140557_advglasses_impersonate",
                    help="Run directory of C3 attack")
    ap.add_argument("--r2b-run", default="results/innovation/20261106_132526_advglasses_impersonate",
                    help="Run directory of R2b attack")
    ap.add_argument("--frames-dir", default="data/innovation/frames/owner",
                    help="Directory containing owner session folders")
    ap.add_argument("--gallery-dir", default="data/innovation/gallery",
                    help="Directory containing 1:N gallery folders")
    ap.add_argument("--checkpoint", default="checkpoints/method3_best.pth",
                    help="Path to FaceRecognizer checkpoint (method3)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap and EOT")
    ap.add_argument("--control-seed", type=int, default=0, help="Random seed for control random spatial shuffle")
    ap.add_argument("--bootstrap-iterations", type=int, default=2000, help="Number of bootstrap iterations for CI")
    ap.add_argument("--output-dir", default=None, help="Explicit output directory (default: new run in results/)")
    args = ap.parse_args()

    run_signatures_study(
        c3_run=Path(args.c3_run),
        r2b_run=Path(args.r2b_run),
        frames_dir=Path(args.frames_dir),
        gallery_dir=Path(args.gallery_dir),
        checkpoint=Path(args.checkpoint),
        seed=args.seed,
        control_seed=args.control_seed,
        n_boot=args.bootstrap_iterations,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
