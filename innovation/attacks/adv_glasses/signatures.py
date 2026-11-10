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
import re
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
    query_idx: int | None = None,
    query_emb: np.ndarray | None = None,
    min_cos: float = 0.99,
) -> dict[str, np.ndarray] | None:
    """
    Returns a copy of gallery_embs where target_id has one embedding removed.
    If query_emb is provided, drops the gallery embedding with the highest cosine similarity,
    only when that cosine >= min_cos (otherwise returns None).
    If query_idx is provided, drops target_id[query_idx].
    """
    target_embs = gallery_embs[target_id]
    if len(target_embs) <= 1:
        raise ValueError(f"Cannot perform leave-one-out on {target_id} with <= 1 embeddings")

    if query_emb is not None:
        sims = target_embs @ query_emb
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) < min_cos:
            return None
        remove_idx = best_idx
    elif query_idx is not None:
        remove_idx = query_idx
    else:
        remove_idx = 0

    loo = dict(gallery_embs)
    loo[target_id] = np.delete(target_embs, remove_idx, axis=0)
    return loo


def merge_galleries(
    base_gallery_embs: dict[str, np.ndarray],
    extra_gallery_embs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Merge extra gallery embeddings into base gallery.
    Raises ValueError if any identity already exists in base gallery.
    """
    duplicates = set(base_gallery_embs.keys()) & set(extra_gallery_embs.keys())
    if duplicates:
        raise ValueError(f"Duplicate identities found when merging extra gallery: {sorted(duplicates)}")
    merged = dict(base_gallery_embs)
    merged.update(extra_gallery_embs)
    return merged


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


def compute_operating_points(
    all_rows: list[dict[str, Any]],
    people_sessions: list[str] | None = None,
    tpr_target: float = 0.95,
    pos_sets: list[str] | None = None,
) -> dict[str, Any]:
    """Compute threshold at TPR 95% on pooled attacks (default: A_C3 + A_C3_EOT + A_R2b + A_R2b_EOT)
    and report FPR across negative groups: R_owner (+EOT), R_owner_controls, R_lfw,
    and for each person session: R_<tag>, R_<tag>_controls, R_<tag>_all.
    """
    if pos_sets is None:
        pos_sets = ["A_C3", "A_C3_EOT", "A_R2b", "A_R2b_EOT"]
    groups_def: dict[str, list[str]] = {
        "R_owner": ["R_owner", "R_owner_EOT"],
        "R_owner_controls": ["R_owner_controls"],
        "R_lfw": ["R_lfw"],
    }
    if people_sessions:
        for ps in people_sessions:
            ps_clean = ps.strip().replace("\\", "/")
            if not ps_clean:
                continue
            tag = ps_clean.replace("/", "_")
            groups_def[f"R_{tag}"] = [f"R_{tag}"]
            groups_def[f"R_{tag}_controls"] = [f"R_{tag}_controls"]
            groups_def[f"R_{tag}_all"] = [f"R_{tag}", f"R_{tag}_controls"]

    res: dict[str, Any] = {}
    for sig in ["s2_maxdrop", "s2_conc"]:
        pos_vals = [float(r[sig]) for r in all_rows if r.get("set") in pos_sets and r.get(sig) not in ("", None)]
        if not pos_vals:
            res[sig] = {
                "threshold_tpr95": None,
                "n_pos": 0,
                "groups": {},
            }
            continue

        pct = (1.0 - tpr_target) * 100.0  # 5th percentile
        threshold = float(np.percentile(pos_vals, pct))

        sig_groups: dict[str, dict[str, Any]] = {}
        for grp_name, grp_sets in groups_def.items():
            neg_vals = [float(r[sig]) for r in all_rows if r.get("set") in grp_sets and r.get(sig) not in ("", None)]
            n_neg = len(neg_vals)
            if n_neg == 0:
                sig_groups[grp_name] = {"fpr": None, "n_neg": 0, "n_fp": 0}
            else:
                n_fp = sum(1 for v in neg_vals if v >= threshold)
                fpr = float(n_fp / n_neg)
                sig_groups[grp_name] = {
                    "fpr": round(fpr, 4),
                    "n_neg": n_neg,
                    "n_fp": n_fp,
                }

        sig_res: dict[str, Any] = {
            "threshold_tpr95": round(threshold, 4),
            "n_pos": len(pos_vals),
            "groups": sig_groups,
        }
        for grp_name, grp_stats in sig_groups.items():
            sig_res[grp_name] = grp_stats

        res[sig] = sig_res

    return res


def evaluate_rule_check(
    eval_results: dict[str, dict[str, Any]],
    operating_points: dict[str, Any],
    session_recogs: dict[str, dict[str, Any]],
    people_sessions: list[str] | None = None,
    primary_session: str | None = None,
) -> dict[str, Any]:
    """Evaluate S2 defense rules on extra people sessions according to experiment_log.md."""
    if not people_sessions:
        return {
            "primary_session": primary_session,
            "s2_holds_for_other_people": None,
            "sessions": {},
            "variants": {"s2_maxdrop": {}, "s2_conc": {}},
            "s2_maxdrop": {},
            "s2_conc": {},
        }

    sessions_dict: dict[str, Any] = {}
    variants_dict: dict[str, dict[str, Any]] = {"s2_maxdrop": {}, "s2_conc": {}}

    for ps in people_sessions:
        ps_clean = ps.strip().replace("\\", "/")
        if not ps_clean:
            continue
        tag = ps_clean.replace("/", "_")
        recog = session_recogs.get(ps_clean, {"recognized_frames": 0, "total_face_frames": 0, "recognition_rate": 0.0})

        c3_key = f"A_C3_vs_R_{tag}_all"
        r2b_key = f"A_R2b_vs_R_{tag}_all"
        comp_c3 = eval_results.get(c3_key, {})
        comp_r2b = eval_results.get(r2b_key, {})

        sess_variants: dict[str, Any] = {}
        for sig in ["s2_maxdrop", "s2_conc"]:
            auc_c3 = comp_c3.get(sig, {}).get("auc") if isinstance(comp_c3, dict) else None
            auc_r2b = comp_r2b.get(sig, {}).get("auc") if isinstance(comp_r2b, dict) else None

            if auc_c3 is not None and auc_r2b is not None:
                min_auc = round(min(auc_c3, auc_r2b), 4)
            elif auc_c3 is not None:
                min_auc = round(auc_c3, 4)
            elif auc_r2b is not None:
                min_auc = round(auc_r2b, 4)
            else:
                min_auc = None

            fpr = operating_points.get(sig, {}).get(f"R_{tag}_all", {}).get("fpr")
            is_pass = bool(min_auc is not None and fpr is not None and min_auc >= 0.90 and fpr <= 0.10)

            variant_entry = {
                "min_auc": min_auc,
                "auc_c3": auc_c3,
                "auc_r2b": auc_r2b,
                "fpr": fpr,
                "pass": is_pass,
            }
            sess_variants[sig] = variant_entry
            variants_dict[sig][ps_clean] = variant_entry

        sessions_dict[ps_clean] = {
            "recognized_frames": recog.get("recognized_frames", 0),
            "total_face_frames": recog.get("total_face_frames", 0),
            "recognition_rate": recog.get("recognition_rate", 0.0),
            "s2_maxdrop": sess_variants["s2_maxdrop"],
            "s2_conc": sess_variants["s2_conc"],
        }

    s2_holds: Optional[bool] = None
    if primary_session:
        prim_clean = primary_session.strip().replace("\\", "/")
        if prim_clean in sessions_dict:
            s2_holds = bool(
                sessions_dict[prim_clean]["s2_maxdrop"]["pass"] or
                sessions_dict[prim_clean]["s2_conc"]["pass"]
            )
        else:
            s2_holds = False

    return {
        "primary_session": primary_session,
        "s2_holds_for_other_people": s2_holds,
        "sessions": sessions_dict,
        "variants": variants_dict,
        "s2_maxdrop": variants_dict["s2_maxdrop"],
        "s2_conc": variants_dict["s2_conc"],
    }


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


def plot_extra_signature_histograms(
    frames: list[dict],
    extra_name: str,
    signatures: list[str],
    out_dir: Path,
) -> None:
    """Generates hist_<extra_name>_<signature>.png for each scalar signature comparing
    extra attack (A_<extra_name> pooled) vs pooled negatives.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    neg_frames = [r for r in frames if r["label"] == 0]
    extra_frames = [r for r in frames if r["set"] in (f"A_{extra_name}", f"A_{extra_name}_EOT")]

    for sig in signatures:
        neg_vals = [float(r[sig]) for r in neg_frames if r[sig] != "" and r[sig] is not None]
        extra_vals = [float(r[sig]) for r in extra_frames if r[sig] != "" and r[sig] is not None]

        if not neg_vals and not extra_vals:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        all_vals = neg_vals + extra_vals
        vmin, vmax = min(all_vals), max(all_vals)
        bins = np.linspace(vmin, vmax, 30) if vmin < vmax else 10

        if neg_vals:
            ax.hist(neg_vals, bins=bins, alpha=0.5, label=f"Real / Negatives (N={len(neg_vals)})", color="forestgreen", density=True)
        if extra_vals:
            ax.hist(extra_vals, bins=bins, alpha=0.5, label=f"A_{extra_name} + EOT (N={len(extra_vals)})", color="purple", density=True)

        ax.set_title(f"Signature Distribution ({extra_name}): {sig}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{sig} (higher = more suspect)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        plot_path = out_dir / f"hist_{extra_name}_{sig}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# ==============================================================================
# Pipeline & Data Collection
# ==============================================================================

def load_texture_assets(run_dir: Path, control_seed: int = 0) -> dict[str, Any]:
    """Load mask, colormap and colors for D and controls from an attack run."""
    attack = json.loads((run_dir / "attack.json").read_text(encoding="utf-8"))
    shape = glasses.shape_from_attack(attack)
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


def build_session_streams(
    c3_assets: dict[str, Any],
    r2b_assets: Optional[dict[str, Any]] = None,
    target_person: str = "owner",
    set_prefix: str = "R_owner",
    include_attacks: bool = True,
    extra_assets: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Build list of streams for continuous session processing.
    When include_attacks is False, generates only negative streams with target_id=target_person.
    """
    if include_attacks:
        streams = [
            {"set": set_prefix, "tier": "baseline", "assets": None, "label": 0, "target_id": target_person},
            {"set": f"{set_prefix}_EOT", "tier": "baseline_EOT", "assets": None, "label": 0, "target_id": target_person},
            {"set": "A_C3", "tier": "D", "assets": c3_assets, "label": 1, "target_id": c3_assets["victim"]},
            {"set": "A_C3_EOT", "tier": "D_EOT", "assets": c3_assets, "label": 1, "target_id": c3_assets["victim"]},
            {"set": f"{set_prefix}_controls", "tier": "control_random_C3", "assets": c3_assets, "label": 0, "target_id": target_person},
            {"set": f"{set_prefix}_controls", "tier": "control_plain_C3", "assets": c3_assets, "label": 0, "target_id": target_person},
        ]
        if r2b_assets is not None:
            streams.extend([
                {"set": "A_R2b", "tier": "D", "assets": r2b_assets, "label": 1, "target_id": r2b_assets["victim"]},
                {"set": "A_R2b_EOT", "tier": "D_EOT", "assets": r2b_assets, "label": 1, "target_id": r2b_assets["victim"]},
                {"set": f"{set_prefix}_controls", "tier": "control_random_R2b", "assets": r2b_assets, "label": 0, "target_id": target_person},
                {"set": f"{set_prefix}_controls", "tier": "control_plain_R2b", "assets": r2b_assets, "label": 0, "target_id": target_person},
            ])
        if extra_assets is not None:
            for extra_name, assets in extra_assets.items():
                streams.extend([
                    {"set": f"A_{extra_name}", "tier": "D", "assets": assets, "label": 1, "target_id": assets["victim"]},
                    {"set": f"A_{extra_name}_EOT", "tier": "D_EOT", "assets": assets, "label": 1, "target_id": assets["victim"]},
                    {"set": f"{set_prefix}_controls", "tier": f"control_random_{extra_name}", "assets": assets, "label": 0, "target_id": target_person},
                    {"set": f"{set_prefix}_controls", "tier": f"control_plain_{extra_name}", "assets": assets, "label": 0, "target_id": target_person},
                ])
    else:
        # Negative-only streams for extra people sessions
        # Baseline and baseline_EOT share set = set_prefix; controls share f"{set_prefix}_controls"
        streams = [
            {"set": set_prefix, "tier": "baseline", "assets": None, "label": 0, "target_id": target_person},
            {"set": set_prefix, "tier": "baseline_EOT", "assets": None, "label": 0, "target_id": target_person},
            {"set": f"{set_prefix}_controls", "tier": "control_random_C3", "assets": c3_assets, "label": 0, "target_id": target_person},
            {"set": f"{set_prefix}_controls", "tier": "control_plain_C3", "assets": c3_assets, "label": 0, "target_id": target_person},
        ]
        if r2b_assets is not None:
            streams.extend([
                {"set": f"{set_prefix}_controls", "tier": "control_random_R2b", "assets": r2b_assets, "label": 0, "target_id": target_person},
                {"set": f"{set_prefix}_controls", "tier": "control_plain_R2b", "assets": r2b_assets, "label": 0, "target_id": target_person},
            ])
        if extra_assets is not None:
            for extra_name, assets in extra_assets.items():
                streams.extend([
                    {"set": f"{set_prefix}_controls", "tier": f"control_random_{extra_name}", "assets": assets, "label": 0, "target_id": target_person},
                    {"set": f"{set_prefix}_controls", "tier": f"control_plain_{extra_name}", "assets": assets, "label": 0, "target_id": target_person},
                ])
    return streams


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
    target_person: str = "owner",
    set_prefix: str = "R_owner",
    include_attacks: bool = True,
    session_source: Optional[str] = None,
    extra_assets: Optional[dict[str, dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Process one continuous session (e.g. s2_day2, s5_vary, s6_vary, or people session).
    Renders attacks/baselines/controls, runs S1/S2/S3, and keeps valid frames.
    Returns (session_rows, recognition_stats).
    """
    empty_stats = {"recognized_frames": 0, "total_face_frames": 0, "recognition_rate": 0.0}
    files = faces.list_images(session_dir)
    if not files:
        return [], empty_stats

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
        return [], empty_stats

    streams = build_session_streams(
        c3_assets=c3_assets,
        r2b_assets=r2b_assets,
        target_person=target_person,
        set_prefix=set_prefix,
        include_attacks=include_attacks,
        extra_assets=extra_assets,
    )

    session_rows: list[dict[str, Any]] = []
    total_face_frames = 0
    recognized_frames = 0

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

            if tier == "baseline":
                total_face_frames += 1

            emb = recognizer.extract_embedding(face)
            scores = score_identities(emb, gallery_embs)
            (top1_id, top1_score), (top2_id, top2_score) = get_top_k(scores, 2)

            if tier == "baseline" and top1_id == target_id and top1_score >= 0.5:
                recognized_frames += 1

            # Update stream history for S3
            stream_history.append({"t": t, "top1_id": top1_id, "scores": scores})

            # Check criteria
            if label == 1:
                # Attack frame must match victim with score >= 0.5
                if top1_id != target_id or top1_score < 0.5:
                    skipped_counts["attack_failed"] += 1
                    continue
            else:
                # Negative frame must match target_person with score >= 0.5
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
                "source": session_source or session_dir.name,
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

    recog_rate = round(float(recognized_frames / total_face_frames), 4) if total_face_frames > 0 else 0.0
    recog_stats = {
        "recognized_frames": int(recognized_frames),
        "total_face_frames": int(total_face_frames),
        "recognition_rate": recog_rate,
    }
    return session_rows, recog_stats


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

            # LOO gallery: remove embedding with max cosine similarity, only if >= 0.99
            loo_embs = make_leave_one_out_gallery(full_gallery_embs, p_name, query_emb=emb, min_cos=0.99)
            if loo_embs is None:
                skipped_counts["loo_mismatch"] = skipped_counts.get("loo_mismatch", 0) + 1
                continue
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
    extra_gallery: Optional[Path] = None,
    people_sessions: Optional[list[str]] = None,
    primary_session: Optional[str] = None,
    seed: int = 0,
    control_seed: int = 0,
    n_boot: int = 2000,
    output_dir: Optional[Path] = None,
    extra_attack: Optional[list[str]] = None,
) -> Path:
    """Main execution of signatures feasibility evaluation (v2 supporting extra gallery and people sessions)."""
    run_dir = output_dir or new_run("signatures", {
        "c3_run": str(c3_run),
        "r2b_run": str(r2b_run),
        "frames_dir": str(frames_dir),
        "gallery_dir": str(gallery_dir),
        "checkpoint": str(checkpoint),
        "extra_gallery": str(extra_gallery) if extra_gallery else None,
        "people_sessions": people_sessions or [],
        "primary_session": primary_session,
        "extra_attack": extra_attack or [],
        "seed": seed,
        "control_seed": control_seed,
        "bootstrap_iterations": n_boot,
    })

    rng = np.random.default_rng(seed)
    recognizer = FaceRecognizer(threshold=0.5, ckpt_path=str(checkpoint), strict=True)
    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    gallery, _ = faces.build_gallery(gallery_dir, recognizer, detector, max_per_id=faces.GALLERY_PER_ID, cache_dir=Path("results/innovation/_cache"))
    gallery_embs = dict(gallery.embeddings)
    extra_gallery_ids: list[str] = []

    if extra_gallery is not None:
        extra_path = Path(extra_gallery)
        if extra_path.exists():
            extra_gal, _ = faces.build_gallery(extra_path, recognizer, detector, max_per_id=faces.GALLERY_PER_ID, cache_dir=Path("results/innovation/_cache"))
            gallery_embs = merge_galleries(gallery_embs, extra_gal.embeddings)
            extra_gallery_ids = sorted(extra_gal.embeddings.keys())
        else:
            raise FileNotFoundError(f"Extra gallery directory not found: {extra_path}")

    extra_attacks: dict[str, Path] = {}
    if extra_attack:
        for item in extra_attack:
            if "=" not in item:
                raise SystemExit(f"invalid --extra-attack format, expected NAME=RUN_DIR, got: {item}")
            name, run_dir_str = item.split("=", 1)
            name = name.strip()
            run_dir_path = Path(run_dir_str.strip())
            if not re.match(r"^[A-Za-z0-9]+$", name):
                raise SystemExit(f"Extra attack NAME must match [A-Za-z0-9]+, got '{name}'")
            if name in ("C3", "R2b"):
                raise SystemExit(f"Extra attack NAME cannot be C3 or R2b: '{name}'")
            if not run_dir_path.exists():
                raise FileNotFoundError(f"Extra attack run directory not found: {run_dir_path}")
            extra_attacks[name] = run_dir_path

    extra_assets_all: dict[str, dict[str, Any]] = {}
    allowed_sessions_map: dict[str, set[str]] = {}
    for name, r_path in extra_attacks.items():
        assets = load_texture_assets(r_path, control_seed=control_seed)
        extra_assets_all[name] = assets
        split_path = r_path / "split.json"
        if not split_path.exists():
            raise FileNotFoundError(f"split.json not found in extra attack run: {r_path}")
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        train_sessions = {Path(p).parent.name for p in split_data.get("train", [])}
        dev_sessions = {Path(p).parent.name for p in split_data.get("dev", [])}
        eval_sessions = {Path(p).parent.name for p in split_data.get("eval", [])}
        allowed = eval_sessions - train_sessions - dev_sessions
        allowed_sessions_map[name] = allowed
        applied = sorted([s for s in ("s2_day2", "s5_vary", "s6_vary") if s in allowed])
        print(f"Extra attack '{name}': applied to sessions {applied}")

    c3_assets = load_texture_assets(c3_run, control_seed=control_seed)
    r2b_assets = load_texture_assets(r2b_run, control_seed=control_seed)

    skipped_counts = {"dim_frames": 0, "attack_failed": 0, "negative_failed": 0, "loo_mismatch": 0}
    all_rows: list[dict[str, Any]] = []

    # 1. Process s2_day2: only C3 applies (C3 trained on s1)
    extra_assets_s2 = {name: assets for name, assets in extra_assets_all.items() if "s2_day2" in allowed_sessions_map[name]}
    s2_dir = frames_dir / "s2_day2"
    if s2_dir.exists():
        rows_s2, _ = process_continuous_session(
            s2_dir, detector, eyes_finder, recognizer, gallery_embs, rng,
            c3_assets=c3_assets, r2b_assets=None, skipped_counts=skipped_counts,
            target_person="owner", set_prefix="R_owner", include_attacks=True,
            extra_assets=extra_assets_s2,
        )
        all_rows.extend(rows_s2)

    # 2. Process s5_vary: both C3 and R2b apply (R2b trained on s1 + s2)
    extra_assets_s5 = {name: assets for name, assets in extra_assets_all.items() if "s5_vary" in allowed_sessions_map[name]}
    s5_dir = frames_dir / "s5_vary"
    if s5_dir.exists():
        rows_s5, _ = process_continuous_session(
            s5_dir, detector, eyes_finder, recognizer, gallery_embs, rng,
            c3_assets=c3_assets, r2b_assets=r2b_assets, skipped_counts=skipped_counts,
            target_person="owner", set_prefix="R_owner", include_attacks=True,
            extra_assets=extra_assets_s5,
        )
        all_rows.extend(rows_s5)

    # 3. Process s6_vary: both C3 and R2b apply
    extra_assets_s6 = {name: assets for name, assets in extra_assets_all.items() if "s6_vary" in allowed_sessions_map[name]}
    s6_dir = frames_dir / "s6_vary"
    if s6_dir.exists():
        rows_s6, _ = process_continuous_session(
            s6_dir, detector, eyes_finder, recognizer, gallery_embs, rng,
            c3_assets=c3_assets, r2b_assets=r2b_assets, skipped_counts=skipped_counts,
            target_person="owner", set_prefix="R_owner", include_attacks=True,
            extra_assets=extra_assets_s6,
        )
        all_rows.extend(rows_s6)

    # 4. Process people sessions (negative streams only)
    session_recogs: dict[str, dict[str, Any]] = {}
    if people_sessions:
        for ps in people_sessions:
            ps_clean = ps.strip().replace("\\", "/")
            if not ps_clean:
                continue
            parts = ps_clean.split("/")
            if len(parts) != 2:
                raise ValueError(f"people-sessions must be in NAME/SESSION format, got: {ps_clean}")
            person, session = parts
            session_path = frames_dir.parent / person / session
            if not session_path.exists():
                raise FileNotFoundError(f"People session directory not found: {session_path}")

            tag = f"{person}_{session}"
            rows_ps, recog_stats = process_continuous_session(
                session_dir=session_path,
                detector=detector,
                eyes_finder=eyes_finder,
                recognizer=recognizer,
                gallery_embs=gallery_embs,
                rng=rng,
                c3_assets=c3_assets,
                r2b_assets=r2b_assets,
                skipped_counts=skipped_counts,
                target_person=person,
                set_prefix=f"R_{tag}",
                include_attacks=False,
                session_source=tag,
                extra_assets=extra_assets_all,
            )
            all_rows.extend(rows_ps)
            session_recogs[ps_clean] = recog_stats

    # 5. Process R_lfw
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

    # Add extra comparisons for people sessions:
    # A_C3_vs_R_<person>_<session>_all and A_R2b_vs_R_<person>_<session>_all
    if people_sessions:
        for ps in people_sessions:
            ps_clean = ps.strip().replace("\\", "/")
            if not ps_clean:
                continue
            tag = ps_clean.replace("/", "_")
            neg_sets = [f"R_{tag}", f"R_{tag}_controls"]
            comparisons_def[f"A_C3_vs_R_{tag}_all"] = (["A_C3", "A_C3_EOT"], neg_sets)
            if r2b_assets is not None:
                comparisons_def[f"A_R2b_vs_R_{tag}_all"] = (["A_R2b", "A_R2b_EOT"], neg_sets)

    if extra_attacks:
        for extra_name in extra_attacks:
            extra_pos = [f"A_{extra_name}", f"A_{extra_name}_EOT"]
            comparisons_def[f"A_{extra_name}_vs_pooled_negatives"] = (
                extra_pos,
                ["R_owner", "R_owner_EOT", "R_owner_controls", "R_lfw"],
            )
            comparisons_def[f"A_{extra_name}_vs_R_owner_controls"] = (
                extra_pos,
                ["R_owner_controls"],
            )
            comparisons_def[f"A_{extra_name}_vs_R_lfw"] = (
                extra_pos,
                ["R_lfw"],
            )
            if people_sessions:
                for ps in people_sessions:
                    ps_clean = ps.strip().replace("\\", "/")
                    if not ps_clean:
                        continue
                    tag = ps_clean.replace("/", "_")
                    neg_sets = [f"R_{tag}", f"R_{tag}_controls"]
                    comparisons_def[f"A_{extra_name}_vs_R_{tag}_all"] = (extra_pos, neg_sets)

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

    # Operating points at 95% TPR on pooled positives
    operating_points = compute_operating_points(all_rows, people_sessions=people_sessions, tpr_target=0.95)

    # Rule check for other people
    rule_check = evaluate_rule_check(
        eval_results=eval_results,
        operating_points=operating_points,
        session_recogs=session_recogs,
        people_sessions=people_sessions,
        primary_session=primary_session,
    )

    operating_points_extra: dict[str, dict[str, Any]] = {}
    rule_check_extra: dict[str, dict[str, Any]] = {}
    if extra_attacks:
        for extra_name in extra_attacks:
            op_extra = compute_operating_points(
                all_rows,
                people_sessions=people_sessions,
                tpr_target=0.95,
                pos_sets=[f"A_{extra_name}", f"A_{extra_name}_EOT"],
            )
            operating_points_extra[extra_name] = op_extra

            if primary_session:
                prim_clean = primary_session.strip().replace("\\", "/")
                tag = prim_clean.replace("/", "_")
                comp_key = f"A_{extra_name}_vs_R_{tag}_all"
                comp_res = eval_results.get(comp_key, {})
                rule_name: dict[str, Any] = {
                    "primary_session": primary_session,
                }
                for sig in ["s2_maxdrop", "s2_conc"]:
                    auc_val = comp_res.get(sig, {}).get("auc") if isinstance(comp_res, dict) else None
                    fpr_val = op_extra.get(sig, {}).get(f"R_{tag}_all", {}).get("fpr") if isinstance(op_extra.get(sig), dict) else None
                    is_pass = bool(auc_val is not None and fpr_val is not None and auc_val >= 0.90 and fpr_val <= 0.10)
                    rule_name[sig] = {
                        "auc_vs_primary": auc_val,
                        "fpr_primary_at_tpr95": fpr_val,
                        "pass": is_pass,
                    }
                rule_name["s2_holds"] = bool(rule_name["s2_maxdrop"]["pass"] or rule_name["s2_conc"]["pass"])
                rule_check_extra[extra_name] = rule_name
            else:
                rule_check_extra[extra_name] = {
                    "primary_session": None,
                    "s2_maxdrop": {"auc_vs_primary": None, "fpr_primary_at_tpr95": None, "pass": False},
                    "s2_conc": {"auc_vs_primary": None, "fpr_primary_at_tpr95": None, "pass": False},
                    "s2_holds": None,
                }

    # Counts by set
    set_counts: dict[str, int] = {}
    for r in all_rows:
        set_counts[r["set"]] = set_counts.get(r["set"], 0) + 1

    summary = {
        "run_dir": str(run_dir),
        "c3_run": str(c3_run),
        "r2b_run": str(r2b_run),
        "extra_gallery_ids": extra_gallery_ids,
        "primary_session": primary_session,
        "people_sessions": people_sessions or [],
        "comparisons": eval_results,
        "operating_points": operating_points,
        "rule_check": rule_check,
        "counts": set_counts,
        "skipped": skipped_counts,
    }
    if extra_attacks:
        summary["extra_attacks"] = {k: str(v) for k, v in extra_attacks.items()}
        summary["operating_points_extra"] = operating_points_extra
        summary["rule_check_extra"] = rule_check_extra
    write_json(run_dir / "summary.json", summary)

    # Plot histograms
    plot_signature_histograms(all_rows, SIGNATURE_SCALARS, run_dir)
    if extra_attacks:
        for extra_name in extra_attacks:
            plot_extra_signature_histograms(all_rows, extra_name, SIGNATURE_SCALARS, run_dir)

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

    # Print Operating Points table
    if operating_points:
        print("\n" + "=" * 88)
        print("OPERATING POINTS AT 95% TPR ON POOLED ATTACKS")
        print("=" * 88)
        print(f"{'Signature':<14} {'Thresh@95':<12} {'Negative Group':<32} {'FPR':<8} {'N_neg':<7} {'N_fp':<7}")
        print("-" * 88)
        for sig, op in operating_points.items():
            thresh_str = f"{op['threshold_tpr95']:.4f}" if op.get("threshold_tpr95") is not None else "N/A"
            groups = op.get("groups", {})
            for grp_name, g_stats in groups.items():
                fpr_str = f"{g_stats['fpr']:.4f}" if g_stats.get("fpr") is not None else "N/A"
                print(f"{sig:<14} {thresh_str:<12} {grp_name:<32} {fpr_str:<8} {g_stats['n_neg']:<7} {g_stats['n_fp']:<7}")
            print("-" * 88)

    # Print Rule Check summary
    if people_sessions and rule_check.get("sessions"):
        print("\n" + "=" * 88)
        print("RULE CHECK: S2 HOLDS FOR OTHER PEOPLE")
        print("=" * 88)
        for ps_name, s_info in rule_check["sessions"].items():
            is_prim = (ps_name == primary_session)
            prim_tag = " (PRIMARY)" if is_prim else ""
            recog_str = f"{s_info['recognized_frames']}/{s_info['total_face_frames']} ({s_info['recognition_rate']*100:.1f}%)"
            print(f"Session: {ps_name}{prim_tag} | Baseline Recognition: {recog_str}")
            for sig in ["s2_maxdrop", "s2_conc"]:
                v = s_info[sig]
                status = "PASS" if v["pass"] else "FAIL"
                min_auc_str = f"{v['min_auc']:.4f}" if v['min_auc'] is not None else "N/A"
                fpr_str = f"{v['fpr']:.4f}" if v['fpr'] is not None else "N/A"
                print(f"  {sig:<12}: min_AUC = {min_auc_str} (>=0.90?), FPR@95% = {fpr_str} (<=0.10?) -> {status}")
            print("-" * 88)
        holds = rule_check.get("s2_holds_for_other_people")
        holds_str = "PASS (S2 holds for other people)" if holds else "FAIL (S2 does not hold on primary session)"
        print(f"Conclusion: s2_holds_for_other_people = {holds} -> {holds_str}")
        print("=" * 88 + "\n")

    # Print Operating Points for Extra Attacks
    if extra_attacks and operating_points_extra:
        for extra_name, op_extra in operating_points_extra.items():
            print("\n" + "=" * 88)
            print(f"OPERATING POINTS AT 95% TPR ON EXTRA ATTACK: {extra_name}")
            print("=" * 88)
            print(f"{'Signature':<14} {'Thresh@95':<12} {'Negative Group':<32} {'FPR':<8} {'N_neg':<7} {'N_fp':<7}")
            print("-" * 88)
            for sig, op in op_extra.items():
                thresh_str = f"{op['threshold_tpr95']:.4f}" if op.get("threshold_tpr95") is not None else "N/A"
                groups = op.get("groups", {})
                for grp_name, g_stats in groups.items():
                    fpr_str = f"{g_stats['fpr']:.4f}" if g_stats.get("fpr") is not None else "N/A"
                    print(f"{sig:<14} {thresh_str:<12} {grp_name:<32} {fpr_str:<8} {g_stats['n_neg']:<7} {g_stats['n_fp']:<7}")
                print("-" * 88)

    # Print Rule Check for Extra Attacks
    if extra_attacks and rule_check_extra and primary_session:
        print("\n" + "=" * 88)
        print("RULE CHECK: S2 ON EXTRA ATTACKS")
        print("=" * 88)
        for extra_name, r_info in rule_check_extra.items():
            print(f"Extra Attack: {extra_name} | Primary Session: {primary_session}")
            for sig in ["s2_maxdrop", "s2_conc"]:
                v = r_info[sig]
                status = "PASS" if v["pass"] else "FAIL"
                auc_str = f"{v['auc_vs_primary']:.4f}" if v['auc_vs_primary'] is not None else "N/A"
                fpr_str = f"{v['fpr_primary_at_tpr95']:.4f}" if v['fpr_primary_at_tpr95'] is not None else "N/A"
                print(f"  {sig:<12}: AUC_vs_primary = {auc_str} (>=0.90?), FPR@95% = {fpr_str} (<=0.10?) -> {status}")
            holds = r_info.get("s2_holds")
            holds_str = "PASS (S2 holds for extra attack)" if holds else "FAIL (S2 does not hold on primary session)"
            print(f"Conclusion for {extra_name}: s2_holds = {holds} -> {holds_str}")
            print("-" * 88)
        print("=" * 88 + "\n")

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
    ap.add_argument("--extra-gallery", default=None,
                    help="Directory containing extra gallery identities (e.g. data/innovation/gallery_people)")
    ap.add_argument("--people-sessions", nargs="*", default=None,
                    help="Relative session paths for extra people (e.g. p01/s2 p01/s1)")
    ap.add_argument("--primary-session", default=None,
                    help="Primary session for rule check decision (e.g. p01/s2)")
    ap.add_argument("--extra-attack", nargs="*", default=None,
                    help="Extra attack runs in NAME=RUN_DIR format (e.g. H09=results/innovation/20261107_214538_advglasses_impersonate)")
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
        extra_gallery=Path(args.extra_gallery) if args.extra_gallery else None,
        people_sessions=args.people_sessions,
        primary_session=args.primary_session,
        seed=args.seed,
        control_seed=args.control_seed,
        n_boot=args.bootstrap_iterations,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        extra_attack=args.extra_attack,
    )


if __name__ == "__main__":
    main()
