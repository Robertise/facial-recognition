"""
eval_purifier.py - Evaluate Purifier U-Net, gray baseline, and identity consistency detector
through the full backend recognition & liveness pipeline.

CLI:
  python -m innovation.defense.eval_purifier --modes none gray purifier \\
      --purifier-ckpt results/innovation/<train_run>/best.pt \\
      --extra-gallery data/innovation/gallery_people --liveness
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from backend.face_recognizer import FaceRecognizer
from innovation.attacks.adv_glasses import faces, glasses, printability
from innovation.attacks.adv_glasses.evaluate import (
    frame_times,
    jpeg,
    liveness_step,
    paint,
    wilson,
)
from innovation.attacks.adv_glasses.optimize import control_random_texture, upsample_logits
from innovation.defense.band import band_in_crop, load_band_template
from innovation.defense.purifier import (
    PurifierUNet,
    apply_gray,
    apply_purifier,
    load_purifier_checkpoint,
)
from innovation.eval.metrics import auc
from innovation.eval.runs import new_run, write_json

DEFAULT_H09_RUN = Path("results/innovation/20261107_214538_advglasses_impersonate")
DEFAULT_R2B_RUN = Path("results/innovation/20261106_132526_advglasses_impersonate")
DEFAULT_OWNER_S6 = Path("data/innovation/frames/owner/s6_vary")
DEFAULT_P01_ROOT = Path("data/innovation/frames/p01")
DEFAULT_GALLERY = Path("data/innovation/gallery")
DEFAULT_CHECKPOINT = Path("checkpoints/method3_best.pth")


def parse_attack_spec(spec: str) -> tuple[str, Path, Path]:
    """Parse NAME=RUN:SESSIONDIR safely across POSIX and Windows drive-letter paths."""
    if "=" not in spec:
        raise SystemExit(f"Invalid --attack format (expected NAME=RUN:SESSIONDIR): {spec}")
    name, rest = spec.split("=", 1)
    name = name.strip()
    rest = rest.strip()
    if not name:
        raise SystemExit(f"Empty attack NAME in --attack: {spec}")

    # Find ':' that separates RUN and SESSIONDIR (ignoring Windows drive letter ':' at index 1)
    sep_idx = None
    for i, ch in enumerate(rest):
        if ch == ":":
            if i == 1 and len(rest) > 2 and rest[0].isalpha() and rest[2] in ("/", "\\"):
                continue
            sep_idx = i
            break

    if sep_idx is None:
        raise SystemExit(f"Invalid --attack format (missing ':SESSIONDIR' in NAME=RUN:SESSIONDIR): {spec}")

    run_str = rest[:sep_idx].strip()
    sess_str = rest[sep_idx + 1:].strip()
    if not run_str or not sess_str:
        raise SystemExit(f"Invalid --attack paths in: {spec}")
    return name, Path(run_str), Path(sess_str)


def load_attack_assets(run_dir: Path, control_seed: int = 0) -> dict[str, Any]:
    """Load attack metadata, mask, seen texture, and control textures from an optimization run."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Attack run directory not found: {run_dir}")
    attack = json.loads((run_dir / "attack.json").read_text(encoding="utf-8"))
    shape = glasses.shape_from_attack(attack)
    mask = torch.from_numpy(shape.mask())[None]
    cmap = printability.ColorMap.load(run_dir / "colormap.json")
    tex_data = torch.load(run_dir / "texture.pt", map_location="cpu", weights_only=False)
    logits_raw = tex_data["logits"]
    logits_full = upsample_logits(logits_raw)
    T = torch.sigmoid(logits_full)
    seen = cmap(T)
    T_random = control_random_texture(logits_raw, seed=control_seed)
    seen_random = cmap(T_random)
    flat = seen.reshape(3, -1)
    plain = flat.mean(1, keepdim=True).expand_as(flat).reshape(seen.shape).clone()
    return {
        "run_dir": str(run_dir),
        "attack": attack,
        "victim": attack["victim"],
        "attacker_id": attack.get("attacker_id", "owner"),
        "mask": mask,
        "seen": seen,
        "seen_random": seen_random,
        "plain": plain,
    }


def check_session_not_in_train_dev(run_dir: Path, session_dir: Path) -> None:
    """Verify that none of the images in `session_dir` belong to `train` or `dev` in `run_dir/split.json`."""
    split_path = Path(run_dir) / "split.json"
    if not split_path.exists():
        return
    split = json.loads(split_path.read_text(encoding="utf-8"))
    leak_set: set[str] = set()
    for k in ("train", "dev"):
        for f_str in split.get(k, []):
            fp = Path(f_str)
            leak_set.add(str(fp))
            leak_set.add(fp.as_posix())
            try:
                leak_set.add(str(fp.resolve()))
            except Exception:
                pass
    for f in faces.list_images(session_dir):
        f_vars = {str(f), f.as_posix()}
        try:
            f_vars.add(str(f.resolve()))
        except Exception:
            pass
        if any(v in leak_set for v in f_vars):
            raise SystemExit(
                f"Evaluation session '{session_dir}' contains frame '{f}' which is in train/dev of {run_dir}"
            )


def transform_crop112(
    crop112: np.ndarray,
    band112: Optional[np.ndarray],
    mode: str,
    purifier_model: Optional[PurifierUNet] = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    if mode == "none" or band112 is None:
        return crop112.copy()
    if mode == "gray":
        return apply_gray(crop112, band112)
    if mode == "purifier":
        if purifier_model is None:
            raise ValueError("purifier_model is required when mode == 'purifier'")
        return apply_purifier(purifier_model, crop112, band112, device=device)
    raise ValueError(f"Unknown defense mode: {mode}")


def build_gallery_for_mode(
    root: Path,
    recognizer: Any,
    detector: Any,
    eyes_finder: Any,
    mode: str,
    band_template: np.ndarray,
    purifier_model: Optional[PurifierUNet] = None,
    purifier_tag: str = "none",
    device: str | torch.device = "cpu",
    max_per_id: int = faces.GALLERY_PER_ID,
    cache_dir: Optional[Path] = None,
) -> tuple[faces.Gallery, dict[str, Any]]:
    """Build 1:N gallery where every enrolled image goes through the same defense mode (none/gray/purifier)."""
    root = Path(root)
    ckpt = getattr(recognizer, "ckpt_path", None)

    candidates_by_person: dict[str, list[tuple[Path, np.ndarray, Optional[np.ndarray]]]] = {}
    skipped_by_person: dict[str, int] = {}
    used_records: list[str] = []

    for person in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        valid_items = []
        skipped = 0
        for f in faces.list_images(person)[: max_per_id * 3]:
            if len(valid_items) >= max_per_id:
                break
            img = cv2.imread(str(f))
            bbox = detector.detect(img) if img is not None else None
            if bbox is None:
                skipped += 1
                continue
            face, _ = faces.enhance(faces.crop(img, bbox))
            if face.size == 0:
                skipped += 1
                continue
            crop112 = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
            eyes = eyes_finder.eyes(img)
            band112 = band_in_crop(img, bbox, eyes, band_template=band_template, size=112) if eyes is not None else None
            valid_items.append((f, crop112, band112))
            st = f.stat()
            rel = f.relative_to(root).as_posix()
            used_records.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}:{int(band112 is not None)}")
        candidates_by_person[person.name] = valid_items
        skipped_by_person[person.name] = skipped

    band_hash = hashlib.sha1(np.ascontiguousarray(band_template).tobytes()).hexdigest()[:8]
    used_hash = hashlib.sha1("\n".join(used_records).encode()).hexdigest()[:12]
    key_str = f"{root.resolve()}|{ckpt}|{max_per_id}|{mode}|{purifier_tag}|{band_hash}|{used_hash}"
    key = hashlib.sha1(key_str.encode()).hexdigest()[:12]
    cache = (Path(cache_dir) / f"gallery_defense_{mode}_{key}.npz") if cache_dir else None

    if cache and cache.exists() and (ckpt is None or Path(str(ckpt)).exists()):
        data = np.load(cache, allow_pickle=True)
        return faces.Gallery(dict(data["embeddings"].item())), dict(data["report"].item())

    embeddings: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for person_name, valid_items in candidates_by_person.items():
        embs = []
        band_ok_count = 0
        for _, crop112, band112 in valid_items:
            if band112 is not None:
                band_ok_count += 1
            proc = transform_crop112(crop112, band112, mode=mode, purifier_model=purifier_model, device=device)
            embs.append(recognizer.extract_embedding(proc))
        if embs:
            embeddings[person_name] = np.stack(embs)
        report[person_name] = {
            "used": len(embs),
            "band_ok": band_ok_count,
            "skipped_no_face": skipped_by_person[person_name],
        }
    report["uneven"] = sorted(k for k, v in report.items() if isinstance(v, dict) and v.get("used", 0) < max_per_id)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=np.array(embeddings, dtype=object), report=np.array(report, dtype=object))
    return faces.Gallery(embeddings), report


def score_against_gallery(emb: np.ndarray, gallery_embs: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute max cosine similarity per enrolled identity."""
    emb_norm = emb / max(float(np.linalg.norm(emb)), 1e-12)
    scores: dict[str, float] = {}
    for person_id, mat in gallery_embs.items():
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_norm = mat / np.clip(norms, 1e-12, None)
        sims = mat_norm @ emb_norm
        scores[person_id] = float(np.max(sims))
    return scores


def _consistency_score(row: dict[str, Any]) -> float:
    """Return consistency detector score (`top1_drop`), treating `band_ok == 0` as `+inf` (maximally suspect)."""
    if int(row.get("band_ok", 0)) == 0 or int(row.get("rejected_no_band", 0)) == 1:
        return float("inf")
    return float(row["top1_drop"])


def _percentile_with_inf(vals: np.ndarray, pct: float) -> float:
    """Compute percentile safely when `vals` may contain `+inf` without producing NaN from `inf - inf`."""
    s = np.sort(np.asarray(vals, dtype=np.float64))
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(s[0])
    pos = (pct / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    frac = pos - lo
    a, b = float(s[lo]), float(s[hi])
    if np.isinf(a):
        return float("inf")
    if np.isinf(b):
        return a if frac == 0.0 else float("inf")
    return a + frac * (b - a)


def compute_consistency_metrics(
    rows_by_mode: dict[str, list[dict[str, Any]]],
    defense_modes: list[str],
    attack_sets: list[str],
    genuine_sets: list[str],
    tpr_target: float = 0.95,
) -> dict[str, Any]:
    """Compute AUC and FPR@95%TPR for the identity consistency detector (`top1_drop`)."""
    out: dict[str, Any] = {}
    for mode in defense_modes:
        if mode == "none" or mode not in rows_by_mode:
            continue
        m_rows = [
            r for r in rows_by_mode[mode]
            if r["face"] == 1
            and r["enhancer_applied"] == 0
            and (r["top1_drop"] != "" or int(r.get("band_ok", 0)) == 0 or int(r.get("rejected_no_band", 0)) == 1)
        ]

        pos_rows = [r for r in m_rows if r["set"] in attack_sets]
        pos_vals = np.array([_consistency_score(r) for r in pos_rows], dtype=np.float64)

        if len(pos_vals) == 0:
            out[mode] = {"threshold_tpr95": None, "n_pos": 0, "groups": {}, "by_attack": {}}
            continue

        pct = (1.0 - tpr_target) * 100.0
        threshold = _percentile_with_inf(pos_vals, pct)

        # Define negative groups
        neg_groups: dict[str, list[str]] = {}
        for gs in genuine_sets:
            neg_groups[gs] = [gs]
        ctrl_sets = [gs for gs in genuine_sets if "control" in gs]
        if ctrl_sets:
            neg_groups["owner_controls"] = ctrl_sets
        if genuine_sets:
            neg_groups["pooled_genuine"] = list(genuine_sets)

        groups_res: dict[str, Any] = {}
        for grp_name, grp_set_list in neg_groups.items():
            neg_rows = [r for r in m_rows if r["set"] in grp_set_list]
            neg_vals = np.array([_consistency_score(r) for r in neg_rows], dtype=np.float64)
            n_neg = int(len(neg_vals))
            if n_neg == 0:
                groups_res[grp_name] = {"auc": None, "fpr_at_tpr95": None, "n_pos": int(len(pos_vals)), "n_neg": 0, "n_fp": 0}
                continue
            roc_auc = round(float(auc(pos_vals, neg_vals)), 4)
            n_fp = int(np.sum(neg_vals >= threshold))
            fpr = round(float(n_fp / n_neg), 4)
            groups_res[grp_name] = {
                "auc": roc_auc,
                "fpr_at_tpr95": fpr,
                "n_pos": int(len(pos_vals)),
                "n_neg": n_neg,
                "n_fp": n_fp,
            }

        by_attack: dict[str, Any] = {}
        for atk_name in attack_sets:
            atk_vals = np.array([_consistency_score(r) for r in pos_rows if r["set"] == atk_name], dtype=np.float64)
            if len(atk_vals) == 0:
                continue
            atk_thresh = _percentile_with_inf(atk_vals, pct)
            atk_groups: dict[str, Any] = {}
            for grp_name, grp_set_list in neg_groups.items():
                neg_vals = np.array([_consistency_score(r) for r in m_rows if r["set"] in grp_set_list], dtype=np.float64)
                if len(neg_vals) == 0:
                    continue
                atk_groups[grp_name] = {
                    "auc": round(float(auc(atk_vals, neg_vals)), 4),
                    "fpr_at_tpr95": round(float(np.sum(neg_vals >= atk_thresh) / len(neg_vals)), 4),
                    "n_pos": int(len(atk_vals)),
                    "n_neg": int(len(neg_vals)),
                }
            by_attack[atk_name] = {
                "threshold_tpr95": round(atk_thresh, 4) if np.isfinite(atk_thresh) else float("inf"),
                "n_pos": int(len(atk_vals)),
                "groups": atk_groups,
            }

        out[mode] = {
            "threshold_tpr95": round(threshold, 4) if np.isfinite(threshold) else float("inf"),
            "n_pos": int(len(pos_vals)),
            "groups": groups_res,
            "by_attack": by_attack,
        }
    return out


def evaluate_purifier_rules(
    results: dict[str, dict[str, Any]],
    consistency: dict[str, Any],
    attack_sets: list[str],
    has_p01_s2: bool,
) -> dict[str, Any]:
    """Evaluate pre-registered decision rules from `experiment_log.md` (2026-11-09 'Luật đặt trước: purifier')."""
    # (1) Block attacks: success rate in mode purifier <= 10% for H09 and R2b (and any extra attack)
    atk_rates_purifier = {}
    for atk in attack_sets:
        entry = results.get(f"purifier/{atk}")
        atk_rates_purifier[atk] = entry.get("rate") if entry else None

    if atk_rates_purifier and all(v is not None for v in atk_rates_purifier.values()):
        rule1_block_attacks = all(v <= 0.10 for v in atk_rates_purifier.values())
    else:
        rule1_block_attacks = None

    # (2) Preserve genuine users: GAR(purifier) >= 0.90 * GAR(none) on owner_s6 AND on p01_s2
    gar_none_owner = results.get("none/owner_s6", {}).get("rate")
    gar_pur_owner = results.get("purifier/owner_s6", {}).get("rate")
    gar_gray_owner = results.get("gray/owner_s6", {}).get("rate")

    gar_none_p01 = results.get("none/p01_s2", {}).get("rate") if has_p01_s2 else None
    gar_pur_p01 = results.get("purifier/p01_s2", {}).get("rate") if has_p01_s2 else None
    gar_gray_p01 = results.get("gray/p01_s2", {}).get("rate") if has_p01_s2 else None

    owner_ok_pur = (
        bool(gar_pur_owner >= 0.90 * gar_none_owner)
        if (gar_pur_owner is not None and gar_none_owner is not None)
        else None
    )
    owner_ok_gray = (
        bool(gar_gray_owner >= 0.90 * gar_none_owner)
        if (gar_gray_owner is not None and gar_none_owner is not None)
        else None
    )

    if has_p01_s2:
        p01_ok_pur = (
            bool(gar_pur_p01 >= 0.90 * gar_none_p01)
            if (gar_pur_p01 is not None and gar_none_p01 is not None)
            else None
        )
        p01_ok_gray = (
            bool(gar_gray_p01 >= 0.90 * gar_none_p01)
            if (gar_gray_p01 is not None and gar_none_p01 is not None)
            else None
        )
        rule2_preserve_genuine = (
            bool(owner_ok_pur and p01_ok_pur)
            if (owner_ok_pur is not None and p01_ok_pur is not None)
            else None
        )
        gray_preserve_genuine = (
            bool(owner_ok_gray and p01_ok_gray)
            if (owner_ok_gray is not None and p01_ok_gray is not None)
            else None
        )
    else:
        p01_ok_pur = None
        p01_ok_gray = None
        rule2_preserve_genuine = None
        gray_preserve_genuine = None

    purifier_pass = (
        bool(rule1_block_attacks and rule2_preserve_genuine)
        if (rule1_block_attacks is not None and rule2_preserve_genuine is not None)
        else None
    )

    # Beats gray: GAR higher by >= 5 percentage points on both real sets, OR gray fails (2) while purifier passes (2)
    beats_gray = None
    if gar_pur_owner is not None and gar_gray_owner is not None:
        if has_p01_s2 and gar_pur_p01 is not None and gar_gray_p01 is not None:
            margin_both = (gar_pur_owner - gar_gray_owner >= 0.05) and (gar_pur_p01 - gar_gray_p01 >= 0.05)
        else:
            margin_both = (gar_pur_owner - gar_gray_owner >= 0.05)
        gray_fail_pur_pass = bool(gray_preserve_genuine is False and rule2_preserve_genuine is True)
        beats_gray = bool(margin_both or gray_fail_pur_pass)

    # Consistency detector rule: AUC >= 0.90 and FPR <= 0.10 at TPR 95% on each genuine set
    cons_pur = consistency.get("purifier", {}).get("groups", {})
    req_groups = ["owner_s6", "owner_controls"]
    if has_p01_s2:
        req_groups.append("p01_s2")
    cons_checks = {}
    for g in req_groups:
        g_stats = cons_pur.get(g)
        if g_stats and g_stats.get("auc") is not None and g_stats.get("fpr_at_tpr95") is not None:
            cons_checks[g] = bool(g_stats["auc"] >= 0.90 and g_stats["fpr_at_tpr95"] <= 0.10)
        else:
            cons_checks[g] = None

    consistency_pass = (
        all(cons_checks.values())
        if (cons_checks and all(v is not None for v in cons_checks.values()))
        else None
    )

    return {
        "rule1_block_attacks": rule1_block_attacks,
        "attack_rates_purifier": atk_rates_purifier,
        "rule2_preserve_genuine": rule2_preserve_genuine,
        "owner_s6_gar": {"none": gar_none_owner, "gray": gar_gray_owner, "purifier": gar_pur_owner, "pass": owner_ok_pur},
        "p01_s2_gar": {"none": gar_none_p01, "gray": gar_gray_p01, "purifier": gar_pur_p01, "pass": p01_ok_pur},
        "purifier_pass": purifier_pass,
        "beats_gray": beats_gray,
        "consistency_group_pass": cons_checks,
        "consistency_pass": consistency_pass,
    }


def run_eval_purifier(
    modes: list[str] = ["none", "gray", "purifier"],
    purifier_ckpt: Optional[Path] = None,
    h09_run: Path = DEFAULT_H09_RUN,
    r2b_run: Path = DEFAULT_R2B_RUN,
    owner_s6_dir: Path = DEFAULT_OWNER_S6,
    p01_root: Path = DEFAULT_P01_ROOT,
    extra_attacks: Optional[list[str]] = None,
    gallery_dir: Path = DEFAULT_GALLERY,
    extra_gallery: Optional[list[str]] = None,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    liveness: bool = False,
    control_seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Optional[Path] = None,
    purifier_model: Optional[PurifierUNet] = None,
    band_template: Optional[np.ndarray] = None,
) -> Path:
    """Run purifier evaluation across modes, attack sets, and genuine user sets."""
    for m in modes:
        if m not in ("none", "gray", "purifier"):
            raise SystemExit(f"Unsupported mode '{m}'. Choose from: none, gray, purifier")

    if "purifier" in modes and purifier_model is None:
        if purifier_ckpt is None:
            raise SystemExit("--purifier-ckpt is required when 'purifier' is in --modes")
        purifier_model = load_purifier_checkpoint(purifier_ckpt, device=device)

    meta_args = {
        "modes": list(modes),
        "purifier_ckpt": str(purifier_ckpt) if purifier_ckpt else None,
        "h09_run": str(h09_run),
        "r2b_run": str(r2b_run),
        "owner_s6_dir": str(owner_s6_dir),
        "p01_root": str(p01_root),
        "extra_attacks": extra_attacks or [],
        "gallery_dir": str(gallery_dir),
        "extra_gallery": extra_gallery or [],
        "checkpoint": str(checkpoint),
        "liveness": bool(liveness),
        "control_seed": int(control_seed),
        "device": str(device),
    }
    if output_dir is not None:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "meta.json", meta_args)
    else:
        run_dir = new_run("purifier_eval", meta_args)

    tmpl = band_template if band_template is not None else load_band_template()
    recognizer = FaceRecognizer(threshold=0.5, ckpt_path=checkpoint, strict=True)
    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    live = None
    if liveness:
        from liveness.liveness_detector import LivenessDetector
        from liveness.temporal import make_smoother
        live = LivenessDetector()

    # Always include 'none' internally so top1_before and consistency drop are computed against 'none'
    eval_modes = list(dict.fromkeys(["none"] + list(modes)))
    pur_tag = str(Path(purifier_ckpt).resolve()) if purifier_ckpt else "in_memory"
    cache_dir = run_dir.parent / "_cache"

    # 1. Build mode-specific galleries (symmetric transformation of gallery images)
    galleries_by_mode: dict[str, dict[str, np.ndarray]] = {}
    for m in eval_modes:
        gal, _ = build_gallery_for_mode(
            gallery_dir, recognizer, detector, eyes_finder,
            mode=m, band_template=tmpl, purifier_model=purifier_model,
            purifier_tag=pur_tag if m == "purifier" else m,
            device=device, cache_dir=cache_dir,
        )
        embs_m = dict(gal.embeddings)
        if extra_gallery:
            for eg_str in extra_gallery:
                eg_path = Path(eg_str)
                if not eg_path.exists():
                    raise SystemExit(f"Extra gallery directory not found: {eg_path}")
                eg_gal, _ = build_gallery_for_mode(
                    eg_path, recognizer, detector, eyes_finder,
                    mode=m, band_template=tmpl, purifier_model=purifier_model,
                    purifier_tag=pur_tag if m == "purifier" else m,
                    device=device, cache_dir=cache_dir,
                )
                dups = set(embs_m.keys()) & set(eg_gal.embeddings.keys())
                if dups:
                    raise SystemExit(f"Duplicate identities found when merging extra gallery: {sorted(dups)}")
                embs_m.update(eg_gal.embeddings)
        galleries_by_mode[m] = embs_m

    # 2. Load attack assets and build evaluation sets
    h09_assets = load_attack_assets(h09_run, control_seed=control_seed)
    r2b_assets = load_attack_assets(r2b_run, control_seed=control_seed)

    check_session_not_in_train_dev(h09_run, owner_s6_dir)
    check_session_not_in_train_dev(r2b_run, owner_s6_dir)

    # Set specifications: (set_name, kind, session_dir, render_type, assets, target_id)
    set_specs: list[dict[str, Any]] = [
        {"set": "H09", "kind": "attack", "session_dir": Path(owner_s6_dir), "render": "D", "assets": h09_assets, "target_id": h09_assets["victim"]},
        {"set": "R2b", "kind": "attack", "session_dir": Path(owner_s6_dir), "render": "D", "assets": r2b_assets, "target_id": r2b_assets["victim"]},
    ]

    if extra_attacks:
        for spec_str in extra_attacks:
            atk_name, atk_run_dir, atk_sess_dir = parse_attack_spec(spec_str)
            if not atk_sess_dir.exists():
                raise SystemExit(f"Attack session directory not found for '{atk_name}': {atk_sess_dir}")
            check_session_not_in_train_dev(atk_run_dir, atk_sess_dir)
            atk_assets = load_attack_assets(atk_run_dir, control_seed=control_seed)
            set_specs.append({
                "set": atk_name,
                "kind": "attack",
                "session_dir": atk_sess_dir,
                "render": "D",
                "assets": atk_assets,
                "target_id": atk_assets["victim"],
            })

    # Genuine sets
    set_specs.extend([
        {"set": "owner_s6", "kind": "genuine", "session_dir": Path(owner_s6_dir), "render": "baseline", "assets": None, "target_id": "owner"},
        {"set": "owner_control_random", "kind": "genuine", "session_dir": Path(owner_s6_dir), "render": "control_random", "assets": h09_assets, "target_id": "owner"},
        {"set": "owner_control_plain", "kind": "genuine", "session_dir": Path(owner_s6_dir), "render": "control_plain", "assets": h09_assets, "target_id": "owner"},
    ])

    has_p01_s2 = False
    p01_s2_dir = Path(p01_root) / "s2"
    if p01_s2_dir.exists() and "p01" in galleries_by_mode["none"]:
        set_specs.append({
            "set": "p01_s2",
            "kind": "genuine",
            "session_dir": p01_s2_dir,
            "render": "baseline",
            "assets": None,
            "target_id": "p01",
        })
        has_p01_s2 = True

    if Path(p01_root).exists() and "p01" in galleries_by_mode["none"]:
        for acc_dir in sorted(p for p in Path(p01_root).iterdir() if p.is_dir() and p.name.startswith("acc_")):
            if faces.list_images(acc_dir):
                set_specs.append({
                    "set": f"p01_{acc_dir.name}",
                    "kind": "genuine",
                    "session_dir": acc_dir,
                    "render": "baseline",
                    "assets": None,
                    "target_id": "p01",
                })

    attack_set_names = [s["set"] for s in set_specs if s["kind"] == "attack"]
    genuine_set_names = [s["set"] for s in set_specs if s["kind"] == "genuine"]

    # 3. Render frames once per set, run detector + liveness + eye landmarks, then evaluate across modes
    rows_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in eval_modes}

    for spec in set_specs:
        set_name = spec["set"]
        session_dir = spec["session_dir"]
        render_type = spec["render"]
        assets = spec["assets"]
        target_id = spec["target_id"]

        files = faces.list_images(session_dir)
        if not files:
            continue
        times = frame_times(files)
        files.sort(key=lambda f: (f.parent.name, times[str(f)]))

        smoother = None
        if live is not None:
            from liveness.temporal import make_smoother
            smoother = make_smoother(live.config)

        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            t = float(times[str(f)])

            if render_type == "baseline":
                rendered = img
            else:
                eyes_src = eyes_finder.eyes(img)
                if eyes_src is None:
                    continue
                A = glasses.similarity(eyes_src)
                if render_type == "D":
                    rendered = paint(img, assets["seen"], assets["mask"], A)
                elif render_type == "control_random":
                    rendered = paint(img, assets["seen_random"], assets["mask"], A)
                elif render_type == "control_plain":
                    rendered = paint(img, assets["plain"], assets["mask"], A)
                else:
                    rendered = img

            frame = jpeg(rendered, 95)
            bbox = detector.detect(frame)

            is_real = 1
            live_label = ""
            live_score: Any = ""
            if bbox is not None and live is not None:
                real_bool, lbl, p_score = liveness_step(live, smoother, frame, t)
                is_real = int(real_bool)
                live_label = lbl
                live_score = round(float(p_score), 4)

            if bbox is None:
                for m in eval_modes:
                    rows_by_mode[m].append({
                        "mode": m,
                        "set": set_name,
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
                        "target_id": target_id,
                        "target_sim": "",
                        "is_real": 0 if live is not None else 1,
                        "live_label": live_label,
                        "live_score": live_score,
                        "enhancer_applied": "",
                        "success": 0,
                    })
                continue

            face_crop, enh_meta = faces.enhance(faces.crop(frame, bbox))
            enhancer_applied = int(bool(enh_meta and enh_meta.get("applied")))
            crop112 = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)

            eyes_frame = eyes_finder.eyes(frame)
            band112 = band_in_crop(frame, bbox, eyes_frame, band_template=tmpl, size=112) if eyes_frame is not None else None
            band_ok = int(band112 is not None)

            # Evaluate 'none' first to record reference top-1
            ref_top1_id = ""
            ref_top1_sim = 0.0

            for m in eval_modes:
                if m in ("gray", "purifier") and band_ok == 0:
                    rows_by_mode[m].append({
                        "mode": m,
                        "set": set_name,
                        "frame": str(f),
                        "face": 1,
                        "band_ok": 0,
                        "rejected_no_band": 1,
                        "match_id": "",
                        "match_sim": "",
                        "top1_before_id": ref_top1_id,
                        "top1_before_sim": round(float(ref_top1_sim), 4),
                        "top1_after_id": "",
                        "top1_after_sim": "",
                        "top1_drop": "",
                        "target_id": target_id,
                        "target_sim": "",
                        "is_real": int(is_real),
                        "live_label": live_label,
                        "live_score": live_score,
                        "enhancer_applied": enhancer_applied,
                        "success": 0,
                    })
                    continue

                proc_crop = transform_crop112(crop112, band112, mode=m, purifier_model=purifier_model, device=device)
                emb = recognizer.extract_embedding(proc_crop)
                scores = score_against_gallery(emb, galleries_by_mode[m])

                if scores:
                    top1_id, top1_sim = max(scores.items(), key=lambda kv: kv[1])
                else:
                    top1_id, top1_sim = "", 0.0

                if m == "none":
                    ref_top1_id = top1_id
                    ref_top1_sim = top1_sim

                matched = bool(top1_sim >= recognizer.threshold)
                match_id = top1_id if matched else ""
                target_sim = scores.get(target_id, 0.0)
                sim_to_ref_top1 = scores.get(ref_top1_id, 0.0) if ref_top1_id else 0.0
                top1_drop = ref_top1_sim - sim_to_ref_top1

                success = int(bool(is_real) and match_id == target_id)

                rows_by_mode[m].append({
                    "mode": m,
                    "set": set_name,
                    "frame": str(f),
                    "face": 1,
                    "band_ok": band_ok,
                    "rejected_no_band": 0,
                    "match_id": match_id,
                    "match_sim": round(float(top1_sim), 4),
                    "top1_before_id": ref_top1_id,
                    "top1_before_sim": round(float(ref_top1_sim), 4),
                    "top1_after_id": top1_id,
                    "top1_after_sim": round(float(sim_to_ref_top1), 4),
                    "top1_drop": round(float(top1_drop), 4),
                    "target_id": target_id,
                    "target_sim": round(float(target_sim), 4),
                    "is_real": int(is_real),
                    "live_label": live_label,
                    "live_score": live_score,
                    "enhancer_applied": enhancer_applied,
                    "success": success,
                })

    # 4. Write frames.csv for the requested modes
    csv_fields = [
        "mode", "set", "frame", "face", "band_ok", "rejected_no_band",
        "match_id", "match_sim",
        "top1_before_id", "top1_before_sim",
        "top1_after_id", "top1_after_sim", "top1_drop",
        "target_id", "target_sim",
        "is_real", "live_label", "live_score",
        "enhancer_applied", "success",
    ]
    out_rows = [r for m in modes for r in rows_by_mode.get(m, [])]
    with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    # 5. Aggregate per-(mode, set) metrics on normal-light frames (and @dim if present)
    results: dict[str, dict[str, Any]] = {}
    by_mode: dict[str, dict[str, Any]] = {m: {} for m in modes}

    for m in modes:
        for spec in set_specs:
            set_name = spec["set"]
            kind = spec["kind"]
            target_id = spec["target_id"]
            sub = [r for r in rows_by_mode[m] if r["set"] == set_name]
            if not sub:
                continue

            for light_tag, pred_fn in (
                ("", lambda r: r["enhancer_applied"] in (0, "")),
                ("@dim", lambda r: r["enhancer_applied"] == 1),
            ):
                part = [r for r in sub if pred_fn(r)]
                if not part and light_tag == "@dim":
                    continue
                n = len(part)
                k = sum(int(r["success"]) for r in part)
                lo, hi = wilson(k, n)
                face_frames = [r for r in part if r["face"] == 1]
                rate = round(k / n, 4) if n else 0.0
                band_ok_rate = round(sum(int(r["band_ok"]) for r in part) / n, 4) if n else 0.0
                rejected_no_band_rate = round(sum(int(r["rejected_no_band"]) for r in part) / n, 4) if n else 0.0
                target_sims = [float(r["target_sim"]) for r in face_frames if r["target_sim"] != ""]
                drops = [float(r["top1_drop"]) for r in face_frames if r["top1_drop"] != ""]

                entry: dict[str, Any] = {
                    "mode": m,
                    "set": f"{set_name}{light_tag}",
                    "kind": kind,
                    "target_id": target_id,
                    "frames": n,
                    "success": k,
                    "rate": rate,
                    "wilson95": [round(lo, 4), round(hi, 4)],
                    "face_rate": round(len(face_frames) / n, 4) if n else 0.0,
                    "band_ok_rate": band_ok_rate,
                    "rejected_no_band_rate": rejected_no_band_rate,
                    "liveness_pass": round(sum(int(r["is_real"]) for r in face_frames) / len(face_frames), 4) if face_frames else 0.0,
                    "target_sim_mean": round(float(np.mean(target_sims)), 4) if target_sims else 0.0,
                    "top1_drop_mean": round(float(np.mean(drops)), 4) if drops else 0.0,
                }
                if kind == "attack":
                    entry["attack_success_rate"] = rate
                else:
                    entry["gar"] = rate

                key = f"{m}/{set_name}{light_tag}"
                results[key] = entry
                by_mode[m][f"{set_name}{light_tag}"] = entry

    consistency = compute_consistency_metrics(
        rows_by_mode=rows_by_mode,
        defense_modes=modes,
        attack_sets=attack_set_names,
        genuine_sets=genuine_set_names,
        tpr_target=0.95,
    )

    rule_check = evaluate_purifier_rules(
        results=results,
        consistency=consistency,
        attack_sets=attack_set_names,
        has_p01_s2=has_p01_s2,
    )

    summary = {
        **meta_args,
        "run_dir": str(run_dir),
        "attack_sets": attack_set_names,
        "genuine_sets": genuine_set_names,
        "results": results,
        "by_mode": by_mode,
        "consistency": consistency,
        "rule_check": rule_check,
    }
    write_json(run_dir / "summary.json", summary)

    # Print summary table
    print("\n" + "=" * 92)
    print("PURIFIER EVALUATION SUMMARY (NORMAL LIGHT)")
    print("=" * 92)
    print(f"{'Mode':<10} {'Set':<24} {'Kind':<9} {'Rate/GAR':<10} {'95% CI':<18} {'BandOK':<8} {'DropMean':<8}")
    print("-" * 92)
    for k_name, st in results.items():
        if "@dim" in k_name:
            continue
        ci_str = f"[{st['wilson95'][0]:.4f}, {st['wilson95'][1]:.4f}]"
        print(
            f"{st['mode']:<10} {st['set']:<24} {st['kind']:<9} "
            f"{st['rate']:<10.4f} {ci_str:<18} {st['band_ok_rate']:<8.4f} {st['top1_drop_mean']:<8.4f}"
        )
    print("=" * 92)

    if consistency:
        print("\n" + "=" * 92)
        print("IDENTITY CONSISTENCY DETECTOR (TOP1_DROP AT 95% ATTACK TPR)")
        print("=" * 92)
        for m_name, c_info in consistency.items():
            t_val = c_info.get("threshold_tpr95")
            t_str = f"{t_val:.4f}" if t_val is not None else "N/A"
            print(f"Mode: {m_name} | Threshold@95%TPR = {t_str} (N_pos={c_info.get('n_pos', 0)})")
            for grp_name, g_st in c_info.get("groups", {}).items():
                auc_str = f"{g_st['auc']:.4f}" if g_st.get("auc") is not None else "N/A"
                fpr_str = f"{g_st['fpr_at_tpr95']:.4f}" if g_st.get("fpr_at_tpr95") is not None else "N/A"
                print(f"  vs {grp_name:<24}: AUC = {auc_str:<8} FPR@95% = {fpr_str:<8} (N_neg={g_st.get('n_neg', 0)})")
            print("-" * 92)

    print(f"Rule check: purifier_pass={rule_check['purifier_pass']}  beats_gray={rule_check['beats_gray']}  consistency_pass={rule_check['consistency_pass']}")
    print(f"Artifacts written to: {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Purifier defense and identity consistency detector.")
    ap.add_argument("--modes", nargs="+", default=["none", "gray", "purifier"],
                    help="Defense modes to evaluate (subset of: none gray purifier)")
    ap.add_argument("--purifier-ckpt", default=None,
                    help="Path to trained PurifierUNet checkpoint (best.pt)")
    ap.add_argument("--h09-run", default=str(DEFAULT_H09_RUN))
    ap.add_argument("--r2b-run", default=str(DEFAULT_R2B_RUN))
    ap.add_argument("--owner-s6-dir", default=str(DEFAULT_OWNER_S6))
    ap.add_argument("--p01-root", default=str(DEFAULT_P01_ROOT))
    ap.add_argument("--attack", nargs="*", default=None,
                    help="Optional extra attack(s) in NAME=RUN:SESSIONDIR format")
    ap.add_argument("--gallery-dir", default=str(DEFAULT_GALLERY))
    ap.add_argument("--extra-gallery", nargs="*", default=None,
                    help="Extra gallery directories (e.g. data/innovation/gallery_people)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--liveness", action="store_true", help="Gate on backend smoothed liveness")
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    run_eval_purifier(
        modes=args.modes,
        purifier_ckpt=Path(args.purifier_ckpt) if args.purifier_ckpt else None,
        h09_run=Path(args.h09_run),
        r2b_run=Path(args.r2b_run),
        owner_s6_dir=Path(args.owner_s6_dir),
        p01_root=Path(args.p01_root),
        extra_attacks=args.attack,
        gallery_dir=Path(args.gallery_dir),
        extra_gallery=args.extra_gallery,
        checkpoint=Path(args.checkpoint),
        liveness=args.liveness,
        control_seed=args.control_seed,
        device=args.device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
