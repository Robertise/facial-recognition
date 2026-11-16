"""
eval_certified.py - Certified multi-mask consensus evaluation for face defense.

Pipelines:
- none: crop -> enhance -> embed -> 1:N match against gallery none.
- (m, fill), m in {G0, G16, G32, G48, L, F}, fill in {gray, inpaint}:
    crop -> mask112 -> pixel in mask = 128 -> faces.enhance ->
    gray: keep; inpaint: apply_purifier -> extract_embedding ->
    1:N match against gallery (m, fill).

Consensus: Accept X iff every m in S has match_id(m, fill) == X (and is_real == 1 for camera sets).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
from innovation.defense.adaptive import (
    PAD_PX,
    paint_padded,
    similarity_padded,
    upsample_logits_padded,
)
from innovation.defense.band import band_in_crop, load_band_template
from innovation.defense.eval_purifier import (
    check_session_not_in_train_dev,
    load_attack_assets,
    parse_attack_spec,
    score_against_gallery,
)
from innovation.defense.maskset import (
    MASK_NAMES,
    load_maskset,
    mask_in_crop,
    mask_in_window,
    similarity_canvas,
)
from innovation.defense.purifier import (
    PurifierUNet,
    apply_gray,
    apply_purifier,
)
from innovation.defense.train_inpainter import load_inpainter_checkpoint
from innovation.eval.runs import new_run, write_json

DEFAULT_GALLERY = Path("data/innovation/gallery")
DEFAULT_EXTRA_GALLERY = Path("data/innovation/gallery_people")
DEFAULT_CHECKPOINT = Path("checkpoints/method3_best.pth")
DEFAULT_MASKSET_DIR = Path("data/innovation/defense/maskset")

DEFAULT_CANDIDATE_SETS = [
    # Full triplets
    "G48+L+F", "G32+L+F", "G16+L+F", "G0+L+F",
    # Doublets
    "G48+L", "G48+F", "G32+L", "G32+F", "G16+L", "G16+F", "G0+L", "G0+F",
    # Singlets
    "G48", "G32", "G16", "G0",
]

# Digital attacks for test split
TEST_ATTACKS = {
    "H09_s0": Path("results/innovation/20261107_214538_advglasses_impersonate"),
    "H09_s1": Path("results/innovation/20261112_112957_advglasses_impersonate"),
    "H09_s2": Path("results/innovation/20261112_115550_advglasses_impersonate"),
    "R2b": Path("results/innovation/20261106_132526_advglasses_impersonate"),
    "ADA_r8": Path("results/innovation/20261113_074527_adaptive_optimize"),
    "ADA_r16": Path("results/innovation/20261113_074700_adaptive_optimize"),
    "ADA_r32": Path("results/innovation/20261113_074829_adaptive_optimize"),
    "ADA_r48": Path("results/innovation/20261113_075000_adaptive_optimize"),
}


def compute_iou(b1: list[int] | tuple[int, ...], b2: list[int] | tuple[int, ...]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0 else 0.0


def transform_branch_crop(
    face_raw: np.ndarray,
    img_shape: tuple[int, ...],
    bbox: list[int],
    eyes: Optional[np.ndarray],
    mask_canvas: np.ndarray,
    fill: str,
    inpainter: Optional[PurifierUNet] = None,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Transform face crop for branch (m, fill). Returns (crop112, mask112)."""
    c112 = cv2.resize(face_raw, (112, 112), interpolation=cv2.INTER_LINEAR)
    if eyes is None:
        return c112, None

    mask112 = mask_in_crop(img_shape, bbox, eyes, mask_canvas, size=112)
    if mask112 is None:
        return c112, None

    masked112 = c112.copy()
    masked112[mask112 > 0] = 128
    enhanced, _ = faces.enhance(masked112)

    if fill == "gray":
        out = enhanced.copy()
        out[mask112 > 0] = 128
        return out, mask112
    elif fill == "inpaint":
        if inpainter is None:
            raise ValueError("inpainter model is required for fill == 'inpaint'")
        out = apply_purifier(inpainter, enhanced, mask112, device=device)
        return out, mask112
    else:
        raise ValueError(f"Unknown fill: {fill}")


def build_branch_gallery(
    gallery_dir: Path,
    recognizer: Any,
    detector: Any,
    eyes_finder: Any,
    branch_name: str,
    mask_canvas: Optional[np.ndarray] = None,
    inpainter: Optional[PurifierUNet] = None,
    inpainter_tag: str = "none",
    device: str | torch.device = "cpu",
    max_per_id: int = faces.GALLERY_PER_ID,
    cache_dir: Optional[Path] = None,
) -> tuple[faces.Gallery, dict[str, Any]]:
    """Build gallery for a specific branch (none, or <m>/<fill>)."""
    root = Path(gallery_dir)
    ckpt = getattr(recognizer, "ckpt_path", None)

    candidates_by_person: dict[str, list[tuple[Path, np.ndarray]]] = {}
    skipped_by_person: dict[str, int] = {}
    used_records: list[str] = []

    m_name, fill = (branch_name.split("/", 1) if "/" in branch_name else (branch_name, "none"))

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
            face_raw = faces.crop(img, bbox)
            if face_raw.size == 0:
                skipped += 1
                continue

            if branch_name == "none":
                face_enh, _ = faces.enhance(face_raw)
                crop112 = cv2.resize(face_enh, (112, 112), interpolation=cv2.INTER_LINEAR)
            else:
                eyes = eyes_finder.eyes(img)
                if eyes is None or mask_canvas is None:
                    skipped += 1
                    continue
                crop112, m112 = transform_branch_crop(
                    face_raw=face_raw,
                    img_shape=img.shape,
                    bbox=bbox,
                    eyes=eyes,
                    mask_canvas=mask_canvas,
                    fill=fill,
                    inpainter=inpainter,
                    device=device,
                )
                if m112 is None:
                    skipped += 1
                    continue

            valid_items.append((f, crop112))
            st = f.stat()
            rel = f.relative_to(root).as_posix()
            used_records.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
        candidates_by_person[person.name] = valid_items
        skipped_by_person[person.name] = skipped

    mask_hash = (
        hashlib.sha1(np.ascontiguousarray(mask_canvas).tobytes()).hexdigest()[:8]
        if mask_canvas is not None
        else "none"
    )
    used_hash = hashlib.sha1("\n".join(used_records).encode()).hexdigest()[:12]
    key_str = f"{root.resolve()}|{ckpt}|{max_per_id}|{branch_name}|{inpainter_tag}|{mask_hash}|{used_hash}"
    key = hashlib.sha1(key_str.encode()).hexdigest()[:12]
    cache = (Path(cache_dir) / f"gallery_certified_{branch_name.replace('/', '_')}_{key}.npz") if cache_dir else None

    if cache and cache.exists() and (ckpt is None or Path(str(ckpt)).exists()):
        data = np.load(cache, allow_pickle=True)
        return faces.Gallery(dict(data["embeddings"].item())), dict(data["report"].item())

    embeddings: dict[str, np.ndarray] = {}
    report = {"skipped": skipped_by_person, "uneven": []}
    for person_name, valid_items in candidates_by_person.items():
        if len(valid_items) < max_per_id:
            report["uneven"].append(person_name)
        embs = [recognizer.extract_embedding(c) for _, c in valid_items]
        if embs:
            embeddings[person_name] = np.stack(embs)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, embeddings=embeddings, report=report)

    return faces.Gallery(embeddings), report


def load_adaptive_assets(run_dir: Path) -> dict[str, Any]:
    """Load assets from an adaptive optimization run (texture.pt, mask.png)."""
    run_dir = Path(run_dir)
    tex_data = torch.load(run_dir / "texture.pt", map_location="cpu", weights_only=False)
    logits = tex_data["logits"]
    logits_full = upsample_logits_padded(logits)
    T = torch.sigmoid(logits_full)
    mask_png = cv2.imread(str(run_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    mask_t = torch.from_numpy((mask_png > 127).astype(np.float32))[None]
    attack = json.loads((run_dir / "attack.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "attack": attack,
        "victim": attack["victim"],
        "attacker_id": attack.get("attacker_id", "owner"),
        "mask": mask_t,
        "seen": T,
    }


def parse_sets_spec(sets_list: list[str]) -> list[tuple[str, list[str]]]:
    """Parse sets list into list of (set_name, [mask_names])."""
    parsed = []
    for s_spec in sets_list:
        masks = [m.strip() for m in s_spec.split("+") if m.strip()]
        for m in masks:
            if m not in MASK_NAMES:
                raise ValueError(f"Unknown mask '{m}' in set '{s_spec}'")
        parsed.append((s_spec, masks))
    return parsed


def compute_branches_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-branch metrics (GAR for genuine sets, attack success rate for attack sets)."""
    normal_rows = [r for r in rows if r.get("light") == "normal"]
    by_frame: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in normal_rows:
        k = (r["set"], r["frame"])
        if k not in by_frame:
            by_frame[k] = {}
        by_frame[k][r["branch"]] = r

    branch_list = ["none"] + [f"{m}/{fill}" for m in MASK_NAMES for fill in ("gray", "inpaint")]
    branches_summary: dict[str, Any] = {}

    genuine_sets = sorted(list({r["set"] for r in rows if r.get("kind") == "genuine"}))
    attack_sets = sorted(list({r["set"] for r in rows if r.get("kind") == "attack"}))

    for br in branch_list:
        gen_sets_dict: dict[str, Any] = {}
        for s in genuine_sets:
            frames_s = [b_dict for (set_name, _), b_dict in by_frame.items() if set_name == s]
            if not frames_s:
                gen_sets_dict[s] = None
                continue
            tot = len(frames_s)
            pass_cnt = 0
            pass_recog_cnt = 0
            for b_dict in frames_s:
                r = b_dict.get(br)
                if not r:
                    continue
                target = r["target_id"]
                eyes_ok = int(r.get("eyes_ok", 0)) if br != "none" else 1
                is_real_ok = (int(r.get("is_real", 1)) == 1)
                if eyes_ok and r.get("match_id") == target:
                    pass_recog_cnt += 1
                    if is_real_ok:
                        pass_cnt += 1
            gar = round(float(pass_cnt / tot), 4) if tot > 0 else 0.0
            gar_recog = round(float(pass_recog_cnt / tot), 4) if tot > 0 else 0.0
            w_lo, w_hi = wilson(pass_cnt, tot)
            w_lo_r, w_hi_r = wilson(pass_recog_cnt, tot)
            gen_sets_dict[s] = {
                "total": tot,
                "gar": gar,
                "gar_wilson_95": [round(w_lo, 4), round(w_hi, 4)],
                "gar_recog": gar_recog,
                "gar_recog_wilson_95": [round(w_lo_r, 4), round(w_hi_r, 4)],
            }

        # Dev split: pool owner/s3_vary, owner/s4_vary, owner/s5_vary into owner_dev
        dev_names = [s for s in ("owner/s3_vary", "owner/s4_vary", "owner/s5_vary") if s in gen_sets_dict and gen_sets_dict[s] is not None]
        if dev_names:
            dev_tot = sum(gen_sets_dict[s]["total"] for s in dev_names)
            dev_pass = sum(int(round(gen_sets_dict[s]["gar"] * gen_sets_dict[s]["total"])) for s in dev_names)
            dev_pass_r = sum(int(round(gen_sets_dict[s]["gar_recog"] * gen_sets_dict[s]["total"])) for s in dev_names)
            dev_gar = round(float(dev_pass / dev_tot), 4) if dev_tot > 0 else 0.0
            dev_gar_r = round(float(dev_pass_r / dev_tot), 4) if dev_tot > 0 else 0.0
            w_lo, w_hi = wilson(dev_pass, dev_tot)
            w_lo_r, w_hi_r = wilson(dev_pass_r, dev_tot)
            gen_sets_dict["owner_dev"] = {
                "total": dev_tot,
                "gar": dev_gar,
                "gar_wilson_95": [round(w_lo, 4), round(w_hi, 4)],
                "gar_recog": dev_gar_r,
                "gar_recog_wilson_95": [round(w_lo_r, 4), round(w_hi_r, 4)],
            }

        att_sets_dict: dict[str, Any] = {}
        for a in attack_sets:
            frames_a = [b_dict for (set_name, _), b_dict in by_frame.items() if set_name == a]
            if not frames_a:
                att_sets_dict[a] = None
                continue
            tot = len(frames_a)
            suc_cnt = 0
            suc_recog_cnt = 0
            for b_dict in frames_a:
                r = b_dict.get(br)
                if not r:
                    continue
                target = r["target_id"]
                eyes_ok = int(r.get("eyes_ok", 0)) if br != "none" else 1
                is_real_ok = (int(r.get("is_real", 1)) == 1)
                if eyes_ok and r.get("match_id") == target:
                    suc_recog_cnt += 1
                    if is_real_ok:
                        suc_cnt += 1
            rate = round(float(suc_cnt / tot), 4) if tot > 0 else 0.0
            rate_recog = round(float(suc_recog_cnt / tot), 4) if tot > 0 else 0.0
            w_lo, w_hi = wilson(suc_cnt, tot)
            w_lo_r, w_hi_r = wilson(suc_recog_cnt, tot)
            att_sets_dict[a] = {
                "total": tot,
                "success": suc_cnt,
                "rate": rate,
                "wilson_95": [round(w_lo, 4), round(w_hi, 4)],
                "rate_recog": rate_recog,
                "rate_recog_wilson_95": [round(w_lo_r, 4), round(w_hi_r, 4)],
            }

        if "owner_dev" in gen_sets_dict:
            pri = gen_sets_dict["owner_dev"]
        elif gen_sets_dict:
            first_non_null = next((v for v in gen_sets_dict.values() if v is not None), None)
            pri = first_non_null
        else:
            pri = None

        branches_summary[br] = {
            "genuine_sets": gen_sets_dict,
            "genuine_total": pri["total"] if pri else 0,
            "gar": pri["gar"] if pri else 0.0,
            "gar_wilson_95": pri["gar_wilson_95"] if pri else [0.0, 0.0],
            "gar_recog": pri["gar_recog"] if pri and "gar_recog" in pri else (pri["gar"] if pri else 0.0),
            "gar_recog_wilson_95": pri["gar_recog_wilson_95"] if pri and "gar_recog_wilson_95" in pri else (pri["gar_wilson_95"] if pri else [0.0, 0.0]),
            "attacks": att_sets_dict,
        }

    return branches_summary


def compute_sets_summary(
    rows: list[dict[str, Any]],
    candidate_sets: list[tuple[str, list[str]]],
) -> dict[str, Any]:
    """Compute consensus and certification metrics for candidates sets from rows."""
    # Filter to normal light only for official metrics (no fallback to all frames)
    normal_rows = [r for r in rows if r.get("light") == "normal"]
    for s in set(r["set"] for r in rows):
        if not any(r["set"] == s and r.get("light") == "normal" for r in rows):
            print(f"Warning: Set {s} has no normal light frames; reporting null metrics.")

    # Group rows by (set, frame)
    by_frame: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in normal_rows:
        k = (r["set"], r["frame"])
        if k not in by_frame:
            by_frame[k] = {}
        by_frame[k][r["branch"]] = r

    summary_sets: dict[str, Any] = {}
    genuine_set_names = sorted(list({r["set"] for r in rows if r.get("kind") == "genuine"}))
    attack_set_names = sorted(list({r["set"] for r in rows if r.get("kind") == "attack"}))

    for s_name, m_list in candidate_sets:
        summary_sets[s_name] = {}
        for fill in ("gray", "inpaint"):
            fill_branches = [f"{m}/{fill}" for m in m_list]

            gen_sets_summary: dict[str, Any] = {}
            for s in genuine_set_names:
                frames_s = [b_dict for (set_name, _), b_dict in by_frame.items() if set_name == s]
                if not frames_s:
                    gen_sets_summary[s] = None
                    continue

                tot = len(frames_s)
                pass_cnt = 0
                pass_recog_cnt = 0
                cert_cnt = 0
                for b_dict in frames_s:
                    first_r = next(iter(b_dict.values()))
                    target = first_r["target_id"]

                    all_match_x = True
                    first_match = None
                    for br in fill_branches:
                        br_r = b_dict.get(br)
                        if not br_r or not int(br_r.get("eyes_ok", 0)):
                            all_match_x = False
                            break
                        m_id = br_r.get("match_id", "")
                        if not m_id:
                            all_match_x = False
                            break
                        if first_match is None:
                            first_match = m_id
                        elif m_id != first_match:
                            all_match_x = False
                            break

                    is_real_ok = all(int(b_dict[br].get("is_real", 1)) == 1 for br in fill_branches if br in b_dict)
                    s_accepts_id = first_match if (all_match_x and is_real_ok) else None
                    s_accepts_id_recog = first_match if all_match_x else None

                    if s_accepts_id == target:
                        pass_cnt += 1
                    if s_accepts_id_recog == target:
                        pass_recog_cnt += 1

                    # Certified <=> every m in S has match_id in {"", target_id}
                    is_cert = True
                    for br in fill_branches:
                        br_r = b_dict.get(br)
                        if br_r and br_r.get("match_id", "") not in ("", target):
                            is_cert = False
                            break
                    if is_cert:
                        cert_cnt += 1

                gar = round(float(pass_cnt / tot), 4) if tot > 0 else 0.0
                gar_recog = round(float(pass_recog_cnt / tot), 4) if tot > 0 else 0.0
                w_lo, w_hi = wilson(pass_cnt, tot)
                w_lo_r, w_hi_r = wilson(pass_recog_cnt, tot)
                cert_rate = round(float(cert_cnt / tot), 4) if tot > 0 else 0.0
                gen_sets_summary[s] = {
                    "total": tot,
                    "gar": gar,
                    "gar_wilson_95": [round(w_lo, 4), round(w_hi, 4)],
                    "gar_recog": gar_recog,
                    "gar_recog_wilson_95": [round(w_lo_r, 4), round(w_hi_r, 4)],
                    "certified_rate": cert_rate,
                    "certified_rate_recog": cert_rate,
                }

            # Dev split: pool owner/s3_vary, owner/s4_vary, owner/s5_vary into owner_dev
            dev_names = [s for s in ("owner/s3_vary", "owner/s4_vary", "owner/s5_vary") if s in gen_sets_summary and gen_sets_summary[s] is not None]
            if dev_names:
                dev_tot = sum(gen_sets_summary[s]["total"] for s in dev_names)
                dev_pass = sum(int(round(gen_sets_summary[s]["gar"] * gen_sets_summary[s]["total"])) for s in dev_names)
                dev_pass_r = sum(int(round(gen_sets_summary[s]["gar_recog"] * gen_sets_summary[s]["total"])) for s in dev_names)
                dev_cert = sum(int(round(gen_sets_summary[s]["certified_rate"] * gen_sets_summary[s]["total"])) for s in dev_names)
                dev_gar = round(float(dev_pass / dev_tot), 4) if dev_tot > 0 else 0.0
                dev_gar_r = round(float(dev_pass_r / dev_tot), 4) if dev_tot > 0 else 0.0
                w_lo, w_hi = wilson(dev_pass, dev_tot)
                w_lo_r, w_hi_r = wilson(dev_pass_r, dev_tot)
                dev_cert_rate = round(float(dev_cert / dev_tot), 4) if dev_tot > 0 else 0.0
                gen_sets_summary["owner_dev"] = {
                    "total": dev_tot,
                    "gar": dev_gar,
                    "gar_wilson_95": [round(w_lo, 4), round(w_hi, 4)],
                    "gar_recog": dev_gar_r,
                    "gar_recog_wilson_95": [round(w_lo_r, 4), round(w_hi_r, 4)],
                    "certified_rate": dev_cert_rate,
                    "certified_rate_recog": dev_cert_rate,
                }

            att_sets_summary: dict[str, Any] = {}
            for a in attack_set_names:
                frames_a = [b_dict for (set_name, _), b_dict in by_frame.items() if set_name == a]
                if not frames_a:
                    att_sets_summary[a] = None
                    continue

                tot = len(frames_a)
                suc_cnt = 0
                suc_recog_cnt = 0
                for b_dict in frames_a:
                    first_r = next(iter(b_dict.values()))
                    target = first_r["target_id"]

                    all_match_x = True
                    first_match = None
                    for br in fill_branches:
                        br_r = b_dict.get(br)
                        if not br_r or not int(br_r.get("eyes_ok", 0)):
                            all_match_x = False
                            break
                        m_id = br_r.get("match_id", "")
                        if not m_id:
                            all_match_x = False
                            break
                        if first_match is None:
                            first_match = m_id
                        elif m_id != first_match:
                            all_match_x = False
                            break

                    is_real_ok = all(int(b_dict[br].get("is_real", 1)) == 1 for br in fill_branches if br in b_dict)
                    s_accepts_id = first_match if (all_match_x and is_real_ok) else None
                    s_accepts_id_recog = first_match if all_match_x else None

                    if s_accepts_id == target:
                        suc_cnt += 1
                    if s_accepts_id_recog == target:
                        suc_recog_cnt += 1

                rate = round(float(suc_cnt / tot), 4) if tot > 0 else 0.0
                rate_recog = round(float(suc_recog_cnt / tot), 4) if tot > 0 else 0.0
                w_lo, w_hi = wilson(suc_cnt, tot)
                w_lo_r, w_hi_r = wilson(suc_recog_cnt, tot)
                att_sets_summary[a] = {
                    "total": tot,
                    "success": suc_cnt,
                    "rate": rate,
                    "wilson_95": [round(w_lo, 4), round(w_hi, 4)],
                    "rate_recog": rate_recog,
                    "rate_recog_wilson_95": [round(w_lo_r, 4), round(w_hi_r, 4)],
                }

            if "owner_dev" in gen_sets_summary:
                pri = gen_sets_summary["owner_dev"]
            elif gen_sets_summary:
                first_non_null = next((v for v in gen_sets_summary.values() if v is not None), None)
                pri = first_non_null
            else:
                pri = None

            summary_sets[s_name][fill] = {
                "genuine_sets": gen_sets_summary,
                "genuine_total": pri["total"] if pri else 0,
                "gar": pri["gar"] if pri else 0.0,
                "gar_wilson_95": pri["gar_wilson_95"] if pri else [0.0, 0.0],
                "gar_recog": pri["gar_recog"] if pri and "gar_recog" in pri else (pri["gar"] if pri else 0.0),
                "gar_recog_wilson_95": pri["gar_recog_wilson_95"] if pri and "gar_recog_wilson_95" in pri else (pri["gar_wilson_95"] if pri else [0.0, 0.0]),
                "certified_rate": pri["certified_rate"] if pri else 0.0,
                "certified_rate_recog": pri["certified_rate_recog"] if pri and "certified_rate_recog" in pri else (pri["certified_rate"] if pri else 0.0),
                "attacks": att_sets_summary,
            }

    return summary_sets


def evaluate_certified_session(
    frames_dir: Path,
    recognizer: Any,
    detector: Any,
    eyes_finder: Any,
    galleries: dict[str, faces.Gallery],
    masks: dict[str, np.ndarray],
    inpainter: Optional[PurifierUNet],
    target_id: str,
    set_name: str,
    kind: str,
    device: str | torch.device,
    live_detector: Optional[Any] = None,
    limit_frames: Optional[int] = None,
    attack_assets: Optional[dict[str, Any]] = None,
    attack_type: str = "glasses",
    student: Optional[Any] = None,
    thresholds_map: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Evaluate one genuine or attacked session across all branches."""
    from liveness.temporal import make_smoother
    smoother = make_smoother(live_detector.config) if live_detector is not None else None

    files = faces.list_images(frames_dir)
    if limit_frames is not None and limit_frames > 0:
        files = files[:limit_frames]

    times = frame_times(files)
    rows: list[dict[str, Any]] = []

    branch_list = ["none"] + [f"{m}/{fill}" for m in MASK_NAMES for fill in ("gray", "inpaint")]

    for f in files:
        raw_img = cv2.imread(str(f))
        if raw_img is None:
            continue
        t_sec = float(times.get(str(f), 0.0))

        # 1. Clean face landmarks for attack rendering and shift measurement
        eyes_clean = eyes_finder.eyes(raw_img)
        bbox_clean = detector.detect(raw_img)

        # 2. Render attack if kind == "attack"
        if kind == "attack":
            if eyes_clean is None or attack_assets is None:
                continue
            if attack_type == "adaptive":
                A_pad = similarity_padded(eyes_clean)
                rendered = paint_padded(raw_img, attack_assets["seen"], attack_assets["mask"], A_pad)
            else:
                A = glasses.similarity(eyes_clean)
                rendered = paint(raw_img, attack_assets["seen"], attack_assets["mask"], A)
            frame_jpg = jpeg(rendered, 95)
        else:
            frame_jpg = raw_img

        # 3. Liveness and detection on evaluated frame
        bbox_eval = detector.detect(frame_jpg)
        eyes_eval = eyes_finder.eyes(frame_jpg) if bbox_eval else None

        is_real = 1
        live_label = ""
        live_score: Any = ""
        if bbox_eval is not None and live_detector is not None:
            r_bool, lbl, p_sc = liveness_step(live_detector, smoother, frame_jpg, t_sec)
            is_real = int(r_bool)
            live_label = lbl
            live_score = round(float(p_sc), 4)

        # Check light condition via none branch
        if bbox_eval is not None:
            raw_crop_eval = faces.crop(frame_jpg, bbox_eval)
            _, enh_meta = faces.enhance(raw_crop_eval)
            light = "dim" if (enh_meta and enh_meta.get("applied")) else "normal"
        else:
            light = "normal"

        # Shift metrics for attack frame
        eye_shift_px: Any = ""
        bbox_iou_val: Any = ""
        if kind == "attack":
            if bbox_clean is not None and bbox_eval is not None:
                bbox_iou_val = round(compute_iou(bbox_clean, bbox_eval), 4)
            if eyes_clean is not None and eyes_eval is not None:
                ipd_c = np.linalg.norm(eyes_clean[1] - eyes_clean[0])
                if ipd_c > 0:
                    s_left = np.linalg.norm(eyes_eval[0] - eyes_clean[0])
                    s_right = np.linalg.norm(eyes_eval[1] - eyes_clean[1])
                    eye_shift_px = round(float(max(s_left, s_right) * 180.0 / ipd_c), 2)

        # Evaluate across all branches
        for br in branch_list:
            tau = float(thresholds_map.get(br, 0.5)) if thresholds_map else 0.5

            if bbox_eval is None:
                rows.append({
                    "set": set_name,
                    "kind": kind,
                    "frame": Path(f).name,
                    "target_id": target_id,
                    "branch": br,
                    "eyes_ok": 0,
                    "match_id": "",
                    "top1_id": "",
                    "top1_sim": "",
                    "target_sim": "",
                    "threshold": round(tau, 4),
                    "is_real": is_real,
                    "light": light,
                    "eye_shift_px": eye_shift_px,
                    "bbox_iou": bbox_iou_val,
                })
                continue

            gal = galleries[br]
            if br == "none":
                face_enh, _ = faces.enhance(faces.crop(frame_jpg, bbox_eval))
                crop112 = cv2.resize(face_enh, (112, 112), interpolation=cv2.INTER_LINEAR)
                eyes_ok = int(eyes_eval is not None)
            else:
                m_name, fill = br.split("/", 1)
                if eyes_eval is None:
                    # Fail-closed if eyes missing
                    rows.append({
                        "set": set_name,
                        "kind": kind,
                        "frame": Path(f).name,
                        "target_id": target_id,
                        "branch": br,
                        "eyes_ok": 0,
                        "match_id": "",
                        "top1_id": "",
                        "top1_sim": "",
                        "target_sim": "",
                        "threshold": round(tau, 4),
                        "is_real": is_real,
                        "light": light,
                        "eye_shift_px": eye_shift_px,
                        "bbox_iou": bbox_iou_val,
                    })
                    continue

                crop112, m112 = transform_branch_crop(
                    face_raw=faces.crop(frame_jpg, bbox_eval),
                    img_shape=frame_jpg.shape,
                    bbox=bbox_eval,
                    eyes=eyes_eval,
                    mask_canvas=masks[m_name],
                    fill=fill,
                    inpainter=inpainter,
                    device=device,
                )
                eyes_ok = 1

            if br == "none":
                emb = recognizer.extract_embedding(crop112)
            else:
                if student is not None:
                    emb = student.extract_embedding(crop112)
                else:
                    emb = recognizer.extract_embedding(crop112)

            scores = score_against_gallery(emb, gal.embeddings)

            if scores:
                top1_id, top1_sim = max(scores.items(), key=lambda x: x[1])
                target_sim = scores.get(target_id, 0.0)
                match_id = top1_id if top1_sim >= tau else ""
            else:
                top1_id = ""
                top1_sim = 0.0
                target_sim = 0.0
                match_id = ""

            rows.append({
                "set": set_name,
                "kind": kind,
                "frame": Path(f).name,
                "target_id": target_id,
                "branch": br,
                "eyes_ok": eyes_ok,
                "match_id": match_id,
                "top1_id": top1_id,
                "top1_sim": round(float(top1_sim), 4),
                "target_sim": round(float(target_sim), 4),
                "threshold": round(float(tau), 4),
                "is_real": is_real,
                "light": light,
                "eye_shift_px": eye_shift_px,
                "bbox_iou": bbox_iou_val,
            })

    return rows


def cmd_run(args: argparse.Namespace) -> Path:
    """Run certified evaluation on dev or test split."""
    device = args.device
    maskset_dir = Path(args.maskset_dir)
    masks = load_maskset(maskset_dir)

    recognizer = FaceRecognizer(ckpt_path=args.checkpoint, strict=True)
    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()

    inpainter = None
    inpainter_tag = "none"
    if args.inpainter_ckpt:
        p_ckpt = Path(args.inpainter_ckpt)
        if not p_ckpt.exists():
            raise FileNotFoundError(f"Inpainter checkpoint not found: {p_ckpt}")
        inpainter = load_inpainter_checkpoint(p_ckpt, device=device)
        inpainter_tag = p_ckpt.parent.name

    student = None
    if getattr(args, "student_ckpt", None):
        from innovation.defense.train_student import load_student
        p_stu = Path(args.student_ckpt)
        if not p_stu.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {p_stu}")
        student = load_student(p_stu, device=device)

    thresholds_map = None
    if getattr(args, "thresholds", None):
        p_th = Path(args.thresholds)
        if not p_th.exists():
            raise FileNotFoundError(f"Thresholds file not found: {p_th}")
        with open(p_th, "r", encoding="utf-8") as fh:
            raw_th = json.load(fh)
        if "thresholds" in raw_th and isinstance(raw_th["thresholds"], dict):
            thresholds_map = {k: float(v) for k, v in raw_th["thresholds"].items()}
        else:
            thresholds_map = {k: float(v) for k, v in raw_th.items()}

    gallery_mode = getattr(args, "gallery_mode", "symmetric")
    if gallery_mode == "symmetric" and getattr(args, "student_ckpt", None) is not None:
        print(
            "Warning: --student-ckpt is supplied with --gallery-mode symmetric. "
            "In symmetric mode, gallery branch embeddings still use method3. "
            "Certified Face Defense v2 is designed for --gallery-mode clean."
        )

    from liveness.liveness_detector import LivenessDetector
    live_detector = LivenessDetector() if args.liveness else None

    # Determine run directory
    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = new_run("certified_eval")

    cache_dir = run_dir.parent / "_cache"

    # Build galleries for each branch
    branch_list = ["none"] + [f"{m}/{fill}" for m in MASK_NAMES for fill in ("gray", "inpaint")]
    galleries: dict[str, faces.Gallery] = {}

    if gallery_mode == "clean":
        # clean mode: every branch is compared to gallery embeddings of 'none'
        gal_none, _ = build_branch_gallery(
            gallery_dir=Path(args.gallery),
            recognizer=recognizer,
            detector=detector,
            eyes_finder=eyes_finder,
            branch_name="none",
            mask_canvas=None,
            inpainter=inpainter,
            inpainter_tag=inpainter_tag,
            device=device,
            cache_dir=cache_dir,
        )
        if args.extra_gallery:
            for eg in args.extra_gallery:
                eg_gal, _ = build_branch_gallery(
                    gallery_dir=Path(eg),
                    recognizer=recognizer,
                    detector=detector,
                    eyes_finder=eyes_finder,
                    branch_name="none",
                    mask_canvas=None,
                    inpainter=inpainter,
                    inpainter_tag=inpainter_tag,
                    device=device,
                    cache_dir=cache_dir,
                )
                gal_none.embeddings.update(eg_gal.embeddings)
        for br in branch_list:
            galleries[br] = gal_none
    else:
        for br in branch_list:
            if br == "none":
                m_canvas = None
            else:
                m_name, _ = br.split("/", 1)
                m_canvas = masks[m_name]

            gal, _ = build_branch_gallery(
                gallery_dir=Path(args.gallery),
                recognizer=recognizer,
                detector=detector,
                eyes_finder=eyes_finder,
                branch_name=br,
                mask_canvas=m_canvas,
                inpainter=inpainter,
                inpainter_tag=inpainter_tag,
                device=device,
                cache_dir=cache_dir,
            )
            if args.extra_gallery:
                for eg in args.extra_gallery:
                    eg_gal, _ = build_branch_gallery(
                        gallery_dir=Path(eg),
                        recognizer=recognizer,
                        detector=detector,
                        eyes_finder=eyes_finder,
                        branch_name=br,
                        mask_canvas=m_canvas,
                        inpainter=inpainter,
                        inpainter_tag=inpainter_tag,
                        device=device,
                        cache_dir=cache_dir,
                    )
                    gal.embeddings.update(eg_gal.embeddings)
            galleries[br] = gal

    all_rows: list[dict[str, Any]] = []

    if args.split == "dev":
        # Dev split: owner/s3_vary, s4_vary, s5_vary (genuine only)
        dev_sessions = [
            ("owner/s3_vary", Path("data/innovation/frames/owner/s3_vary"), "owner"),
            ("owner/s4_vary", Path("data/innovation/frames/owner/s4_vary"), "owner"),
            ("owner/s5_vary", Path("data/innovation/frames/owner/s5_vary"), "owner"),
        ]
        for sname, sdir, tid in dev_sessions:
            if not sdir.exists():
                raise SystemExit(f"Missing dev session directory: {sdir}")
            rows = evaluate_certified_session(
                frames_dir=sdir,
                recognizer=recognizer,
                detector=detector,
                eyes_finder=eyes_finder,
                galleries=galleries,
                masks=masks,
                inpainter=inpainter,
                target_id=tid,
                set_name=sname,
                kind="genuine",
                device=device,
                live_detector=live_detector,
                limit_frames=args.limit_frames,
                student=student,
                thresholds_map=thresholds_map,
            )
            all_rows.extend(rows)

    elif args.split == "test":
        # Test split: genuine owner/s6_vary and p01/s2
        test_sessions = [
            ("owner/s6_vary", Path("data/innovation/frames/owner/s6_vary"), "owner"),
            ("p01/s2", Path("data/innovation/frames/p01/s2"), "p01"),
        ]
        for sname, sdir, tid in test_sessions:
            if not sdir.exists():
                raise SystemExit(f"Missing test session directory: {sdir}")
            rows = evaluate_certified_session(
                frames_dir=sdir,
                recognizer=recognizer,
                detector=detector,
                eyes_finder=eyes_finder,
                galleries=galleries,
                masks=masks,
                inpainter=inpainter,
                target_id=tid,
                set_name=sname,
                kind="genuine",
                device=device,
                live_detector=live_detector,
                limit_frames=args.limit_frames,
                student=student,
                thresholds_map=thresholds_map,
            )
            all_rows.extend(rows)

        # Attacks pasted on owner/s6_vary
        s6_dir = Path("data/innovation/frames/owner/s6_vary")
        if not s6_dir.exists():
            raise SystemExit(f"Missing test session directory for attacks: {s6_dir}")

        for att_name, run_path in TEST_ATTACKS.items():
            if not run_path.exists():
                raise SystemExit(f"Missing test attack run directory: {run_path}")
            check_session_not_in_train_dev(run_path, s6_dir)
            is_adaptive = "adaptive" in att_name.lower() or "ada" in att_name.lower()
            if is_adaptive:
                assets = load_adaptive_assets(run_path)
                att_type = "adaptive"
            else:
                assets = load_attack_assets(run_path)
                att_type = "glasses"

            rows = evaluate_certified_session(
                frames_dir=s6_dir,
                recognizer=recognizer,
                detector=detector,
                eyes_finder=eyes_finder,
                galleries=galleries,
                masks=masks,
                inpainter=inpainter,
                target_id="LFW_victim",
                set_name=f"attack/{att_name}",
                kind="attack",
                device=device,
                live_detector=live_detector,
                limit_frames=args.limit_frames,
                attack_assets=assets,
                attack_type=att_type,
                student=student,
                thresholds_map=thresholds_map,
            )
            all_rows.extend(rows)

    # Extra genuine sessions if supplied
    if getattr(args, "extra_genuine", None):
        for spec in args.extra_genuine:
            name, rest = spec.split("=", 1)
            dir_str, target_id = rest.split(":", 1)
            sdir = Path(dir_str)
            if not sdir.exists():
                raise SystemExit(f"Missing extra genuine directory: {sdir}")
            rows = evaluate_certified_session(
                frames_dir=sdir,
                recognizer=recognizer,
                detector=detector,
                eyes_finder=eyes_finder,
                galleries=galleries,
                masks=masks,
                inpainter=inpainter,
                target_id=target_id,
                set_name=name,
                kind="genuine",
                device=device,
                live_detector=live_detector,
                limit_frames=args.limit_frames,
                student=student,
                thresholds_map=thresholds_map,
            )
            all_rows.extend(rows)

    # Write frames.csv
    csv_fields = [
        "set", "kind", "frame", "target_id", "branch", "eyes_ok",
        "match_id", "top1_id", "top1_sim", "target_sim", "threshold", "is_real",
        "light", "eye_shift_px", "bbox_iou",
    ]
    with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(all_rows)

    # Count evaluated frames per set
    n_br = len(branch_list)
    set_frame_counts: dict[str, int] = {}
    for r in all_rows:
        s = r["set"]
        set_frame_counts[s] = set_frame_counts.get(s, 0) + 1
    sets_evaluated = {s: cnt // n_br for s, cnt in set_frame_counts.items()}

    # Compute candidate sets summary & branch summary
    candidate_specs = parse_sets_spec(DEFAULT_CANDIDATE_SETS)
    sets_summary = compute_sets_summary(all_rows, candidate_specs)
    branches_summary = compute_branches_summary(all_rows)

    summary = {
        "split": args.split,
        "total_rows": len(all_rows),
        "sets_evaluated": sets_evaluated,
        "branches": branches_summary,
        "sets": sets_summary,
        "config": {
            "gallery_mode": gallery_mode,
            "student_ckpt": getattr(args, "student_ckpt", None),
            "student_fill": getattr(student, "fill", None) if student else None,
            "thresholds": getattr(args, "thresholds", None),
            "inpainter_ckpt": getattr(args, "inpainter_ckpt", None),
            "liveness": bool(getattr(args, "liveness", False)),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir


def cmd_lfwval(args: argparse.Namespace) -> Path:
    """Evaluate LFW validation set across branches and candidate sets."""
    crops_dir = Path(args.crops_dir)
    landmarks_dir = Path(args.landmarks_dir)
    maskset_dir = Path(args.maskset_dir)
    masks = load_maskset(maskset_dir)
    device = args.device

    if not args.inpainter_ckpt:
        raise SystemExit("Missing required argument: --inpainter-ckpt")
    p_ckpt = Path(args.inpainter_ckpt)
    if not p_ckpt.exists():
        raise SystemExit(f"Inpainter checkpoint not found: {p_ckpt}")

    recognizer = FaceRecognizer(ckpt_path=args.checkpoint, strict=True)
    inpainter = load_inpainter_checkpoint(p_ckpt, device=device)

    student = None
    if getattr(args, "student_ckpt", None):
        from innovation.defense.train_student import load_student
        p_stu = Path(args.student_ckpt)
        if not p_stu.exists():
            raise SystemExit(f"Student checkpoint not found: {p_stu}")
        student = load_student(p_stu, device=device)

    thresholds_map = None
    if getattr(args, "thresholds", None):
        p_th = Path(args.thresholds)
        if not p_th.exists():
            raise SystemExit(f"Thresholds file not found: {p_th}")
        with open(p_th, "r", encoding="utf-8") as fh:
            raw_th = json.load(fh)
        if "thresholds" in raw_th and isinstance(raw_th["thresholds"], dict):
            thresholds_map = {k: float(v) for k, v in raw_th["thresholds"].items()}
        else:
            thresholds_map = {k: float(v) for k, v in raw_th.items()}

    gallery_mode = getattr(args, "gallery_mode", "symmetric")
    if gallery_mode == "symmetric" and getattr(args, "student_ckpt", None) is not None:
        print(
            "Warning: --student-ckpt is supplied with --gallery-mode symmetric. "
            "In symmetric mode, gallery branch embeddings still use method3. "
            "Certified Face Defense v2 is designed for --gallery-mode clean."
        )

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = new_run("certified_lfwval")

    # Load val shards with strict pairing by filename
    val_c_map = {p.name: p for p in crops_dir.glob("val_*.npz")}
    val_l_map = {p.name: p for p in landmarks_dir.glob("val_*.npz")}
    if not val_c_map:
        raise SystemExit(f"No val crop shards found in {crops_dir}")
    if not val_l_map:
        raise SystemExit(f"No val landmark shards found in {landmarks_dir}")

    by_identity: dict[str, list[dict[str, Any]]] = {}
    for sname in sorted(val_l_map.keys()):
        if sname not in val_c_map:
            raise SystemExit(f"Missing crop shard for landmark {sname}")
        cs = val_c_map[sname]
        ls = val_l_map[sname]

        c_data = np.load(cs, allow_pickle=True)
        l_data = np.load(ls, allow_pickle=True)

        c_paths = c_data["paths"]
        crops = c_data["crops"]
        identities = c_data["identities"]
        if "paths" in l_data:
            l_paths = l_data["paths"]
            if len(c_paths) != len(l_paths):
                if len(l_paths) < len(c_paths) and np.array_equal(c_paths[:len(l_paths)], l_paths):
                    c_crops = crops[:len(l_paths)]
                    c_paths = c_paths[:len(l_paths)]
                    identities = identities[:len(l_paths)]
                    crops = c_crops
                else:
                    raise SystemExit(
                        f"Row count mismatch for shard {cs.name}: crop {len(c_paths)} vs landmark {len(l_paths)}"
                    )
            if not np.array_equal(c_paths, l_paths):
                raise SystemExit(f"Paths mismatch line-by-line for shard {cs.name}")
        else:
            if len(c_paths) != len(l_data["ok"]):
                raise SystemExit(
                    f"Row count mismatch for shard {cs.name}: crop {len(c_paths)} vs landmark {len(l_data['ok'])}"
                )
        eyes = l_data["eyes"]
        windows = l_data["window"]
        ok_arr = l_data["ok"]

        for i in range(len(ok_arr)):
            if ok_arr[i] != 1:
                continue
            person = str(identities[i])
            if person not in by_identity:
                by_identity[person] = []
            by_identity[person].append({
                "crop": crops[i],
                "path": str(c_paths[i]),
                "eyes": eyes[i],
                "window": tuple(windows[i]),
            })

    # Keep identities with >= 2 valid images
    eligible_ids = sorted(k for k, v in by_identity.items() if len(v) >= 2)
    if not eligible_ids:
        raise SystemExit("No LFW val identities with >= 2 valid images found")

    # Partition A/B if requested
    from innovation.defense.train_student import identity_is_val_a
    val_part = getattr(args, "val_part", "all")
    if val_part == "A":
        eligible_ids = [pid for pid in eligible_ids if identity_is_val_a(pid)]
        set_name = "lfw_val_A"
    elif val_part == "B":
        eligible_ids = [pid for pid in eligible_ids if not identity_is_val_a(pid)]
        set_name = "lfw_val_B"
    else:
        set_name = "lfw_val"

    if not eligible_ids:
        raise SystemExit(f"No LFW val identities found for partition {val_part}")

    branch_list = ["none"] + [f"{m}/{fill}" for m in MASK_NAMES for fill in ("gray", "inpaint")]

    # Build gallery: first image of each eligible identity
    galleries: dict[str, dict[str, np.ndarray]] = {br: {} for br in branch_list}
    probes: list[dict[str, Any]] = []

    for pid in eligible_ids:
        items = sorted(by_identity[pid], key=lambda x: x["path"])
        gal_item = items[0]
        probe_items = items[1:]

        emb_none = recognizer.extract_embedding(gal_item["crop"])
        galleries["none"][pid] = np.atleast_2d(emb_none)

        if gallery_mode == "clean":
            for br in branch_list:
                if br != "none":
                    galleries[br][pid] = galleries["none"][pid]
        else:
            for br in branch_list:
                if br == "none":
                    continue
                m_name, fill = br.split("/", 1)
                m112 = mask_in_window(gal_item["window"], gal_item["eyes"], masks[m_name], size=112)
                if m112 is not None:
                    c = gal_item["crop"].copy()
                    c[m112 > 0] = 128
                    if fill == "inpaint":
                        c = apply_purifier(inpainter, c, m112, device=device)
                    emb = recognizer.extract_embedding(c)
                    galleries[br][pid] = np.atleast_2d(emb)

        for p_it in probe_items:
            probes.append({"person": pid, **p_it})

    if getattr(args, "limit", None) is not None and args.limit > 0:
        probes = probes[:args.limit]

    all_rows = []
    for p_it in probes:
        pid = p_it["person"]
        for br in branch_list:
            tau = float(thresholds_map.get(br, 0.5)) if thresholds_map else 0.5

            if br == "none":
                emb = recognizer.extract_embedding(p_it["crop"])
                scores = score_against_gallery(emb, galleries[br])
                eyes_ok = 1
            else:
                m_name, fill = br.split("/", 1)
                m112 = mask_in_window(p_it["window"], p_it["eyes"], masks[m_name], size=112)
                if m112 is None:
                    # Reject sample when mask cannot be computed (never use unmasked crop)
                    eyes_ok = 0
                    scores = {}
                else:
                    eyes_ok = 1
                    c = p_it["crop"].copy()
                    c[m112 > 0] = 128
                    if fill == "inpaint":
                        c = apply_purifier(inpainter, c, m112, device=device)
                    if student is not None:
                        emb = student.extract_embedding(c)
                    else:
                        emb = recognizer.extract_embedding(c)
                    scores = score_against_gallery(emb, galleries[br])

            if scores:
                top1_id, top1_sim = max(scores.items(), key=lambda x: x[1])
                target_sim = scores.get(pid, 0.0)
                match_id = top1_id if top1_sim >= tau else ""
            else:
                top1_id = ""
                top1_sim = 0.0
                target_sim = 0.0
                match_id = ""

            all_rows.append({
                "set": set_name,
                "kind": "genuine",
                "frame": p_it["path"],
                "target_id": pid,
                "branch": br,
                "eyes_ok": eyes_ok,
                "match_id": match_id,
                "top1_id": top1_id,
                "top1_sim": round(float(top1_sim), 4),
                "target_sim": round(float(target_sim), 4),
                "threshold": round(float(tau), 4),
                "is_real": 1,
                "light": "normal",
                "eye_shift_px": "",
                "bbox_iou": "",
            })

    # Write frames.csv and summary.json
    csv_fields = [
        "set", "kind", "frame", "target_id", "branch", "eyes_ok",
        "match_id", "top1_id", "top1_sim", "target_sim", "threshold", "is_real",
        "light", "eye_shift_px", "bbox_iou",
    ]
    with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(all_rows)

    candidate_specs = parse_sets_spec(DEFAULT_CANDIDATE_SETS)
    sets_summary = compute_sets_summary(all_rows, candidate_specs)
    branches_summary = compute_branches_summary(all_rows)
    summary = {
        "split": set_name,
        "total_probes": len(probes),
        "total_rows": len(all_rows),
        "sets_evaluated": {set_name: len(probes)},
        "branches": branches_summary,
        "sets": sets_summary,
        "config": {
            "val_part": val_part,
            "gallery_mode": gallery_mode,
            "student_ckpt": getattr(args, "student_ckpt", None),
            "student_fill": getattr(student, "fill", None) if student else None,
            "thresholds": getattr(args, "thresholds", None),
            "inpainter_ckpt": getattr(args, "inpainter_ckpt", None),
            "liveness": False,
        },
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir


def cmd_calibrate(args: argparse.Namespace) -> Path:
    """Calibrate branch thresholds on LFW val A so each branch FAR <= FAR_none."""
    crops_dir = Path(args.crops_dir)
    landmarks_dir = Path(args.landmarks_dir)
    maskset_dir = Path(args.maskset_dir)
    masks = load_maskset(maskset_dir)
    device = args.device

    recognizer = FaceRecognizer(ckpt_path=args.checkpoint, strict=True)
    inpainter = None
    if getattr(args, "inpainter_ckpt", None):
        p_ckpt = Path(args.inpainter_ckpt)
        if not p_ckpt.exists():
            raise SystemExit(f"Inpainter checkpoint not found: {p_ckpt}")
        inpainter = load_inpainter_checkpoint(p_ckpt, device=device)

    student = None
    if getattr(args, "student_ckpt", None):
        from innovation.defense.train_student import load_student
        p_stu = Path(args.student_ckpt)
        if not p_stu.exists():
            raise SystemExit(f"Student checkpoint not found: {p_stu}")
        student = load_student(p_stu, device=device)

    gallery_mode = getattr(args, "gallery_mode", "symmetric")
    if gallery_mode == "symmetric" and getattr(args, "student_ckpt", None) is not None:
        print(
            "Warning: --student-ckpt is supplied with --gallery-mode symmetric. "
            "In symmetric mode, gallery branch embeddings still use method3. "
            "Certified Face Defense v2 is designed for --gallery-mode clean."
        )

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = new_run("certified_calib")

    # Load val shards
    val_c_map = {p.name: p for p in crops_dir.glob("val_*.npz")}
    val_l_map = {p.name: p for p in landmarks_dir.glob("val_*.npz")}
    if not val_c_map:
        raise SystemExit(f"No val crop shards found in {crops_dir}")
    if not val_l_map:
        raise SystemExit(f"No val landmark shards found in {landmarks_dir}")

    from innovation.defense.train_student import identity_is_val_a

    by_identity: dict[str, list[dict[str, Any]]] = {}
    for sname in sorted(val_l_map.keys()):
        if sname not in val_c_map:
            continue
        c_data = np.load(val_c_map[sname], allow_pickle=True)
        l_data = np.load(val_l_map[sname], allow_pickle=True)

        c_paths = c_data["paths"]
        crops = c_data["crops"]
        identities = c_data["identities"]
        eyes = l_data["eyes"]
        windows = l_data["window"]
        ok_arr = l_data["ok"]

        n_items = min(len(c_paths), len(ok_arr))
        for i in range(n_items):
            if ok_arr[i] != 1:
                continue
            person = str(identities[i])
            if not identity_is_val_a(person):
                continue
            if person not in by_identity:
                by_identity[person] = []
            by_identity[person].append({
                "crop": crops[i],
                "path": str(c_paths[i]),
                "eyes": eyes[i],
                "window": tuple(windows[i]),
            })

    eligible_ids = sorted(k for k, v in by_identity.items() if len(v) >= 2)
    if not eligible_ids:
        raise SystemExit("No LFW val A identities with >= 2 valid images found")

    branch_list = ["none"] + [f"{m}/{fill}" for m in MASK_NAMES for fill in ("gray", "inpaint")]
    galleries: dict[str, dict[str, np.ndarray]] = {br: {} for br in branch_list}
    probes: list[dict[str, Any]] = []

    for pid in eligible_ids:
        items = sorted(by_identity[pid], key=lambda x: x["path"])
        gal_item = items[0]
        probe_items = items[1:]

        emb_none = recognizer.extract_embedding(gal_item["crop"])
        galleries["none"][pid] = np.atleast_2d(emb_none)

        if gallery_mode == "clean":
            for br in branch_list:
                if br != "none":
                    galleries[br][pid] = galleries["none"][pid]
        else:
            for br in branch_list:
                if br == "none":
                    continue
                m_name, fill = br.split("/", 1)
                m112 = mask_in_window(gal_item["window"], gal_item["eyes"], masks[m_name], size=112)
                if m112 is not None:
                    c = gal_item["crop"].copy()
                    c[m112 > 0] = 128
                    if fill == "inpaint" and inpainter is not None:
                        c = apply_purifier(inpainter, c, m112, device=device)
                    emb = recognizer.extract_embedding(c)
                    galleries[br][pid] = np.atleast_2d(emb)

        for p_it in probe_items:
            probes.append({"person": pid, **p_it})

    if getattr(args, "limit", None) is not None and args.limit > 0:
        probes = probes[:args.limit]

    n_probes = len(probes)
    if n_probes == 0:
        raise SystemExit("No probes available for calibration.")

    max_impostors: dict[str, list[float]] = {br: [] for br in branch_list}

    for p_it in probes:
        pid = p_it["person"]
        for br in branch_list:
            if br == "none":
                emb = recognizer.extract_embedding(p_it["crop"])
                scores = score_against_gallery(emb, galleries[br])
            else:
                m_name, fill = br.split("/", 1)
                m112 = mask_in_window(p_it["window"], p_it["eyes"], masks[m_name], size=112)
                if m112 is None:
                    scores = {}
                else:
                    c = p_it["crop"].copy()
                    c[m112 > 0] = 128
                    if fill == "inpaint" and inpainter is not None:
                        c = apply_purifier(inpainter, c, m112, device=device)
                    if student is not None:
                        emb = student.extract_embedding(c)
                    else:
                        emb = recognizer.extract_embedding(c)
                    scores = score_against_gallery(emb, galleries[br])

            imp_sims = [sim for gid, sim in scores.items() if gid != pid]
            max_imp = max(imp_sims) if imp_sims else 0.0
            max_impostors[br].append(float(max_imp))

    far_none = sum(1 for s in max_impostors["none"] if s >= 0.5) / n_probes

    grid = [round(0.30 + i * 0.005, 4) for i in range(121)]
    calibrated_thresholds: dict[str, float] = {}
    far_by_branch: dict[str, float] = {}

    for br in branch_list:
        scores_b = max_impostors[br]
        best_tau = 0.90
        best_far = sum(1 for s in scores_b if s >= 0.90) / n_probes
        for tau in grid:
            far = sum(1 for s in scores_b if s >= tau) / n_probes
            if far <= far_none:
                best_tau = tau
                best_far = far
                break

        calibrated_thresholds[br] = round(best_tau, 4)
        far_by_branch[br] = round(best_far, 4)

    out_dict = {
        "thresholds": calibrated_thresholds,
        "far_none": round(float(far_none), 4),
        "num_impostors": n_probes,
        "far_by_branch": far_by_branch,
        "config": {
            "gallery_mode": gallery_mode,
            "student_ckpt": getattr(args, "student_ckpt", None),
            "student_fill": getattr(student, "fill", None) if student else None,
            "inpainter_ckpt": getattr(args, "inpainter_ckpt", None),
        },
    }
    write_json(run_dir / "thresholds.json", out_dict)
    write_json(run_dir / "summary.json", out_dict)
    print(f"Calibrated thresholds saved to {run_dir / 'thresholds.json'}")
    print(f"FAR_none: {far_none:.4f}")
    return run_dir


def cmd_summarize(args: argparse.Namespace) -> None:
    """Summarize sets from frames.csv or select winning mask set S on dev runs."""
    metric_key = getattr(args, "metric", "gar") or "gar"
    fill_mode = getattr(args, "fill", "inpaint") or "inpaint"
    if args.select:
        if not args.dev_runs or len(args.dev_runs) < 2:
            raise SystemExit("--select requires two dev runs: --dev-runs <run owner dev> <run lfwval>")
        owner_dev_dir = Path(args.dev_runs[0])
        lfwval_dir = Path(args.dev_runs[1])

        owner_sum_p = owner_dev_dir / "summary.json"
        lfw_sum_p = lfwval_dir / "summary.json"
        owner_sum = json.loads(owner_sum_p.read_text(encoding="utf-8")) if owner_sum_p.exists() else {}
        lfw_sum = json.loads(lfw_sum_p.read_text(encoding="utf-8")) if lfw_sum_p.exists() else {}

        lfw_set_key = "lfw_val_B" if ("lfw_val_B" in lfw_sum.get("sets_evaluated", {}) or lfw_sum.get("split") == "lfw_val_B") else "lfw_val"

        def get_branch_metric(sum_dict: dict[str, Any], branch_name: str, set_key: str, run_dir: Path) -> float:
            br_info = sum_dict.get("branches", {}).get(branch_name, {})
            if "genuine_sets" in br_info and set_key in br_info["genuine_sets"] and br_info["genuine_sets"][set_key] is not None:
                return float(br_info["genuine_sets"][set_key].get(metric_key, br_info["genuine_sets"][set_key].get("gar", 0.0)))
            if metric_key in br_info:
                return float(br_info.get(metric_key, 0.0))
            if "gar" in br_info:
                return float(br_info.get("gar", 0.0))
            fpath = run_dir / "frames.csv"
            if fpath.exists():
                f_rows = list(csv.DictReader(open(fpath, encoding="utf-8")))
                none_r = [r for r in f_rows if r.get("branch") == branch_name and r.get("light") == "normal"]
                if not none_r:
                    none_r = [r for r in f_rows if r.get("branch") == branch_name]
                if metric_key == "gar_recog":
                    pass_cnt = sum(1 for r in none_r if r.get("match_id") == r.get("target_id"))
                else:
                    pass_cnt = sum(1 for r in none_r if r.get("match_id") == r.get("target_id") and int(r.get("is_real", 1)) == 1)
                return float(pass_cnt / len(none_r)) if none_r else 0.0
            return 0.0

        gar_none_owner = get_branch_metric(owner_sum, "none", "owner_dev", owner_dev_dir)
        gar_none_lfw = get_branch_metric(lfw_sum, "none", lfw_set_key, lfwval_dir)

        thresh_owner = 0.90 * gar_none_owner
        thresh_lfw = 0.90 * gar_none_lfw

        print(f"\nSelection on Dev Runs (metric={metric_key}, fill={fill_mode}):")
        print(f"Owner Dev: {metric_key}(none) = {gar_none_owner:.4f}, Pass threshold = {thresh_owner:.4f}")
        print(f"LFW Val ({lfw_set_key}):   {metric_key}(none) = {gar_none_lfw:.4f}, Pass threshold = {thresh_lfw:.4f}\n")

        candidate_order = []
        for r in (48, 32, 16, 0):
            candidate_order.append(f"G{r}+L+F")
        for r in (48, 32, 16, 0):
            candidate_order.append((f"G{r}+L", f"G{r}+F"))
        for r in (48, 32, 16, 0):
            candidate_order.append(f"G{r}")

        def extract_metric(sum_dict: dict[str, Any], candidate: str, set_key: str) -> float:
            c_info = sum_dict.get("sets", {}).get(candidate, {}).get(fill_mode, {})
            if "genuine_sets" in c_info and set_key in c_info["genuine_sets"] and c_info["genuine_sets"][set_key] is not None:
                return float(c_info["genuine_sets"][set_key].get(metric_key, c_info["genuine_sets"][set_key].get("gar", 0.0)))
            return float(c_info.get(metric_key, c_info.get("gar", 0.0)))

        selected_set = None
        pass_table: dict[str, Any] = {}

        for item in candidate_order:
            if isinstance(item, str):
                s = item
                gar_o = extract_metric(owner_sum, s, "owner_dev")
                gar_l = extract_metric(lfw_sum, s, lfw_set_key)
                passed = bool(gar_o >= thresh_owner and gar_l >= thresh_lfw)
                pass_table[s] = {
                    f"{metric_key}_owner": gar_o,
                    f"{metric_key}_lfw": gar_l,
                    "pass": passed,
                }
                if passed and selected_set is None:
                    selected_set = s
            else:
                s1, s2 = item
                gar_o1 = extract_metric(owner_sum, s1, "owner_dev")
                gar_l1 = extract_metric(lfw_sum, s1, lfw_set_key)
                pass1 = bool(gar_o1 >= thresh_owner and gar_l1 >= thresh_lfw)

                gar_o2 = extract_metric(owner_sum, s2, "owner_dev")
                gar_l2 = extract_metric(lfw_sum, s2, lfw_set_key)
                pass2 = bool(gar_o2 >= thresh_owner and gar_l2 >= thresh_lfw)

                pass_table[s1] = {f"{metric_key}_owner": gar_o1, f"{metric_key}_lfw": gar_l1, "pass": pass1}
                pass_table[s2] = {f"{metric_key}_owner": gar_o2, f"{metric_key}_lfw": gar_l2, "pass": pass2}

                if selected_set is None:
                    if pass1 and pass2:
                        min1 = min(gar_o1, gar_l1)
                        min2 = min(gar_o2, gar_l2)
                        selected_set = s1 if min1 >= min2 else s2
                    elif pass1:
                        selected_set = s1
                    elif pass2:
                        selected_set = s2

        selection_res = {
            "fill": fill_mode,
            "metric": metric_key,
            "selected_set": selected_set,
            "thresholds": {"owner_dev": thresh_owner, "lfwval": thresh_lfw},
            "pass_table": pass_table,
        }
        write_json(owner_dev_dir / "selection.json", selection_res)
        print(f"Selection Result: {selected_set}")
        print(json.dumps(selection_res, indent=2))
        return

    # Regular summarize on runs
    if not args.runs:
        raise SystemExit("Must supply --run <dirs> or --select")

    set_specs = parse_sets_spec(args.sets or DEFAULT_CANDIDATE_SETS)
    for rdir_str in args.runs:
        rdir = Path(rdir_str)
        fpath = rdir / "frames.csv"
        if not fpath.exists():
            continue
        rows = list(csv.DictReader(open(fpath, encoding="utf-8")))
        for r in rows:
            r["eyes_ok"] = int(r["eyes_ok"])
            r["is_real"] = int(r.get("is_real", 1))
        summary_sets = compute_sets_summary(rows, set_specs)
        branches_summary = compute_branches_summary(rows)
        write_json(rdir / "sets_summary.json", {"branches": branches_summary, "sets": summary_sets})
        print(f"Wrote sets_summary.json to {rdir}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate certified face defense.")
    sub = ap.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run")
    run_p.add_argument("--split", choices=("dev", "test"), required=True)
    run_p.add_argument("--gallery", default=str(DEFAULT_GALLERY))
    run_p.add_argument("--extra-gallery", nargs="*", default=[str(DEFAULT_EXTRA_GALLERY)])
    run_p.add_argument("--extra-genuine", action="append", default=[], help="Extra genuine sets: NAME=DIR:TARGET")
    run_p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    run_p.add_argument("--maskset-dir", default=str(DEFAULT_MASKSET_DIR))
    run_p.add_argument("--inpainter-ckpt", default=None)
    run_p.add_argument("--student-ckpt", default=None)
    run_p.add_argument("--gallery-mode", choices=("symmetric", "clean"), default="symmetric")
    run_p.add_argument("--thresholds", default=None)
    run_p.add_argument("--liveness", action="store_true")
    run_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run_p.add_argument("--limit-frames", type=int, default=None)
    run_p.add_argument("--output-dir", default=None)

    lfw_p = sub.add_parser("lfwval")
    lfw_p.add_argument("--crops-dir", default=str(Path("data/innovation/defense/lfw_crops")))
    lfw_p.add_argument("--landmarks-dir", default=str(Path("data/innovation/defense/lfw_landmarks")))
    lfw_p.add_argument("--maskset-dir", default=str(DEFAULT_MASKSET_DIR))
    lfw_p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    lfw_p.add_argument("--inpainter-ckpt", default=None)
    lfw_p.add_argument("--student-ckpt", default=None)
    lfw_p.add_argument("--gallery-mode", choices=("symmetric", "clean"), default="symmetric")
    lfw_p.add_argument("--thresholds", default=None)
    lfw_p.add_argument("--val-part", choices=("all", "A", "B"), default="all")
    lfw_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    lfw_p.add_argument("--limit", type=int, default=None)
    lfw_p.add_argument("--output-dir", default=None)

    cal_p = sub.add_parser("calibrate")
    cal_p.add_argument("--crops-dir", default=str(Path("data/innovation/defense/lfw_crops")))
    cal_p.add_argument("--landmarks-dir", default=str(Path("data/innovation/defense/lfw_landmarks")))
    cal_p.add_argument("--maskset-dir", default=str(DEFAULT_MASKSET_DIR))
    cal_p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    cal_p.add_argument("--inpainter-ckpt", default=None)
    cal_p.add_argument("--student-ckpt", default=None)
    cal_p.add_argument("--gallery-mode", choices=("symmetric", "clean"), default="symmetric")
    cal_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cal_p.add_argument("--limit", type=int, default=None)
    cal_p.add_argument("--output-dir", default=None)

    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("--runs", nargs="*", default=[])
    sum_p.add_argument("--sets", nargs="*", default=None)
    sum_p.add_argument("--metric", choices=("gar", "gar_recog"), default="gar")
    sum_p.add_argument("--fill", choices=("inpaint", "gray"), default="inpaint")
    sum_p.add_argument("--select", action="store_true")
    sum_p.add_argument("--dev-runs", nargs="*", default=[])

    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "lfwval":
        cmd_lfwval(args)
    elif args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
