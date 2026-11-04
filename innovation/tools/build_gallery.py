"""
build_gallery.py – Build L1 attack gallery from LFW dataset.
============================================================
Selects:
  - LFW_victim (median baseline similarity to owner s1_day1 frames)
  - (optional) LFW_victim_easy (p90), LFW_victim_hard (p10)
  - LFW_0001 ... LFW_0050 (simulated enrolled class)
Excludes candidates seen during method3 training via:
  1. Name matching with VGGFace2 training metadata (if provided)
  2. ArcFace center similarity calibration on 64x64 domain
Outputs:
  - gallery/<id>/1.jpg ... 5.jpg (unmodified LFW crops)
  - gallery/_build_report.json
  - gallery/_candidates.csv
Never touches gallery/owner/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.face_recognizer import FaceRecognizer, _preprocess_face
from innovation.attacks.adv_glasses.faces import FaceDetector, crop, enhance, list_images, GALLERY_PER_ID


def normalize_name(name: str) -> str:
    """Normalize personal names for comparison: lowercase, strip accents, replace _/- with space, strip punctuation."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = no_accents.lower()
    s = re.sub(r"[_\-\s]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def is_bbox_centered(bbox: list[int], shape: tuple[int, ...], center_tol: float = 0.20) -> bool:
    """Check if the face bbox center is within center_tol of the image center."""
    h, w = shape[:2]
    if w <= 0 or h <= 0:
        return False
    bx = bbox[0] + bbox[2] / 2.0
    by = bbox[1] + bbox[3] / 2.0
    cx = w / 2.0
    cy = h / 2.0
    return (abs(bx - cx) <= center_tol * w) and (abs(by - cy) <= center_tol * h)


def ensure_lfw_extracted(data_home: str | Path = "data/public/lfw") -> Path:
    """Ensure LFW dataset is downloaded and extracted via scikit-learn helper."""
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    from sklearn.datasets import _lfw
    lfw_home, data_folder_path = _lfw._check_fetch_lfw(
        data_home=str(data_home), funneled=True, download_if_missing=True
    )
    return Path(data_folder_path)


def get_git_commit() -> str:
    """Return current git commit hash, or 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_vggface2_train_names(meta_path: Path, train_ids_dir: Path, alt_ids_dir: Optional[Path] = None) -> set[str]:
    """Read VGGFace2 identity_meta.csv and collect normalized names of identities in train_ids_dir."""
    train_ids = set()
    if train_ids_dir and train_ids_dir.is_dir():
        train_ids.update(p.name for p in train_ids_dir.iterdir() if p.is_dir())
    if alt_ids_dir and alt_ids_dir.is_dir():
        alt_ids = {p.name for p in alt_ids_dir.iterdir() if p.is_dir()}
        if len(alt_ids) > len(train_ids):
            train_ids = alt_ids
        else:
            train_ids.update(alt_ids)
    if not train_ids:
        return set()
    train_names = set()
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader, [])]
        class_col = header.index("Class_ID") if "Class_ID" in header else 0
        name_col = header.index("Name") if "Name" in header else 1
        for row in reader:
            if len(row) > max(class_col, name_col):
                cid = row[class_col].strip()
                if cid in train_ids:
                    train_names.add(normalize_name(row[name_col].strip().strip('"')))
    return train_names


def extract_arcface_centers(recognizer: FaceRecognizer) -> np.ndarray:
    """Extract and L2-normalize the 4000 class centers from ArcFace head (shape 4000, 512)."""
    model = recognizer._model
    if hasattr(model, "head") and hasattr(model.head, "weight"):
        w = model.head.weight
    elif hasattr(model, "weight"):
        w = model.weight
    else:
        raise AttributeError("Recognizer model does not have ArcFace head weight.")
    with torch.no_grad():
        w_norm = F.normalize(w, p=2, dim=1)
        return w_norm.detach().cpu().numpy()


def compute_arcface_stats(embeddings: list[np.ndarray], centers: np.ndarray) -> tuple[int, float]:
    """Given a list of L2-normalized embeddings, compute (agree, med_top1) against ArcFace centers."""
    top1_classes = []
    top1_cosines = []
    for emb in embeddings:
        sims = np.dot(centers, emb)
        top1_cls = int(np.argmax(sims))
        top1_cos = float(sims[top1_cls])
        top1_classes.append(top1_cls)
        top1_cosines.append(top1_cos)
    agree = Counter(top1_classes).most_common(1)[0][1] if top1_classes else 0
    med_top1 = float(np.median(top1_cosines)) if top1_cosines else 0.0
    return agree, med_top1


def calibrate_arcface(
    recognizer: FaceRecognizer,
    centers: np.ndarray,
    members_dir: Path,
    pairs_file: Path,
    n_samples: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Calibrate threshold T (5th percentile of members med_top1) and AUC on 64x64 domain without face detection."""
    rng = np.random.RandomState(seed)

    # 1. Members from test_data (each identity in CMU HW2P2 test_data has 2 images)
    member_dirs = sorted([p for p in members_dir.iterdir() if p.is_dir()])
    eligible_members = []
    for d in member_dirs:
        imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        if len(imgs) >= 2:
            eligible_members.append(imgs)
    if len(eligible_members) < n_samples:
        raise ValueError(f"Not enough member identities in {members_dir} (found {len(eligible_members)}, need {n_samples})")

    chosen_indices = rng.choice(len(eligible_members), size=n_samples, replace=False)
    member_meds = []
    for idx in chosen_indices:
        imgs = eligible_members[idx][:5]
        embs = []
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is not None:
                tensor = _preprocess_face(img, recognizer.preprocess)
                with torch.no_grad():
                    emb = recognizer._model.get_embedding(tensor).cpu().numpy().squeeze()
                    embs.append(emb)
        if embs:
            _, med = compute_arcface_stats(embs, centers)
            member_meds.append(med)

    # 2. Non-members from verification_pairs_val.txt (same-identity pairs)
    pairs_dir = pairs_file.parent
    same_pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "1":
                p1 = pairs_dir / parts[0]
                p2 = pairs_dir / parts[1]
                if p1.is_file() and p2.is_file():
                    same_pairs.append((p1, p2))

    if len(same_pairs) < n_samples:
        raise ValueError(f"Not enough non-member pairs in {pairs_file} (found {len(same_pairs)}, need {n_samples})")

    chosen_pairs_indices = rng.choice(len(same_pairs), size=n_samples, replace=False)
    non_member_meds = []
    for idx in chosen_pairs_indices:
        p1, p2 = same_pairs[idx]
        embs = []
        for img_path in (p1, p2):
            img = cv2.imread(str(img_path))
            if img is not None:
                tensor = _preprocess_face(img, recognizer.preprocess)
                with torch.no_grad():
                    emb = recognizer._model.get_embedding(tensor).cpu().numpy().squeeze()
                    embs.append(emb)
        if embs:
            _, med = compute_arcface_stats(embs, centers)
            non_member_meds.append(med)

    calib_T = float(np.percentile(member_meds, 5))

    # Compute ROC AUC between member_meds (class 1) and non_member_meds (class 0)
    from sklearn.metrics import roc_auc_score
    y_true = [1] * len(member_meds) + [0] * len(non_member_meds)
    y_scores = member_meds + non_member_meds
    auc = float(roc_auc_score(y_true, y_scores))

    return calib_T, auc


def load_owner_frame_embeddings(
    owner_frames_dir: Path,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> list[np.ndarray]:
    """Load valid, unglasses, unenhanced frame embeddings from owner session."""
    p_name = owner_frames_dir.name
    if p_name in {"s2_day2", "s3_dim"}:
        raise ValueError(f"--owner-frames points to '{p_name}', but only 's1_day1' is permitted.")

    frames = list_images(owner_frames_dir)
    embs = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        bbox = detector.detect(img)
        if bbox is None:
            continue
        face, meta = enhance(crop(img, bbox))
        if meta and meta.get("applied") is True:
            # Skip frames where enhancer transformed the crop
            continue
        emb = recognizer.extract_embedding(face)
        embs.append(emb)
    if not embs:
        raise ValueError(f"No valid unenhanced owner frames found in {owner_frames_dir}")
    return embs


def build_gallery(
    gallery_dir: Path,
    owner_frames_dir: Path,
    train_ids_dir: Path,
    members_dir: Path,
    pairs_file: Path,
    n_class: int = 50,
    seed: int = 0,
    extra_victims: bool = False,
    vggface2_meta: Optional[Path] = None,
    lfw_root: Optional[Path] = None,
    center_tol: float = 0.20,
    checkpoint: Optional[Path] = None,
    replace: bool = False,
) -> dict:
    gallery_dir = Path(gallery_dir)
    owner_frames_dir = Path(owner_frames_dir)
    train_ids_dir = Path(train_ids_dir)
    members_dir = Path(members_dir)
    pairs_file = Path(pairs_file)

    # 1. Check owner frames constraint
    if owner_frames_dir.name in {"s2_day2", "s3_dim"}:
        raise ValueError(f"--owner-frames points to '{owner_frames_dir.name}', but only 's1_day1' is permitted.")

    # 2. Check destination gallery
    if gallery_dir.exists():
        existing_lfw = [p for p in gallery_dir.iterdir() if p.is_dir() and p.name.startswith("LFW_")]
        if existing_lfw and not replace:
            raise FileExistsError(
                f"Gallery at {gallery_dir} already contains {len(existing_lfw)} LFW folders. "
                f"Pass --replace to overwrite (owner/ will never be touched)."
            )
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # 3. Initialize models
    print(f"[build_gallery] Loading recognizer ({checkpoint or 'default'})...")
    recognizer = FaceRecognizer(ckpt_path=checkpoint, strict=True)
    detector = FaceDetector(model_selection=0)
    centers = extract_arcface_centers(recognizer)

    # 4. Calibrate ArcFace on 64x64 domain
    print(f"[build_gallery] Calibrating ArcFace centers on 64x64 domain (members: {members_dir.name}, pairs: {pairs_file.name})...")
    calib_T, calib_auc = calibrate_arcface(recognizer, centers, members_dir, pairs_file, n_samples=200, seed=seed)
    print(f"[build_gallery] ArcFace calibration: T = {calib_T:.4f} (member 5th percentile), AUC = {calib_auc:.4f}")

    # 5. Load VGGFace2 metadata if provided
    train_names = set()
    used_name_matching = False
    if vggface2_meta and Path(vggface2_meta).is_file():
        train_names = load_vggface2_train_names(Path(vggface2_meta), train_ids_dir, members_dir)
        used_name_matching = True
        print(f"[build_gallery] Loaded {len(train_names)} train names from {vggface2_meta}")
    else:
        print("[build_gallery] No VGGFace2 meta provided or file not found; skipping name matching.")

    # 6. Extract owner frame embeddings (s1_day1 unenhanced)
    print(f"[build_gallery] Loading owner frames from {owner_frames_dir}...")
    owner_embs = load_owner_frame_embeddings(owner_frames_dir, detector, recognizer)
    print(f"[build_gallery] Found {len(owner_embs)} valid unenhanced owner frames.")

    # 7. Locate LFW dataset
    if lfw_root and Path(lfw_root).is_dir():
        lfw_dir = Path(lfw_root)
    else:
        print("[build_gallery] Locating LFW dataset via scikit-learn...")
        lfw_dir = ensure_lfw_extracted("data/public/lfw")
    print(f"[build_gallery] Reading LFW images from: {lfw_dir}")

    # 8. Scan LFW candidates
    person_dirs = sorted([p for p in lfw_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
    n_considered = len(person_dirs)
    print(f"[build_gallery] Scanning {n_considered} identities in LFW...")

    candidates = []
    n_ge5_valid = 0
    excluded_counts = {"fewer_than_5_valid": 0, "name_in_train": 0, "arcface_member": 0}

    for person in person_dirs:
        lfw_name = person.name
        img_files = sorted([p for p in person.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        n_images = len(img_files)

        if n_images < 5:
            candidates.append({
                "lfw_name": lfw_name,
                "person_dir": person,
                "n_images": n_images,
                "n_valid": 0,
                "valid_files": [],
                "embeddings": [],
                "name_in_train": False,
                "agree": None,
                "med_top1": None,
                "s_baseline": None,
                "role": "excluded",
                "excluded_reason": "fewer_than_5_images",
            })
            continue

        valid_files = []
        valid_embs = []
        for f in img_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            bbox = detector.detect(img)
            if bbox is None:
                continue
            if not is_bbox_centered(bbox, img.shape, center_tol=center_tol):
                continue
            face, _ = enhance(crop(img, bbox))
            if face.size == 0:
                continue
            emb = recognizer.extract_embedding(face)
            valid_files.append(f)
            valid_embs.append(emb)
            if len(valid_files) >= GALLERY_PER_ID:
                break

        n_valid = len(valid_files)
        if n_valid < GALLERY_PER_ID:
            excluded_counts["fewer_than_5_valid"] += 1
            candidates.append({
                "lfw_name": lfw_name,
                "person_dir": person,
                "n_images": n_images,
                "n_valid": n_valid,
                "valid_files": valid_files,
                "embeddings": valid_embs,
                "name_in_train": False,
                "agree": None,
                "med_top1": None,
                "s_baseline": None,
                "role": "excluded",
                "excluded_reason": "fewer_than_5_valid",
            })
            continue

        n_ge5_valid += 1
        # Check name matching (§4.1)
        name_matched = False
        if used_name_matching and normalize_name(lfw_name) in train_names:
            name_matched = True

        # Check ArcFace similarity (§4.2)
        agree, med_top1 = compute_arcface_stats(valid_embs, centers)

        # Baseline similarity s(c) to owner
        # mean over owner frames of max cosine with candidate's 5 embeddings
        sims_per_owner_frame = []
        c_embs_mat = np.stack(valid_embs)  # (5, 512)
        for o_emb in owner_embs:
            cos_vals = np.dot(c_embs_mat, o_emb)
            sims_per_owner_frame.append(float(np.max(cos_vals)))
        s_baseline = float(np.mean(sims_per_owner_frame))

        # Check exclusion rules
        excluded_reason = None
        if name_matched:
            excluded_reason = "name_in_train"
            excluded_counts["name_in_train"] += 1
        elif agree >= 3 and med_top1 >= calib_T:
            excluded_reason = "arcface_member"
            excluded_counts["arcface_member"] += 1

        candidates.append({
            "lfw_name": lfw_name,
            "person_dir": person,
            "n_images": n_images,
            "n_valid": n_valid,
            "valid_files": valid_files,
            "embeddings": valid_embs,
            "name_in_train": name_matched,
            "agree": agree,
            "med_top1": round(med_top1, 4),
            "s_baseline": round(s_baseline, 4),
            "role": "excluded" if excluded_reason else "eligible",
            "excluded_reason": excluded_reason,
        })

    print(f"[build_gallery] Candidates with >= 5 valid images: {n_ge5_valid}")
    print(f"[build_gallery] Excluded: {excluded_counts}")

    eligible = [c for c in candidates if c["role"] == "eligible"]
    needed_candidates = n_class + (3 if extra_victims else 1)
    if len(eligible) < needed_candidates:
        raise ValueError(
            f"Not enough eligible candidates (found {len(eligible)}, minimum required: {needed_candidates}). "
            f"Rule cannot be relaxed."
        )

    # 9. Victim selection based on s_baseline distribution
    all_s = np.array([c["s_baseline"] for c in eligible])
    median_s = float(np.median(all_s))
    p90_s = float(np.percentile(all_s, 90))
    p10_s = float(np.percentile(all_s, 10))

    # Primary victim: closest to median s
    victim_cand = min(eligible, key=lambda c: abs(c["s_baseline"] - median_s))
    victim_cand["role"] = "victim"
    selected_victims = [victim_cand]

    # Optional victims
    victim_easy_cand = None
    victim_hard_cand = None
    if extra_victims:
        rem_for_easy = [c for c in eligible if c not in selected_victims]
        victim_easy_cand = min(rem_for_easy, key=lambda c: abs(c["s_baseline"] - p90_s))
        victim_easy_cand["role"] = "victim_easy"
        selected_victims.append(victim_easy_cand)

        rem_for_hard = [c for c in eligible if c not in selected_victims]
        victim_hard_cand = min(rem_for_hard, key=lambda c: abs(c["s_baseline"] - p10_s))
        victim_hard_cand["role"] = "victim_hard"
        selected_victims.append(victim_hard_cand)

    # 10. Simulated class selection (LFW_0001 .. LFW_0050)
    pool = [c for c in eligible if c not in selected_victims]
    # Sort deterministically before sampling
    pool.sort(key=lambda c: c["lfw_name"])
    rng_sim = np.random.RandomState(seed)
    chosen_sim_indices = rng_sim.choice(len(pool), size=n_class, replace=False)
    simulated_candidates = [pool[i] for i in sorted(chosen_sim_indices)]

    for c in pool:
        if c in simulated_candidates:
            c["role"] = "simulated"
        else:
            c["role"] = "unselected"

    # 11. Write out gallery directories and files
    # Clean previous LFW directories if replace is True
    if replace:
        for p in gallery_dir.iterdir():
            if p.is_dir() and p.name.startswith("LFW_"):
                shutil.rmtree(p)

    id_to_lfw_name = {}

    def copy_person_files(cand: dict, target_id: str):
        target_dir = gallery_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for i, src_file in enumerate(cand["valid_files"][:GALLERY_PER_ID], start=1):
            dst_file = target_dir / f"{i}.jpg"
            shutil.copy2(src_file, dst_file)
        id_to_lfw_name[target_id] = cand["lfw_name"]

    # Copy primary victim
    copy_person_files(victim_cand, "LFW_victim")

    # Copy extra victims if requested
    if extra_victims:
        if victim_easy_cand:
            copy_person_files(victim_easy_cand, "LFW_victim_easy")
        if victim_hard_cand:
            copy_person_files(victim_hard_cand, "LFW_victim_hard")

    # Copy simulated class
    for idx, cand in enumerate(simulated_candidates, start=1):
        target_id = f"LFW_{idx:04d}"
        copy_person_files(cand, target_id)

    # Calculate percentile rank of victim(s)
    def pct_rank(val: float) -> float:
        return float(np.mean(all_s <= val) * 100.0)

    # 12. Save _build_report.json
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": get_git_commit(),
        "source_paths": {
            "gallery_dir": str(gallery_dir),
            "owner_frames_dir": str(owner_frames_dir),
            "train_ids_dir": str(train_ids_dir),
            "members_dir": str(members_dir),
            "pairs_file": str(pairs_file),
            "lfw_dir": str(lfw_dir),
            "vggface2_meta": str(vggface2_meta) if vggface2_meta else None,
        },
        "parameters": {
            "n_class": n_class,
            "seed": seed,
            "center_tol": center_tol,
            "extra_victims": extra_victims,
        },
        "counts": {
            "n_lfw_considered": n_considered,
            "n_lfw_ge5_valid": n_ge5_valid,
            "n_excluded_by_reason": excluded_counts,
            "n_eligible": len(eligible),
        },
        "arcface_calibration": {
            "calib_T": round(calib_T, 4),
            "calib_auc": round(calib_auc, 4),
            "members_calib_samples": 200,
            "pairs_calib_samples": 200,
            "heuristic_note": (
                "Calibration was computed on 64x64 domain. "
                "LFW crops pass through backend crop and enhancement, so distribution may shift."
            ),
        },
        "name_matching": {
            "used": used_name_matching,
            "meta_path": str(vggface2_meta) if used_name_matching else None,
        },
        "victim": {
            "id": "LFW_victim",
            "lfw_name": victim_cand["lfw_name"],
            "s_baseline": victim_cand["s_baseline"],
            "percentile": round(pct_rank(victim_cand["s_baseline"]), 2),
            "median_s_baseline": round(median_s, 4),
        },
        "id_to_lfw_name": id_to_lfw_name,
    }

    if extra_victims and victim_easy_cand and victim_hard_cand:
        report_data["victim_easy"] = {
            "id": "LFW_victim_easy",
            "lfw_name": victim_easy_cand["lfw_name"],
            "s_baseline": victim_easy_cand["s_baseline"],
            "percentile": round(pct_rank(victim_easy_cand["s_baseline"]), 2),
            "target_p90_s_baseline": round(p90_s, 4),
        }
        report_data["victim_hard"] = {
            "id": "LFW_victim_hard",
            "lfw_name": victim_hard_cand["lfw_name"],
            "s_baseline": victim_hard_cand["s_baseline"],
            "percentile": round(pct_rank(victim_hard_cand["s_baseline"]), 2),
            "target_p10_s_baseline": round(p10_s, 4),
        }

    report_path = gallery_dir / "_build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[build_gallery] Saved report to {report_path}")

    # 13. Save _candidates.csv
    csv_path = gallery_dir / "_candidates.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lfw_name", "n_images", "n_valid", "name_in_train",
            "agree", "med_top1", "s_baseline", "role", "excluded_reason"
        ])
        for c in candidates:
            writer.writerow([
                c["lfw_name"],
                c["n_images"],
                c["n_valid"],
                c["name_in_train"],
                c["agree"] if c["agree"] is not None else "",
                c["med_top1"] if c["med_top1"] is not None else "",
                c["s_baseline"] if c["s_baseline"] is not None else "",
                c["role"],
                c["excluded_reason"] or "",
            ])
    print(f"[build_gallery] Saved candidate list to {csv_path}")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Build L1 attack gallery from LFW dataset.")
    parser.add_argument("--gallery", default="data/innovation/gallery", help="Target gallery directory")
    parser.add_argument("--owner-frames", default="data/innovation/frames/owner/s1_day1", help="Owner unglasses frames (s1_day1 only)")
    parser.add_argument("--train-ids", default="data/dataset_original/classification_data/train_data", help="Train data directory")
    parser.add_argument("--members-calib", default="data/dataset_original/classification_data/test_data", help="Test data directory for member calibration")
    parser.add_argument("--pairs", default="data/dataset_original/verification_pairs_val.txt", help="Verification pairs file for non-member calibration")
    parser.add_argument("--n-class", type=int, default=50, help="Number of simulated classes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--extra-victims", action="store_true", help="Generate LFW_victim_easy and LFW_victim_hard")
    parser.add_argument("--vggface2-meta", default=None, help="Path to identity_meta.csv")
    parser.add_argument("--lfw-root", default=None, help="Root folder of LFW dataset")
    parser.add_argument("--center-tol", type=float, default=0.20, help="BBox center tolerance relative to image size")
    parser.add_argument("--checkpoint", default="checkpoints/method3_best.pth", help="FaceRecognizer checkpoint path")
    parser.add_argument("--replace", action="store_true", help="Replace existing LFW gallery folders")

    args = parser.parse_args()

    build_gallery(
        gallery_dir=Path(args.gallery),
        owner_frames_dir=Path(args.owner_frames),
        train_ids_dir=Path(args.train_ids),
        members_dir=Path(args.members_calib),
        pairs_file=Path(args.pairs),
        n_class=args.n_class,
        seed=args.seed,
        extra_victims=args.extra_victims,
        vggface2_meta=Path(args.vggface2_meta) if args.vggface2_meta else None,
        lfw_root=Path(args.lfw_root) if args.lfw_root else None,
        center_tol=args.center_tol,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        replace=args.replace,
    )


if __name__ == "__main__":
    main()
