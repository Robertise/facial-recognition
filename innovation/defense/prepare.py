"""
prepare.py - Prepare LFW face crops and eyeglass band masks for training the Purifier U-Net.

1. Scans `data/public/lfw/lfw_home/lfw_funneled`.
2. Excludes every LFW identity used in `data/innovation/gallery/` (mapped via `_build_report.json`
   and `_candidates.csv`; raises SystemExit if any gallery LFW identity cannot be mapped).
3. Splits remaining identities 90% train / 10% val with seed 0 -> writes `split_ids.json`.
4. Runs `FaceDetector` -> `faces.crop` -> `faces.enhance` -> resize 112x112 + `EyeLandmarks` -> `band_in_crop`.
5. Saves `.npz` shards (`crops` uint8 Nx112x112x3, `bands` uint8 Nx112x112) and `prepare_report.json`
   into `data/innovation/defense/lfw_crops/`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from innovation.attacks.adv_glasses import faces
from innovation.defense.band import band_in_crop, load_band_template
from innovation.eval.runs import write_json

DEFAULT_LFW_DIR = Path("data/public/lfw/lfw_home/lfw_funneled")
DEFAULT_GALLERY_DIR = Path("data/innovation/gallery")
DEFAULT_OUT_DIR = Path("data/innovation/defense/lfw_crops")


def resolve_gallery_lfw_names(gallery_dir: Path, lfw_dir: Optional[Path] = None) -> tuple[dict[str, str], set[str]]:
    """Map every non-owner identity folder in `gallery_dir` to its original LFW identity name.
    Raises SystemExit if any gallery identity cannot be mapped.
    Returns (id_to_lfw_name_dict, all_excluded_lfw_names_set).
    """
    gallery_dir = Path(gallery_dir)
    if not gallery_dir.exists():
        raise SystemExit(f"Gallery directory does not exist: {gallery_dir}")

    gallery_folders = sorted(
        p.name for p in gallery_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name != "owner"
    )

    known_map: dict[str, str] = {}
    extra_excluded: set[str] = set()

    # 1. Read _build_report.json if present
    report_path = gallery_dir / "_build_report.json"
    if report_path.exists():
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            id_map = rep.get("id_to_lfw_name") or {}
            for k, v in id_map.items():
                if k and v:
                    known_map[str(k)] = str(v)
                    extra_excluded.add(str(v))
            for role_key in ("victim", "victim_easy", "victim_hard"):
                role_info = rep.get(role_key)
                if isinstance(role_info, dict) and role_info.get("id") and role_info.get("lfw_name"):
                    known_map[str(role_info["id"])] = str(role_info["lfw_name"])
                    extra_excluded.add(str(role_info["lfw_name"]))
        except Exception as e:
            raise SystemExit(f"Failed to read {report_path}: {e}") from e

    # 2. Read _candidates.csv if present
    cand_path = gallery_dir / "_candidates.csv"
    if cand_path.exists():
        try:
            sim_idx = 0
            with open(cand_path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    lfw_name = (row.get("lfw_name") or "").strip()
                    role = (row.get("role") or "").strip()
                    if not lfw_name or not role or role in ("excluded", "unselected", "eligible_unused"):
                        continue
                    extra_excluded.add(lfw_name)
                    if role in ("victim", "victim_easy", "victim_hard"):
                        gid = f"LFW_{role}"
                        known_map.setdefault(gid, lfw_name)
                    elif role == "simulated":
                        sim_idx += 1
                        known_map.setdefault(f"LFW_{sim_idx:04d}", lfw_name)
                    elif role.startswith("LFW_"):
                        known_map.setdefault(role, lfw_name)
        except Exception as e:
            raise SystemExit(f"Failed to read {cand_path}: {e}") from e

    # 3. Map every gallery folder
    lfw_existing = set()
    if lfw_dir is not None and Path(lfw_dir).exists():
        lfw_existing = {p.name for p in Path(lfw_dir).iterdir() if p.is_dir() and not p.name.startswith("_")}

    resolved: dict[str, str] = {}
    unmapped: list[str] = []
    for folder in gallery_folders:
        if folder in known_map:
            resolved[folder] = known_map[folder]
        elif folder in lfw_existing:
            resolved[folder] = folder
        else:
            unmapped.append(folder)

    if unmapped:
        raise SystemExit(
            f"Could not map gallery identity folder(s) to original LFW names: {unmapped}"
        )
    if not resolved and not extra_excluded:
        raise SystemExit(
            f"No LFW gallery identities could be mapped from {gallery_dir}"
        )

    excluded = set(resolved.values()) | extra_excluded
    return resolved, excluded


def split_lfw_identities(
    lfw_dir: Path = DEFAULT_LFW_DIR,
    gallery_dir: Path = DEFAULT_GALLERY_DIR,
    seed: int = 0,
) -> dict[str, Any]:
    """Exclude gallery LFW identities and split remaining LFW identities 90% train / 10% val by identity."""
    lfw_dir = Path(lfw_dir)
    if not lfw_dir.exists():
        raise SystemExit(f"LFW directory not found: {lfw_dir}")

    id_to_lfw, excluded_set = resolve_gallery_lfw_names(gallery_dir, lfw_dir=lfw_dir)

    all_ids = sorted(
        p.name for p in lfw_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    if not all_ids:
        raise SystemExit(f"No identity folders found in LFW directory: {lfw_dir}")

    eligible_ids = [i for i in all_ids if i not in excluded_set]
    if len(eligible_ids) < 2:
        raise SystemExit(f"Need at least 2 eligible LFW identities after gallery exclusion, got {len(eligible_ids)}")

    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(eligible_ids))
    n_train = int(round(len(perm) * 0.9))
    n_train = min(max(1, n_train), len(perm) - 1)

    train_ids = sorted(str(x) for x in perm[:n_train])
    val_ids = sorted(str(x) for x in perm[n_train:])

    excluded_present = sorted(set(all_ids) & excluded_set)
    return {
        "seed": int(seed),
        "total_lfw_identities": len(all_ids),
        "excluded_gallery_identities": sorted(excluded_set),
        "excluded_present_in_lfw": excluded_present,
        "id_to_lfw_name": id_to_lfw,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }


def _save_shards(
    split_name: str,
    crops: list[np.ndarray],
    bands: list[np.ndarray],
    identities: list[str],
    paths: list[str],
    out_dir: Path,
    shard_size: int,
) -> list[str]:
    if not crops:
        return []
    shard_files: list[str] = []
    n = len(crops)
    for idx, start in enumerate(range(0, n, shard_size)):
        end = min(start + shard_size, n)
        fname = f"{split_name}_{idx:03d}.npz"
        fpath = out_dir / fname
        np.savez_compressed(
            fpath,
            crops=np.stack(crops[start:end], axis=0).astype(np.uint8),
            bands=np.stack(bands[start:end], axis=0).astype(np.uint8),
            identities=np.array(identities[start:end]),
            paths=np.array(paths[start:end]),
        )
        shard_files.append(fname)
    return shard_files


def prepare_lfw_dataset(
    lfw_dir: Path = DEFAULT_LFW_DIR,
    gallery_dir: Path = DEFAULT_GALLERY_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 0,
    shard_size: int = 1000,
    limit: Optional[int] = None,
    detector: Optional[Any] = None,
    eyes_finder: Optional[Any] = None,
    band_template: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Process LFW images into 112x112 crops + band masks and write train/val shards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove any old shard files in out_dir so re-runs are clean
    for old_npz in out_dir.glob("*.npz"):
        old_npz.unlink()

    split_info = split_lfw_identities(lfw_dir=lfw_dir, gallery_dir=gallery_dir, seed=seed)
    write_json(out_dir / "split_ids.json", split_info)

    det = detector if detector is not None else faces.FaceDetector()
    eyes_fn = eyes_finder if eyes_finder is not None else faces.EyeLandmarks()
    tmpl = band_template if band_template is not None else load_band_template()

    if limit is not None and limit > 0:
        limit_val = max(1, int(round(limit * 0.1))) if limit >= 2 else 0
        limit_train = max(1, limit - limit_val)
    else:
        limit_train = None
        limit_val = None

    stats = {
        "images_seen": 0,
        "skipped_unreadable": 0,
        "skipped_no_face": 0,
        "skipped_no_eyes": 0,
        "enhancer_applied": 0,
        "used_train": 0,
        "used_val": 0,
    }

    def collect_split(id_list: list[str], max_images: Optional[int]) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
        s_crops: list[np.ndarray] = []
        s_bands: list[np.ndarray] = []
        s_ids: list[str] = []
        s_paths: list[str] = []
        seen_in_split = 0

        for person_id in id_list:
            person_dir = Path(lfw_dir) / person_id
            if not person_dir.exists():
                continue
            for img_path in faces.list_images(person_dir):
                if max_images is not None and seen_in_split >= max_images:
                    return s_crops, s_bands, s_ids, s_paths
                seen_in_split += 1
                stats["images_seen"] += 1

                img = cv2.imread(str(img_path))
                if img is None:
                    stats["skipped_unreadable"] += 1
                    continue
                bbox = det.detect(img)
                if bbox is None:
                    stats["skipped_no_face"] += 1
                    continue
                eyes = eyes_fn.eyes(img)
                if eyes is None:
                    stats["skipped_no_eyes"] += 1
                    continue
                raw_crop = faces.crop(img, bbox)
                if raw_crop.size == 0:
                    stats["skipped_no_face"] += 1
                    continue
                face, meta = faces.enhance(raw_crop)
                if face.size == 0:
                    stats["skipped_no_face"] += 1
                    continue
                if meta and meta.get("applied"):
                    stats["enhancer_applied"] += 1

                band112 = band_in_crop(img, bbox, eyes, band_template=tmpl, size=112)
                if band112 is None:
                    stats["skipped_no_eyes"] += 1
                    continue

                crop112 = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                s_crops.append(crop112)
                s_bands.append(band112.astype(np.uint8))
                s_ids.append(person_id)
                try:
                    rel = img_path.relative_to(lfw_dir).as_posix()
                except ValueError:
                    rel = str(img_path)
                s_paths.append(rel)

        return s_crops, s_bands, s_ids, s_paths

    tr_crops, tr_bands, tr_ids, tr_paths = collect_split(split_info["train_ids"], limit_train)
    va_crops, va_bands, va_ids, va_paths = collect_split(split_info["val_ids"], limit_val)

    stats["used_train"] = len(tr_crops)
    stats["used_val"] = len(va_crops)

    train_shards = _save_shards("train", tr_crops, tr_bands, tr_ids, tr_paths, out_dir, shard_size)
    val_shards = _save_shards("val", va_crops, va_bands, va_ids, va_paths, out_dir, shard_size)

    report = {
        "lfw_dir": str(lfw_dir),
        "gallery_dir": str(gallery_dir),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "limit": limit,
        "shard_size": int(shard_size),
        "total_lfw_identities": split_info["total_lfw_identities"],
        "n_excluded_gallery_identities": len(split_info["excluded_gallery_identities"]),
        "n_train_identities": len(split_info["train_ids"]),
        "n_val_identities": len(split_info["val_ids"]),
        **stats,
        "train_shards": train_shards,
        "val_shards": val_shards,
    }
    write_json(out_dir / "prepare_report.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare LFW crops and band masks for Purifier training.")
    ap.add_argument("--lfw-dir", default=str(DEFAULT_LFW_DIR))
    ap.add_argument("--gallery-dir", default=str(DEFAULT_GALLERY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="Maximum total images to inspect (for quick testing)")
    args = ap.parse_args()

    rep = prepare_lfw_dataset(
        lfw_dir=Path(args.lfw_dir),
        gallery_dir=Path(args.gallery_dir),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        shard_size=args.shard_size,
        limit=args.limit,
    )
    print(
        f"Prepared LFW crops in {rep['out_dir']}: "
        f"seen={rep['images_seen']}  train={rep['used_train']}  val={rep['used_val']}  "
        f"no_face={rep['skipped_no_face']}  no_eyes={rep['skipped_no_eyes']}  "
        f"excluded_ids={rep['n_excluded_gallery_identities']}"
    )


if __name__ == "__main__":
    main()
