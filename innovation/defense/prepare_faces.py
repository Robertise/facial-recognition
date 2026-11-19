"""
prepare_faces.py - Build crop and landmark shards from labeled face datasets (e.g. VGGFace2 test).

Reads images from --images-root (<root>/<class_id>/<img>.jpg), checks --meta (Class_ID, Name, ...),
excludes any identity whose normalized name matches LFW gallery identities or LFW validation identities,
takes up to --per-id images per identity, runs detection + enhance + FaceMesh landmarks, and writes
matching 1000-sample shards to <out_dir>/crops/train_XXX.npz and <out_dir>/landmarks/train_XXX.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp_module
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from innovation.attacks.adv_glasses import faces
from innovation.defense.band import band_in_crop, load_band_template
from innovation.eval.runs import write_json

DEFAULT_IMAGES_ROOT = Path("data/vggface2/test")
DEFAULT_META = Path("data/vggface2/identity_meta.csv")
DEFAULT_GALLERY = Path("data/innovation/gallery")
DEFAULT_LFW_CROPS = Path("data/innovation/defense/lfw_crops")
DEFAULT_OUT_DIR = Path("data/innovation/defense/vgg_test")


def normalize_name(s: str) -> str:
    """Normalize identity name: strip quotes, lowercase, replace spaces/punctuation with underscores."""
    s = s.strip().strip('"\'').strip()
    s = s.replace(" ", "_").lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def get_excluded_lfw_names(
    gallery_dir: Path = DEFAULT_GALLERY,
    lfw_crops_dir: Path = DEFAULT_LFW_CROPS,
) -> tuple[set[str], dict[str, set[str]]]:
    """Collect and normalize LFW identity names in gallery, LFW val, and LFW train shards."""
    by_source: dict[str, set[str]] = {
        "gallery": set(),
        "lfw_val": set(),
        "lfw_train": set(),
    }

    # (a) Gallery LFW identities
    g_dir = Path(gallery_dir)
    if g_dir.exists():
        from innovation.defense import prepare
        try:
            _, gallery_lfw = prepare.resolve_gallery_lfw_names(g_dir)
            for name in gallery_lfw:
                norm = normalize_name(name)
                if norm:
                    by_source["gallery"].add(norm)
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"Failed to resolve gallery LFW names from {g_dir}: {e}")

    # (b) LFW val identities
    c_dir = Path(lfw_crops_dir)
    if c_dir.exists():
        for sf in sorted(c_dir.glob("val_*.npz")):
            try:
                data = np.load(sf, allow_pickle=True)
                if "identities" in data:
                    for ident in data["identities"]:
                        norm = normalize_name(str(ident))
                        if norm:
                            by_source["lfw_val"].add(norm)
            except Exception:
                pass

    # (c) LFW train identities
    if c_dir.exists():
        for sf in sorted(c_dir.glob("train_*.npz")):
            try:
                data = np.load(sf, allow_pickle=True)
                if "identities" in data:
                    for ident in data["identities"]:
                        norm = normalize_name(str(ident))
                        if norm:
                            by_source["lfw_train"].add(norm)
            except Exception:
                pass

    all_excluded = by_source["gallery"] | by_source["lfw_val"] | by_source["lfw_train"]
    return all_excluded, by_source


# Global variables for worker processes
_WORKER_DETECTOR: Optional[faces.FaceDetector] = None
_WORKER_MESH_DETECTOR: Optional[Any] = None
_WORKER_TMPL: Optional[np.ndarray] = None


def _init_worker() -> None:
    global _WORKER_DETECTOR, _WORKER_MESH_DETECTOR, _WORKER_TMPL
    import mediapipe as mp
    _WORKER_DETECTOR = faces.FaceDetector()
    _WORKER_MESH_DETECTOR = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    _WORKER_TMPL = load_band_template()


def _process_one_image(item: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str, str]:
    """
    Process single image: detect -> crop -> enhance -> resize 112 -> FaceMesh -> band_in_crop.
    item: (abs_img_path, class_id, rel_path)
    Returns: (crop112, eyes, window, mesh_pts, ok, rel_path, class_id)
    """
    global _WORKER_DETECTOR, _WORKER_MESH_DETECTOR, _WORKER_TMPL
    if _WORKER_DETECTOR is None:
        _init_worker()

    img_path, class_id, rel_path = item
    img = cv2.imread(img_path)

    crop112 = np.zeros((112, 112, 3), dtype=np.uint8)
    eyes = np.zeros((2, 2), dtype=np.float32)
    window = np.zeros((4,), dtype=np.int32)
    mesh_pts = np.zeros((468, 2), dtype=np.float32)
    ok = 0

    if img is None or img.size == 0:
        return crop112, eyes, window, mesh_pts, ok, rel_path, class_id

    h, w = img.shape[:2]
    bbox = _WORKER_DETECTOR.detect(img)
    if bbox is None:
        return crop112, eyes, window, mesh_pts, ok, rel_path, class_id

    raw_crop = faces.crop(img, bbox)
    if raw_crop.size == 0:
        return crop112, eyes, window, mesh_pts, ok, rel_path, class_id

    face_enh, _ = faces.enhance(raw_crop)
    if face_enh.size == 0:
        return crop112, eyes, window, mesh_pts, ok, rel_path, class_id

    crop112 = cv2.resize(face_enh, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    win_tuple = faces.crop_window(bbox, img.shape)
    window = np.array(win_tuple, dtype=np.int32)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _WORKER_MESH_DETECTOR.process(img_rgb)
    if not res.multi_face_landmarks:
        return crop112, eyes, window, mesh_pts, ok, rel_path, class_id

    lm = res.multi_face_landmarks[0].landmark
    mesh_pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
    pt_l = np.mean([[lm[idx].x * w, lm[idx].y * h] for idx in faces.LEFT_EYE], axis=0)
    pt_r = np.mean([[lm[idx].x * w, lm[idx].y * h] for idx in faces.RIGHT_EYE], axis=0)
    eyes = np.array([pt_l, pt_r], dtype=np.float32)

    b_comp = band_in_crop(img, bbox, eyes, band_template=_WORKER_TMPL, size=112)
    if b_comp is not None:
        ok = 1

    return crop112, eyes, window, mesh_pts, ok, rel_path, class_id


def prepare_faces_dataset(
    images_root: Path = DEFAULT_IMAGES_ROOT,
    meta_path: Optional[Path] = DEFAULT_META,
    gallery_dir: Path = DEFAULT_GALLERY,
    lfw_crops_dir: Path = DEFAULT_LFW_CROPS,
    per_id: int = 60,
    out_dir: Path = DEFAULT_OUT_DIR,
    workers: int = 0,
    shard_size: int = 1000,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    images_root = Path(images_root)
    out_dir = Path(out_dir)
    crops_out = out_dir / "crops"
    landmarks_out = out_dir / "landmarks"
    crops_out.mkdir(parents=True, exist_ok=True)
    landmarks_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Resolve excluded names
    all_excluded, excluded_by_src_sets = get_excluded_lfw_names(gallery_dir, lfw_crops_dir)

    # 2. Parse metadata CSV if available
    meta_name_by_id: dict[str, str] = {}
    excluded_identities_report: list[dict[str, Any]] = []

    if meta_path is not None:
        p_meta = Path(meta_path)
        if not p_meta.exists():
            raise SystemExit(f"Metadata file not found: {p_meta}")
        with open(p_meta, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cleaned_row = {k.strip(): (v.strip() if v else "") for k, v in row.items() if k is not None}
                cid = cleaned_row.get("Class_ID", "")
                name = cleaned_row.get("Name", "")
                if not cid:
                    continue
                norm = normalize_name(name) if name else ""
                meta_name_by_id[cid] = name
                if norm and norm in all_excluded:
                    matched_srcs = [src for src in ("gallery", "lfw_val", "lfw_train") if norm in excluded_by_src_sets[src]]
                    excluded_identities_report.append({
                        "class_id": cid,
                        "name": name,
                        "normalized": norm,
                        "sources": matched_srcs,
                        "reason": "matches_lfw",
                    })

    excluded_class_ids = {x["class_id"] for x in excluded_identities_report}

    # 3. Discover identity folders in images_root
    if not images_root.exists():
        raise SystemExit(f"Images root directory not found: {images_root}")

    all_class_dirs = sorted(p for p in images_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not all_class_dirs:
        raise SystemExit(f"No identity directories found in images root: {images_root}")
    usable_class_dirs = [p for p in all_class_dirs if p.name not in excluded_class_ids]

    tasks: list[tuple[str, str, str]] = []
    identities_used: set[str] = set()

    for cdir in usable_class_dirs:
        cid = cdir.name
        img_files = faces.list_images(cdir)
        img_files = sorted(img_files, key=lambda p: p.name)[:per_id]
        if not img_files:
            continue
        identities_used.add(cid)
        for img_p in img_files:
            try:
                rel = img_p.relative_to(images_root).as_posix()
            except ValueError:
                rel = img_p.name
            tasks.append((str(img_p), cid, rel))
            if limit is not None and len(tasks) >= limit:
                break
        if limit is not None and len(tasks) >= limit:
            break

    total_images = len(tasks)
    print(
        f"[prepare_faces] Found {len(all_class_dirs)} identity dirs, "
        f"excluded {len(all_class_dirs) - len(usable_class_dirs)} (matching LFW), "
        f"processing {len(tasks)} images with per_id<={per_id} (workers={workers})...",
        flush=True,
    )

    # 4. Process images (single process or multiprocessing)
    shard_names: list[str] = []
    shard_idx = 0
    total_ok = 0

    curr_crops: list[np.ndarray] = []
    curr_eyes: list[np.ndarray] = []
    curr_windows: list[np.ndarray] = []
    curr_meshes: list[np.ndarray] = []
    curr_oks: list[int] = []
    curr_paths: list[str] = []
    curr_ids: list[str] = []

    def _flush_shard() -> None:
        nonlocal shard_idx, total_ok
        if not curr_crops:
            return
        t_shard = time.time()
        sname = f"train_{shard_idx:03d}.npz"
        c_path = crops_out / sname
        l_path = landmarks_out / sname

        np.savez_compressed(
            c_path,
            crops=np.stack(curr_crops, axis=0).astype(np.uint8),
            identities=np.array(curr_ids),
            paths=np.array(curr_paths),
        )
        np.savez_compressed(
            l_path,
            eyes=np.stack(curr_eyes, axis=0).astype(np.float32),
            window=np.stack(curr_windows, axis=0).astype(np.int32),
            mesh=np.stack(curr_meshes, axis=0).astype(np.float32),
            ok=np.array(curr_oks, dtype=np.uint8),
            paths=np.array(curr_paths),
        )
        shard_names.append(sname)
        n_ok = sum(curr_oks)
        total_ok += n_ok
        print(
            f"Saved shard {sname}: total={len(curr_crops)}, ok={n_ok}, "
            f"elapsed={time.time() - t_shard:.2f}s",
            flush=True,
        )
        shard_idx += 1
        curr_crops.clear()
        curr_eyes.clear()
        curr_windows.clear()
        curr_meshes.clear()
        curr_oks.clear()
        curr_paths.clear()
        curr_ids.clear()

    if workers > 1:
        ctx = mp_module.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
            for res in pool.imap(_process_one_image, tasks, chunksize=16):
                crop112, eyes, window, mesh_pts, ok, rel_path, cid = res
                curr_crops.append(crop112)
                curr_eyes.append(eyes)
                curr_windows.append(window)
                curr_meshes.append(mesh_pts)
                curr_oks.append(ok)
                curr_paths.append(rel_path)
                curr_ids.append(cid)
                if len(curr_crops) >= shard_size:
                    _flush_shard()
    else:
        _init_worker()
        for task_item in tasks:
            crop112, eyes, window, mesh_pts, ok, rel_path, cid = _process_one_image(task_item)
            curr_crops.append(crop112)
            curr_eyes.append(eyes)
            curr_windows.append(window)
            curr_meshes.append(mesh_pts)
            curr_oks.append(ok)
            curr_paths.append(rel_path)
            curr_ids.append(cid)
            if len(curr_crops) >= shard_size:
                _flush_shard()

    _flush_shard()

    excluded_by_source_counts = {
        "gallery": sum(1 for x in excluded_identities_report if "gallery" in x.get("sources", [])),
        "lfw_val": sum(1 for x in excluded_identities_report if "lfw_val" in x.get("sources", [])),
        "lfw_train": sum(1 for x in excluded_identities_report if "lfw_train" in x.get("sources", [])),
    }

    report = {
        "images_root": str(images_root),
        "meta_file": str(meta_path) if meta_path else None,
        "out_dir": str(out_dir),
        "per_id": int(per_id),
        "shard_size": int(shard_size),
        "identities_used": len(identities_used),
        "identities_excluded": len(excluded_identities_report),
        "excluded_by_source": excluded_by_source_counts,
        "excluded_list": excluded_identities_report,
        "total_images": total_images,
        "total_ok": total_ok,
        "time_seconds": round(time.time() - t0, 2),
        "shards": shard_names,
    }
    write_json(out_dir / "report.json", report)
    print(
        f"[prepare_faces] Finished: {len(identities_used)} identities used, "
        f"{len(excluded_identities_report)} excluded, {total_images} images, "
        f"{total_ok} ok, in {time.time() - t0:.1f}s. Report -> {out_dir / 'report.json'}",
        flush=True,
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build crop + landmark shards from labeled face datasets.")
    ap.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT), help="Path to images directory (<root>/<class_id>/<img>.jpg)")
    ap.add_argument("--meta", default=str(DEFAULT_META), help="Path to identity_meta.csv")
    ap.add_argument("--gallery-dir", default=str(DEFAULT_GALLERY), help="Path to gallery directory")
    ap.add_argument("--lfw-crops-dir", default=str(DEFAULT_LFW_CROPS), help="Path to lfw_crops directory")
    ap.add_argument("--per-id", type=int, default=60, help="Max images per identity")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for crops and landmarks")
    ap.add_argument("--workers", type=int, default=0, help="Multiprocessing workers (0=single-process)")
    ap.add_argument("--shard-size", type=int, default=1000, help="Shard size (default: 1000)")
    ap.add_argument("--limit", type=int, default=None, help="Debug limit on total images to process")
    args = ap.parse_args()

    prepare_faces_dataset(
        images_root=Path(args.images_root),
        meta_path=Path(args.meta) if args.meta else None,
        gallery_dir=Path(args.gallery_dir),
        lfw_crops_dir=Path(args.lfw_crops_dir),
        per_id=args.per_id,
        out_dir=Path(args.out_dir),
        workers=args.workers,
        shard_size=args.shard_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
