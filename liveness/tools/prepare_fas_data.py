# -*- coding: utf-8 -*-
"""
liveness/tools/prepare_fas_data.py
==================================
Prepares CelebA-Spoof dataset from Hugging Face for Face Anti-Spoofing training.

1. Downloads specified train shards and test shards from Hugging Face via huggingface_hub.
   Avoids re-downloading files already present in the HF cache (try_to_load_from_cache).
2. Verifies schema and label mappings (0 <-> live, 1 <-> spoof).
3. Inspects cropped_image["path"] in Step 0 to determine validation strategy:
   - Branch 1 (subject-based split) if subject IDs can be parsed.
   - Branch 2 (held-out shards split) if subject IDs cannot be parsed.
4. Pre-allocates NumPy arrays to minimize RAM overhead during ingestion.
5. Calibrates MediaPipe crop parameters (scale_w, scale_h, shift_x, shift_y) on live faces.
6. Saves train.npz, val.npz, test.npz, meta.json, calib_hist.png, and samples.png.

Usage:
    python -m liveness.tools.prepare_fas_data --out-dir ~/fas_data
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
import random
from pathlib import Path

# Use headless Agg backend for matplotlib (avoids errors in subprocess environments like Kaggle/vast.ai)
os.environ["MPLBACKEND"] = "Agg"

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liveness.fas_common import resize_face

TRAIN_REPO = "nguyenkhoa/celeba-spoof-for-face-antispoofing"
TEST_REPO = "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"

SUBJECT_RE = re.compile(r"(?:^|/)(?P<subj>[^/]+)/(?:live|spoof)/", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CelebA-Spoof dataset for FAS training")
    parser.add_argument("--out-dir", type=str, default=str(Path.home() / "fas_data"),
                        help="Output directory (outside repo)")
    parser.add_argument("--cache-dir", type=str, default=str(Path.home() / ".cache" / "fas_hf"),
                        help="Hugging Face download cache directory")
    parser.add_argument("--n-train-shards", type=int, default=46,
                        help="Number of train shards to download and process (out of 93, default 46)")
    parser.add_argument("--shards", type=str, default="",
                        help="Explicit list of train shard indices (e.g. '0,1,2' or '0-5')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shard selection, sampling, and splits")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of subjects (Branch 1) reserved for validation")
    parser.add_argument("--min-side", type=int, default=40,
                        help="Minimum width/height for raw face image to be accepted")
    parser.add_argument("--max-rows-per-shard", type=int, default=0,
                        help="Maximum rows to read per shard (0 = unlimited, for smoke testing)")
    parser.add_argument("--calib-n", type=int, default=2000,
                        help="Number of live train images to use for crop calibration")
    parser.add_argument("--delete-after-processing", action="store_true", default=False,
                        help="Delete parquet files and blobs of train/val shards after processing to save disk space")
    return parser.parse_args()


def parse_shards_arg(shards_str: str) -> list[int]:
    """Parse comma/dash separated shard string into sorted integer list."""
    if not shards_str.strip():
        return []
    indices = set()
    parts = [p.strip() for p in shards_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            for i in range(int(start), int(end) + 1):
                indices.add(i)
        else:
            indices.add(int(part))
    return sorted(indices)


def extract_raw_bytes(item) -> bytes | None:
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
        return bytes(item["bytes"])
    return None


def fetch_file(repo_id: str, filename: str, cache_dir: Path, hf_token: str | None, stats: dict) -> Path:
    """Load file from cache if already present; otherwise download from HF."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
        repo_type="dataset",
    )
    if cached is not None and isinstance(cached, (str, Path)) and os.path.exists(cached):
        print(f"  [cache] {filename}")
        stats["cached_files"] += 1
        return Path(cached)

    print(f"  [download] {filename}...")
    local_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        token=hf_token,
    )
    stats["downloaded_files"] += 1
    try:
        sz = os.path.getsize(local_file)
        stats["downloaded_bytes"] += sz
    except Exception:
        pass
    return Path(local_file)


def cleanup_shard(local_file: Path, stats: dict):
    """Delete parquet file and underlying blob in cache to free disk space."""
    try:
        real_blob = Path(os.path.realpath(local_file))
        symlink_file = Path(local_file)
        sz = 0
        if real_blob.exists():
            sz = real_blob.stat().st_size
            real_blob.unlink()
        if symlink_file.exists() and symlink_file != real_blob:
            symlink_file.unlink()
        stats["freed_bytes"] += sz
        print(f"  [delete] Freed shard {local_file.name} ({sz / (1024**3):.2f} GB, total freed: {stats['freed_bytes'] / (1024**3):.2f} GB)")
    except Exception as e:
        print(f"  [WARNING] Failed to delete {local_file}: {e}")


def calibrate_crop(calib_images: list[np.ndarray], out_dir: Path) -> dict[str, float]:
    """
    Calibrate MediaPipe crop scale and shift parameters on live face images.
    Pad 50% replicate -> MediaPipe face detection -> convert to original coordinates.
    """
    import mediapipe as mp
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    scale_w_list = []
    scale_h_list = []
    shift_x_list = []
    shift_y_list = []

    detected_count = 0
    total_images = len(calib_images)

    print(f"\n[Calibrate] Running MediaPipe crop calibration on {total_images} live face images...")

    for img_bgr in calib_images:
        orig_h, orig_w = img_bgr.shape[:2]
        pad_top = int(round(orig_h * 0.5))
        pad_bottom = int(round(orig_h * 0.5))
        pad_left = int(round(orig_w * 0.5))
        pad_right = int(round(orig_w * 0.5))

        padded_bgr = cv2.copyMakeBorder(
            img_bgr,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        padded_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
        pad_h, pad_w = padded_rgb.shape[:2]

        results = mp_fd.process(padded_rgb)
        if not results.detections:
            continue

        best_detection = max(results.detections, key=lambda d: d.score[0] if d.score else 0.0)
        bbox_rel = best_detection.location_data.relative_bounding_box

        det_x = bbox_rel.xmin * pad_w
        det_y = bbox_rel.ymin * pad_h
        det_w = bbox_rel.width * pad_w
        det_h = bbox_rel.height * pad_h

        box_x = det_x - pad_left
        box_y = det_y - pad_top
        box_w = det_w
        box_h = det_h

        if box_w <= 0 or box_h <= 0:
            continue

        detected_count += 1

        crop_cx = orig_w / 2.0
        crop_cy = orig_h / 2.0
        box_cx = box_x + box_w / 2.0
        box_cy = box_y + box_h / 2.0

        scale_w_list.append(orig_w / float(box_w))
        scale_h_list.append(orig_h / float(box_h))
        shift_x_list.append((crop_cx - box_cx) / float(box_w))
        shift_y_list.append((crop_cy - box_cy) / float(box_h))

    mp_fd.close()

    det_rate = (detected_count / total_images) if total_images > 0 else 0.0
    print(f"[Calibrate] Detection success rate: {det_rate * 100:.2f}% ({detected_count}/{total_images})")

    if det_rate < 0.50:
        print(f"[ERROR] Crop calibration detection rate is below 50% ({det_rate * 100:.2f}% < 50%)! Aborting.")
        sys.exit(1)

    med_scale_w = float(np.clip(np.median(scale_w_list), 0.5, 3.0))
    med_scale_h = float(np.clip(np.median(scale_h_list), 0.5, 3.0))
    med_shift_x = float(np.clip(np.median(shift_x_list), -0.5, 0.5))
    med_shift_y = float(np.clip(np.median(shift_y_list), -0.5, 0.5))

    crop_cfg = {
        "scale_w": round(med_scale_w, 2),
        "scale_h": round(med_scale_h, 2),
        "shift_x": round(med_shift_x, 2),
        "shift_y": round(med_shift_y, 2),
    }

    calib_stats = {
        "detected_count": detected_count,
        "total_images": total_images,
        "detection_rate": round(det_rate, 4),
        "scale_w_p10_p50_p90": [round(float(p), 3) for p in np.percentile(scale_w_list, [10, 50, 90])],
        "scale_h_p10_p50_p90": [round(float(p), 3) for p in np.percentile(scale_h_list, [10, 50, 90])],
        "shift_x_p10_p50_p90": [round(float(p), 3) for p in np.percentile(shift_x_list, [10, 50, 90])],
        "shift_y_p10_p50_p90": [round(float(p), 3) for p in np.percentile(shift_y_list, [10, 50, 90])],
        "crop_cfg": crop_cfg,
    }

    print("[Calibrate] Percentiles (p10, p50, p90):")
    print(f"  scale_w: {calib_stats['scale_w_p10_p50_p90']} -> selected: {crop_cfg['scale_w']}")
    print(f"  scale_h: {calib_stats['scale_h_p10_p50_p90']} -> selected: {crop_cfg['scale_h']}")
    print(f"  shift_x: {calib_stats['shift_x_p10_p50_p90']} -> selected: {crop_cfg['shift_x']}")
    print(f"  shift_y: {calib_stats['shift_y_p10_p50_p90']} -> selected: {crop_cfg['shift_y']}")

    # Plot histogram
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(scale_w_list, bins=30, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(crop_cfg["scale_w"], color="red", linestyle="--", label=f"Median={crop_cfg['scale_w']}")
    axes[0, 0].set_title("scale_w")
    axes[0, 0].legend()

    axes[0, 1].hist(scale_h_list, bins=30, color="teal", edgecolor="white")
    axes[0, 1].axvline(crop_cfg["scale_h"], color="red", linestyle="--", label=f"Median={crop_cfg['scale_h']}")
    axes[0, 1].set_title("scale_h")
    axes[0, 1].legend()

    axes[1, 0].hist(shift_x_list, bins=30, color="darkorange", edgecolor="white")
    axes[1, 0].axvline(crop_cfg["shift_x"], color="red", linestyle="--", label=f"Median={crop_cfg['shift_x']}")
    axes[1, 0].set_title("shift_x")
    axes[1, 0].legend()

    axes[1, 1].hist(shift_y_list, bins=30, color="purple", edgecolor="white")
    axes[1, 1].axvline(crop_cfg["shift_y"], color="red", linestyle="--", label=f"Median={crop_cfg['shift_y']}")
    axes[1, 1].set_title("shift_y")
    axes[1, 1].legend()

    plt.suptitle(f"Crop Calibration Histograms (N={detected_count})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    hist_path = out_dir / "calib_hist.png"
    plt.savefig(hist_path, dpi=120)
    plt.close()
    print(f"[Calibrate] Saved calibration histogram to: {hist_path}")

    return calib_stats


def save_sample_grid(images: np.ndarray, labels: list[int], out_path: Path):
    """Save an 8x8 sample grid (32 live + 32 spoof) after resizing."""
    live_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    spoof_indices = [i for i, lbl in enumerate(labels) if lbl == 1]

    n_live = min(32, len(live_indices))
    n_spoof = min(32, len(spoof_indices))

    selected_live = live_indices[:n_live]
    selected_spoof = spoof_indices[:n_spoof]
    selected = selected_live + selected_spoof

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < len(selected):
            img_bgr = images[selected[idx]]
            lbl = labels[selected[idx]]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            title = "Live" if lbl == 0 else "Spoof"
            color = "green" if lbl == 0 else "red"
            ax.set_title(title, fontsize=8, color=color, pad=2)

    plt.suptitle("Dataset Samples (80x80): 32 Live vs 32 Spoof", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[Samples] Saved sample grid to: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  CelebA-Spoof Dataset Preparation for Face Anti-Spoofing (FAS)")
    print(f"  Out Dir: {out_dir}")
    print(f"  Cache Dir: {cache_dir}")
    print(f"  Seed: {args.seed}  |  Train Shards (target): {args.n_train_shards}/93")
    print(f"  Delete Parquet Shards After Processing: {args.delete_after_processing}")
    print("=" * 70)

    try:
        from huggingface_hub import try_to_load_from_cache
        import pyarrow.parquet as pq
    except ImportError as e:
        print(f"[ERROR] Required packages missing: {e}")
        print("Please install: pip install huggingface_hub pyarrow")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")

    stats = {
        "cached_files": 0,
        "downloaded_files": 0,
        "downloaded_bytes": 0,
        "freed_bytes": 0,
    }
    discard_counts = {"decode_error": 0, "too_small": 0}

    # 1. Scan local HF cache for existing train shards
    print("\n[1] Scanning cache for existing train shards...")
    cached_train_shards = []
    for i in range(93):
        fn = f"data/train-{i:05d}-of-00093.parquet"
        c = try_to_load_from_cache(TRAIN_REPO, fn, cache_dir=str(cache_dir), repo_type="dataset")
        if c is not None and os.path.exists(c):
            cached_train_shards.append(i)
    print(f"  Found {len(cached_train_shards)} train shards in cache: {cached_train_shards}")

    # 2. Step 0: Inspect first shard to verify schema & cropped_image['path']
    print("\n[Step 0] Inspecting first shard for schema and subject ID...")
    if cached_train_shards:
        inspect_shard_idx = cached_train_shards[0]
    elif args.shards:
        inspect_shard_idx = parse_shards_arg(args.shards)[0]
    else:
        inspect_shard_idx = 0

    first_shard_fn = f"data/train-{inspect_shard_idx:05d}-of-00093.parquet"
    first_shard_file = fetch_file(TRAIN_REPO, first_shard_fn, cache_dir, hf_token, stats)
    pfile0 = pq.ParquetFile(first_shard_file)
    batch0 = next(pfile0.iter_batches(batch_size=1024))
    b0_dict = batch0.to_pydict()

    if "cropped_image" not in b0_dict or "labels" not in b0_dict or "labelNames" not in b0_dict:
        print(f"[ERROR] Schema mismatch! Columns found: {list(b0_dict.keys())}")
        sys.exit(1)

    col_img0 = b0_dict["cropped_image"]
    first_5_paths = [
        (item.get("path") if isinstance(item, dict) else getattr(item, "path", None))
        for item in col_img0[:5]
    ]
    print(f"  [Step 0] First 5 cropped_image['path'] values: {first_5_paths}")

    # Check regex for subject ID
    can_parse_subject = False
    sample_subject_match = None
    for item in col_img0:
        p_val = item.get("path") if isinstance(item, dict) else getattr(item, "path", None)
        if p_val and isinstance(p_val, str):
            m = SUBJECT_RE.search(p_val)
            if m:
                can_parse_subject = True
                sample_subject_match = (m.group(1), p_val)
                break

    if can_parse_subject:
        val_mode = "subject"
        print(f"  [Step 0] Successfully parsed subject_id: '{sample_subject_match[0]}' from path '{sample_subject_match[1]}'.")
        print("  [Step 0] Active validation strategy: Branch 1 (subject-based split).")
        n_total = args.n_train_shards
    else:
        val_mode = "heldout_shards"
        print("  [Step 0] Cannot parse subject_id from cropped_image['path'] (values are None or do not match regex).")
        print("  [Step 0] Active validation strategy: Branch 2 (held-out shards split).")
        print("  [WARNING] Val split uses held-out shards. This only approximates subject-level separation under the assumption that subjects are clustered within shards.")
        n_total = args.n_train_shards + 2

    # 3. Shard Selection (A2 & A3)
    test_shard_indices = list(range(10))
    if args.shards:
        explicit_shards = parse_shards_arg(args.shards)
        if val_mode == "subject":
            train_shard_indices = explicit_shards
            val_shard_indices = []
        else:
            if len(explicit_shards) >= 3:
                val_shard_indices = explicit_shards[-2:]
                train_shard_indices = explicit_shards[:-2]
            elif len(explicit_shards) == 2:
                val_shard_indices = explicit_shards[-1:]
                train_shard_indices = explicit_shards[:1]
            else:
                val_shard_indices = explicit_shards
                train_shard_indices = explicit_shards
    else:
        cached_set = set(cached_train_shards)
        uncached_list = sorted([i for i in range(93) if i not in cached_set])

        if val_mode == "subject":
            needed_train = n_total - len(cached_train_shards)
            if needed_train > 0:
                train_shard_indices = sorted(cached_train_shards + rng.sample(uncached_list, min(needed_train, len(uncached_list))))
            else:
                train_shard_indices = sorted(cached_train_shards[:n_total])
            val_shard_indices = []
        else:
            # Branch 2: held-out shards
            if len(cached_train_shards) >= n_total:
                # All required shards already in cache
                selected_all = sorted(cached_train_shards[:n_total])
                val_shard_indices = selected_all[-2:]
                train_shard_indices = selected_all[:-2]
            else:
                # Val = 2 held-out shards chosen randomly from uncached shards
                n_val = min(2, len(uncached_list))
                val_shard_indices = sorted(rng.sample(uncached_list, n_val)) if n_val > 0 else []
                remaining_uncached = [i for i in uncached_list if i not in val_shard_indices]

                needed_train = args.n_train_shards - len(cached_train_shards)
                if needed_train > 0:
                    new_train = rng.sample(remaining_uncached, min(needed_train, len(remaining_uncached)))
                    train_shard_indices = sorted(cached_train_shards + new_train)
                else:
                    train_shard_indices = sorted(cached_train_shards[:args.n_train_shards])

    print(f"\n[2] Selected Shards (val_mode='{val_mode}'):")
    print(f"  Train shards ({len(train_shard_indices)}): {train_shard_indices}")
    if val_shard_indices:
        print(f"  Val shards ({len(val_shard_indices)}): {val_shard_indices}")
    print(f"  Test shards ({len(test_shard_indices)}): {test_shard_indices}")

    # 4. Disk space check (A4)
    all_needed_train_val = sorted(list(set(train_shard_indices + val_shard_indices)))
    uncached_train_val_count = 0
    for idx in all_needed_train_val:
        fn = f"data/train-{idx:05d}-of-00093.parquet"
        c = try_to_load_from_cache(TRAIN_REPO, fn, cache_dir=str(cache_dir), repo_type="dataset")
        if c is None or not os.path.exists(c):
            uncached_train_val_count += 1

    uncached_test_count = 0
    for idx in test_shard_indices:
        fn = f"data/test-{idx:05d}-of-00010.parquet"
        c = try_to_load_from_cache(TEST_REPO, fn, cache_dir=str(cache_dir), repo_type="dataset")
        if c is None or not os.path.exists(c):
            uncached_test_count += 1

    total_uncached_files = uncached_train_val_count + uncached_test_count
    if args.delete_after_processing:
        peak_files = min(uncached_train_val_count, 1) + uncached_test_count
        needed_gb = peak_files * 0.55 + 7.0
    else:
        needed_gb = total_uncached_files * 0.55 + 7.0

    needed_bytes = int(needed_gb * (1024**3))
    free_bytes = shutil.disk_usage(out_dir).free
    free_gb = free_bytes / (1024**3)

    print(f"\n[Disk] Space check: {total_uncached_files} uncached files, estimated required space: ~{needed_gb:.2f} GB")
    print(f"[Disk] Free space on {out_dir}: {free_gb:.2f} GB")
    if free_bytes < needed_bytes:
        print(f"[ERROR] Insufficient disk space in {out_dir}!")
        print(f"  Required: {needed_gb:.2f} GB, but only {free_gb:.2f} GB available.")
        print("  Tip: Free disk space or run with '--delete-after-processing' to delete shards immediately after ingestion.")
        sys.exit(1)

    # 5. Process Train Shards (RAM pre-allocation)
    print("\n[3] Ingesting Train Shards...")
    calib_pool_bgr = []
    live_images_seen = 0

    if not args.delete_after_processing:
        # Pre-fetch all train files to compute exact total capacity from Parquet metadata
        train_files = []
        cap_train = 0
        for shard_idx in train_shard_indices:
            fn = f"data/train-{shard_idx:05d}-of-00093.parquet"
            local_file = fetch_file(TRAIN_REPO, fn, cache_dir, hf_token, stats)
            nr = pq.ParquetFile(local_file).metadata.num_rows
            if args.max_rows_per_shard > 0:
                nr = min(nr, args.max_rows_per_shard)
            cap_train += nr
            train_files.append((shard_idx, local_file))
        train_imgs_arr = np.empty((cap_train, 80, 80, 3), dtype=np.uint8)
        train_lbls_arr = np.empty((cap_train,), dtype=np.int64)
        train_shards_to_process = train_files
    else:
        train_imgs_arr = np.empty((0, 80, 80, 3), dtype=np.uint8)
        train_lbls_arr = np.empty((0,), dtype=np.int64)
        train_shards_to_process = [(idx, None) for idx in train_shard_indices]

    train_subjs_list = []
    train_write_idx = 0

    for shard_idx, local_file in train_shards_to_process:
        if local_file is None:
            fn = f"data/train-{shard_idx:05d}-of-00093.parquet"
            local_file = fetch_file(TRAIN_REPO, fn, cache_dir, hf_token, stats)
        pfile = pq.ParquetFile(local_file)
        if args.delete_after_processing:
            nr = pfile.metadata.num_rows
            if args.max_rows_per_shard > 0:
                nr = min(nr, args.max_rows_per_shard)
            if train_write_idx + nr > len(train_imgs_arr):
                new_cap = max(len(train_imgs_arr) * 2, train_write_idx + nr)
                train_imgs_arr = np.resize(train_imgs_arr, (new_cap, 80, 80, 3))
                train_lbls_arr = np.resize(train_lbls_arr, (new_cap,))

        rows_read = 0

        for batch in pfile.iter_batches(batch_size=1024):
            batch_dict = batch.to_pydict()
            col_img = batch_dict.get("cropped_image", [])
            col_lbl = batch_dict.get("labels", [])
            col_names = batch_dict.get("labelNames", [])
            n_rows = len(col_lbl)

            for i in range(n_rows):
                if args.max_rows_per_shard > 0 and rows_read >= args.max_rows_per_shard:
                    break
                rows_read += 1

                raw_lbl = int(col_lbl[i])
                name = str(col_names[i]).lower()
                if raw_lbl == 0 and name != "live":
                    print(f"[ERROR] Label mapping conflict! Label 0 has name '{name}' != 'live'")
                    sys.exit(1)
                if raw_lbl == 1 and name != "spoof":
                    print(f"[ERROR] Label mapping conflict! Label 1 has name '{name}' != 'spoof'")
                    sys.exit(1)

                raw_bytes = extract_raw_bytes(col_img[i])
                if raw_bytes is None:
                    discard_counts["decode_error"] += 1
                    continue

                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None or img_bgr.size == 0:
                    discard_counts["decode_error"] += 1
                    continue

                h, w = img_bgr.shape[:2]
                if min(h, w) < args.min_side:
                    discard_counts["too_small"] += 1
                    continue

                # Reservoir sampling for live face calibration pool
                if raw_lbl == 0:
                    live_images_seen += 1
                    if len(calib_pool_bgr) < args.calib_n:
                        calib_pool_bgr.append(img_bgr.copy())
                    else:
                        j = rng.randint(0, live_images_seen - 1)
                        if j < args.calib_n:
                            calib_pool_bgr[j] = img_bgr.copy()

                # Extract subject_id if in Branch 1
                if val_mode == "subject":
                    p_val = col_img[i].get("path") if isinstance(col_img[i], dict) else getattr(col_img[i], "path", None)
                    subj = "unknown"
                    if p_val and isinstance(p_val, str):
                        m = SUBJECT_RE.search(p_val)
                        if m:
                            subj = m.group(1)
                    train_subjs_list.append(subj)

                face80 = resize_face(img_bgr)
                assert train_write_idx < len(train_imgs_arr), f"train_write_idx {train_write_idx} exceeds array capacity {len(train_imgs_arr)}"
                train_imgs_arr[train_write_idx] = face80
                train_lbls_arr[train_write_idx] = raw_lbl
                train_write_idx += 1

            if args.max_rows_per_shard > 0 and rows_read >= args.max_rows_per_shard:
                break

        if args.delete_after_processing:
            cleanup_shard(local_file, stats)

    # Trim train arrays
    train_imgs_arr = train_imgs_arr[:train_write_idx]
    train_lbls_arr = train_lbls_arr[:train_write_idx]
    print(f"  Finished Train Shards: {train_write_idx:,} valid images ingested.")

    # 6. Process Val Shards or Partition by Subject
    if val_mode == "subject":
        print(f"\n[4] Partitioning Train into Train & Val by subject ({args.val_frac * 100:.1f}%)...")
        all_unique_subjs = sorted(list(set(train_subjs_list)))
        n_val_subjs = max(1, int(round(len(all_unique_subjs) * args.val_frac)))
        val_subjs_set = set(rng.sample(all_unique_subjs, n_val_subjs))
        train_subjs_set = set(all_unique_subjs) - val_subjs_set

        # Verify subject sets are strictly disjoint
        assert len(train_subjs_set.intersection(val_subjs_set)) == 0, "Train and Val subjects must not overlap!"

        val_mask = np.array([s in val_subjs_set for s in train_subjs_list], dtype=bool)
        train_mask = ~val_mask

        val_imgs_arr = train_imgs_arr[val_mask]
        val_lbls_arr = train_lbls_arr[val_mask]
        train_imgs_arr = train_imgs_arr[train_mask]
        train_lbls_arr = train_lbls_arr[train_mask]
        print(f"  Subject split: {len(train_subjs_set)} train subjects ({len(train_lbls_arr):,} imgs), "
              f"{len(val_subjs_set)} val subjects ({len(val_lbls_arr):,} imgs).")
    else:
        # Branch 2: Ingest held-out val shards directly into val arrays
        print(f"\n[4] Ingesting Held-out Val Shards ({val_shard_indices})...")
        if not args.delete_after_processing:
            val_files = []
            cap_val = 0
            for shard_idx in val_shard_indices:
                fn = f"data/train-{shard_idx:05d}-of-00093.parquet"
                local_file = fetch_file(TRAIN_REPO, fn, cache_dir, hf_token, stats)
                nr = pq.ParquetFile(local_file).metadata.num_rows
                if args.max_rows_per_shard > 0:
                    nr = min(nr, args.max_rows_per_shard)
                cap_val += nr
                val_files.append((shard_idx, local_file))
            val_imgs_arr = np.empty((cap_val, 80, 80, 3), dtype=np.uint8)
            val_lbls_arr = np.empty((cap_val,), dtype=np.int64)
            val_shards_to_process = val_files
        else:
            val_imgs_arr = np.empty((0, 80, 80, 3), dtype=np.uint8)
            val_lbls_arr = np.empty((0,), dtype=np.int64)
            val_shards_to_process = [(idx, None) for idx in val_shard_indices]

        val_write_idx = 0

        for shard_idx, local_file in val_shards_to_process:
            if local_file is None:
                fn = f"data/train-{shard_idx:05d}-of-00093.parquet"
                local_file = fetch_file(TRAIN_REPO, fn, cache_dir, hf_token, stats)
            pfile = pq.ParquetFile(local_file)
            if args.delete_after_processing:
                nr = pfile.metadata.num_rows
                if args.max_rows_per_shard > 0:
                    nr = min(nr, args.max_rows_per_shard)
                if val_write_idx + nr > len(val_imgs_arr):
                    new_cap = max(len(val_imgs_arr) * 2, val_write_idx + nr)
                    val_imgs_arr = np.resize(val_imgs_arr, (new_cap, 80, 80, 3))
                    val_lbls_arr = np.resize(val_lbls_arr, (new_cap,))

            rows_read = 0

            for batch in pfile.iter_batches(batch_size=1024):
                batch_dict = batch.to_pydict()
                col_img = batch_dict.get("cropped_image", [])
                col_lbl = batch_dict.get("labels", [])
                n_rows = len(col_lbl)

                for i in range(n_rows):
                    if args.max_rows_per_shard > 0 and rows_read >= args.max_rows_per_shard:
                        break
                    rows_read += 1

                    raw_lbl = int(col_lbl[i])
                    raw_bytes = extract_raw_bytes(col_img[i])
                    if raw_bytes is None:
                        discard_counts["decode_error"] += 1
                        continue

                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img_bgr is None or img_bgr.size == 0:
                        discard_counts["decode_error"] += 1
                        continue

                    h, w = img_bgr.shape[:2]
                    if min(h, w) < args.min_side:
                        discard_counts["too_small"] += 1
                        continue

                    face80 = resize_face(img_bgr)
                    assert val_write_idx < len(val_imgs_arr), f"val_write_idx {val_write_idx} exceeds array capacity {len(val_imgs_arr)}"
                    val_imgs_arr[val_write_idx] = face80
                    val_lbls_arr[val_write_idx] = raw_lbl
                    val_write_idx += 1

                if args.max_rows_per_shard > 0 and rows_read >= args.max_rows_per_shard:
                    break

            if args.delete_after_processing:
                cleanup_shard(local_file, stats)

        val_imgs_arr = val_imgs_arr[:val_write_idx]
        val_lbls_arr = val_lbls_arr[:val_write_idx]
        print(f"  Finished Val Shards: {val_write_idx:,} valid images ingested.")

    # 7. Calibrate Crop (Reservoir Sampling on Train Live Faces)
    print(f"\n[Calibrate] Total live images seen: {live_images_seen:,}, sampled into calibration pool: {len(calib_pool_bgr):,}")
    calib_stats = calibrate_crop(calib_pool_bgr, out_dir)
    crop_cfg = calib_stats["crop_cfg"]
    del calib_pool_bgr

    # 8. Process Test Shards (NEVER deleted)
    print("\n[5] Ingesting Test Shards (all 10 shards, never deleted)...")
    if not args.delete_after_processing:
        test_files = []
        cap_test = 0
        for shard_idx in test_shard_indices:
            fn = f"data/test-{shard_idx:05d}-of-00010.parquet"
            local_file = fetch_file(TEST_REPO, fn, cache_dir, hf_token, stats)
            nr = pq.ParquetFile(local_file).metadata.num_rows
            if args.max_rows_per_shard > 0:
                nr = min(nr, args.max_rows_per_shard)
            cap_test += nr
            test_files.append((shard_idx, local_file))
        test_imgs_arr = np.empty((cap_test, 80, 80, 3), dtype=np.uint8)
        test_lbls_arr = np.empty((cap_test,), dtype=np.int64)
        test_shards_to_process = test_files
    else:
        test_imgs_arr = np.empty((0, 80, 80, 3), dtype=np.uint8)
        test_lbls_arr = np.empty((0,), dtype=np.int64)
        test_shards_to_process = [(idx, None) for idx in test_shard_indices]

    test_write_idx = 0

    for shard_idx, local_file in test_shards_to_process:
        if local_file is None:
            fn = f"data/test-{shard_idx:05d}-of-00010.parquet"
            local_file = fetch_file(TEST_REPO, fn, cache_dir, hf_token, stats)
        pfile = pq.ParquetFile(local_file)
        if args.delete_after_processing:
            nr = pfile.metadata.num_rows
            if args.max_rows_per_shard > 0:
                nr = min(nr, args.max_rows_per_shard)
            if test_write_idx + nr > len(test_imgs_arr):
                new_cap = max(len(test_imgs_arr) * 2, test_write_idx + nr)
                test_imgs_arr = np.resize(test_imgs_arr, (new_cap, 80, 80, 3))
                test_lbls_arr = np.resize(test_lbls_arr, (new_cap,))

        rows_read = 0

        for batch in pfile.iter_batches(batch_size=1024):
            batch_dict = batch.to_pydict()
            col_img = batch_dict.get("cropped_image", [])
            col_lbl = batch_dict.get("labels", [])
            n_rows = len(col_lbl)

            for i in range(n_rows):
                if args.max_rows_per_shard > 0 and rows_read >= args.max_rows_per_shard:
                    break
                rows_read += 1

                raw_lbl = int(col_lbl[i])
                raw_bytes = extract_raw_bytes(col_img[i])
                if raw_bytes is None:
                    discard_counts["decode_error"] += 1
                    continue

                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None or img_bgr.size == 0:
                    discard_counts["decode_error"] += 1
                    continue

                h, w = img_bgr.shape[:2]
                if min(h, w) < args.min_side:
                    discard_counts["too_small"] += 1
                    continue

                face80 = resize_face(img_bgr)
                assert test_write_idx < len(test_imgs_arr), f"test_write_idx {test_write_idx} exceeds array capacity {len(test_imgs_arr)}"
                test_imgs_arr[test_write_idx] = face80
                test_lbls_arr[test_write_idx] = raw_lbl
                test_write_idx += 1

            if args.max_rows_per_shard > 0 and rows_read >= args.max_rows_per_shard:
                break

    test_imgs_arr = test_imgs_arr[:test_write_idx]
    test_lbls_arr = test_lbls_arr[:test_write_idx]
    print(f"  Finished Test Shards: {test_write_idx:,} valid images ingested.")

    # 9. Save NPZ files
    print("\n[6] Saving compressed NPZ files to out-dir...")
    np.savez_compressed(out_dir / "train.npz", images=train_imgs_arr, labels=train_lbls_arr)
    np.savez_compressed(out_dir / "val.npz", images=val_imgs_arr, labels=val_lbls_arr)
    np.savez_compressed(out_dir / "test.npz", images=test_imgs_arr, labels=test_lbls_arr)

    # 10. Save Samples Grid
    save_sample_grid(train_imgs_arr, train_lbls_arr.tolist(), out_dir / "samples.png")

    # 11. Save Meta JSON
    meta = {
        "dataset_source": {"train": TRAIN_REPO, "test": TEST_REPO},
        "prep_args": vars(args),
        "val_mode": val_mode,
        "train_shards": train_shard_indices,
        "test_shards": test_shard_indices,
        "seed": args.seed,
        "crop_cfg": crop_cfg,
        "calibration": calib_stats,
        "discarded": discard_counts,
        "download_stats": {
            "cached_files": stats["cached_files"],
            "downloaded_files": stats["downloaded_files"],
            "downloaded_gb": round(stats["downloaded_bytes"] / (1024**3), 3),
            "freed_gb": round(stats["freed_bytes"] / (1024**3), 3),
        },
        "splits": {
            "train": {
                "total": int(len(train_lbls_arr)),
                "live": int(np.sum(train_lbls_arr == 0)),
                "spoof": int(np.sum(train_lbls_arr == 1)),
            },
            "val": {
                "total": int(len(val_lbls_arr)),
                "live": int(np.sum(val_lbls_arr == 0)),
                "spoof": int(np.sum(val_lbls_arr == 1)),
            },
            "test": {
                "total": int(len(test_lbls_arr)),
                "live": int(np.sum(test_lbls_arr == 0)),
                "spoof": int(np.sum(test_lbls_arr == 1)),
            },
        },
    }

    if val_mode == "heldout_shards":
        meta["val_shards"] = val_shard_indices
    else:
        meta["train_subjects_count"] = len(train_subjs_set)
        meta["val_subjects_count"] = len(val_subjs_set)

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Meta] Saved metadata to: {meta_path}")

    # 12. Final summary table
    print("\n" + "=" * 70)
    print("  DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Validation Mode: {val_mode}")
    if val_mode == "heldout_shards":
        print(f"  Val Shards     : {val_shard_indices}")
    else:
        print(f"  Subjects       : {len(train_subjs_set)} train vs {len(val_subjs_set)} val")
    print(f"  {'Split':<10} {'Total':>10} {'Live (0)':>12} {'Spoof (1)':>12} {'Live %':>10}")
    print("  " + "-" * 56)
    for sp in ["train", "val", "test"]:
        st = meta["splits"][sp]
        live_pct = (st["live"] / st["total"] * 100) if st["total"] > 0 else 0
        print(f"  {sp:<10} {st['total']:>10,} {st['live']:>12,} {st['spoof']:>12,} {live_pct:>9.1f}%")
    print("  " + "-" * 56)
    print(f"  Discarded images: decode_error={discard_counts['decode_error']}, too_small={discard_counts['too_small']}")
    print(f"  Crop configuration: {crop_cfg}")
    print("  " + "-" * 56)
    print("  Cache & Download Summary:")
    print(f"    Files from cache    : {stats['cached_files']}")
    print(f"    New files downloaded: {stats['downloaded_files']}")
    print(f"    Data downloaded     : {stats['downloaded_bytes'] / (1024**3):.2f} GB")
    print(f"    Disk space freed    : {stats['freed_bytes'] / (1024**3):.2f} GB")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
