"""
prepare_landmarks.py - Extract face landmarks and eye coordinates for LFW shards.

Reads existing LFW crop shards in `data/innovation/defense/lfw_crops/{train,val}_*.npz`,
detects bbox, window, FaceMesh 468 landmarks, and eye centers on the original images.
Verifies consistency by checking IoU >= 0.95 between re-rendered band and stored band.
Saves matching shards to `data/innovation/defense/lfw_landmarks/`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from innovation.attacks.adv_glasses import faces
from innovation.defense.band import band_in_crop, load_band_template
from innovation.eval.runs import write_json

DEFAULT_CROPS_DIR = Path("data/innovation/defense/lfw_crops")
DEFAULT_LFW_DIR = Path("data/public/lfw/lfw_home/lfw_funneled")
DEFAULT_OUT_DIR = Path("data/innovation/defense/lfw_landmarks")


def prepare_landmarks(
    crops_dir: Path = DEFAULT_CROPS_DIR,
    lfw_dir: Path = DEFAULT_LFW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    crops_dir = Path(crops_dir)
    lfw_dir = Path(lfw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(crops_dir.glob("*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {crops_dir}")

    det = faces.FaceDetector()
    import mediapipe as mp
    mesh_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    tmpl = load_band_template()

    total_images = 0
    ok_count = 0
    band_mismatches = 0
    ious: list[float] = []

    t0 = time.time()
    if limit is not None and limit > 0:
        limit_val = max(1, int(round(limit * 0.1))) if limit >= 2 else 0
        limit_train = max(1, limit - limit_val)
    else:
        limit_train = None
        limit_val = None

    train_shards = sorted(crops_dir.glob("train_*.npz"))
    val_shards = sorted(crops_dir.glob("val_*.npz"))
    split_plans = [(train_shards, limit_train), (val_shards, limit_val)]
    shards_report: dict[str, Any] = {}

    for shards, split_limit in split_plans:
        remaining_limit = split_limit
        for sf in shards:
            if remaining_limit is not None and remaining_limit <= 0:
                break

            t_shard = time.time()
            data = np.load(sf, allow_pickle=True)
            paths = data["paths"]
            bands = data["bands"]
            n_shard = len(paths)

            if remaining_limit is not None and n_shard > remaining_limit:
                n_process = remaining_limit
            else:
                n_process = n_shard

            shard_eyes = np.zeros((n_process, 2, 2), dtype=np.float32)
            shard_window = np.zeros((n_process, 4), dtype=np.int32)
            shard_mesh = np.zeros((n_process, 468, 2), dtype=np.float32)
            shard_ok = np.zeros((n_process,), dtype=np.uint8)
            shard_paths = paths[:n_process]

            shard_ok_cnt = 0
            shard_mismatches = 0
            shard_ious: list[float] = []

            for i in range(n_process):
                rel_path = str(paths[i])
                img_path = lfw_dir / rel_path
                img = cv2.imread(str(img_path)) if img_path.exists() else None

                bbox = None
                window = (0, 0, 0, 0)
                eyes = None
                mesh_pts = None
                iou = 0.0
                ok = 0

                if img is not None:
                    bbox = det.detect(img)
                    if bbox is not None:
                        window = faces.crop_window(bbox, img.shape)
                        h, w = img.shape[:2]
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        res = mesh_detector.process(rgb)
                        if res.multi_face_landmarks:
                            lm = res.multi_face_landmarks[0].landmark
                            mesh_pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
                            pt_l = np.mean([[lm[idx].x * w, lm[idx].y * h] for idx in faces.LEFT_EYE], axis=0)
                            pt_r = np.mean([[lm[idx].x * w, lm[idx].y * h] for idx in faces.RIGHT_EYE], axis=0)
                            eyes = np.array([pt_l, pt_r], dtype=np.float32)

                            b_computed = band_in_crop(img, bbox, eyes, band_template=tmpl, size=112)
                            if b_computed is not None:
                                b_saved = (bands[i] > 0).astype(np.uint8)
                                b_comp = (b_computed > 0).astype(np.uint8)
                                inter = int(np.sum((b_saved > 0) & (b_comp > 0)))
                                union = int(np.sum((b_saved > 0) | (b_comp > 0)))
                                iou = float(inter / union) if union > 0 else 1.0
                                if iou >= 0.95:
                                    ok = 1
                                else:
                                    band_mismatches += 1
                                    shard_mismatches += 1

                if eyes is not None:
                    shard_eyes[i] = eyes
                shard_window[i] = np.array(window, dtype=np.int32)
                if mesh_pts is not None:
                    shard_mesh[i] = mesh_pts
                shard_ok[i] = ok

                ious.append(iou)
                shard_ious.append(iou)
                if ok == 1:
                    ok_count += 1
                    shard_ok_cnt += 1
                total_images += 1

            out_shard_path = out_dir / sf.name
            np.savez_compressed(
                out_shard_path,
                eyes=shard_eyes,
                window=shard_window,
                mesh=shard_mesh,
                ok=shard_ok,
                paths=shard_paths,
            )

            elapsed_shard = time.time() - t_shard
            print(
                f"Shard {sf.name}: total={n_process}, ok={shard_ok_cnt}, time={elapsed_shard:.2f}s",
                flush=True,
            )

            shards_report[sf.name] = {
                "total": n_process,
                "ok": shard_ok_cnt,
                "band_mismatches": shard_mismatches,
                "iou_mean": round(float(np.mean(shard_ious)), 4) if shard_ious else 0.0,
            }

            if remaining_limit is not None:
                remaining_limit -= n_process

    # Verify that all crop shards have matching landmark shards with identical row counts
    if limit is None:
        all_crop_shards = sorted(crops_dir.glob("*.npz"))
        for cs in all_crop_shards:
            ls = out_dir / cs.name
            if not ls.exists():
                raise SystemExit(f"Missing landmark shard: {ls.name}")
            c_data = np.load(cs, allow_pickle=True)
            l_data = np.load(ls, allow_pickle=True)
            if len(c_data["paths"]) != len(l_data["paths"]):
                raise SystemExit(
                    f"Row count mismatch in shard {cs.name}: "
                    f"crop {len(c_data['paths'])} vs landmark {len(l_data['paths'])}"
                )

    elapsed = time.time() - t0
    report = {
        "total_images": total_images,
        "ok_count": ok_count,
        "band_mismatches": band_mismatches,
        "iou_min": round(float(np.min(ious)), 4) if ious else 0.0,
        "iou_mean": round(float(np.mean(ious)), 4) if ious else 0.0,
        "time_seconds": round(elapsed, 2),
        "shards": shards_report,
    }

    write_json(out_dir / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description="Extract landmarks and eye coordinates for LFW shards.")
    ap.add_argument("--crops-dir", default=str(DEFAULT_CROPS_DIR))
    ap.add_argument("--lfw-dir", default=str(DEFAULT_LFW_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rep = prepare_landmarks(
        crops_dir=Path(args.crops_dir),
        lfw_dir=Path(args.lfw_dir),
        out_dir=Path(args.out_dir),
        limit=args.limit,
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
