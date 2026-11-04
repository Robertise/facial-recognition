"""
evaluate.py – measure an optimised frame on held-out frames, through the
backend's own code: JPEG q95 (as the frontend sends) -> detection ->
liveness exactly as check-in decides it (when --liveness:
backend.processing._liveness_decision, i.e. the distance gate "Move closer" /
"Move back" and then the temporal smoother) -> crop -> lighting enhancer ->
recogniser -> find_match against the gallery (1:N, attacker included).

Evaluation frames should come from another day under the same normal light as
training. Frames where the lighting enhancer fires (dim light) are reported as
their own tier suffix `@dim`, never mixed into the main rate, because the
enhancer changes the input the frame was optimised for (plan_L1.md §3).

    python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<optimise run> \\
        --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth

Tiers (plan_L1.md §6):
  baseline              attacker without glasses (zero-effort)
  D                     the optimised frame
  control_random        same mask, random spatial cells at tex_res (same palette and smoothness)
  control_plain         same mask, one flat colour (the frame's mean)
  D_EOT                 the optimised frame under unseen placement, light, blur, distance and JPEG
  baseline_EOT          attacker without glasses under unseen EOT transforms
  control_random_EOT    control_random under unseen EOT transforms
  control_plain_EOT     control_plain under unseen EOT transforms
Transfer models run baseline and D with their own gallery embeddings.

Writes frames.csv (one row per frame, tier and model) and summary.json with
Wilson 95% intervals and the share of 3 s windows with at least one success.
gate_to_stage_B.pass is only true/false when liveness was gated; without
--liveness it is null (the gate is defined on the full check-in decision).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from innovation.attacks.adv_glasses import eot, faces, glasses, pipeline, printability
from innovation.attacks.adv_glasses.optimize import control_random_texture, upsample_logits
from innovation.eval.runs import new_run, write_json

WINDOW_S = 3.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def frame_times(files: list[Path]) -> dict[str, float]:
    """Seconds since the session start, from the capture meta.csv (index order otherwise, 0.2 s apart)."""
    out = {}
    for session in {f.parent for f in files}:
        meta = session / "meta.csv"
        ms = {}
        if meta.exists():
            with open(meta, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    ms[int(row["index"])] = float(row["client_ms"]) / 1000
        for f in faces.list_images(session):
            try:
                idx = int(f.stem)
            except ValueError:
                idx = 0
            out[str(f)] = ms.get(idx, idx * 0.2)
    return out


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def unseen_frame(img: np.ndarray, rng: np.random.Generator, crop_side: float) -> np.ndarray:
    """Whole-frame version of eot.UNSEEN: light, colour, blur, distance and JPEG.

    Distance as in eot.photometric: the face crop is brought to a side drawn from
    UNSEEN.distance_px (the frame is down-scaled by side / crop_side, then back up
    so the detector sees the usual frame size). Never up-scales.
    """
    c = eot.UNSEEN
    x = img.astype(np.float32) * (1 + rng.uniform(-c.brightness, c.brightness))
    x = (x - x.mean()) * (1 + rng.uniform(-c.contrast, c.contrast)) + x.mean()
    x = np.clip(x * (1 + rng.uniform(-c.channel_gain, c.channel_gain, 3)), 0, 255).astype(np.uint8)
    s = rng.uniform(0, c.blur_sigma)
    if s > 0.3:
        x = cv2.GaussianBlur(x, (0, 0), s)
    side = rng.uniform(c.distance_px[0], c.distance_px[1])
    k = min(1.0, side / max(crop_side, 1.0))
    h, w = x.shape[:2]
    if k < 1.0:
        x = cv2.resize(cv2.resize(x, (max(8, int(w * k)), max(8, int(h * k))), interpolation=cv2.INTER_AREA), (w, h))
    return jpeg(x, int(rng.integers(c.jpeg_q[0], c.jpeg_q[1] + 1)))


def paint(img: np.ndarray, colours_rgb: torch.Tensor, mask: torch.Tensor, A: np.ndarray) -> np.ndarray:
    full = (0, 0, img.shape[1], img.shape[0])
    out = glasses.render(pipeline.to_tensor(img), colours_rgb, mask, A, full)
    return out.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()


def stage_b_gate(results: dict, liveness_gated: bool) -> dict | None:
    """plan_L1 §6: method3/D >= 80% and both controls <= 20%, on normal-light frames.

    The gate is defined on the full check-in decision, so without liveness the
    verdict is null (rates still reported).
    """
    d = results.get("method3/D", {}).get("rate")
    if d is None:
        return None
    c = max(results.get(f"method3/{t}", {}).get("rate", 0.0) for t in ("control_random", "control_plain"))
    return {"D_rate": d, "control_max": c, "liveness_gated": liveness_gated,
            "pass": bool(d >= 0.8 and c <= 0.2) if liveness_gated else None}


def liveness_step(live, smoother, frame: np.ndarray, t: float) -> tuple[bool, str, float]:
    """Check-in's liveness decision (backend.processing.process_attendance_frame).

    With a smoother: _liveness_decision (distance gate, then smoothing). Without one
    (smoothing disabled in the liveness config): the plain live.check branch.
    Returns (is_real, label, score).
    """
    from backend.processing import _liveness_decision
    if smoother is None:
        is_real, p, label, _ = live.check(frame)
        return bool(is_real) and label != "No Face", label, float(p)
    found, is_real, info, _ = _liveness_decision(live, frame, smoother, t)
    if not found:
        return False, "No Face", 0.0
    return bool(is_real), info["label"], float(info["score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an optimize.py results folder")
    ap.add_argument("--tiers", nargs="*", default=[
        "baseline", "D", "control_random", "control_plain",
        "D_EOT", "baseline_EOT", "control_random_EOT", "control_plain_EOT"
    ])
    ap.add_argument("--liveness", action="store_true", help="gate on the backend's smoothed liveness (needs its checkpoint)")
    ap.add_argument("--transfer", nargs="*", default=[], help="name=checkpoint pairs for the transfer study")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--control-seed", type=int, default=0, help="random seed for control_random spatial shuffling (default: 0)")
    args = ap.parse_args()

    src = Path(args.run)
    attack = json.loads((src / "attack.json").read_text())
    split = json.loads((src / "split.json").read_text())
    files = [Path(f) for f in split["eval"]]
    times = frame_times(files)
    files.sort(key=lambda f: (f.parent.name, times[str(f)]))
    run = new_run(f"advglasses_eval_{attack['mode']}", {**vars(args), "source_run": str(src)})
    rng = np.random.default_rng(args.seed)

    shape = glasses.FrameShape(**attack["frame_shape"])
    mask = torch.from_numpy(shape.mask())[None]
    cmap = printability.ColorMap.load(src / "colormap.json")
    tex_data = torch.load(src / "texture.pt", map_location="cpu")
    logits_raw = tex_data["logits"]
    tex_res = tex_data.get("tex_res") or attack.get("tex_res") or list(logits_raw.shape[-2:])
    logits_full = upsample_logits(logits_raw)
    T = torch.sigmoid(logits_full)
    seen = cmap(T)                                              # the camera's view of the printed frame
    T_random = control_random_texture(logits_raw, seed=args.control_seed)
    seen_random = cmap(T_random)
    flat = seen.reshape(3, -1)
    plain = flat.mean(1, keepdim=True).expand_as(flat).reshape(seen.shape).clone()
    colours = {"D": seen, "control_random": seen_random, "control_plain": plain, "D_EOT": seen}

    from backend.face_recognizer import FaceRecognizer
    detector, eyes_finder = faces.FaceDetector(), faces.EyeLandmarks()
    models = {"method3": attack["checkpoint"], **dict(t.split("=", 1) for t in args.transfer)}

    live = None
    if args.liveness:
        from liveness.liveness_detector import LivenessDetector
        from liveness.temporal import make_smoother
        live = LivenessDetector()

    # placement and the D_EOT frames are drawn once, so every model sees the same inputs
    prepared = []
    for f in files:
        img = cv2.imread(str(f))
        eyes = eyes_finder.eyes(img) if img is not None else None
        if eyes is None:
            continue
        A = glasses.similarity(eyes)
        A_unseen = glasses.jitter(A, eot.UNSEEN.shift_px, eot.UNSEEN.rot_deg, eot.UNSEEN.scale, rng)
        bbox0 = detector.detect(img)
        side = float(max(bbox0[2], bbox0[3]) * (1 + 2 * faces.CROP_PAD)) if bbox0 is not None else float(max(img.shape[:2]))
        prepared.append((f, img, A, A_unseen, side, int(rng.integers(1 << 30))))

    rows, model_info = [], {}
    for model, ckpt in models.items():
        recognizer = FaceRecognizer(threshold=0.5, ckpt_path=ckpt, strict=True)  # stops on key mismatch
        model_info[model] = recognizer.load_report
        gallery, _ = faces.build_gallery(Path(attack["gallery"]), recognizer, detector,
                                         attack.get("gallery_per_id", faces.GALLERY_PER_ID),
                                         cache_dir=src.parent / "_cache")
        db = gallery.as_db()
        tiers = args.tiers if model == "method3" else [t for t in args.tiers if t in ("baseline", "D")]
        for tier in tiers:
            smoothers = {}
            for f, img, A, A_unseen, side, seed in prepared:
                if tier == "baseline":
                    frame = img
                elif tier == "baseline_EOT":
                    frame = unseen_frame(img, np.random.default_rng(seed), side)
                elif tier.endswith("_EOT"):
                    base_tier = tier[:-4]
                    frame = unseen_frame(paint(img, colours[base_tier], mask, A_unseen), np.random.default_rng(seed), side)
                else:
                    frame = paint(img, colours[tier], mask, A)
                frame = jpeg(frame, 95)                          # what the frontend sends
                row = {"model": model, "tier": tier, "frame": str(f), "session": f.parent.name,
                       "t": round(times[str(f)], 3), "face": 0, "live_label": "", "live_score": "",
                       "is_real": 1, "enhancer_applied": "", "match_id": "", "match_sim": "",
                       "victim_sim": "", "attacker_sim": ""}
                bbox = detector.detect(frame)
                if bbox is not None:
                    row["face"] = 1
                    if live is not None:
                        if f.parent.name not in smoothers:      # one smoother per session, like one check-in socket
                            smoothers[f.parent.name] = make_smoother(live.config)
                        real, label, p = liveness_step(live, smoothers[f.parent.name], frame, times[str(f)])
                        row.update(live_label=label, live_score=round(p, 4), is_real=int(real))
                    face, meta = faces.enhance(faces.crop(frame, bbox))
                    row["enhancer_applied"] = int(bool(meta and meta.get("applied")))
                    emb = recognizer.extract_embedding(face)
                    m = recognizer.find_match(emb, db)
                    # match_sim is the best gallery score; match_id only when it clears the threshold
                    row["match_sim"] = round(m["similarity"], 4)
                    row["match_id"] = m["employee_id"] if m["matched"] else ""
                    if attack.get("victim") in gallery.embeddings:
                        row["victim_sim"] = round(float((gallery.embeddings[attack["victim"]] @ emb).max()), 4)
                    row["attacker_sim"] = round(float((gallery.embeddings[attack["attacker_id"]] @ emb).max()), 4)
                if attack["mode"] == "impersonate":
                    hit = row["match_id"] == attack["victim"]
                else:
                    hit = row["face"] == 1 and row["match_id"] != attack["attacker_id"]
                row["match_ok"] = int(bool(hit))
                row["success"] = int(bool(hit) and row["is_real"] == 1 and row["face"] == 1)
                rows.append(row)

    with open(run / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = {"source_run": str(src), "mode": attack["mode"], "victim": attack.get("victim"),
               "attacker": attack["attacker_id"], "tex_res": tex_res, "control_seed": args.control_seed,
               "liveness_gated": live is not None,
               "eval_frames": len(prepared), "coverage_train": attack.get("coverage_train"),
               "color_map_calibrated": attack.get("color_map_calibrated"),
               "gallery_per_id": attack.get("gallery_per_id"), "models": model_info, "results": {}}
    # normal light and dim light (enhancer fired) are separate tiers, never pooled
    for model in models:
        for tier in args.tiers:
            for light, suffix in (("normal", ""), ("dim", "@dim")):
                sel = [r for r in rows if r["model"] == model and r["tier"] == tier
                       and (r["enhancer_applied"] == 1) == (light == "dim")]
                if not sel:
                    continue
                n, k = len(sel), sum(r["success"] for r in sel)
                windows = {}
                for r in sel:
                    windows.setdefault((r["session"], int(r["t"] // WINDOW_S)), []).append(r["success"])
                vs = [r["victim_sim"] for r in sel if r["victim_sim"] != ""]
                summary["results"][f"{model}/{tier}{suffix}"] = {
                    "frames": n, "success": k, "rate": k / n, "wilson95": wilson(k, n),
                    "match_ignoring_liveness": sum(r["match_ok"] for r in sel) / n,
                    "liveness_pass": sum(r["is_real"] for r in sel if r["face"]) / max(1, sum(r["face"] for r in sel)),
                    "no_face": sum(1 - r["face"] for r in sel) / n,
                    "enhancer_applied": sum(1 for r in sel if r["enhancer_applied"] == 1) / n,
                    "victim_sim_mean": round(float(np.mean(vs)), 4) if vs else None,
                    "windows_3s": len(windows), "windows_with_success": sum(any(v) for v in windows.values()) / max(1, len(windows)),
                }
    if attack["mode"] == "impersonate":
        g = stage_b_gate(summary["results"], live is not None)
        if g is not None:
            summary["gate_to_stage_B"] = g
    write_json(run / "summary.json", summary)
    print(f"results: {run}")
    for k, v in summary["results"].items():
        lo, hi = v["wilson95"]
        print(f"  {k:<26} success {v['success']}/{v['frames']} = {v['rate']:.2f} [{lo:.2f}, {hi:.2f}]  "
              f"match {v['match_ignoring_liveness']:.2f}  live {v['liveness_pass']:.2f}  "
              f"3s-windows {v['windows_with_success']:.2f}  victim_sim {v['victim_sim_mean']}")
    if "gate_to_stage_B" in summary:
        print("  gate to stage B:", summary["gate_to_stage_B"])
        if live is None:
            print("  (no verdict: rerun with --liveness)")


if __name__ == "__main__":
    main()
