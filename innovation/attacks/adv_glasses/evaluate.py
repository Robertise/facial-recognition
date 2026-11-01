"""
evaluate.py – measure an optimised frame on held-out frames, through the
backend's own code: JPEG q95 (as the frontend sends) -> detection ->
smoothed liveness (when --liveness) -> crop -> lighting enhancer ->
recogniser -> find_match against the gallery (1:N, attacker included).

    python -m innovation.attacks.adv_glasses.evaluate --run results/innovation/<optimise run> \\
        --liveness --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth

Tiers (plan_L1.md §6):
  baseline        attacker without glasses (zero-effort)
  D               the optimised frame
  control_random  same mask, the frame's own colours shuffled (same palette, no structure)
  control_plain   same mask, one flat colour (the frame's mean)
  D_EOT           the optimised frame under unseen placement, light, blur, distance and JPEG
Transfer models run baseline and D with their own gallery embeddings.

Writes frames.csv (one row per frame, tier and model) and summary.json with
Wilson 95% intervals and the share of 3 s windows with at least one success.
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

from innovation.attacks.adv_glasses import eot, faces, glasses, pipeline, printability
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


def unseen_frame(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Whole-frame version of eot.UNSEEN: light, colour, blur, distance (down/up-scale) and JPEG."""
    c = eot.UNSEEN
    x = img.astype(np.float32) * (1 + rng.uniform(-c.brightness, c.brightness))
    x = (x - x.mean()) * (1 + rng.uniform(-c.contrast, c.contrast)) + x.mean()
    x = np.clip(x * (1 + rng.uniform(-c.channel_gain, c.channel_gain, 3)), 0, 255).astype(np.uint8)
    s = rng.uniform(0, c.blur_sigma)
    if s > 0.3:
        x = cv2.GaussianBlur(x, (0, 0), s)
    k = rng.uniform(0.5, 0.8)
    h, w = x.shape[:2]
    x = cv2.resize(cv2.resize(x, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA), (w, h))
    return jpeg(x, int(rng.integers(c.jpeg_q[0], c.jpeg_q[1] + 1)))


def paint(img: np.ndarray, colours_rgb: torch.Tensor, mask: torch.Tensor, A: np.ndarray) -> np.ndarray:
    full = (0, 0, img.shape[1], img.shape[0])
    out = glasses.render(pipeline.to_tensor(img), colours_rgb, mask, A, full)
    return out.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an optimize.py results folder")
    ap.add_argument("--tiers", nargs="*", default=["baseline", "D", "control_random", "control_plain", "D_EOT"])
    ap.add_argument("--liveness", action="store_true", help="gate on the backend's smoothed liveness (needs its checkpoint)")
    ap.add_argument("--transfer", nargs="*", default=[], help="name=checkpoint pairs for the transfer study")
    ap.add_argument("--seed", type=int, default=1)
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
    T = torch.sigmoid(torch.load(src / "texture.pt")["logits"])
    seen = cmap(T)                                              # the camera's view of the printed frame
    flat = seen.reshape(3, -1)
    shuffled = flat[:, torch.from_numpy(rng.permutation(flat.shape[1]))].reshape(seen.shape)
    plain = flat.mean(1, keepdim=True).expand_as(flat).reshape(seen.shape).clone()
    colours = {"D": seen, "control_random": shuffled, "control_plain": plain, "D_EOT": seen}

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
        prepared.append((f, img, A, A_unseen, int(rng.integers(1 << 30))))

    rows = []
    for model, ckpt in models.items():
        recognizer = FaceRecognizer(threshold=0.5, ckpt_path=ckpt)
        gallery, _ = faces.build_gallery(Path(attack["gallery"]), recognizer, detector,
                                         cache_dir=src.parent / "_cache")
        db = gallery.as_db()
        tiers = args.tiers if model == "method3" else [t for t in args.tiers if t in ("baseline", "D")]
        for tier in tiers:
            smoothers = {}
            for f, img, A, A_unseen, seed in prepared:
                if tier == "baseline":
                    frame = img
                elif tier == "D_EOT":
                    frame = unseen_frame(paint(img, colours[tier], mask, A_unseen), np.random.default_rng(seed))
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
                        sm = smoothers.setdefault(f.parent.name, make_smoother(live.config))
                        _, p, label, _ = live.check(frame)
                        state = label
                        if sm is not None and label != "No Face":
                            state, mean = sm.update(times[str(f)], p, bbox)
                            p = mean if mean is not None else p
                        row.update(live_label=state, live_score=round(float(p), 4), is_real=int(state == "Real"))
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
               "attacker": attack["attacker_id"], "liveness_gated": live is not None,
               "eval_frames": len(prepared), "coverage_train": attack.get("coverage_train"),
               "color_map_calibrated": attack.get("color_map_calibrated"), "results": {}}
    for model in models:
        for tier in args.tiers:
            sel = [r for r in rows if r["model"] == model and r["tier"] == tier]
            if not sel:
                continue
            n, k = len(sel), sum(r["success"] for r in sel)
            windows = {}
            for r in sel:
                windows.setdefault((r["session"], int(r["t"] // WINDOW_S)), []).append(r["success"])
            vs = [r["victim_sim"] for r in sel if r["victim_sim"] != ""]
            summary["results"][f"{model}/{tier}"] = {
                "frames": n, "success": k, "rate": k / n, "wilson95": wilson(k, n),
                "match_ignoring_liveness": sum(r["match_ok"] for r in sel) / n,
                "liveness_pass": sum(r["is_real"] for r in sel if r["face"]) / max(1, sum(r["face"] for r in sel)),
                "no_face": sum(1 - r["face"] for r in sel) / n,
                "enhancer_applied": sum(1 for r in sel if r["enhancer_applied"] == 1) / n,
                "victim_sim_mean": float(np.mean(vs)) if vs else None,
                "windows_3s": len(windows), "windows_with_success": sum(any(v) for v in windows.values()) / max(1, len(windows)),
            }
    d = summary["results"].get("method3/D", {}).get("rate")
    c = max([summary["results"].get(f"method3/{t}", {}).get("rate", 0) for t in ("control_random", "control_plain")], default=0)
    if attack["mode"] == "impersonate" and d is not None:
        summary["gate_to_stage_B"] = {"D_rate": d, "control_max": c, "pass": bool(d >= 0.8 and c <= 0.2)}
    write_json(run / "summary.json", summary)
    print(f"results: {run}")
    for k, v in summary["results"].items():
        lo, hi = v["wilson95"]
        print(f"  {k:<26} success {v['success']}/{v['frames']} = {v['rate']:.2f} [{lo:.2f}, {hi:.2f}]  "
              f"match {v['match_ignoring_liveness']:.2f}  live {v['liveness_pass']:.2f}  "
              f"3s-windows {v['windows_with_success']:.2f}  victim_sim {v['victim_sim_mean']}")
    if "gate_to_stage_B" in summary:
        print("  gate to stage B:", summary["gate_to_stage_B"])


if __name__ == "__main__":
    main()
