"""
optimize.py – optimise the texture of a printed eyeglass frame.

    # impersonation: the attacker's frames should match the victim, top-1 in the class
    python -m innovation.attacks.adv_glasses.optimize \\
        --attacker-frames data/innovation/frames/owner --gallery data/innovation/gallery \\
        --attacker-id owner --victim VOX_id10270 --mode impersonate \\
        --color-map data/innovation/chart/colormap.json --steps 1500 --device cuda

    # dodging: the attacker should no longer match themself
    python -m innovation.attacks.adv_glasses.optimize ... --mode dodge

Inputs
- --attacker-frames: frames captured in the Lab (<person>/<session>/*.jpg). With two
  or more sessions the last one (or --eval-sessions) is held out for evaluation;
  with one session 30% of frames are held out at random.
- --gallery: one folder per enrolled identity (victim, attacker, the simulated class),
  embedded through the backend path. The attacker must be in it (plan_L1.md §2).

The texture T lives in print colours. Each step: T -> M(T) (what the camera sees)
-> placed on the eyes by FaceMesh with placement jitter -> composited on the
crop -> light, blur, distance and JPEG (EOT) -> the recogniser. T itself is
the file to print. Outputs go to results/innovation/<run>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from innovation.attacks.adv_glasses import eot, faces, glasses, pipeline, printability
from innovation.eval.runs import new_run, write_json

TEX_H, TEX_W = 15, 40           # default parameter resolution; upsampled bicubic to 150 x 400 template


def upsample_logits(logits: torch.Tensor, out_h: int = glasses.TEMPLATE_H, out_w: int = glasses.TEMPLATE_W) -> torch.Tensor:
    """Upsample parameter logits to template resolution using bicubic interpolation."""
    if logits.shape[-2:] == (out_h, out_w):
        return logits
    if logits.dim() == 3:
        return F.interpolate(logits[None], size=(out_h, out_w), mode="bicubic", align_corners=False)[0]
    return F.interpolate(logits, size=(out_h, out_w), mode="bicubic", align_corners=False)


CONTROL_SEED = 0


def permute_cells(logits: torch.Tensor, seed: int = CONTROL_SEED) -> torch.Tensor:
    """Randomly permute spatial cells (H x W) of logits at parameter resolution,
    preserving the 3 channels of each cell.
    Works for (3, H, W) or (B, 3, H, W).
    """
    squeeze = False
    if logits.dim() == 4:
        logits = logits[0]
        squeeze = True
    c, h, w = logits.shape
    rng = np.random.default_rng(seed)
    perm = torch.from_numpy(rng.permutation(h * w)).to(logits.device)
    flat = logits.reshape(c, h * w)
    shuffled = flat[:, perm].reshape(c, h, w)
    return shuffled[None] if squeeze else shuffled


def control_random_texture(logits: torch.Tensor, seed: int = CONTROL_SEED,
                           out_h: int = glasses.TEMPLATE_H,
                           out_w: int = glasses.TEMPLATE_W) -> torch.Tensor:
    """Generate control_random print texture:
    Randomly permute spatial cells (H x W) of logits at parameter resolution (preserving 3 channels per cell),
    then upsample bicubic to (out_h, out_w) and apply sigmoid, exactly like the real texture.
    Returns T in print colours with shape (3, out_h, out_w) in [0, 1].
    """
    shuffled_logits = permute_cells(logits, seed=seed)
    logits_full = upsample_logits(shuffled_logits, out_h=out_h, out_w=out_w)
    return torch.sigmoid(logits_full)


make_control_random = control_random_texture


# ── frames and split ────────────────────────────────────────────────────────

def split_frames(root: Path, eval_sessions: list[str] | None, seed: int) -> tuple[list[Path], list[Path], dict]:
    sessions = sorted(p for p in Path(root).iterdir() if p.is_dir())
    if not sessions:  # a flat folder of frames
        sessions = [Path(root)]
    by = {s.name: faces.list_images(s) for s in sessions}
    if eval_sessions:
        held = set(eval_sessions)
    elif len(by) >= 2:
        held = {sorted(by)[-1]}
    else:
        held = set()
    if held:
        train = [f for s, fs in by.items() if s not in held for f in fs]
        test = [f for s, fs in by.items() if s in held for f in fs]
        how = f"held-out sessions: {sorted(held)}"
    else:
        fs = list(next(iter(by.values())))
        rng = np.random.default_rng(seed)
        rng.shuffle(fs)
        k = int(round(len(fs) * 0.7))
        train, test, how = sorted(fs[:k]), sorted(fs[k:]), "single session, random 70/30"
    return train, test, {"how": how, "sessions": {s: len(fs) for s, fs in by.items()},
                         "train": [str(f) for f in train], "eval": [str(f) for f in test]}


def prepare(frames: list[Path], detector, eyes_finder) -> tuple[list[dict], dict]:
    """Detection, crop window, eye placement and enhancer check for each frame."""
    items, report = [], {"no_face": 0, "no_mesh": 0, "enhancer_applied": 0}
    for f in frames:
        img = cv2.imread(str(f))
        bbox = detector.detect(img) if img is not None else None
        if bbox is None:
            report["no_face"] += 1
            continue
        _, meta = faces.enhance(faces.crop(img, bbox))
        if meta and meta.get("applied"):
            report["enhancer_applied"] += 1        # plan_L1 §3: optimise only on normal light
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            report["no_mesh"] += 1
            continue
        win = faces.crop_window(bbox, img.shape)
        x0, y0, x1, y1 = win
        items.append({"file": str(f), "window": win, "A": glasses.similarity(eyes),
                      "region": pipeline.to_tensor(img[y0:y1, x0:x1])})
    report["used"] = len(items)
    return items, report


# ── losses ──────────────────────────────────────────────────────────────────

def soft_max_cos(emb: torch.Tensor, E: torch.Tensor, tau: float) -> torch.Tensor:
    """Smooth version of find_match's max over one identity's enrolled images.

    The softmax-weighted mean sum_i softmax(cos/tau)_i * cos_i: never above the
    true max (so never above 1), tends to it as tau -> 0, and has no log(n) term,
    so identities are not shifted by how many images they have (the earlier
    tau * (logsumexp - log n) moved an identity with 10 images by tau*log 10 =
    0.115 against tau*log 5 = 0.080 for one with 5, a 0.035 bias in the margin).
    Galleries also hold 5 images per identity (faces.GALLERY_PER_ID).
    """
    cos = emb @ E.T
    return (torch.softmax(cos / tau, dim=1) * cos).sum(dim=1)


def tv(t: torch.Tensor) -> torch.Tensor:
    return (t[:, 1:, :] - t[:, :-1, :]).abs().mean() + (t[:, :, 1:] - t[:, :, :-1]).abs().mean()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attacker-frames", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--attacker-id", required=True, help="the attacker's folder name in the gallery")
    ap.add_argument("--victim", help="the victim's folder name in the gallery (impersonate)")
    ap.add_argument("--mode", choices=("impersonate", "dodge"), default="impersonate")
    ap.add_argument("--checkpoint", default="checkpoints/method3_best.pth")
    ap.add_argument("--color-map", help="colormap.json from printability fit; identity when omitted")
    ap.add_argument("--chart", help="chart.json, for the NPS palette (default: the standard chart colours)")
    ap.add_argument("--thickness", type=float, default=35.0)
    ap.add_argument("--rim-scale", type=float, default=1.1)
    ap.add_argument("--tex-res", type=int, nargs=2, default=[15, 40], metavar=("H", "W"),
                    help="parameter grid resolution (H, W); upsampled bicubic to 150x400 (default: 15 40)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--lambda-tv", type=float, default=2.5)
    ap.add_argument("--lambda-nps", type=float, default=0.5)
    ap.add_argument("--eval-sessions", nargs="*")
    ap.add_argument("--max-gallery", type=int, default=faces.GALLERY_PER_ID,
                    help="images per identity in the gallery (5 = the 5 Register angles)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()
    if args.mode == "impersonate" and not args.victim:
        ap.error("--victim is required for impersonation")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    from backend.face_recognizer import FaceRecognizer
    recognizer = FaceRecognizer(ckpt_path=args.checkpoint, strict=True)   # stops on missing/unexpected keys
    pipe = pipeline.TorchPipeline(recognizer, args.device)
    detector, eyes_finder = faces.FaceDetector(), faces.EyeLandmarks()

    run = new_run(f"advglasses_{args.mode}", vars(args))
    gallery, g_report = faces.build_gallery(Path(args.gallery), recognizer, detector, args.max_gallery,
                                            cache_dir=run.parent / "_cache")
    if g_report.get("uneven"):
        print(f"WARNING: fewer than {args.max_gallery} usable images for {g_report['uneven']}")
    if args.attacker_id not in gallery.embeddings:
        raise SystemExit(f"attacker '{args.attacker_id}' has no usable images in the gallery")
    if args.mode == "impersonate" and args.victim not in gallery.embeddings:
        raise SystemExit(f"victim '{args.victim}' has no usable images in the gallery")

    train_files, _, split = split_frames(Path(args.attacker_frames), args.eval_sessions, args.seed)
    items, p_report = prepare(train_files, detector, eyes_finder)
    if len(items) < 4:
        raise SystemExit(f"only {len(items)} usable training frames: {p_report}")

    E = {k: torch.from_numpy(v).float().to(args.device) for k, v in gallery.embeddings.items()}
    target = args.victim if args.mode == "impersonate" else args.attacker_id
    others = [k for k in E if k != target]

    shape = glasses.FrameShape(args.thickness, args.rim_scale)
    mask = torch.from_numpy(shape.mask())[None].to(args.device)
    cover = [glasses.coverage(mask.cpu(), it["A"], it["window"]) for it in items]
    cmap = printability.ColorMap.load(Path(args.color_map)) if args.color_map else printability.ColorMap.identity()
    palette = torch.tensor(json.loads(Path(args.chart).read_text())["colours_rgb"] if args.chart
                           else printability.chart_colours(), dtype=torch.float32, device=args.device)

    tex_h, tex_w = args.tex_res[0], args.tex_res[1]
    logits = (0.05 * torch.randn(3, tex_h, tex_w, device=args.device)).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=args.lr)
    log_rows, t0 = [], time.time()

    for step in range(1, args.steps + 1):
        logits_full = upsample_logits(logits)
        T = torch.sigmoid(logits_full)                  # print colours
        seen = cmap(T)                                  # what the camera sees
        batch = [items[i] for i in rng.choice(len(items), size=min(args.batch, len(items)), replace=False)]
        crops = []
        for it in batch:
            A = glasses.jitter(it["A"], eot.TRAIN.shift_px, eot.TRAIN.rot_deg, eot.TRAIN.scale, rng)
            out = glasses.render(it["region"].to(args.device), seen, mask, A, it["window"])
            crops.append(eot.photometric(out, eot.TRAIN, rng))
        emb = pipe.embed(crops)

        s_target = soft_max_cos(emb, E[target], args.tau)
        s_others = torch.stack([soft_max_cos(emb, E[k], args.tau) for k in others], dim=1) if others else None
        if args.mode == "impersonate":
            l_main = (1 - s_target).mean()
            l_margin = F.relu(s_others - s_target[:, None] + args.margin).sum(1).mean() if others else torch.zeros(())
            l_margin_att = F.relu(s_others[:, others.index(args.attacker_id)] - s_target + args.margin).mean()
        else:
            l_main = s_target.mean()
            l_margin = l_margin_att = torch.zeros((), device=args.device)
        l_tv, l_nps = tv(T), printability.nps(T, palette)
        loss = l_main + l_margin + args.lambda_tv * l_tv + args.lambda_nps * l_nps
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                hard = {k: (emb @ E[k].T).max(1).values for k in E}
                best = torch.stack([hard[k] for k in sorted(E)], 1).max(1)
                top_ids = [sorted(E)[i] for i in best.indices.tolist()]
                if args.mode == "impersonate":
                    ok = [(t == target and v >= 0.5) for t, v in zip(top_ids, best.values.tolist())]
                else:
                    ok = [not (t == target and v >= 0.5) for t, v in zip(top_ids, best.values.tolist())]
            row = {"step": step, "loss": float(loss), "main": float(l_main), "margin": float(l_margin),
                   "margin_vs_attacker": float(l_margin_att), "tv": float(l_tv), "nps": float(l_nps),
                   "target_cos": float(hard[target].mean()), "batch_success": float(np.mean(ok)),
                   "seconds": round(time.time() - t0, 1)}
            log_rows.append(row)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    with open(run / "loss.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
        w.writeheader(); w.writerows(log_rows)

    logits_full = upsample_logits(logits)
    T = torch.sigmoid(logits_full).detach().cpu()
    tex_bgr = (T.flip(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    cv2.imwrite(str(run / "texture_print.png"), tex_bgr)              # print colours: the file to print
    cv2.imwrite(str(run / "mask.png"), (shape.mask() * 255).astype(np.uint8))
    torch.save({"logits": logits.detach().cpu(), "shape": vars(shape), "tex_res": [tex_h, tex_w]}, run / "texture.pt")
    cmap.save(run / "colormap.json")
    write_json(run / "split.json", split)
    write_json(run / "attack.json", {
        "mode": args.mode, "attacker_id": args.attacker_id, "victim": args.victim, "target": target,
        "checkpoint": args.checkpoint, "model": recognizer.load_report,
        "gallery": args.gallery, "gallery_per_id": args.max_gallery, "gallery_report": g_report,
        "identities": sorted(E), "frame_shape": vars(shape), "tex_res": [tex_h, tex_w],
        "coverage_train": {"mean": float(np.mean(cover)), "min": float(np.min(cover)), "max": float(np.max(cover))},
        "prepare_report": p_report, "color_map_calibrated": cmap.calibrated,
        "final": log_rows[-1]})
    print(f"results: {run}")


if __name__ == "__main__":
    main()
