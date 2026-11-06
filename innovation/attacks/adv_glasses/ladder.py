"""
ladder.py – sequential evaluation of configuration ladder for E1a contingency (fallback 1).

Round 1 configs (C0–C4, physical constraint ladder):
  C0: --tex-res 75 200 --no-eot --thickness 55 --rim-scale 1.2   (least constrained, unconstrained digital)
  C1: --tex-res 75 200          --thickness 55 --rim-scale 1.2   (high res + EOT)
  C2: --tex-res 30 80           --thickness 55 --rim-scale 1.2   (medium res + EOT)
  C3: --tex-res 15 40           --thickness 55 --rim-scale 1.2   (printable res + EOT, large frame)
  C4: --tex-res 15 40           --thickness 35 --rim-scale 1.1   (printable res + EOT, standard frame)

Round 2 configs (R2a–R2c, data diversity + coverage expansion):
  R2a: --tex-res 15 40 --thickness 55 --rim-scale 1.2             (~11.1% coverage, baseline C3 geometry)
  R2b: --tex-res 15 40 --thickness 85 --rim-scale 1.42            (~15.0% coverage, thicker frame)
  R2c: --tex-res 30 80 --thickness 85 --rim-scale 1.42            (~15.0% coverage, medium resolution)

Round 2 common parameters:
  --eval-sessions s3_vary --dev-frac 0.2 --dev-mode block --steps 3000 --lr 0.03 --victim LFW_victim --seed 0

IMPORTANT:
This script ONLY optimizes on the train split and evaluates on the held-out DEV split.
It NEVER evaluates on the test session (e.g. s3_vary, which is reserved for the single official evaluation).
Outputs ladder.csv to results/innovation/<ts>_ladder/.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from innovation.eval.runs import new_run

LADDER_CONFIGS: dict[str, dict[str, Any]] = {
    # Round 1
    "C0": {
        "tex_res": [75, 200],
        "no_eot": True,
        "thickness": 55.0,
        "rim_scale": 1.2,
        "description": "75x200, no EOT, thickness 55, rim 1.2 (unconstrained digital)",
    },
    "C1": {
        "tex_res": [75, 200],
        "no_eot": False,
        "thickness": 55.0,
        "rim_scale": 1.2,
        "description": "75x200, EOT, thickness 55, rim 1.2 (high res + EOT)",
    },
    "C2": {
        "tex_res": [30, 80],
        "no_eot": False,
        "thickness": 55.0,
        "rim_scale": 1.2,
        "description": "30x80, EOT, thickness 55, rim 1.2 (medium res + EOT)",
    },
    "C3": {
        "tex_res": [15, 40],
        "no_eot": False,
        "thickness": 55.0,
        "rim_scale": 1.2,
        "description": "15x40, EOT, thickness 55, rim 1.2 (printable low-res, large frame)",
    },
    "C4": {
        "tex_res": [15, 40],
        "no_eot": False,
        "thickness": 35.0,
        "rim_scale": 1.1,
        "description": "15x40, EOT, thickness 35, rim 1.1 (printable low-res, standard frame)",
    },
    # Round 2
    "R2a": {
        "tex_res": [15, 40],
        "no_eot": False,
        "thickness": 55.0,
        "rim_scale": 1.2,
        "description": "15x40, EOT, thickness 55, rim 1.2 (~11.1% coverage, C3 geometry)",
    },
    "R2b": {
        "tex_res": [15, 40],
        "no_eot": False,
        "thickness": 85.0,
        "rim_scale": 1.42,
        "description": "15x40, EOT, thickness 85, rim 1.42 (~15.0% coverage)",
    },
    "R2c": {
        "tex_res": [30, 80],
        "no_eot": False,
        "thickness": 85.0,
        "rim_scale": 1.42,
        "description": "30x80, EOT, thickness 85, rim 1.42 (~15.0% coverage, medium res)",
    },
}

ROUND1_ORDER = ["C0", "C1", "C2", "C3", "C4"]
ROUND2_ORDER = ["R2a", "R2b", "R2c"]


def measure_coverage_s1(attacker_frames: Path | str = "data/innovation/frames/owner") -> list[dict[str, Any]]:
    """Measure mean coverage on usable normal-light s1 frames for candidate thickness/rim-scale pairs."""
    s1_dir = Path(attacker_frames) / "s1_day1"
    if not s1_dir.exists():
        s1_dirs = list(Path(attacker_frames).glob("*s1*"))
        if s1_dirs:
            s1_dir = s1_dirs[0]
        else:
            raise FileNotFoundError(f"s1 session directory not found in {attacker_frames}")

    import cv2
    import torch
    from innovation.attacks.adv_glasses import faces, glasses

    detector = faces.FaceDetector()
    eyes_finder = faces.EyeLandmarks()
    frames = faces.list_images(s1_dir)
    items = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        bbox = detector.detect(img)
        if bbox is None:
            continue
        _, meta = faces.enhance(faces.crop(img, bbox))
        if meta and meta.get("applied"):
            continue
        eyes = eyes_finder.eyes(img)
        if eyes is None:
            continue
        win = faces.crop_window(bbox, img.shape)
        items.append({"A": glasses.similarity(eyes), "window": win})

    candidates = [
        (55.0, 1.20, "R2a baseline (C3 geometry)"),
        (65.0, 1.30, "intermediate"),
        (75.0, 1.35, "intermediate"),
        (80.0, 1.40, "candidate ~14.8%"),
        (80.0, 1.42, "candidate ~14.9%"),
        (85.0, 1.42, "candidate ~15.0% (selected for R2b/R2c)"),
        (100.0, 1.40, "candidate ~15.0%"),
    ]
    results = []
    print(f"\nMeasuring eyeglass coverage across {len(items)} usable normal-light s1 frames:\n" + "-" * 75)
    for t, r, note in candidates:
        shape = glasses.FrameShape(t, r)
        mask = torch.from_numpy(shape.mask())[None]
        covs = [glasses.coverage(mask, it["A"], it["window"]) for it in items]
        mean_cov = float(np.mean(covs))
        min_cov = float(np.min(covs))
        max_cov = float(np.max(covs))
        results.append({
            "thickness": t, "rim_scale": r, "mean_coverage": mean_cov,
            "min_coverage": min_cov, "max_coverage": max_cov, "note": note
        })
        print(f"  thickness={t:5.1f}, rim_scale={r:4.2f} -> mean={mean_cov*100:5.2f}% "
              f"[{min_cov*100:5.2f}%, {max_cov*100:5.2f}%]  ({note})")
    print("-" * 75)
    return results


def select_config(rows: list[dict[str, Any]], priority_order: list[str] | None = None) -> tuple[dict[str, Any], str]:
    """Preregistered selection rule:
    - Prioritize order (e.g. R2a -> R2b -> R2c).
    - Pick the first config with dev_success >= 0.80.
    - If none reach 0.80, pick the config with the highest dev_success (breaking ties by priority order).
    """
    if not rows:
        raise ValueError("No results to select from.")
    by_config = {r["config"]: r for r in rows}
    order = [c for c in (priority_order or []) if c in by_config]
    if not order:
        order = [r["config"] for r in rows]

    # First in priority order with dev_success >= 0.80
    for c in order:
        r = by_config[c]
        if float(r.get("dev_success", 0.0)) >= 0.80:
            return r, f"First config in priority order with dev_success >= 80%: {c} ({float(r['dev_success']):.1%})"

    # None >= 0.80 -> highest dev_success
    best_config = max(order, key=lambda c: float(by_config[c].get("dev_success", 0.0)))
    r = by_config[best_config]
    return r, f"No config reached >= 80%; selected highest dev_success: {best_config} ({float(r['dev_success']):.1%})"


def execute_command(cmd: list[str]) -> str:
    """Execute command, stream output to stdout, and return the run directory from 'results: <path>'."""
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    run_dir = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if "results:" in line:
            run_dir = line.split("results:", 1)[1].strip()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command exited with code {proc.returncode}: {' '.join(cmd)}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attacker-frames", default="data/innovation/frames/owner")
    ap.add_argument("--gallery", default="data/innovation/gallery")
    ap.add_argument("--attacker-id", default="owner")
    ap.add_argument("--victim", default="LFW_victim")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--dev-mode", choices=("random", "block"), default="block",
                    help="dev split mode: 'block' (default for round 2) or 'random'")
    ap.add_argument("--eval-sessions", nargs="*", default=["s3_vary"],
                    help="sessions held out for final eval (default: s3_vary)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoints/method3_best.pth")
    ap.add_argument("--device", default="cuda" if subprocess.run(
        [sys.executable, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"]
    ).returncode == 0 else "cpu")
    ap.add_argument("--configs", nargs="*", default=ROUND2_ORDER,
                    help="configurations to run in order (default: R2a R2b R2c)")
    ap.add_argument("--measure-coverage", action="store_true",
                    help="measure frame coverage on s1 frames and print candidate table without running ladder")
    args = ap.parse_args()

    if args.measure_coverage:
        measure_coverage_s1(args.attacker_frames)
        return

    ladder_run = new_run("ladder", vars(args))
    csv_rows = []

    print(f"================================================================================")
    print(f"STARTING LADDER EVALUATION")
    print(f"Results directory: {ladder_run}")
    print(f"Configurations to run: {args.configs}")
    print(f"Common parameters: dev_frac={args.dev_frac}, dev_mode={args.dev_mode}, steps={args.steps}, lr={args.lr}, seed={args.seed}")
    print(f"Held-out eval sessions: {args.eval_sessions}")
    print(f"================================================================================")

    for cfg_name in args.configs:
        if cfg_name not in LADDER_CONFIGS:
            raise ValueError(f"Unknown configuration '{cfg_name}'. Valid: {list(LADDER_CONFIGS.keys())}")
        cfg = LADDER_CONFIGS[cfg_name]
        print(f"\n{'='*70}")
        print(f"RUNNING CONFIG {cfg_name}: {cfg['description']}")
        print(f"{'='*70}")

        # 1. Optimize
        cmd_opt = [
            sys.executable, "-m", "innovation.attacks.adv_glasses.optimize",
            "--attacker-frames", str(args.attacker_frames),
            "--gallery", str(args.gallery),
            "--attacker-id", str(args.attacker_id),
            "--victim", str(args.victim),
            "--mode", "impersonate",
            "--checkpoint", str(args.checkpoint),
            "--thickness", str(cfg["thickness"]),
            "--rim-scale", str(cfg["rim_scale"]),
            "--tex-res", str(cfg["tex_res"][0]), str(cfg["tex_res"][1]),
            "--steps", str(args.steps),
            "--lr", str(args.lr),
            "--dev-frac", str(args.dev_frac),
            "--dev-mode", str(args.dev_mode),
            "--seed", str(args.seed),
            "--device", str(args.device),
        ]
        if args.eval_sessions:
            cmd_opt.extend(["--eval-sessions", *args.eval_sessions])
        if cfg.get("no_eot"):
            cmd_opt.append("--no-eot")

        opt_dir_str = execute_command(cmd_opt)
        opt_dir = Path(opt_dir_str)
        if not opt_dir.exists():
            raise FileNotFoundError(f"Optimize directory not found: {opt_dir}")

        # 2. Evaluate on dev split with liveness
        cmd_eval = [
            sys.executable, "-m", "innovation.attacks.adv_glasses.evaluate",
            "--run", str(opt_dir),
            "--split", "dev",
            "--liveness",
            "--seed", str(args.seed + 1),
        ]
        eval_dir_str = execute_command(cmd_eval)
        eval_dir = Path(eval_dir_str)
        if not eval_dir.exists():
            raise FileNotFoundError(f"Evaluate directory not found: {eval_dir}")

        # 3. Read metrics
        attack = json.loads((opt_dir / "attack.json").read_text(encoding="utf-8"))
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))

        cov_info = attack.get("coverage_train", {})
        cov_mean = cov_info.get("mean")
        final_info = attack.get("final", {})
        target_cos = final_info.get("target_cos")
        margin_att = final_info.get("margin_vs_attacker")

        dev_res = summary.get("results", {}).get("method3/D")
        if dev_res is not None:
            succ = dev_res.get("success", 0)
            frames = dev_res.get("frames", 0)
            dev_succ_frames = f"{succ}/{frames}"
            w_lo, w_hi = dev_res.get("wilson95", (0.0, 0.0))
            wilson_str = f"[{w_lo:.4f}, {w_hi:.4f}]"
            vic_sim = dev_res.get("victim_sim_mean")
            rate = dev_res.get("rate", 0.0)
            match_ign = dev_res.get("match_ignoring_liveness", 0.0)
        else:
            dev_succ_frames = "0/0"
            wilson_str = "[0.0000, 0.0000]"
            vic_sim = None
            rate = 0.0
            match_ign = 0.0

        row = {
            "config": cfg_name,
            "run_optimize": opt_dir_str,
            "run_eval_dev": eval_dir_str,
            "coverage": round(float(cov_mean), 4) if cov_mean is not None else "",
            "dev_success_frames": dev_succ_frames,
            "wilson95": wilson_str,
            "victim_sim_mean": round(float(vic_sim), 4) if vic_sim is not None else "",
            "final_train_target_cos": round(float(target_cos), 4) if target_cos is not None else "",
            "margin_vs_attacker": round(float(margin_att), 4) if margin_att is not None else "",
            "dev_success": round(float(rate), 4),
            "dev_match_ignoring_liveness": round(float(match_ign), 4),
        }
        csv_rows.append(row)

    # 4. Write ladder.csv
    fieldnames = [
        "config", "run_optimize", "run_eval_dev", "coverage",
        "dev_success_frames", "wilson95", "victim_sim_mean",
        "final_train_target_cos", "margin_vs_attacker",
        "dev_success", "dev_match_ignoring_liveness",
    ]
    csv_path = ladder_run / "ladder.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # 5. Print summary report
    print("\n" + "=" * 115)
    print("LADDER SUMMARY RESULTS (DEV SPLIT ONLY)")
    print("=" * 115)
    hdr = f"{'Config':<8} {'Coverage':<10} {'Dev Success':<14} {'Wilson 95%':<22} {'Victim Sim':<12} {'Target Cos':<12} {'Dev Succ Rate':<15} {'Dev Match (No Live)':<20}"
    print(hdr)
    print("-" * 115)
    for r in csv_rows:
        print(f"{r['config']:<8} {str(r['coverage']):<10} {r['dev_success_frames']:<14} {r['wilson95']:<22} {str(r['victim_sim_mean']):<12} {str(r['final_train_target_cos']):<12} {str(r['dev_success']):<15} {str(r['dev_match_ignoring_liveness']):<20}")
    print("=" * 115)
    print(f"Results CSV written to: {csv_path}\n")

    # Preregistered selection rule evaluation
    priority = ROUND2_ORDER if any(r["config"].startswith("R2") for r in csv_rows) else ROUND1_ORDER
    selected, reason = select_config(csv_rows, priority)
    print(f">>> PREREGISTERED SELECTION RESULT: {reason}")
    print(f"    Selected configuration: {selected['config']}")
    print(f"    Optimize run:           {selected['run_optimize']}")

    eval_cmd = (
        f"python -m innovation.attacks.adv_glasses.evaluate "
        f"--run {selected['run_optimize']} "
        f"--liveness "
        f"--transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth "
        f"method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth"
    )
    print("\n>>> FINAL EVALUATION COMMAND ON TEST SESSION (RUN MANUALLY ONCE, NOT EXECUTED AUTOMATICALLY):")
    print(f"    {eval_cmd}\n")


if __name__ == "__main__":
    main()
