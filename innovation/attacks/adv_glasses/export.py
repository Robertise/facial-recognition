"""
export.py – the optimised texture as a print-ready PNG at true size.

    # single run export
    python -m innovation.attacks.adv_glasses.export --run results/innovation/<optimise run> --ipd-mm 63

    # multi-frame A4 print sheet
    python -m innovation.attacks.adv_glasses.export --sheet run1 run2 --controls --ipd-mm 63

The template is placed by the eye centres, which are 180 template pixels
apart, so the real size follows the wearer's interpupillary distance (IPD):
measure it (a ruler in a mirror, or a phone app) and pass it in millimetres.
The texture is printed as optimised, in print colours: the colour map M was
already applied during optimisation, so do not correct the colours again.
The sheet has the frame (cut along the grey outline), a 50 mm ruler to check
the printer did not rescale, and the run id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from innovation.attacks.adv_glasses import faces, glasses, pipeline, printability
from innovation.attacks.adv_glasses.optimize import control_random_texture


def load_run_data(run_dir: Path | str) -> dict:
    """Load run texture, mask, and tex_res (defaulting to [75, 200] for legacy runs)."""
    run = Path(run_dir)
    tex_path = run / "texture.pt"
    attack_path = run / "attack.json"
    tex_res = [75, 200]
    tex_data = {}
    if tex_path.exists():
        tex_data = torch.load(tex_path, map_location="cpu")
        if "tex_res" in tex_data:
            tex_res = list(tex_data["tex_res"])
    attack_data = {}
    if attack_path.exists():
        try:
            attack_data = json.loads(attack_path.read_text(encoding="utf-8"))
            if "tex_res" in attack_data and tex_res == [75, 200]:
                tex_res = list(attack_data["tex_res"])
        except Exception:
            pass

    tex = cv2.imread(str(run / "texture_print.png"))
    if tex is None and "logits" in tex_data:
        logits = tex_data["logits"]
        if logits.shape[-2:] != (glasses.TEMPLATE_H, glasses.TEMPLATE_W):
            logits = F.interpolate(logits[None], size=(glasses.TEMPLATE_H, glasses.TEMPLATE_W), mode="bicubic", align_corners=False)[0]
        T = torch.sigmoid(logits)
        tex = (T.flip(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    mask = cv2.imread(str(run / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        shape_payload = dict(attack_data)
        if "frame_shape" not in shape_payload and "shape" in tex_data:
            shape_payload["frame_shape"] = tex_data["shape"]
        shape = glasses.shape_from_attack(shape_payload)
        mask = (shape.mask() * 255).astype(np.uint8)

    return {
        "run_dir": run,
        "tex": tex,
        "mask": mask,
        "tex_res": tex_res,
        "attack": attack_data,
        "tex_data": tex_data,
    }


def draw_ruler(sheet: np.ndarray, x0: int, y: int, length_mm: int = 50, dpi: int = 300, label_suffix: str = ""):
    """Draw a calibration ruler of length_mm with millimeter tick marks and label."""
    mm = lambda v: int(round(v / 25.4 * dpi))
    cv2.line(sheet, (x0, y), (x0 + mm(length_mm), y), (0, 0, 0), 2)
    for k in range(0, length_mm + 1, 5):
        h = mm(3) if k % 10 == 0 else mm(1.5)
        cv2.line(sheet, (x0 + mm(k), y - h), (x0 + mm(k), y), (0, 0, 0), 2)
    label = f"ruler {length_mm} mm (must measure exactly {length_mm} mm when printed at 100% scale)"
    if label_suffix:
        label = f"{label}  |  {label_suffix}"
    cv2.putText(sheet, label, (x0, y + mm(8)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 600, (0, 0, 0), 2, cv2.LINE_AA)


def make_camera_preview(run_dir: Path | str, ipd_mm: float = 63.0, frame_path: Path | str | None = None, out_path: Path | str | None = None) -> Path:
    """Render glasses onto a real owner frame, downsample to camera resolution, and zoom for inspection.

    Saved to results/<run>/camera_preview.png.
    """
    run_dir = Path(run_dir)
    data = load_run_data(run_dir)
    tex_bgr = data["tex"]
    mask_arr = data["mask"]

    # Locate real owner frame
    img = None
    if frame_path is not None and Path(frame_path).exists():
        img = cv2.imread(str(frame_path))
    if img is None and (run_dir / "split.json").exists():
        try:
            sp = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
            for cand in sp.get("eval", []) + sp.get("train", []):
                p = Path(cand)
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        break
        except Exception:
            pass
    if img is None:
        cands = sorted(Path("data/innovation/frames/owner").glob("*/*.jpg"))
        if cands:
            img = cv2.imread(str(cands[0]))
    if img is None:
        # Fallback synthetic frame if no frame is present on disk (e.g. testing)
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.circle(img, (320, 240), 120, (180, 160, 150), -1)

    # Detect eyes
    eyes = None
    try:
        eyes = faces.EyeLandmarks().eyes(img)
    except Exception:
        pass
    if eyes is None:
        eyes = np.array([[288.0, 192.0], [356.0, 190.0]], dtype=np.float32)

    measured_ipd_px = float(np.linalg.norm(eyes[1] - eyes[0]))

    # Compute seen texture M(T)
    cmap_path = run_dir / "colormap.json"
    cmap = printability.ColorMap.load(cmap_path) if cmap_path.exists() else printability.ColorMap.identity()
    tex_rgb = torch.from_numpy(tex_bgr).float().flip(2).permute(2, 0, 1) / 255.0
    seen = cmap(tex_rgb)
    mask_t = torch.from_numpy(mask_arr.astype(np.float32) / 255.0)[None]

    A = glasses.similarity(eyes)
    full_win = (0, 0, img.shape[1], img.shape[0])
    rendered_t = glasses.render(pipeline.to_tensor(img), seen, mask_t, A, full_win)
    rendered_bgr = rendered_t.clamp(0, 255).round().byte().permute(1, 2, 0).numpy()

    # Crop glasses region on camera frame (at actual webcam resolution)
    mid_x = float(eyes[:, 0].mean())
    mid_y = float(eyes[:, 1].mean())
    D = max(measured_ipd_px, 30.0)

    crop_w = int(round(2.5 * D))
    crop_h = int(round(1.5 * D))
    x0 = max(0, int(round(mid_x - crop_w / 2)))
    x1 = min(img.shape[1], x0 + crop_w)
    y0 = max(0, int(round(mid_y - crop_h / 2)))
    y1 = min(img.shape[0], y0 + crop_h)

    cam_crop = rendered_bgr[y0:y1, x0:x1]

    # Zoom camera crop 4x nearest neighbor to see camera sensor pixelation
    zoom_scale = 4
    zoomed = cv2.resize(cam_crop, (cam_crop.shape[1] * zoom_scale, cam_crop.shape[0] * zoom_scale), interpolation=cv2.INTER_NEAREST)

    # Face context crop (around eyes)
    face_w = int(round(3.2 * D))
    face_h = int(round(3.6 * D))
    fx0 = max(0, int(round(mid_x - face_w / 2)))
    fx1 = min(img.shape[1], fx0 + face_w)
    fy0 = max(0, int(round(mid_y - 0.4 * face_h)))
    fy1 = min(img.shape[0], fy0 + face_h)
    face_crop = rendered_bgr[fy0:fy1, fx0:fx1]
    target_h = zoomed.shape[0]
    face_crop_res = cv2.resize(face_crop, (int(round(face_crop.shape[1] * (target_h / face_crop.shape[0]))), target_h))

    # Compose side-by-side: Face context (left) | Camera crop 4x (right)
    header_h = 60
    pad = 20
    total_w = face_crop_res.shape[1] + zoomed.shape[1] + 3 * pad
    total_h = target_h + header_h + 2 * pad
    preview = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    tex_res_str = f"{data['tex_res'][0]}x{data['tex_res'][1]}"
    title = f"Camera Preview: {run_dir.name} (tex_res={tex_res_str}) | IPD: {measured_ipd_px:.1f}px ({ipd_mm:.1f}mm)"
    subtitle = "Left: Frame at webcam resolution  |  Right: Glasses region zoomed 4x (webcam sensor pixelation)"
    cv2.putText(preview, title, (pad, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(preview, subtitle, (pad, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    y_off = header_h + pad
    x_off = pad
    preview[y_off:y_off + target_h, x_off:x_off + face_crop_res.shape[1]] = face_crop_res
    x_off += face_crop_res.shape[1] + pad
    preview[y_off:y_off + target_h, x_off:x_off + zoomed.shape[1]] = zoomed

    cv2.rectangle(preview, (pad, y_off), (pad + face_crop_res.shape[1], y_off + target_h), (180, 180, 180), 1)
    cv2.rectangle(preview, (x_off, y_off), (x_off + zoomed.shape[1], y_off + target_h), (180, 180, 180), 1)

    dest = Path(out_path) if out_path is not None else run_dir / "camera_preview.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), preview)
    return dest


def export_single_run(run_dir: Path | str, ipd_mm: float = 63.0, dpi: int = 300, frame_path: Path | str | None = None,
                      preview: bool = True) -> dict:
    """Export a single run to print-ready PNG with 50 mm ruler and generate camera preview."""
    run = Path(run_dir)
    data = load_run_data(run)
    tex = data["tex"]
    mask = data["mask"]
    tex_res = data["tex_res"]

    eye_px = float(glasses.EYES[1, 0] - glasses.EYES[0, 0])
    mm_per_tpx = ipd_mm / eye_px
    dots_per_tpx = mm_per_tpx / 25.4 * dpi
    W, H = int(round(glasses.TEMPLATE_W * dots_per_tpx)), int(round(glasses.TEMPLATE_H * dots_per_tpx))
    tex_big = cv2.resize(tex, (W, H), interpolation=cv2.INTER_CUBIC)
    m_big = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR) > 127

    mm = lambda v: int(round(v / 25.4 * dpi))
    margin = mm(10)
    sheet = np.full((H + 3 * margin + mm(12), max(W, mm(60)) + 2 * margin, 3), 255, np.uint8)
    region = sheet[margin:margin + H, margin:margin + W]
    region[m_big] = tex_big[m_big]
    contours, _ = cv2.findContours(m_big.astype(np.uint8), cv2.RETR_EXTERNAL | cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(region, contours, -1, (150, 150, 150), 1)

    y = margin * 2 + H
    x0 = margin
    cv2.line(sheet, (x0, y), (x0 + mm(50), y), (0, 0, 0), 2)
    for k in range(0, 51, 5):
        cv2.line(sheet, (x0 + mm(k), y - (mm(3) if k % 10 == 0 else mm(1.5))), (x0 + mm(k), y), (0, 0, 0), 2)
    label = f"{run.name}  IPD {ipd_mm:g} mm  {dpi} dpi  ruler 50 mm"
    cv2.putText(sheet, label, (x0, y + mm(8)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 600, (0, 0, 0), 2, cv2.LINE_AA)

    out = run / f"print_ipd{ipd_mm:g}mm_{dpi}dpi.png"
    cv2.imwrite(str(out), sheet)
    info = {"file": out.name, "ipd_mm": ipd_mm, "dpi": dpi,
            "tex_res": tex_res,
            "frame_width_mm": round(glasses.TEMPLATE_W * mm_per_tpx, 1),
            "frame_height_mm": round(glasses.TEMPLATE_H * mm_per_tpx, 1),
            "note": "print at 100% scale (no fit-to-page); the ruler must measure 50 mm"}
    (run / "print_info.json").write_text(json.dumps(info, indent=1))

    if preview:
        try:
            make_camera_preview(run, ipd_mm=ipd_mm, frame_path=frame_path)
        except Exception as e:
            print(f"Warning: could not generate camera preview: {e}")

    return info


def export_sheet(runs: list[Path | str], out_path: Path | str | None = None,
                 include_controls: bool = False, ipd_mm: float = 63.0, dpi: int = 300,
                 preview: bool = True, frame_path: Path | str | None = None,
                 control_seed: int = 0) -> dict:
    """Export multiple runs onto A4 print sheets (300 DPI) with cut contours, labels, and 50 mm ruler."""
    mm = lambda v: int(round(v / 25.4 * dpi))
    eye_px = float(glasses.EYES[1, 0] - glasses.EYES[0, 0])
    mm_per_tpx = ipd_mm / eye_px
    dots_per_tpx = mm_per_tpx / 25.4 * dpi
    W = int(round(glasses.TEMPLATE_W * dots_per_tpx))
    H = int(round(glasses.TEMPLATE_H * dots_per_tpx))

    items = []
    for r in runs:
        data = load_run_data(r)
        run_name = Path(r).name
        tex = data["tex"]
        mask = data["mask"]
        res = data["tex_res"]
        # Tier D
        items.append({
            "run": run_name,
            "tier": "D",
            "tex": tex,
            "mask": mask,
            "label": f"{run_name} [D] (res {res[0]}x{res[1]})",
        })
        if include_controls:
            # Control random: permute cells at tex_res, bicubic upsample, sigmoid
            logits_raw = data["tex_data"].get("logits")
            if logits_raw is not None:
                T_rand = control_random_texture(logits_raw, seed=control_seed)
                tex_rand = (T_rand.flip(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            else:
                T_est = torch.from_numpy(tex).float().flip(2).permute(2, 0, 1) / 255.0
                T_est = torch.clamp(T_est, 1e-4, 1.0 - 1e-4)
                logits_est = torch.log(T_est / (1.0 - T_est))
                if logits_est.shape[-2:] != tuple(res):
                    logits_est = F.interpolate(logits_est[None], size=tuple(res), mode="area")[0]
                T_rand = control_random_texture(logits_est, seed=control_seed)
                tex_rand = (T_rand.flip(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            items.append({
                "run": run_name,
                "tier": "control_random",
                "tex": tex_rand,
                "mask": mask,
                "label": f"{run_name} [control_random] (res {res[0]}x{res[1]}, seed {control_seed})",
            })
            # Control plain
            mean_color = tex[mask > 127].mean(axis=0).round().astype(np.uint8)
            plain = np.full_like(tex, mean_color)
            items.append({
                "run": run_name,
                "tier": "control_plain",
                "tex": plain,
                "mask": mask,
                "label": f"{run_name} [control_plain]",
            })
        if preview:
            try:
                make_camera_preview(r, ipd_mm=ipd_mm, frame_path=frame_path)
            except Exception as e:
                print(f"Warning: could not generate preview for {run_name}: {e}")

    # Standard A4 size at 300 DPI: 210 x 297 mm
    page_w_port, page_h_port = mm(210), mm(297)
    page_w_land, page_h_land = mm(297), mm(210)

    # Up to 4 frames fit portrait; 5 or 6 frames fit landscape (2 cols x 3 rows)
    use_landscape = (4 < len(items) <= 6)
    sheet_w = page_w_land if use_landscape else page_w_port
    sheet_h = page_h_land if use_landscape else page_h_port

    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    if not use_landscape:
        # Portrait layout (1 column, up to 4 items)
        x_left = (sheet_w - W) // 2
        top_margin = mm(12)
        bottom_area = mm(28)
        avail_h = sheet_h - top_margin - bottom_area
        slot_h = H + mm(10)
        n = max(1, min(len(items), 4))
        gap = max(mm(4), (avail_h - n * slot_h) // (n + 1))

        for idx, item in enumerate(items[:4]):
            y_top = top_margin + gap + idx * (slot_h + gap)
            tex_big = cv2.resize(item["tex"], (W, H), interpolation=cv2.INTER_CUBIC)
            m_big = cv2.resize(item["mask"], (W, H), interpolation=cv2.INTER_LINEAR) > 127
            region = sheet[y_top:y_top + H, x_left:x_left + W]
            region[m_big] = tex_big[m_big]
            contours, _ = cv2.findContours(m_big.astype(np.uint8), cv2.RETR_EXTERNAL | cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(region, contours, -1, (150, 150, 150), 1)
            lbl = f"{item['label']}  |  IPD {ipd_mm:g} mm  {dpi} dpi"
            cv2.putText(sheet, lbl, (x_left, y_top + H + mm(6)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 700, (0, 0, 0), 2, cv2.LINE_AA)

        ruler_y = sheet_h - mm(16)
        draw_ruler(sheet, x_left, ruler_y, 50, dpi, f"Sheet: {len(items)} frame(s)")
    else:
        # Landscape layout (2 columns x 3 rows)
        col_w = W
        slot_h = H + mm(8)
        x_cols = [mm(8), mm(8) + col_w + mm(8)]
        y_rows = [mm(8), mm(8) + slot_h + mm(4), mm(8) + 2 * (slot_h + mm(4))]

        for idx, item in enumerate(items[:6]):
            col = idx % 2
            row = idx // 2
            x_left = x_cols[col]
            y_top = y_rows[row]
            tex_big = cv2.resize(item["tex"], (W, H), interpolation=cv2.INTER_CUBIC)
            m_big = cv2.resize(item["mask"], (W, H), interpolation=cv2.INTER_LINEAR) > 127
            region = sheet[y_top:y_top + H, x_left:x_left + W]
            region[m_big] = tex_big[m_big]
            contours, _ = cv2.findContours(m_big.astype(np.uint8), cv2.RETR_EXTERNAL | cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(region, contours, -1, (150, 150, 150), 1)
            lbl = f"{item['label']} | IPD {ipd_mm:g}mm"
            cv2.putText(sheet, lbl, (x_left, y_top + H + mm(5)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 800, (0, 0, 0), 1, cv2.LINE_AA)

        ruler_y = sheet_h - mm(10)
        draw_ruler(sheet, mm(15), ruler_y, 50, dpi, f"Sheet (Landscape): {len(items)} frames")

    if out_path is None:
        out_path = Path("results/innovation/sheet.png")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)

    info = {
        "file": str(out_path),
        "frames_count": len(items),
        "runs": [str(Path(r).name) for r in runs],
        "include_controls": include_controls,
        "control_seed": control_seed,
        "ipd_mm": ipd_mm,
        "dpi": dpi,
        "width_px": sheet_w,
        "height_px": sheet_h,
        "frames": [{"run": it["run"], "tier": it["tier"], "label": it["label"]} for it in items],
        "note": "print at 100% scale (no fit-to-page); the ruler must measure 50 mm"
    }
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="an optimize.py results folder")
    ap.add_argument("--sheet", nargs="+", help="paths of runs to export together on an A4 sheet")
    ap.add_argument("--controls", "--include-controls", action="store_true", dest="include_controls",
                    help="include control_random and control_plain for each run on the sheet")
    ap.add_argument("--out", help="output path for the sheet (default: results/innovation/sheet.png)")
    ap.add_argument("--frame", help="path to custom owner frame for camera preview")
    ap.add_argument("--no-preview", action="store_true", help="skip generating camera preview")
    ap.add_argument("--control-seed", type=int, default=0, help="random seed for control_random spatial shuffling (default: 0)")
    ap.add_argument("--ipd-mm", type=float, default=63.0, help="wearer's interpupillary distance in mm")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not args.run and not args.sheet:
        ap.error("either --run or --sheet is required")

    if args.sheet:
        info = export_sheet(args.sheet, out_path=args.out, include_controls=args.include_controls,
                            ipd_mm=args.ipd_mm, dpi=args.dpi, preview=(not args.no_preview),
                            frame_path=args.frame, control_seed=args.control_seed)
        print(json.dumps(info, indent=1))
    else:
        info = export_single_run(args.run, ipd_mm=args.ipd_mm, dpi=args.dpi,
                                 frame_path=args.frame, preview=(not args.no_preview))
        print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
