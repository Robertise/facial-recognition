"""
export.py – the optimised texture as a print-ready PNG at true size.

    python -m innovation.attacks.adv_glasses.export --run results/innovation/<optimise run> --ipd-mm 63

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

from innovation.attacks.adv_glasses import glasses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ipd-mm", type=float, default=63.0, help="wearer's interpupillary distance in mm")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    run = Path(args.run)
    tex = cv2.imread(str(run / "texture_print.png"))
    mask = cv2.imread(str(run / "mask.png"), cv2.IMREAD_GRAYSCALE)
    eye_px = float(glasses.EYES[1, 0] - glasses.EYES[0, 0])
    mm_per_tpx = args.ipd_mm / eye_px
    dots_per_tpx = mm_per_tpx / 25.4 * args.dpi
    W, H = int(round(glasses.TEMPLATE_W * dots_per_tpx)), int(round(glasses.TEMPLATE_H * dots_per_tpx))
    tex_big = cv2.resize(tex, (W, H), interpolation=cv2.INTER_CUBIC)
    m_big = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR) > 127

    mm = lambda v: int(round(v / 25.4 * args.dpi))
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
    label = f"{run.name}  IPD {args.ipd_mm:g} mm  {args.dpi} dpi  ruler 50 mm"
    cv2.putText(sheet, label, (x0, y + mm(8)), cv2.FONT_HERSHEY_SIMPLEX, args.dpi / 600, (0, 0, 0), 2, cv2.LINE_AA)

    out = run / f"print_ipd{args.ipd_mm:g}mm_{args.dpi}dpi.png"
    cv2.imwrite(str(out), sheet)
    info = {"file": out.name, "ipd_mm": args.ipd_mm, "dpi": args.dpi,
            "frame_width_mm": round(glasses.TEMPLATE_W * mm_per_tpx, 1),
            "frame_height_mm": round(glasses.TEMPLATE_H * mm_per_tpx, 1),
            "note": "print at 100% scale (no fit-to-page); the ruler must measure 50 mm"}
    (run / "print_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
