"""
printability.py – the colour chart, the printer-to-camera colour map M, and
the non-printability score (NPS).

Workflow: `make_chart` writes a 10 x 10 chart with ArUco markers in the
corners; print it on the paper the frame will use, photograph it with the
attendance webcam under the light of the trial, then `fit_from_photo` reads
the cells and fits M (print colour -> colour the camera sees). M is applied
inside the optimisation, so the optimised texture T is the file to print.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

GRID = 10
ARUCO = cv2.aruco.DICT_4X4_50


def chart_colours() -> np.ndarray:
    """100 RGB colours in 0..1: the 5-level cube (125) thinned to 100, keeping greys and corners."""
    levels = np.linspace(0, 1, 5)
    cube = np.array([[r, g, b] for r in levels for g in levels for b in levels])
    grey = np.all(cube == cube[:, :1], axis=1)
    corner = np.all((cube == 0) | (cube == 1), axis=1)
    keep = grey | corner
    rest = np.flatnonzero(~keep)
    drop = rest[np.linspace(0, len(rest) - 1, len(cube) - GRID * GRID).round().astype(int)]
    return np.delete(cube, drop, axis=0)


def make_chart(path: Path, cell_mm: float = 12.0, dpi: int = 300) -> Path:
    """Write the printable chart PNG (and chart.json with the colours) to `path`."""
    px = lambda mm: int(round(mm / 25.4 * dpi))
    cell, marker, margin = px(cell_mm), px(cell_mm * 1.2), px(cell_mm * 0.6)
    grid = cell * GRID
    side = grid + 2 * (marker + margin)
    img = np.full((side, side, 3), 255, np.uint8)
    cols = chart_colours()
    off = marker + margin
    for i, c in enumerate(cols):
        r, k = divmod(i, GRID)
        bgr = (np.array(c[::-1]) * 255).round().astype(np.uint8).tolist()
        cv2.rectangle(img, (off + k * cell, off + r * cell), (off + (k + 1) * cell - 1, off + (r + 1) * cell - 1), bgr, -1)
    d = cv2.aruco.getPredefinedDictionary(ARUCO)
    corners = [(0, 0), (side - marker, 0), (side - marker, side - marker), (0, side - marker)]
    for mid, (x, y) in enumerate(corners):
        m = cv2.aruco.generateImageMarker(d, mid, marker)
        img[y:y + marker, x:x + marker] = m[..., None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    meta = {"grid": GRID, "cell_px": cell, "marker_px": marker, "margin_px": margin, "side_px": side,
            "dpi": dpi, "cell_mm": cell_mm, "colours_rgb": cols.tolist()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path


def read_chart(photo_bgr: np.ndarray, meta: dict) -> np.ndarray:
    """Observed RGB (100, 3) in 0..1 for each cell, from a photo of the printed chart."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO))
    corners, ids, _ = det.detectMarkers(photo_bgr)
    if ids is None or not {0, 1, 2, 3} <= set(ids.ravel().tolist()):
        raise ValueError("could not find all four corner markers in the chart photo")
    side, marker = meta["side_px"], meta["marker_px"]
    # outer corner of each marker, in chart pixels and in the photo
    chart_pts = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    which = {0: 0, 1: 1, 2: 2, 3: 3}
    photo_pts = np.zeros((4, 2), np.float32)
    for c, i in zip(corners, ids.ravel()):
        if i in which:
            photo_pts[which[i]] = c[0][which[i]]
    H = cv2.getPerspectiveTransform(chart_pts, photo_pts)
    cell, off = meta["cell_px"], meta["marker_px"] + meta["margin_px"]
    out = np.zeros((GRID * GRID, 3), np.float32)
    for i in range(GRID * GRID):
        r, k = divmod(i, GRID)
        # sample the inner half of the cell
        xs = np.linspace(off + (k + 0.25) * cell, off + (k + 0.75) * cell, 7)
        ys = np.linspace(off + (r + 0.25) * cell, off + (r + 0.75) * cell, 7)
        pts = np.array([[x, y] for y in ys for x in xs], np.float32)[None]
        p = cv2.perspectiveTransform(pts, H)[0]
        px = [photo_bgr[int(round(y)), int(round(x))] for x, y in p
              if 0 <= int(round(y)) < photo_bgr.shape[0] and 0 <= int(round(x)) < photo_bgr.shape[1]]
        out[i] = np.median(np.array(px, np.float32), axis=0)[::-1] / 255.0
    return out


def _features(c: torch.Tensor) -> torch.Tensor:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    one = torch.ones_like(r)
    return torch.stack([one, r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], -1)


@dataclass
class ColorMap:
    """Second-order polynomial from print RGB to observed RGB (both 0..1), differentiable."""
    W: np.ndarray                     # (10, 3)
    calibrated: bool = True

    @classmethod
    def identity(cls) -> "ColorMap":
        W = np.zeros((10, 3), np.float32)
        W[1, 0] = W[2, 1] = W[3, 2] = 1.0
        return cls(W, calibrated=False)

    @classmethod
    def fit(cls, printed: np.ndarray, observed: np.ndarray, ridge: float = 1e-3) -> "ColorMap":
        X = _features(torch.from_numpy(printed.astype(np.float32))).numpy()
        A = X.T @ X + ridge * np.eye(X.shape[1])
        return cls((np.linalg.solve(A, X.T @ observed)).astype(np.float32))

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        """(..., 3) or (3, h, w) RGB in 0..1."""
        chw = rgb.dim() == 3 and rgb.shape[0] == 3
        x = rgb.permute(1, 2, 0) if chw else rgb
        y = (_features(x) @ torch.from_numpy(self.W).to(x)).clamp(0, 1)
        return y.permute(2, 0, 1) if chw else y

    def save(self, path: Path, extra: dict | None = None) -> None:
        Path(path).write_text(json.dumps({"W": self.W.tolist(), "calibrated": self.calibrated, **(extra or {})}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "ColorMap":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["W"], np.float32), d.get("calibrated", True))


def fit_from_photo(photo_path: Path, chart_json: Path) -> tuple[ColorMap, dict]:
    meta = json.loads(Path(chart_json).read_text())
    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise FileNotFoundError(photo_path)
    observed = read_chart(photo, meta)
    printed = np.array(meta["colours_rgb"], np.float32)
    cmap = ColorMap.fit(printed, observed)
    pred = cmap(torch.from_numpy(printed)).numpy()
    err = np.abs(pred - observed).mean(axis=1) * 255
    return cmap, {"fit_mae_8bit": float(err.mean()), "fit_max_8bit": float(err.max()),
                  "observed_range": [observed.min(0).tolist(), observed.max(0).tolist()]}


def nps(texture_rgb: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Mean distance from each texture colour to the nearest chart colour (where M was measured)."""
    t = texture_rgb.reshape(3, -1).T                     # (n, 3)
    d = torch.cdist(t, palette.to(t))                    # (n, 100)
    return d.min(dim=1).values.mean()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Colour chart for printability: make it, then fit the colour map from a photo.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chart", help="write the printable chart PNG and its chart.json")
    c.add_argument("out", nargs="?", default="data/innovation/chart/chart.png")
    c.add_argument("--cell-mm", type=float, default=12.0)
    f = sub.add_parser("fit", help="fit the colour map from a webcam photo of the printed chart")
    f.add_argument("photo")
    f.add_argument("--chart", default="data/innovation/chart/chart.json")
    f.add_argument("--out", default="data/innovation/chart/colormap.json")
    a = ap.parse_args()
    if a.cmd == "chart":
        p = make_chart(Path(a.out), a.cell_mm)
        print(f"wrote {p} and {p.with_suffix('.json')}; print at 100% scale on the frame's paper")
    else:
        cmap, report = fit_from_photo(Path(a.photo), Path(a.chart))
        cmap.save(Path(a.out), {"photo": a.photo, **report})
        print(json.dumps(report, indent=1), f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
