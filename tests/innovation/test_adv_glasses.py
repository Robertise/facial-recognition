"""
tests/innovation/test_adv_glasses.py
====================================
L1 attack tooling, without real checkpoints: the crop matches the backend,
the torch pipeline matches the backend's embedding, the frame is placed and
painted differentiably, JPEG is straight-through, the colour chart round-trips,
and the frame-capture endpoint is gated on research mode.
"""
import importlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("mediapipe")

from backend.processing import crop_face_from_bbox
from innovation.attacks.adv_glasses import eot, faces, glasses, pipeline, printability
from innovation.attacks.adv_glasses.evaluate import wilson
from innovation.attacks.adv_glasses.optimize import soft_max_cos


def test_crop_window_matches_backend_crop():
    rng = np.random.default_rng(0)
    img = np.zeros((480, 640, 3), np.uint8)
    for _ in range(200):
        bbox = [int(rng.integers(-20, 600)), int(rng.integers(-20, 450)), int(rng.integers(10, 300)), int(rng.integers(10, 300))]
        bbox = [max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]]
        x0, y0, x1, y1 = faces.crop_window(bbox, img.shape)
        assert crop_face_from_bbox(img, bbox).shape[:2] == (max(0, y1 - y0), max(0, x1 - x0))


def test_similarity_puts_template_eyes_on_target():
    eyes = np.array([[300.0, 200.0], [380.0, 190.0]], np.float32)
    A = glasses.similarity(eyes)
    mapped = (A[:, :2] @ glasses.EYES.T).T + A[:, 2]
    np.testing.assert_allclose(mapped, eyes, atol=1e-3)
    J = glasses.jitter(A, 0, 0, 0, np.random.default_rng(0))
    np.testing.assert_allclose(J, A, atol=1e-5)


def test_render_paints_only_the_frame_and_has_gradient():
    region = torch.full((3, 200, 200), 100.0)
    A = glasses.similarity(np.array([[60.0, 100.0], [140.0, 100.0]], np.float32))
    mask = torch.from_numpy(glasses.FrameShape(20).mask())[None]
    tex = torch.full((3, 75, 200), 0.5, requires_grad=True)
    out = glasses.render(region, tex, mask, A, (0, 0, 200, 200))
    painted = (out - 100.0).abs().sum(0) > 1
    assert 0.02 < painted.float().mean() < 0.6
    assert torch.allclose(out[:, 0, 0], torch.full((3,), 100.0))       # far corner untouched
    out.sum().backward()
    assert tex.grad.abs().sum() > 0
    zero = glasses.render(region, tex.detach(), torch.zeros_like(mask), A, (0, 0, 200, 200))
    assert torch.allclose(zero, region)


def test_coverage_grows_with_thickness():
    A = glasses.similarity(np.array([[60.0, 100.0], [140.0, 100.0]], np.float32))
    cov = [glasses.coverage(torch.from_numpy(glasses.FrameShape(t).mask())[None], A, (0, 0, 200, 200)) for t in (8, 20, 40)]
    assert cov[0] < cov[1] < cov[2]


def test_torch_pipeline_matches_backend_embedding(monkeypatch, tmp_path):
    from backend.face_recognizer import FaceRecognizer
    torch.manual_seed(0)
    rec = FaceRecognizer(ckpt_path=tmp_path / "missing.pth")   # random weights are enough for parity
    pipe = pipeline.TorchPipeline(rec)
    rng = np.random.default_rng(0)
    crops = [cv2.GaussianBlur(rng.integers(0, 256, (int(rng.integers(90, 260)), int(rng.integers(90, 260)), 3), dtype=np.uint8), (0, 0), 2)
             for _ in range(10)]
    assert pipeline.parity(pipe, crops).min() >= 0.999


def test_jpeg_is_straight_through():
    x = (torch.rand(3, 64, 64) * 255).requires_grad_(True)
    y = eot.jpeg_bpda(x, 70)
    assert not torch.allclose(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_photometric_keeps_shape_and_range():
    rng = np.random.default_rng(0)
    out = eot.photometric(torch.rand(3, 150, 150) * 255, eot.UNSEEN, rng)
    assert out.shape[0] == 3 and out.min() >= 0 and out.max() <= 255


def test_colour_chart_round_trip(tmp_path):
    chart = printability.make_chart(tmp_path / "chart.png")
    img = cv2.imread(str(chart))
    h, w = img.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                    np.float32([[40, 30], [w * 0.6, 20], [w * 0.62, h * 0.6], [25, h * 0.58]]))
    photo = cv2.warpPerspective(img, H, (int(w * 0.7), int(h * 0.7)), borderValue=(180, 180, 180)).astype(np.float32)
    photo = np.clip((photo / 255) ** 1.15 * np.array([0.92, 1.0, 1.06]) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "photo.png"), photo)
    cmap, report = printability.fit_from_photo(tmp_path / "photo.png", chart.with_suffix(".json"))
    assert report["fit_mae_8bit"] < 3
    cmap.save(tmp_path / "cm.json")
    again = printability.ColorMap.load(tmp_path / "cm.json")
    x = torch.rand(3, 4, 4)
    assert torch.allclose(again(x), cmap(x))
    assert printability.ColorMap.identity()(x).sub(x).abs().max() < 1e-6


def test_soft_max_between_mean_and_max():
    emb = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    E = torch.nn.functional.normalize(torch.randn(5, 16), dim=1)
    s = soft_max_cos(emb, E, 0.05)
    cos = emb @ E.T
    assert torch.all(s <= cos.max(1).values + 1e-6) and torch.all(s >= cos.mean(1) - 1e-6)


def test_wilson_interval():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95
    assert wilson(0, 0) == (0.0, 0.0)


def test_frame_capture_needs_research_mode(monkeypatch, tmp_path):
    pytest.importorskip("av")              # the innovation API decodes media with PyAV
    from fastapi.testclient import TestClient
    import backend.database as database
    import innovation.api as api
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "attendance.db")
    monkeypatch.setenv("INNOVATION_ENABLED", "1")
    monkeypatch.setattr(api, "FRAMES_DIR", tmp_path / "frames")
    import backend.main as main
    main = importlib.reload(main)
    client = TestClient(main.app)
    jpg = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    send = lambda **kw: client.post("/api/innovation/frames", data={"person": "owner", "session": "s1", "index": 0, "client_ms": 12.5, **kw},
                                    files={"file": ("0.jpg", jpg, "image/jpeg")})
    monkeypatch.delenv("INNOVATION_RESEARCH_MODE", raising=False)
    assert send().status_code == 403
    monkeypatch.setenv("INNOVATION_RESEARCH_MODE", "1")
    assert send().status_code == 200
    assert (tmp_path / "frames" / "owner" / "s1" / "0000.jpg").exists()
    assert (tmp_path / "frames" / "owner" / "s1" / "meta.csv").read_text().splitlines()[1] == "0,12.5," + str(len(jpg))
    assert send(person="../etc").status_code == 400
    monkeypatch.delenv("INNOVATION_ENABLED")
    importlib.reload(main)
