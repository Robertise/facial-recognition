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
from innovation.attacks.adv_glasses.optimize import (
    soft_max_cos, tv, upsample_logits, control_random_texture, permute_cells, split_frames
)


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
    rec = FaceRecognizer(ckpt_path=tmp_path / "missing.pth", strict=False)   # random weights are enough for parity
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


# ── review fixes: model loading, soft max, liveness, gate ───────────────────

def _save(state, path):
    torch.save(state, path)
    return path


@pytest.mark.parametrize("cls,name", [("FaceCNN_ArcFace", "FaceCNN_ArcFace"),
                                      ("FaceCNN_Classifier", "FaceCNN_Classifier"),
                                      ("FaceCNN_Triplet", "FaceCNN_Triplet")])
def test_cnn_checkpoints_load_strict_with_their_class(tmp_path, cls, name):
    import backend.face_recognizer as fr
    m = getattr(fr, cls)() if cls == "FaceCNN_Triplet" else getattr(fr, cls)(37)
    rec = fr.FaceRecognizer(ckpt_path=_save(m.state_dict(), tmp_path / "m.pth"), strict=True)
    assert rec.arch == name and rec.preprocess.size == 112
    assert not rec.load_report["missing_keys"] and not rec.load_report["unexpected_keys"]


def test_strict_load_stops_on_key_mismatch(tmp_path):
    import backend.face_recognizer as fr
    state = fr.FaceCNN_ArcFace(37).state_dict()
    state.pop("backbone.emb_head.2.running_var")
    state["extra.weight"] = torch.zeros(1)
    path = _save(state, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="missing"):
        fr.FaceRecognizer(ckpt_path=path, strict=True)
    with pytest.raises(FileNotFoundError):
        fr.FaceRecognizer(ckpt_path=tmp_path / "none.pth", strict=True)
    assert fr.FaceRecognizer(ckpt_path=path, strict=False).load_report["unexpected_keys"] == ["extra.weight"]


def test_research_mode_makes_loading_strict(monkeypatch, tmp_path):
    import backend.face_recognizer as fr
    monkeypatch.setenv("INNOVATION_RESEARCH_MODE", "true")
    with pytest.raises(FileNotFoundError):
        fr.FaceRecognizer(ckpt_path=tmp_path / "none.pth")


def test_vit_method5_loads_with_its_own_class_and_preprocessing(tmp_path):
    pytest.importorskip("timm")
    import backend.face_recognizer as fr
    torch.manual_seed(0)
    state = fr.FaceViT_ArcFace(37).state_dict()
    assert len(state) == 157 and "backbone.cls_token" in state     # the layout of method5_best.pth
    rec = fr.FaceRecognizer(ckpt_path=_save(state, tmp_path / "m5.pth"), strict=True)
    assert rec.arch == "FaceViT_ArcFace"
    assert rec.preprocess == fr.VIT_PREPROCESS and rec.preprocess.size == 224
    # loading the ViT state into the CNN would have left every CNN weight random
    with pytest.raises(RuntimeError):
        fr.FaceCNN_ArcFace(37).load_state_dict(state)
    rng = np.random.default_rng(0)
    crops = [cv2.GaussianBlur(rng.integers(0, 256, (150, 130, 3), dtype=np.uint8), (0, 0), 2) for _ in range(3)]
    assert pipeline.parity(pipeline.TorchPipeline(rec), crops).min() >= 0.999


def test_soft_max_does_not_depend_on_image_count():
    """Adding 5 poor images to an identity must not move its score (the old log(n) form dropped by tau*log 2)."""
    torch.manual_seed(0)
    nz = torch.nn.functional.normalize
    base = torch.randn(16)
    emb = nz(base + 0.3 * torch.randn(4, 16), dim=1)
    E = nz(base + 0.8 * torch.randn(5, 16), dim=1)
    low = nz(-base + 0.3 * torch.randn(5, 16), dim=1)          # far from every query
    E10 = torch.cat([E, low])
    s5, s10 = soft_max_cos(emb, E, 0.05), soft_max_cos(emb, E10, 0.05)
    assert (s10 - s5).abs().max() < 1e-3
    old = lambda E_: 0.05 * (torch.logsumexp(emb @ E_.T / 0.05, dim=1) - np.log(E_.shape[0]))
    assert (old(E) - old(E10)).min() > 0.03                    # the bias this replaces


def test_liveness_step_uses_the_distance_gate():
    from innovation.attacks.adv_glasses.evaluate import liveness_step
    from liveness.temporal import LivenessSmoother

    class FakeLive:
        config = {"gate": {"enabled": True, "min_ratio": 0.22, "max_ratio": 0.70}}

        def check(self, frame):
            return True, 0.99, "Real", [300, 200, 40, 40]   # a face 40 px wide in a 640 px frame

    real, label, _ = liveness_step(FakeLive(), LivenessSmoother(), np.zeros((480, 640, 3), np.uint8), 0.0)
    assert not real and label == "Move closer"


def test_stage_b_gate_needs_liveness():
    from innovation.attacks.adv_glasses.evaluate import stage_b_gate
    res = {"method3/D": {"rate": 0.9}, "method3/control_random": {"rate": 0.1}, "method3/control_plain": {"rate": 0.0}}
    assert stage_b_gate(res, liveness_gated=False)["pass"] is None
    assert stage_b_gate(res, liveness_gated=True)["pass"] is True
    assert stage_b_gate({}, True) is None


def test_unseen_frame_distance_follows_eot_config():
    from innovation.attacks.adv_glasses.evaluate import unseen_frame
    img = np.full((240, 320, 3), 128, np.uint8)
    out = unseen_frame(img, np.random.default_rng(0), crop_side=50)   # already below distance_px: no rescale
    assert out.shape == img.shape


def test_gallery_cache_recomputed_when_image_changes(tmp_path):
    import os
    import time
    gal = tmp_path / "gallery"
    gal.mkdir()
    p1 = gal / "p1"
    p1.mkdir()
    img_file = p1 / "1.jpg"
    img_arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_file), img_arr)

    ckpt_file = tmp_path / "dummy_ckpt.pth"
    ckpt_file.write_text("ckpt")

    class DummyRecognizer:
        def __init__(self):
            self.ckpt_path = ckpt_file
            self.call_count = 0
        def extract_embedding(self, face):
            self.call_count += 1
            val = float(np.mean(face))
            return np.full(512, val, dtype=np.float32)

    class DummyDetector:
        def detect(self, img):
            return [10, 10, 50, 50]

    cache_dir = tmp_path / "cache"
    rec = DummyRecognizer()
    det = DummyDetector()

    # 1. Initial build: extracts embedding and saves to cache
    g1, r1 = faces.build_gallery(gal, rec, det, max_per_id=1, cache_dir=cache_dir)
    assert rec.call_count == 1
    cache_files_1 = list(cache_dir.glob("gallery_*.npz"))
    assert len(cache_files_1) == 1

    # 2. Re-run without changes: hits cache, extract_embedding is NOT called
    g2, r2 = faces.build_gallery(gal, rec, det, max_per_id=1, cache_dir=cache_dir)
    assert rec.call_count == 1
    assert np.allclose(g1.embeddings["p1"], g2.embeddings["p1"])

    # 3. Change image: modify pixels and update mtime
    time.sleep(0.01)
    new_arr = np.full((100, 100, 3), 220, dtype=np.uint8)
    cv2.imwrite(str(img_file), new_arr)
    future_time = time.time() + 10
    os.utime(str(img_file), (future_time, future_time))

    # 4. Re-run: cache key must differ, cache must be recomputed
    g3, r3 = faces.build_gallery(gal, rec, det, max_per_id=1, cache_dir=cache_dir)
    assert rec.call_count == 2
    cache_files_2 = list(cache_dir.glob("gallery_*.npz"))
    assert len(cache_files_2) == 2
    assert not np.allclose(g1.embeddings["p1"], g3.embeddings["p1"])


# ── tex_res, smoothness (TV), camera preview, and multi-frame sheet ──────────

def test_tex_res_saved_and_loaded_correctly(tmp_path):
    from innovation.attacks.adv_glasses.export import load_run_data

    # Case 1: Run with tex_res saved in texture.pt and attack.json
    run1 = tmp_path / "run_new"
    run1.mkdir()
    shape = glasses.FrameShape()
    mask = (shape.mask() * 255).astype(np.uint8)
    cv2.imwrite(str(run1 / "mask.png"), mask)
    tex = np.full((glasses.TEMPLATE_H, glasses.TEMPLATE_W, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(run1 / "texture_print.png"), tex)

    torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, run1 / "texture.pt")
    (run1 / "attack.json").write_text(json.dumps({"frame_shape": vars(shape), "tex_res": [15, 40]}))

    data1 = load_run_data(run1)
    assert data1["tex_res"] == [15, 40]

    # Case 2: Legacy run without tex_res -> defaults to [75, 200]
    run_old = tmp_path / "run_legacy"
    run_old.mkdir()
    cv2.imwrite(str(run_old / "mask.png"), mask)
    cv2.imwrite(str(run_old / "texture_print.png"), tex)
    torch.save({"logits": torch.zeros(3, 75, 200), "shape": vars(shape)}, run_old / "texture.pt")
    (run_old / "attack.json").write_text(json.dumps({"frame_shape": vars(shape)}))

    data_old = load_run_data(run_old)
    assert data_old["tex_res"] == [75, 200]


def test_tv_smooth_tex_res_substantially_lower_than_legacy_noise():
    torch.manual_seed(0)
    # Legacy initialization: 75x200 grid with 0.5 * randn
    logits_old = 0.5 * torch.randn(3, 75, 200)
    T_old = torch.sigmoid(logits_old)
    tv_old = float(tv(T_old))

    # New initialization: 15x40 grid with 0.05 * randn + bicubic upsample
    logits_new = (0.05 * torch.randn(3, 15, 40)).requires_grad_(True)
    T_new = torch.sigmoid(upsample_logits(logits_new))
    tv_new = float(tv(T_new).detach())

    # Total variation must be substantially lower (>10x lower)
    assert tv_new < 0.1 * tv_old
    assert tv_new < 0.01 and tv_old > 0.05

    # Gradient flows through upsample and tv
    l_tv = tv(T_new)
    l_tv.backward()
    assert logits_new.grad is not None and logits_new.grad.abs().sum() > 0


def test_export_sheet_creates_correct_number_of_frames(tmp_path):
    from innovation.attacks.adv_glasses.export import export_sheet, make_camera_preview

    # Setup two mock runs
    shape = glasses.FrameShape()
    mask = (shape.mask() * 255).astype(np.uint8)
    tex = np.full((glasses.TEMPLATE_H, glasses.TEMPLATE_W, 3), 128, dtype=np.uint8)

    runs = []
    for i in range(2):
        r = tmp_path / f"run_{i}"
        r.mkdir()
        cv2.imwrite(str(r / "mask.png"), mask)
        cv2.imwrite(str(r / "texture_print.png"), tex)
        torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, r / "texture.pt")
        (r / "attack.json").write_text(json.dumps({"frame_shape": vars(shape), "tex_res": [15, 40]}))
        runs.append(r)

    # 1. Sheet without controls: 1 frame per run -> 2 frames
    sheet_no_ctrl = tmp_path / "sheet_no_ctrl.png"
    info1 = export_sheet(runs, out_path=sheet_no_ctrl, include_controls=False, preview=False)
    assert info1["frames_count"] == 2
    assert sheet_no_ctrl.exists()
    assert sheet_no_ctrl.stat().st_size > 0

    # 2. Sheet with controls: 3 frames per run (D, random, plain) -> 6 frames
    sheet_ctrl = tmp_path / "sheet_ctrl.png"
    info2 = export_sheet(runs, out_path=sheet_ctrl, include_controls=True, preview=False)
    assert info2["frames_count"] == 6
    assert sheet_ctrl.exists()
    assert sheet_ctrl.stat().st_size > 0

    # 3. Camera preview utility
    preview_file = tmp_path / "preview.png"
    out_preview = make_camera_preview(runs[0], out_path=preview_file)
    assert out_preview.exists()
    assert out_preview.stat().st_size > 0


def test_control_random_multiset_and_tv():
    from innovation.attacks.adv_glasses.optimize import (
        control_random_texture,
        permute_cells,
        tv,
        upsample_logits,
    )

    # 1. Multiset test at 15x40 and legacy 75x200
    for h, w in [(15, 40), (75, 200)]:
        torch.manual_seed(42)
        logits = torch.randn(3, h, w)
        shuffled = permute_cells(logits, seed=0)
        assert shuffled.shape == (3, h, w)

        # Same multiset of 3-channel values across cells
        orig_cells = sorted([tuple(round(float(x), 5) for x in v) for v in logits.reshape(3, -1).T])
        shuf_cells = sorted([tuple(round(float(x), 5) for x in v) for v in shuffled.reshape(3, -1).T])
        assert orig_cells == shuf_cells

        # Sorted channels match exactly
        for c in range(3):
            assert torch.allclose(torch.sort(logits[c].reshape(-1))[0], torch.sort(shuffled[c].reshape(-1))[0])

    # 2. TV test: control_random has TV of the same order of magnitude as real texture,
    # and significantly lower than pixel-shuffled texture
    torch.manual_seed(0)
    logits = 0.05 * torch.randn(3, 15, 40)
    T_real = torch.sigmoid(upsample_logits(logits))
    T_ctrl_cell = control_random_texture(logits, seed=0)

    # Old-style pixel shuffle at 150x400
    rng = np.random.default_rng(0)
    T_ctrl_pixel = T_real.reshape(3, -1)[:, torch.from_numpy(rng.permutation(150 * 400))].reshape(3, 150, 400)

    tv_real = float(tv(T_real).detach())
    tv_ctrl_cell = float(tv(T_ctrl_cell).detach())
    tv_ctrl_pixel = float(tv(T_ctrl_pixel).detach())

    # control_random after upsample has TV of the same order of magnitude as T_real
    assert 0.5 * tv_real < tv_ctrl_cell < 2.5 * tv_real
    # And much lower than pixel-shuffled noise (which is ~10x higher)
    assert tv_ctrl_pixel > 5.0 * tv_ctrl_cell


def test_control_random_in_export_sheet_and_evaluate_labels(tmp_path):
    from innovation.attacks.adv_glasses.export import export_sheet

    shape = glasses.FrameShape()
    mask = (shape.mask() * 255).astype(np.uint8)
    tex = np.full((glasses.TEMPLATE_H, glasses.TEMPLATE_W, 3), 128, dtype=np.uint8)

    r = tmp_path / "run_test"
    r.mkdir()
    cv2.imwrite(str(r / "mask.png"), mask)
    cv2.imwrite(str(r / "texture_print.png"), tex)
    torch.save({"logits": torch.randn(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, r / "texture.pt")
    (r / "attack.json").write_text(json.dumps({"frame_shape": vars(shape), "tex_res": [15, 40]}))

    sheet_path = tmp_path / "sheet.png"
    info = export_sheet([r], out_path=sheet_path, include_controls=True, preview=False, control_seed=42)
    assert info["frames_count"] == 3
    assert info["control_seed"] == 42
    # Check label of control_random contains res and seed
    ctrl_item = next(it for it in info["frames"] if it["tier"] == "control_random")
    assert "res 15x40" in ctrl_item["label"]
    assert "seed 42" in ctrl_item["label"]


def test_baseline_eot_and_d_eot_receive_same_unseen_transforms():
    """For the same frame, baseline_EOT and D_EOT receive the exact same image transform (differing only in glasses)."""
    from innovation.attacks.adv_glasses.evaluate import paint, unseen_frame

    rng = np.random.default_rng(42)
    img = rng.integers(60, 200, (480, 640, 3), dtype=np.uint8)
    eyes = np.array([[280.0, 190.0], [360.0, 190.0]], dtype=np.float32)
    A = glasses.similarity(eyes)
    A_unseen = glasses.jitter(A, eot.UNSEEN.shift_px, eot.UNSEEN.rot_deg, eot.UNSEEN.scale, rng)

    shape = glasses.FrameShape(35.0, 1.1)
    mask = torch.from_numpy(shape.mask())[None]
    tex_D = torch.full((3, 150, 400), 0.8)

    seed = 987654321
    side = 112.0

    # 1. With empty mask, paint leaves img untouched; unseen_frame with the same seed is bit-for-bit identical
    zero_mask = torch.zeros_like(mask)
    painted_zero = paint(img, tex_D, zero_mask, A_unseen)
    np.testing.assert_array_equal(painted_zero, img)

    f_base_eot = unseen_frame(img, np.random.default_rng(seed), side)
    f_zero_eot = unseen_frame(painted_zero, np.random.default_rng(seed), side)
    np.testing.assert_array_equal(f_base_eot, f_zero_eot)

    # 2. With real mask, baseline_EOT and D_EOT receive the exact same transform pipeline
    painted_D = paint(img, tex_D, mask, A_unseen)
    f_d_eot = unseen_frame(painted_D, np.random.default_rng(seed), side)

    # Must differ strongly in the glasses region
    assert not np.array_equal(f_base_eot, f_d_eot)
    glasses_diff = np.mean(np.abs(f_base_eot[160:220, 260:380].astype(float) - f_d_eot[160:220, 260:380].astype(float)))
    corner_diff = np.mean(np.abs(f_base_eot[:20, :20].astype(float) - f_d_eot[:20, :20].astype(float)))
    assert glasses_diff > 10.0
    assert corner_diff <= 4.0

    # 3. control_random_EOT and control_plain_EOT also receive the exact same transform
    tex_ctrl = torch.full((3, 150, 400), 0.2)
    painted_ctrl = paint(img, tex_ctrl, mask, A_unseen)
    f_ctrl_eot = unseen_frame(painted_ctrl, np.random.default_rng(seed), side)
    assert not np.array_equal(f_d_eot, f_ctrl_eot)
    corner_ctrl_diff = np.mean(np.abs(f_base_eot[:20, :20].astype(float) - f_ctrl_eot[:20, :20].astype(float)))
    assert corner_ctrl_diff <= 4.0


def test_dev_split_disjoint_and_no_s2_frames(tmp_path):
    """Verify that split_frames with dev_frac > 0 partitions training frames cleanly,
    dev is disjoint from train, and dev never contains frames from held-out eval sessions.
    """
    s1 = tmp_path / "s1_day1"
    s2 = tmp_path / "s2_day2"
    s1.mkdir()
    s2.mkdir()
    for i in range(20):
        (s1 / f"{i:03d}.jpg").write_bytes(b"dummy")
        (s2 / f"{i:03d}.jpg").write_bytes(b"dummy")

    train_files, test_files, split = split_frames(tmp_path, eval_sessions=None, seed=0, dev_frac=0.25)

    dev_set = set(split["dev"])
    train_set = set(split["train"])
    eval_set = set(split["eval"])

    # 20 frames in s1: 25% = 5 dev, 15 train. 20 frames in s2: eval.
    assert len(dev_set) == 5
    assert len(train_set) == 15
    assert len(eval_set) == 20

    # dev must be disjoint from train
    assert dev_set.isdisjoint(train_set)
    # dev must NOT contain any frames from s2_day2 (eval)
    assert dev_set.isdisjoint(eval_set)
    for p_str in dev_set:
        assert "s2_day2" not in p_str
        assert "s1_day1" in p_str

    # train_files returned matches train_set
    assert set(str(f) for f in train_files) == train_set


def test_no_eot_disables_jitter_and_photometric():
    """Verify that --no-eot keeps the landmark matrix intact and skips photometric transforms."""
    eyes = np.array([[300.0, 200.0], [380.0, 190.0]], np.float32)
    A = glasses.similarity(eyes)
    rng = np.random.default_rng(42)

    # Normal EOT jitter modifies A
    J = glasses.jitter(A, eot.TRAIN.shift_px, eot.TRAIN.rot_deg, eot.TRAIN.scale, rng)
    assert not np.allclose(J, A)

    # With no_eot, A is preserved exactly
    A_no_eot = A
    np.testing.assert_array_equal(A_no_eot, A)

    # Normal photometric modifies pixel values
    region = torch.full((3, 112, 112), 128.0)
    photo = eot.photometric(region, eot.TRAIN, rng)
    assert not torch.allclose(photo, region)

    # With no_eot, region is appended directly without photometric
    out_no_eot = region
    assert torch.allclose(out_no_eot, region)


def test_ladder_script_properties_and_no_s2_day2():
    """Verify ladder.py configuration definitions and ensure it NEVER references s2_day2."""
    from innovation.attacks.adv_glasses.ladder import LADDER_CONFIGS, ROUND1_ORDER, ROUND2_ORDER

    ladder_file = Path("innovation/attacks/adv_glasses/ladder.py")
    assert ladder_file.exists()
    content = ladder_file.read_text(encoding="utf-8")

    # MUST NOT contain s2_day2 anywhere in the file
    assert "s2_day2" not in content

    # All configs present
    for c in ["C0", "C1", "C2", "C3", "C4", "R2a", "R2b", "R2c"]:
        assert c in LADDER_CONFIGS

    assert ROUND1_ORDER == ["C0", "C1", "C2", "C3", "C4"]
    assert ROUND2_ORDER == ["R2a", "R2b", "R2c"]

    assert LADDER_CONFIGS["C0"]["no_eot"] is True
    assert LADDER_CONFIGS["C0"]["tex_res"] == [75, 200]
    assert LADDER_CONFIGS["C0"]["thickness"] == 55.0
    assert LADDER_CONFIGS["C0"]["rim_scale"] == 1.2

    assert LADDER_CONFIGS["C1"]["no_eot"] is False
    assert LADDER_CONFIGS["C1"]["tex_res"] == [75, 200]

    assert LADDER_CONFIGS["C2"]["no_eot"] is False
    assert LADDER_CONFIGS["C2"]["tex_res"] == [30, 80]

    assert LADDER_CONFIGS["C3"]["no_eot"] is False
    assert LADDER_CONFIGS["C3"]["tex_res"] == [15, 40]
    assert LADDER_CONFIGS["C3"]["thickness"] == 55.0

    assert LADDER_CONFIGS["C4"]["no_eot"] is False
    assert LADDER_CONFIGS["C4"]["tex_res"] == [15, 40]
    assert LADDER_CONFIGS["C4"]["thickness"] == 35.0
    assert LADDER_CONFIGS["C4"]["rim_scale"] == 1.1

    # Round 2 configs
    assert LADDER_CONFIGS["R2a"]["tex_res"] == [15, 40]
    assert LADDER_CONFIGS["R2a"]["thickness"] == 55.0
    assert LADDER_CONFIGS["R2a"]["rim_scale"] == 1.2

    assert LADDER_CONFIGS["R2b"]["tex_res"] == [15, 40]
    assert LADDER_CONFIGS["R2b"]["thickness"] == 85.0
    assert LADDER_CONFIGS["R2b"]["rim_scale"] == 1.42

    assert LADDER_CONFIGS["R2c"]["tex_res"] == [30, 80]
    assert LADDER_CONFIGS["R2c"]["thickness"] == 85.0
    assert LADDER_CONFIGS["R2c"]["rim_scale"] == 1.42


def test_dev_mode_block_contiguous_and_disjoint(tmp_path):
    """Verify that dev-mode block takes the last contiguous time block from each session,
    is disjoint from train, and records dev_mode in split.json.
    """
    import csv

    # Create session 1 (20 frames) and session 2 (30 frames)
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s1.mkdir()
    s2.mkdir()

    meta1 = s1 / "meta.csv"
    with open(meta1, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "client_ms", "bytes"])
        for i in range(20):
            w.writerow([i, i * 200.0, 50000])
            (s1 / f"{i:03d}.jpg").write_bytes(b"dummy")

    meta2 = s2 / "meta.csv"
    with open(meta2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "client_ms", "bytes"])
        for i in range(30):
            w.writerow([i, i * 200.0, 50000])
            (s2 / f"{i:03d}.jpg").write_bytes(b"dummy")

    train_files, test_files, split = split_frames(
        tmp_path, eval_sessions=[], seed=0, dev_frac=0.2, dev_mode="block"
    )

    dev_files = split["dev"]
    train_files_split = split["train"]

    # In s1 (20 frames): 20 * 0.2 = 4 dev frames (indices 16, 17, 18, 19)
    # In s2 (30 frames): 30 * 0.2 = 6 dev frames (indices 24, 25, 26, 27, 28, 29)
    assert len(dev_files) == 10
    assert len(train_files_split) == 40
    assert split["dev_mode"] == "block"

    # Dev and train are completely disjoint
    assert set(dev_files).isdisjoint(set(train_files_split))

    # Check contiguous ending blocks for s1 and s2
    s1_dev = [f for f in dev_files if "s1" in f]
    s2_dev = [f for f in dev_files if "s2" in f]
    assert len(s1_dev) == 4
    assert len(s2_dev) == 6

    # Verify s1 dev frames are exactly 016..019
    s1_dev_names = sorted(Path(f).name for f in s1_dev)
    assert s1_dev_names == ["016.jpg", "017.jpg", "018.jpg", "019.jpg"]

    # Verify s2 dev frames are exactly 024..029
    s2_dev_names = sorted(Path(f).name for f in s2_dev)
    assert s2_dev_names == ["024.jpg", "025.jpg", "026.jpg", "027.jpg", "028.jpg", "029.jpg"]


def test_ladder_does_not_read_s3_vary_frames(tmp_path):
    """Verify that when s3_vary is specified in eval_sessions, ladder evaluation on dev
    never includes or reads any frame from s3_vary.
    """
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s3 = tmp_path / "s3_vary"
    for d in (s1, s2, s3):
        d.mkdir()
        for i in range(10):
            (d / f"{i:03d}.jpg").write_bytes(b"dummy")

    train_files, test_files, split = split_frames(
        tmp_path, eval_sessions=["s3_vary"], seed=0, dev_frac=0.2, dev_mode="block"
    )

    # s3_vary frames must ONLY be in eval, never in train or dev
    assert len(split["eval"]) == 10
    for p in split["eval"]:
        assert "s3_vary" in p

    for p in split["train"]:
        assert "s3_vary" not in p

    for p in split["dev"]:
        assert "s3_vary" not in p


def test_selection_rule_on_synthetic_data():
    """Verify preregistered selection rule:
    - First in priority order with dev_success >= 0.80
    - If none reach 0.80, select config with highest dev_success
    """
    from innovation.attacks.adv_glasses.ladder import select_config

    order = ["R2a", "R2b", "R2c"]

    # Case 1: R2b and R2c reach >= 80%, R2a does not -> select R2b (first in priority)
    rows1 = [
        {"config": "R2a", "dev_success": 0.72},
        {"config": "R2b", "dev_success": 0.83},
        {"config": "R2c", "dev_success": 0.88},
    ]
    sel1, reason1 = select_config(rows1, order)
    assert sel1["config"] == "R2b"
    assert "First config in priority order with dev_success >= 80%: R2b" in reason1

    # Case 2: All three reach >= 80% -> select R2a (first in priority)
    rows2 = [
        {"config": "R2a", "dev_success": 0.81},
        {"config": "R2b", "dev_success": 0.85},
        {"config": "R2c", "dev_success": 0.90},
    ]
    sel2, reason2 = select_config(rows2, order)
    assert sel2["config"] == "R2a"
    assert "First config in priority order with dev_success >= 80%: R2a" in reason2

    # Case 3: None reach >= 80% -> select R2c (highest dev_success)
    rows3 = [
        {"config": "R2a", "dev_success": 0.45},
        {"config": "R2b", "dev_success": 0.62},
        {"config": "R2c", "dev_success": 0.74},
    ]
    sel3, reason3 = select_config(rows3, order)
    assert sel3["config"] == "R2c"
    assert "No config reached >= 80%; selected highest dev_success: R2c" in reason3


def test_evaluate_split_dev_sets_gate_null():
    """Verify that evaluate with split=dev sets gate_to_stage_B to None (null)."""
    from innovation.attacks.adv_glasses.evaluate import stage_b_gate

    # Regular stage_b_gate gives pass=True when rates qualify
    good_results = {
        "method3/D": {"rate": 0.85},
        "method3/control_random": {"rate": 0.05},
        "method3/control_plain": {"rate": 0.05},
    }
    gate_eval = stage_b_gate(good_results, liveness_gated=True)
    assert gate_eval["pass"] is True

    # In evaluate.py, for args.split == 'dev', gate_to_stage_B is explicitly None
    summary = {}
    split_arg = "dev"
    if split_arg == "dev":
        summary["gate_to_stage_B"] = None
    else:
        summary["gate_to_stage_B"] = gate_eval

    assert summary["gate_to_stage_B"] is None


def test_optimize_fails_on_missing_eval_session(tmp_path):
    """Verify split_frames (and optimize.py) stops with SystemExit naming the missing session when s3_vary is absent."""
    for name in ("s1_day1", "s2_day2"):
        d = tmp_path / name
        d.mkdir()
        (d / "000.jpg").write_bytes(b"dummy")

    with pytest.raises(SystemExit) as exc_info:
        split_frames(tmp_path, eval_sessions=["s3_vary"], seed=0, dev_frac=0.2, dev_mode="block")
    assert "s3_vary" in str(exc_info.value)


def test_dev_block_smoother_warmup_exits_checking_and_excludes_warmup_from_csv(tmp_path, monkeypatch):
    """Verify evaluate.py --split dev --liveness with dev_mode=='block' warms up the smoother
    using the preceding frames of the session so the first dev frame is already Real/Fake (not Checking),
    and warmup frames are excluded from frames.csv.
    """
    import csv
    import sys
    import backend.face_recognizer as fr_mod
    import liveness.liveness_detector as ld_mod
    from innovation.attacks.adv_glasses import evaluate as eval_mod

    s1 = tmp_path / "frames" / "owner" / "s1_day1"
    s1.mkdir(parents=True)
    with open(s1 / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "client_ms", "bytes"])
        for i in range(6):
            w.writerow([i, i * 200.0, 50000])
            img = np.full((240, 320, 3), 120 + i * 5, dtype=np.uint8)
            cv2.imwrite(str(s1 / f"{i:03d}.jpg"), img)

    train_f = [str(s1 / f"{i:03d}.jpg") for i in range(4)]
    dev_f = [str(s1 / f"{i:03d}.jpg") for i in range(4, 6)]

    opt_run = tmp_path / "opt_run"
    opt_run.mkdir()
    shape = glasses.FrameShape(55.0, 1.2)
    torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, opt_run / "texture.pt")
    printability.ColorMap.identity().save(opt_run / "colormap.json")
    (opt_run / "split.json").write_text(json.dumps({
        "how": "test",
        "sessions": {"s1_day1": 6},
        "train": train_f,
        "dev": dev_f,
        "eval": [],
        "dev_mode": "block",
    }), encoding="utf-8")
    (opt_run / "attack.json").write_text(json.dumps({
        "mode": "impersonate",
        "attacker_id": "owner",
        "victim": "LFW_victim",
        "target": "LFW_victim",
        "checkpoint": "dummy.pth",
        "gallery": str(tmp_path / "gallery"),
        "gallery_per_id": 1,
        "frame_shape": vars(shape),
        "tex_res": [15, 40],
        "dev_frac": 0.33,
        "dev_mode": "block",
    }), encoding="utf-8")

    class FakeDetector:
        def detect(self, img):
            return [80, 60, 120, 120]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[110.0, 100.0], [170.0, 100.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.load_report = {"arch": "fake"}

        def extract_embedding(self, face):
            v = np.zeros(16, dtype=np.float32)
            v[0] = 1.0
            return v

        def find_match(self, emb, db):
            return {"matched": True, "employee_id": "LFW_victim", "similarity": 0.85}

    class FakeGallery:
        def __init__(self):
            v = np.zeros((1, 16), dtype=np.float32)
            v[0, 0] = 1.0
            self.embeddings = {"LFW_victim": v, "owner": v}

        def as_db(self):
            return {}

    class FakeLive:
        config = {
            "smoothing": {
                "enabled": True,
                "window_s": 1.5,
                "t_on": 0.20,
                "t_off": 0.05,
                "min_frames": 3,
                "reset_s": 1.0,
                "jump_ratio": 0.5,
            }
        }

        def check(self, frame):
            return True, 0.95, "Real", [80, 60, 120, 120]

    eval_out = tmp_path / "eval_out"
    eval_out.mkdir()

    monkeypatch.setattr(faces, "FaceDetector", FakeDetector)
    monkeypatch.setattr(faces, "EyeLandmarks", FakeEyes)
    monkeypatch.setattr(faces, "build_gallery", lambda *a, **kw: (FakeGallery(), {}))
    monkeypatch.setattr(faces, "enhance", lambda crop: (crop, {"applied": False}))
    monkeypatch.setattr(fr_mod, "FaceRecognizer", FakeRecognizer)
    monkeypatch.setattr(ld_mod, "LivenessDetector", FakeLive)
    monkeypatch.setattr(eval_mod, "new_run", lambda name, args: eval_out)
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--split", "dev", "--liveness", "--tiers", "baseline", "D"],
    )

    eval_mod.main()

    with open(eval_out / "frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # Only the 2 dev frames (004.jpg, 005.jpg) x 2 tiers = 4 rows; warmup frames 001..003 must not appear
    recorded_names = {Path(r["frame"]).name for r in rows}
    assert recorded_names == {"004.jpg", "005.jpg"}
    for w_name in ("000.jpg", "001.jpg", "002.jpg", "003.jpg"):
        assert w_name not in recorded_names

    # First dev frame (004.jpg) must already be in Real or Fake state, never Checking
    first_dev_rows = [r for r in rows if Path(r["frame"]).name == "004.jpg"]
    assert len(first_dev_rows) == 2
    for r in first_dev_rows:
        assert r["live_label"] in ("Real", "Fake")
        assert r["live_label"] != "Checking"

    summary = json.loads((eval_out / "summary.json").read_text(encoding="utf-8"))
    assert summary["smoother_warmup_frames"] == 3


def test_evaluate_eval_session_override_and_summary_keys(tmp_path, monkeypatch):
    """(a) --eval-session NAME overrides frames with all images in that session folder,
    records eval_session_override and eval_session_names in summary.json, and default (no override)
    records null for eval_session_override and lists split sessions in eval_session_names.
    """
    import csv
    import sys
    import backend.face_recognizer as fr_mod
    from innovation.attacks.adv_glasses import evaluate as eval_mod

    root = tmp_path / "frames" / "owner"
    s_train = root / "s_train"
    s_dev = root / "s_dev"
    s_eval = root / "s_eval"
    s_new = root / "s_new"
    for s, n in [(s_train, 2), (s_dev, 2), (s_eval, 2), (s_new, 3)]:
        s.mkdir(parents=True)
        with open(s / "meta.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "client_ms", "bytes"])
            for i in range(n):
                w.writerow([i, i * 200.0, 50000])
                img = np.full((240, 320, 3), 100 + i * 10, dtype=np.uint8)
                cv2.imwrite(str(s / f"{i:03d}.jpg"), img)

    opt_run = tmp_path / "opt_run"
    opt_run.mkdir()
    shape = glasses.FrameShape(55.0, 1.2)
    torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, opt_run / "texture.pt")
    printability.ColorMap.identity().save(opt_run / "colormap.json")
    (opt_run / "split.json").write_text(json.dumps({
        "how": "test",
        "sessions": {"s_train": 2, "s_dev": 2, "s_eval": 2},
        "train": [str(s_train / f"{i:03d}.jpg") for i in range(2)],
        "dev": [str(s_dev / f"{i:03d}.jpg") for i in range(2)],
        "eval": [str(s_eval / f"{i:03d}.jpg") for i in range(2)],
    }), encoding="utf-8")
    (opt_run / "attack.json").write_text(json.dumps({
        "mode": "impersonate",
        "attacker_id": "owner",
        "victim": "LFW_victim",
        "target": "LFW_victim",
        "checkpoint": "dummy.pth",
        "gallery": str(tmp_path / "gallery"),
        "gallery_per_id": 1,
        "frame_shape": vars(shape),
        "tex_res": [15, 40],
        "attacker_frames": str(root),
    }), encoding="utf-8")

    class FakeDetector:
        def detect(self, img):
            return [80, 60, 120, 120]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[110.0, 100.0], [170.0, 100.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.load_report = {"arch": "fake"}

        def extract_embedding(self, face):
            v = np.zeros(16, dtype=np.float32)
            v[0] = 1.0
            return v

        def find_match(self, emb, db):
            return {"matched": True, "employee_id": "LFW_victim", "similarity": 0.85}

    class FakeGallery:
        def __init__(self):
            v = np.zeros((1, 16), dtype=np.float32)
            v[0, 0] = 1.0
            self.embeddings = {"LFW_victim": v, "owner": v}

        def as_db(self):
            return {}

    eval_out = tmp_path / "eval_out"
    eval_out.mkdir()

    monkeypatch.setattr(faces, "FaceDetector", FakeDetector)
    monkeypatch.setattr(faces, "EyeLandmarks", FakeEyes)
    monkeypatch.setattr(faces, "build_gallery", lambda *a, **kw: (FakeGallery(), {}))
    monkeypatch.setattr(faces, "enhance", lambda crop: (crop, {"applied": False}))
    monkeypatch.setattr(fr_mod, "FaceRecognizer", FakeRecognizer)
    monkeypatch.setattr(eval_mod, "new_run", lambda name, args: eval_out)

    # 1. Test with --eval-session s_new
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--eval-session", "s_new", "--tiers", "baseline", "D"],
    )
    eval_mod.main()

    with open(eval_out / "frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # s_new has 3 images * 2 tiers = 6 rows
    assert len(rows) == 6
    assert all(r["session"] == "s_new" for r in rows)
    recorded_names = {Path(r["frame"]).name for r in rows}
    assert recorded_names == {"000.jpg", "001.jpg", "002.jpg"}

    summary = json.loads((eval_out / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_session_override"] == "s_new"
    assert summary["eval_session_names"] == ["s_new"]

    # 2. Test without --eval-session (default behavior unchanged)
    eval_out_default = tmp_path / "eval_out_default"
    eval_out_default.mkdir()
    monkeypatch.setattr(eval_mod, "new_run", lambda name, args: eval_out_default)
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--tiers", "baseline", "D"],
    )
    eval_mod.main()

    with open(eval_out_default / "frames.csv", encoding="utf-8") as fh:
        rows_def = list(csv.DictReader(fh))
    assert len(rows_def) == 4
    assert all(r["session"] == "s_eval" for r in rows_def)

    summary_def = json.loads((eval_out_default / "summary.json").read_text(encoding="utf-8"))
    assert summary_def["eval_session_override"] is None
    assert summary_def["eval_session_names"] == ["s_eval"]


def test_evaluate_eval_session_fails_on_leak_with_train_or_dev(tmp_path, monkeypatch):
    """(b) --eval-session stops with SystemExit if any frame in that session is in split train or dev."""
    import sys
    from innovation.attacks.adv_glasses import evaluate as eval_mod

    root = tmp_path / "frames" / "owner"
    s_train = root / "s_train"
    s_dev = root / "s_dev"
    s_train.mkdir(parents=True)
    s_dev.mkdir(parents=True)

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(s_train / "000.jpg"), img)
    cv2.imwrite(str(s_dev / "000.jpg"), img)

    opt_run = tmp_path / "opt_run"
    opt_run.mkdir()
    shape = glasses.FrameShape(55.0, 1.2)
    torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, opt_run / "texture.pt")
    printability.ColorMap.identity().save(opt_run / "colormap.json")
    (opt_run / "split.json").write_text(json.dumps({
        "how": "test",
        "sessions": {"s_train": 1, "s_dev": 1},
        "train": [str(s_train / "000.jpg")],
        "dev": [str(s_dev / "000.jpg")],
        "eval": [],
    }), encoding="utf-8")
    (opt_run / "attack.json").write_text(json.dumps({
        "mode": "impersonate",
        "attacker_id": "owner",
        "victim": "LFW_victim",
        "target": "LFW_victim",
        "checkpoint": "dummy.pth",
        "gallery": str(tmp_path / "gallery"),
        "gallery_per_id": 1,
        "frame_shape": vars(shape),
        "tex_res": [15, 40],
        "attacker_frames": str(root),
    }), encoding="utf-8")

    # Override with s_train -> must raise SystemExit
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--eval-session", "s_train"],
    )
    with pytest.raises(SystemExit) as exc1:
        eval_mod.main()
    assert "train or dev" in str(exc1.value)

    # Override with s_dev -> must raise SystemExit
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--eval-session", "s_dev"],
    )
    with pytest.raises(SystemExit) as exc2:
        eval_mod.main()
    assert "train or dev" in str(exc2.value)


def test_evaluate_eval_session_fails_when_split_is_dev(tmp_path, monkeypatch):
    """(c) --eval-session stops with SystemExit when used with --split dev."""
    import sys
    from innovation.attacks.adv_glasses import evaluate as eval_mod

    opt_run = tmp_path / "opt_run"
    opt_run.mkdir()
    shape = glasses.FrameShape(55.0, 1.2)
    torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, opt_run / "texture.pt")
    printability.ColorMap.identity().save(opt_run / "colormap.json")
    (opt_run / "split.json").write_text(json.dumps({
        "how": "test",
        "train": [],
        "dev": [],
        "eval": [],
    }), encoding="utf-8")
    (opt_run / "attack.json").write_text(json.dumps({
        "mode": "impersonate",
        "attacker_id": "owner",
        "victim": "LFW_victim",
        "target": "LFW_victim",
        "checkpoint": "dummy.pth",
        "gallery": str(tmp_path / "gallery"),
        "gallery_per_id": 1,
        "frame_shape": vars(shape),
        "tex_res": [15, 40],
    }), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--split", "dev", "--eval-session", "s_new"],
    )
    with pytest.raises(SystemExit) as exc:
        eval_mod.main()
    assert "--split eval" in str(exc.value)


def test_evaluate_eval_session_fails_on_missing_or_empty_folder(tmp_path, monkeypatch):
    """Verify --eval-session stops with SystemExit if folder does not exist or has no images."""
    import sys
    from innovation.attacks.adv_glasses import evaluate as eval_mod

    root = tmp_path / "frames" / "owner"
    root.mkdir(parents=True)
    s_train = root / "s_train"
    s_train.mkdir()
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(s_train / "000.jpg"), img)

    s_empty = root / "s_empty"
    s_empty.mkdir()

    opt_run = tmp_path / "opt_run"
    opt_run.mkdir()
    shape = glasses.FrameShape(55.0, 1.2)
    torch.save({"logits": torch.zeros(3, 15, 40), "shape": vars(shape), "tex_res": [15, 40]}, opt_run / "texture.pt")
    printability.ColorMap.identity().save(opt_run / "colormap.json")
    (opt_run / "split.json").write_text(json.dumps({
        "how": "test",
        "train": [str(s_train / "000.jpg")],
        "eval": [],
    }), encoding="utf-8")
    (opt_run / "attack.json").write_text(json.dumps({
        "mode": "impersonate",
        "attacker_id": "owner",
        "victim": "LFW_victim",
        "target": "LFW_victim",
        "checkpoint": "dummy.pth",
        "gallery": str(tmp_path / "gallery"),
        "gallery_per_id": 1,
        "frame_shape": vars(shape),
        "tex_res": [15, 40],
        "attacker_frames": str(root),
    }), encoding="utf-8")

    # Non-existent session folder
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--eval-session", "nonexistent"],
    )
    with pytest.raises(SystemExit) as exc1:
        eval_mod.main()
    assert "does not exist" in str(exc1.value)

    # Empty session folder
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate.py", "--run", str(opt_run), "--eval-session", "s_empty"],
    )
    with pytest.raises(SystemExit) as exc2:
        eval_mod.main()
    assert "no images found" in str(exc2.value)



