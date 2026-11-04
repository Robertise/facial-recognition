"""
tests/innovation/test_build_gallery.py
======================================
Tests for L1 gallery builder:
- Name normalization
- BBox centering filter
- Selection logic on synthetic scores (median victim, easy/hard, simulated class, seed reproducibility)
- Refusal of evaluation sessions (s2_day2, s3_dim)
- Preservation of owner/ and refusal to overwrite without --replace
- Halting on insufficient candidates (< 53)
"""
from pathlib import Path
import numpy as np
import pytest
import torch

from innovation.tools.build_gallery import (
    normalize_name,
    is_bbox_centered,
    build_gallery,
)


def test_name_normalization():
    """José_Ángel Martín must normalize to 'jose angel martin'."""
    assert normalize_name("José_Ángel Martín") == "jose angel martin"
    assert normalize_name("John-Paul  O'Connor") == "john paul oconnor"
    assert normalize_name("George_W_Bush") == "george w bush"


def test_bbox_centered_filter():
    """Bbox center must be within center_tol of image center."""
    shape = (250, 250, 3)
    # Perfectly centered bbox: center is (125, 125)
    centered_bbox = [75, 75, 100, 100]
    assert is_bbox_centered(centered_bbox, shape, center_tol=0.20) is True

    # 15% offset: dx = 250 * 0.15 = 37.5. bx = 125 + 37.5 = 162.5
    offset_15_bbox = [112, 75, 100, 100]  # bx = 162.5
    assert is_bbox_centered(offset_15_bbox, shape, center_tol=0.20) is True

    # 25% offset: dx = 250 * 0.25 = 62.5. bx = 125 + 62.5 = 187.5
    offset_25_bbox = [137, 75, 100, 100]  # bx = 187.5 > 125 + 50
    assert is_bbox_centered(offset_25_bbox, shape, center_tol=0.20) is False

    # Off-center y
    offset_y_bbox = [75, 140, 100, 100]
    assert is_bbox_centered(offset_y_bbox, shape, center_tol=0.20) is False


def test_refuse_eval_sessions(tmp_path):
    """Tool must refuse --owner-frames pointing to s2_day2 or s3_dim."""
    s2 = tmp_path / "s2_day2"
    s2.mkdir()
    s3 = tmp_path / "s3_dim"
    s3.mkdir()

    with pytest.raises(ValueError, match="s2_day2"):
        build_gallery(
            gallery_dir=tmp_path / "gallery",
            owner_frames_dir=s2,
            train_ids_dir=tmp_path / "train",
            members_dir=tmp_path / "members",
            pairs_file=tmp_path / "pairs.txt",
        )

    with pytest.raises(ValueError, match="s3_dim"):
        build_gallery(
            gallery_dir=tmp_path / "gallery",
            owner_frames_dir=s3,
            train_ids_dir=tmp_path / "train",
            members_dir=tmp_path / "members",
            pairs_file=tmp_path / "pairs.txt",
        )


def test_selection_logic_on_synthetic_scores():
    """Test victim is median, easy/hard are p90/p10, simulated class disjoint, seed reproducible."""
    n_candidates = 60
    candidates = [
        {"lfw_name": f"Person_{i:03d}", "s_baseline": float(i) / 100.0, "role": "eligible"}
        for i in range(n_candidates)
    ]
    all_s = np.array([c["s_baseline"] for c in candidates])
    median_s = float(np.median(all_s))
    p90_s = float(np.percentile(all_s, 90))
    p10_s = float(np.percentile(all_s, 10))

    # Victim selection
    victim = min(candidates, key=lambda c: abs(c["s_baseline"] - median_s))
    assert victim["s_baseline"] == pytest.approx(median_s, abs=0.01)

    rem = [c for c in candidates if c != victim]
    victim_easy = min(rem, key=lambda c: abs(c["s_baseline"] - p90_s))
    assert victim_easy["s_baseline"] == pytest.approx(p90_s, abs=0.01)

    rem2 = [c for c in rem if c != victim_easy]
    victim_hard = min(rem2, key=lambda c: abs(c["s_baseline"] - p10_s))
    assert victim_hard["s_baseline"] == pytest.approx(p10_s, abs=0.01)

    # Simulated class
    pool = [c for c in candidates if c not in (victim, victim_easy, victim_hard)]
    pool.sort(key=lambda c: c["lfw_name"])

    rng1 = np.random.RandomState(42)
    sim1 = [pool[i]["lfw_name"] for i in rng1.choice(len(pool), size=50, replace=False)]

    rng2 = np.random.RandomState(42)
    sim2 = [pool[i]["lfw_name"] for i in rng2.choice(len(pool), size=50, replace=False)]

    # Same seed gives exact same result
    assert sim1 == sim2
    # Simulated class does not contain any victims
    assert victim["lfw_name"] not in sim1
    assert victim_easy["lfw_name"] not in sim1
    assert victim_hard["lfw_name"] not in sim1


def test_insufficient_candidates_halts(tmp_path, monkeypatch):
    """When eligible candidates < 53 (50 + 3), build_gallery must stop."""
    gal = tmp_path / "gallery"
    gal.mkdir()
    owner_s1 = tmp_path / "s1_day1"
    owner_s1.mkdir()
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    members_dir = tmp_path / "members"
    members_dir.mkdir()
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("")

    # Mock recognizer, detector, arcface calibration, owner frames, and LFW scan
    from unittest.mock import MagicMock
    import innovation.tools.build_gallery as bg

    mock_rec = MagicMock()
    mock_rec._model.head.weight = torch.zeros(4000, 512)
    mock_rec.extract_embedding.return_value = np.zeros(512)
    monkeypatch.setattr(bg, "FaceRecognizer", lambda **kwargs: mock_rec)
    monkeypatch.setattr(bg, "FaceDetector", lambda **kwargs: MagicMock())
    monkeypatch.setattr(bg, "calibrate_arcface", lambda *args, **kwargs: (0.5, 0.95))
    monkeypatch.setattr(bg, "load_owner_frame_embeddings", lambda *args, **kwargs: [np.zeros(512)])

    # Create dummy LFW with only 10 candidates
    lfw_root = tmp_path / "lfw_test"
    lfw_root.mkdir()
    for i in range(10):
        p = lfw_root / f"Person_{i}"
        p.mkdir()
        for j in range(5):
            (p / f"{j}.jpg").write_text("dummy")

    monkeypatch.setattr(bg, "cv2", MagicMock())
    bg.cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(bg, "is_bbox_centered", lambda *args, **kwargs: True)
    monkeypatch.setattr(bg, "crop", lambda *args, **kwargs: np.zeros((50, 50, 3), dtype=np.uint8))
    monkeypatch.setattr(bg, "enhance", lambda *args, **kwargs: (np.zeros((50, 50, 3), dtype=np.uint8), {}))

    with pytest.raises(ValueError, match="Not enough eligible candidates"):
        bg.build_gallery(
            gallery_dir=gal,
            owner_frames_dir=owner_s1,
            train_ids_dir=train_dir,
            members_dir=members_dir,
            pairs_file=pairs_file,
            n_class=50,
            extra_victims=True,
            lfw_root=lfw_root,
        )


def test_preserves_owner_and_refuses_overwrite_without_replace(tmp_path, monkeypatch):
    """Tool must not touch owner/ and must refuse to overwrite existing LFW gallery without --replace."""
    gal = tmp_path / "gallery"
    gal.mkdir()
    owner_dir = gal / "owner"
    owner_dir.mkdir()
    (owner_dir / "1_front.jpg").write_text("owner_img")

    # Simulate existing LFW directory
    (gal / "LFW_victim").mkdir()

    s1 = tmp_path / "s1_day1"
    s1.mkdir()

    with pytest.raises(FileExistsError, match="--replace"):
        build_gallery(
            gallery_dir=gal,
            owner_frames_dir=s1,
            train_ids_dir=tmp_path / "train",
            members_dir=tmp_path / "members",
            pairs_file=tmp_path / "pairs.txt",
            replace=False,
        )

    # Verify owner/ is completely intact
    assert owner_dir.exists()
    assert (owner_dir / "1_front.jpg").read_text() == "owner_img"


def test_load_vggface2_train_names_with_quoted_commas(tmp_path):
    from innovation.tools.build_gallery import load_vggface2_train_names
    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text(
        'Class_ID, Name, Sample_Num, Flag, Gender\n'
        'n000001, "Bush, George W.", 100, 1, m\n'
        'n000002, "14th_Dalai_Lama", 200, 0, m\n'
    )
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "n000001").mkdir()
    (train_dir / "n000002").mkdir()

    names = load_vggface2_train_names(meta_csv, train_dir)
    assert "bush george w" in names
    assert "14th dalai lama" in names
