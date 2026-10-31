"""
tests/innovation/test_p0.py
===========================
P0 infrastructure: media decoding, the new tables and the feature flag.
"""
import importlib
import io
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

import backend.database as database
from innovation import config as cfg
from innovation.media import decode_media


def make_webm(seconds=4.0, fps=30, width=640, height=480, audio_rate=48000) -> bytes:
    """A clip shaped like browser MediaRecorder output: VP8 video + Opus audio in webm."""
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="webm") as out:
        v = out.add_stream("libvpx", rate=fps)
        v.width, v.height, v.pix_fmt = width, height, "yuv420p"
        a = out.add_stream("libopus", rate=audio_rate)
        a.layout = "mono"
        for i in range(int(seconds * fps)):
            img = np.full((height, width, 3), (i * 7) % 255, np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for pkt in v.encode(frame):
                out.mux(pkt)
        t = np.arange(int(seconds * audio_rate)) / audio_rate
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        step = 960
        for k in range(0, len(tone) - step + 1, step):
            af = av.AudioFrame.from_ndarray(tone[k:k + step].reshape(1, -1), format="flt", layout="mono")
            af.sample_rate, af.pts, af.time_base = audio_rate, k, Fraction(1, audio_rate)
            for pkt in a.encode(af):
                out.mux(pkt)
        for s in (v, a):
            for pkt in s.encode(None):
                out.mux(pkt)
    return buf.getvalue()


@pytest.fixture
def temp_db(monkeypatch):
    path = Path(tempfile.mkdtemp()) / "attendance.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()
    return path


# ── media ───────────────────────────────────────────────────────────────────

def test_decode_webm_frames_fps_and_16k_audio():
    clip = decode_media(make_webm())
    s = clip.summary()
    assert s["video"]["frames"] >= 115
    assert 28 <= s["video"]["fps"] <= 32
    assert (s["video"]["width"], s["video"]["height"]) == (640, 480)
    assert clip.frames[0].dtype == np.uint8 and clip.frames[0].shape == (480, 640, 3)
    assert s["audio"]["sample_rate"] == 16000
    assert abs(s["audio"]["duration_s"] - 4.0) < 0.15
    assert np.all(np.diff(clip.frame_times) > 0)


def test_decode_audio_only_and_empty_container():
    clip = decode_media(make_webm(), with_video=False)
    assert clip.frames == [] and clip.audio_duration_s > 3.8
    with pytest.raises(Exception):
        decode_media(b"not a media file")


# ── tables ──────────────────────────────────────────────────────────────────

def test_innovation_tables_and_attendance_method(temp_db):
    from innovation.db import init_innovation_db, list_attempts
    init_innovation_db()
    init_innovation_db()  # idempotent
    database.create_employee("A", "Alice")
    database.log_attendance("A", "Happy", 0.9, 0.8)
    with database.get_conn() as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        method = conn.execute("SELECT method FROM attendance").fetchone()["method"]
        conn.execute("INSERT INTO voice_embeddings (employee_id, embedding, source) VALUES ('A', x'00', 'test')")
        conn.execute("INSERT INTO verification_attempts (candidate_id, decision, reason_codes) VALUES ('A', 'reject', '[\"VOICE_MISMATCH\"]')")
    assert {"voice_embeddings", "challenges", "verification_attempts"} <= tables
    assert method == "quick"
    assert list_attempts()[0]["reason_codes"] == ["VOICE_MISMATCH"]
    assert list_attempts(decision="accept") == []

    database.delete_employee("A")
    with database.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0
        # research records are kept when the person is deleted
        assert conn.execute("SELECT COUNT(*) FROM verification_attempts").fetchone()[0] == 1


# ── flag and API ────────────────────────────────────────────────────────────

def test_flag_off_leaves_the_api_unchanged(monkeypatch, temp_db):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("INNOVATION_ENABLED", raising=False)
    import backend.main as main
    main = importlib.reload(main)
    client = TestClient(main.app)
    assert client.get("/api/innovation/config").status_code == 404
    assert client.get("/api/attempts").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


def test_flag_on_mounts_router_and_probe_works(monkeypatch, temp_db):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("INNOVATION_ENABLED", "1")
    import backend.main as main
    main = importlib.reload(main)
    from innovation.db import init_innovation_db
    init_innovation_db()

    client = TestClient(main.app)
    conf = client.get("/api/innovation/config").json()
    assert conf["challenge"]["digits"] == 6 and conf["research_mode"] is False
    assert client.get("/api/attempts").json() == []

    r = client.post("/api/innovation/media/probe", files={"file": ("clip.webm", make_webm(), "video/webm")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["checks"] == {"fps_ok": True, "audio_ok": True}
    assert client.post("/api/innovation/media/probe", files={"file": ("x.webm", b"junk", "video/webm")}).status_code == 400

    monkeypatch.delenv("INNOVATION_ENABLED")
    importlib.reload(main)


def test_config_flags(monkeypatch):
    monkeypatch.setenv("INNOVATION_ENABLED", "true")
    monkeypatch.setenv("INNOVATION_RESEARCH_MODE", "0")
    assert cfg.enabled() and not cfg.research_mode()
    assert cfg.load_config()["challenge"]["window_s"] == 8.0
