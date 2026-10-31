"""
tests/innovation/test_voice.py
==============================
P1 voice enrolment: activity detection, quality checks, scoring and the API
flow, with a fake encoder so no model download is needed.
"""
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")

import backend.database as database
from innovation import voice
from tests.innovation.media_utils import make_webm, speechlike


def resample16(x48: np.ndarray) -> np.ndarray:
    return x48[::3].copy()


# ── VAD and quality ─────────────────────────────────────────────────────────

def test_quality_passes_clean_speech():
    q = voice.quality(resample16(speechlike(4.0)))
    assert q.ok, q.as_dict()
    assert 1.5 < q.speech_s < 3.5 and q.snr_db > 20 and q.clip_fraction == 0


@pytest.mark.parametrize("audio,issue", [
    (resample16(speechlike(1.2)), "TOO_SHORT"),
    (resample16(speechlike(4.0, level=0.003)), "TOO_QUIET"),
    (np.clip(resample16(speechlike(4.0, level=3.0)), -1, 1), "CLIPPING"),
    (resample16(speechlike(4.0, level=0.05, noise=0.03)), "NOISY"),
    (np.zeros(16000 * 3, np.float32), "TOO_SHORT"),
])
def test_quality_flags_each_problem(audio, issue):
    assert issue in voice.quality(audio).issues


def test_speech_only_drops_pauses():
    a = resample16(speechlike(4.0, pause_s=0.6))
    s = voice.speech_only(a)
    assert 0.3 * len(a) < len(s) < 0.7 * len(a)


def test_score_uses_centroid_and_is_cosine():
    rng = np.random.default_rng(1)
    me = rng.standard_normal(192)
    enrolled = [me + 0.1 * rng.standard_normal(192) for _ in range(3)]
    other = rng.standard_normal(192)
    assert voice.score(me, enrolled) > 0.95
    assert abs(voice.score(other, enrolled)) < 0.3
    assert voice.score(me, []) is None


def test_random_digits_shape():
    for _ in range(200):
        d = voice.random_digits(6)
        assert len(d) == 6 and d.isdigit()
        assert all(a != b for a, b in zip(d, d[1:]))


# ── API flow ────────────────────────────────────────────────────────────────

class FakeEncoder:
    """Embedding from the clip's loudness profile: deterministic, no model."""
    def embed(self, audio, sr=16000):
        rng = np.random.default_rng(int(np.abs(audio).sum() * 1000) % (2 ** 32))
        return voice.l2norm(rng.standard_normal(192))


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(database, "DB_PATH", Path(tempfile.mkdtemp()) / "attendance.db")
    monkeypatch.setenv("INNOVATION_ENABLED", "1")
    monkeypatch.setattr(voice, "get_encoder", lambda: FakeEncoder())
    import backend.main as main
    main = importlib.reload(main)
    database.init_db()
    from innovation.db import init_innovation_db
    init_innovation_db()
    database.create_employee("SV001", "Owner")
    yield TestClient(main.app)
    monkeypatch.delenv("INNOVATION_ENABLED")
    importlib.reload(main)


def submit(client, emp, ch, audio48):
    return client.post("/api/voice/enroll/submit",
                       data={"employee_id": emp, "challenge_id": ch},
                       files={"file": ("v.webm", make_webm(4.0, audio=audio48, video=False), "audio/webm")})


def test_enrolment_flow(client):
    start = client.post("/api/voice/enroll/start", json={"employee_id": "SV001"}).json()
    assert len(start["prompts"]) == 3 and all(len(p["digits"]) == 6 for p in start["prompts"])

    # a bad recording is rejected with its issue and the prompt stays open
    p0 = start["prompts"][0]["challenge_id"]
    bad = submit(client, "SV001", p0, speechlike(4.0, level=0.003)).json()
    assert bad["ok"] is False and "TOO_QUIET" in bad["quality"]["issues"]

    for i, p in enumerate(start["prompts"]):
        r = submit(client, "SV001", p["challenge_id"], speechlike(4.0, seed=i)).json()
        assert r["ok"] is True, r
    assert r["enrolled"] is True and r["n_samples"] == 3

    # each prompt counts once
    assert submit(client, "SV001", p0, speechlike(4.0)).status_code == 409
    assert client.get("/api/voice/status/SV001").json()["enrolled"] is True
    assert client.get("/api/voice/status").json()["counts"] == {"SV001": 3}

    # reset clears the samples
    assert client.delete("/api/voice/SV001").json()["deleted"] == 3
    assert client.get("/api/voice/status/SV001").json()["n_samples"] == 0


def test_enrolment_guards(client):
    assert client.post("/api/voice/enroll/start", json={"employee_id": "NOPE"}).status_code == 404
    database.create_employee("SV002", "Someone else")
    start = client.post("/api/voice/enroll/start", json={"employee_id": "SV001"}).json()
    # another person cannot use SV001's prompt
    assert submit(client, "SV002", start["prompts"][0]["challenge_id"], speechlike(4.0)).status_code == 400
    # expired prompt
    from innovation import db
    db.create_challenge("old", "SV001", "123456", 1.0, mode="enroll", now=0)
    assert submit(client, "SV001", "old", speechlike(4.0)).status_code == 410


def test_voice_deleted_with_person(client):
    start = client.post("/api/voice/enroll/start", json={"employee_id": "SV001"}).json()
    submit(client, "SV001", start["prompts"][0]["challenge_id"], speechlike(4.0))
    database.delete_employee("SV001")
    assert client.get("/api/voice/status").json()["counts"] == {}
