"""
api.py – REST endpoints of the innovation extension, mounted by backend/main.py
only when INNOVATION_ENABLED=1.

- P0: the read-only config, the attempts list and a media probe.
- P1: voice enrolment (start, submit, status, reset).
- L1: frame capture for the attack experiments (research mode only).
"""
from __future__ import annotations

import csv
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from innovation import config as cfg
from innovation import db
from innovation import voice
from innovation.db import list_attempts
from innovation.media import decode_media

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_AUDIO_DIR = PROJECT_ROOT / "data" / "innovation" / "voice"
FRAMES_DIR = PROJECT_ROOT / "data" / "innovation" / "frames"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

router = APIRouter(prefix="/api", tags=["innovation"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.get("/innovation/config")
def get_config():
    return {"research_mode": cfg.research_mode(), **cfg.load_config()}


@router.get("/attempts")
def get_attempts(limit: int = 100, decision: Optional[str] = None, attack_label: Optional[str] = None):
    return list_attempts(min(max(limit, 1), 1000), decision, attack_label)


@router.post("/innovation/media/probe")
async def probe_media(file: UploadFile = File(...)):
    """Decode an uploaded clip and report frames, measured fps and audio, for P0's recorder check."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "The upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The clip is larger than 50 MB.")
    try:
        clip = decode_media(data)
    except Exception as e:  # FFmpeg raises many error types for a bad container
        raise HTTPException(400, f"The clip could not be decoded: {type(e).__name__}: {e}")
    summary = clip.summary()
    min_fps = cfg.load_config()["media"]["min_fps"]
    summary["checks"] = {
        "fps_ok": summary["video"]["fps"] >= min_fps,
        "audio_ok": summary["audio"]["sample_rate"] == 16000 and summary["audio"]["samples"] > 0,
    }
    summary["content_type"] = file.content_type
    summary["bytes"] = len(data)
    print(f"[innovation] media probe: {summary}")
    return summary


# ── Voice enrolment (P1) ────────────────────────────────────────────────────

class EnrollStart(BaseModel):
    employee_id: str
    reset: bool = False


@router.post("/voice/enroll/start")
def voice_enroll_start(body: EnrollStart):
    """Issue the digit strings a person reads to enrol their voice."""
    if not db.employee_exists(body.employee_id):
        raise HTTPException(404, f"No registered person has the ID {body.employee_id}.")
    v = cfg.load_config()["voice"]
    if body.reset:
        db.delete_voice(body.employee_id)
    prompts = []
    for _ in range(v["enroll_samples"]):
        ch = db.create_challenge(uuid.uuid4().hex, body.employee_id, voice.random_digits(v["enroll_digits"]),
                                 v["enroll_window_s"], mode="enroll")
        prompts.append({"challenge_id": ch["id"], "digits": ch["digits"]})
    return {"employee_id": body.employee_id, "prompts": prompts, "samples_needed": v["enroll_samples"],
            "n_samples": len(db.get_voice_embeddings(body.employee_id)),
            "expires_at": prompts and db.get_challenge(prompts[0]["challenge_id"])["expires_at"]}


@router.post("/voice/enroll/submit")
async def voice_enroll_submit(employee_id: str = Form(...), challenge_id: str = Form(...),
                              keep_audio: bool = Form(False), file: UploadFile = File(...)):
    """Check one recording, and store its speaker embedding when it is good enough.

    A recording that fails a quality check is not stored and its prompt stays
    open, so the person can record it again. Whether the right digits were
    spoken is checked once ASR is in place (P2).
    """
    ch = db.get_challenge(challenge_id)
    if not ch or ch["mode"] != "enroll" or ch["candidate_id"] != employee_id:
        raise HTTPException(400, "This prompt does not belong to that person. Start the enrolment again.")
    if ch["used"]:
        raise HTTPException(409, "This prompt was already recorded.")
    if time.time() > ch["expires_at"]:
        raise HTTPException(410, "This prompt has expired. Start the enrolment again.")

    data = await file.read()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "The recording is empty or larger than 50 MB.")
    try:
        clip = decode_media(data, with_video=False)
    except Exception as e:
        raise HTTPException(400, f"The recording could not be decoded: {type(e).__name__}: {e}")

    q = voice.quality(clip.audio)
    if not q.ok:
        return {"ok": False, "quality": q.as_dict(), "n_samples": len(db.get_voice_embeddings(employee_id))}

    t0 = time.perf_counter()
    emb = await run_in_threadpool(voice.get_encoder().embed, clip.audio)
    embed_ms = (time.perf_counter() - t0) * 1000
    if not db.consume_challenge(challenge_id):
        raise HTTPException(409, "This prompt was already recorded.")
    db.save_voice_embedding(employee_id, emb, source=f"enroll:{challenge_id}")
    if keep_audio:
        VOICE_AUDIO_DIR.joinpath(employee_id).mkdir(parents=True, exist_ok=True)
        VOICE_AUDIO_DIR.joinpath(employee_id, f"{challenge_id}.webm").write_bytes(data)
    n = len(db.get_voice_embeddings(employee_id))
    need = cfg.load_config()["voice"]["enroll_samples"]
    return {"ok": True, "quality": q.as_dict(), "n_samples": n, "enrolled": n >= need,
            "embed_ms": round(embed_ms, 1)}


@router.get("/voice/status")
def voice_status_all():
    return {"samples_needed": cfg.load_config()["voice"]["enroll_samples"], "counts": db.voice_counts()}


@router.get("/voice/status/{employee_id}")
def voice_status(employee_id: str):
    n = len(db.get_voice_embeddings(employee_id))
    need = cfg.load_config()["voice"]["enroll_samples"]
    return {"employee_id": employee_id, "n_samples": n, "samples_needed": need, "enrolled": n >= need}


@router.delete("/voice/{employee_id}")
def voice_reset(employee_id: str):
    return {"employee_id": employee_id, "deleted": db.delete_voice(employee_id)}


# ── Frame capture for the L1 experiments (research mode) ────────────────────

@router.post("/innovation/frames")
async def save_frame(person: str = Form(...), session: str = Form(...), index: int = Form(...),
                     client_ms: float = Form(0.0), file: UploadFile = File(...)):
    """Store one captured frame exactly as check-in would send it (640x480 JPEG q0.95).

    Saved as data/innovation/frames/<person>/<session>/<index>.jpg with a
    meta.csv row (index, client time in ms, bytes) so the sequence timing is kept.
    """
    if not cfg.research_mode():
        raise HTTPException(403, "Frame capture needs INNOVATION_RESEARCH_MODE=1.")
    if not SAFE_NAME.match(person) or not SAFE_NAME.match(session):
        raise HTTPException(400, "Person and session may use letters, digits, '-' and '_' only.")
    if not 0 <= index < 100000:
        raise HTTPException(400, "Index out of range.")
    data = await file.read()
    if not data or len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "The frame is empty or larger than 5 MB.")
    folder = FRAMES_DIR / person / session
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{index:04d}.jpg").write_bytes(data)
    meta = folder / "meta.csv"
    new = not meta.exists()
    with open(meta, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["index", "client_ms", "bytes"])
        w.writerow([index, round(client_ms, 1), len(data)])
    return {"saved": f"{person}/{session}/{index:04d}.jpg"}
