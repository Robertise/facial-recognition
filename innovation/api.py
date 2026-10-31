"""
api.py – REST endpoints of the innovation extension, mounted by backend/main.py
only when INNOVATION_ENABLED=1.

P0 ships the read-only config, the attempts list and a media probe (decode a
recorded clip and report what came out). Enrolment, challenges and
verification arrive in P1 to P3.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from innovation import config as cfg
from innovation.db import list_attempts
from innovation.media import decode_media

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
