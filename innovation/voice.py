"""
voice.py – speaker embeddings for voice enrolment and 1:1 verification.

- `speech_mask` / `quality`: an energy voice-activity detector and the checks a
  recording must pass (enough speech, loud enough, not clipped, not noisy).
- `VoiceEncoder`: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), 192-dim
  L2-normalised embeddings of the speech-only audio.
- `score`: cosine between a clip and the mean of a person's enrolled embeddings.
  Verification is 1:1 against the candidate face recognition proposed, never
  a search of every enrolled voice.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from innovation.config import load_config

SR = 16000
FRAME_S = 0.03


# ── Voice activity and quality ──────────────────────────────────────────────

def _frame_rms(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    n = max(1, int(FRAME_S * sr))
    usable = len(audio) // n * n
    if usable == 0:
        return np.zeros(0, np.float32)
    frames = audio[:usable].reshape(-1, n)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)


def speech_mask(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Per-frame speech flags (30 ms frames).

    A frame is speech when it is 10 dB above the noise floor (the 10th
    percentile frame) and above an absolute floor of -50 dBFS. Isolated frames
    are dropped and gaps up to 150 ms between speech frames are filled.
    """
    rms = _frame_rms(audio, sr)
    if rms.size == 0:
        return np.zeros(0, bool)
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    floor = np.percentile(db, 10)
    mask = (db > floor + 10) & (db > -50)
    # fill short gaps (≤ 5 frames), then drop isolated single frames
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 6:
            mask[a:b] = True
    for i in range(len(mask)):
        if mask[i] and not ((i > 0 and mask[i - 1]) or (i + 1 < len(mask) and mask[i + 1])):
            mask[i] = False
    return mask


def speech_only(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    mask = speech_mask(audio, sr)
    n = int(FRAME_S * sr)
    if not mask.any():
        return audio[:0]
    return np.concatenate([audio[i * n:(i + 1) * n] for i in np.flatnonzero(mask)])


@dataclass
class Quality:
    duration_s: float
    speech_s: float
    rms_dbfs: Optional[float]
    snr_db: Optional[float]
    clip_fraction: float
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"duration_s": round(self.duration_s, 2), "speech_s": round(self.speech_s, 2),
                "rms_dbfs": None if self.rms_dbfs is None else round(self.rms_dbfs, 1),
                "snr_db": None if self.snr_db is None else round(self.snr_db, 1),
                "clip_fraction": round(self.clip_fraction, 4), "issues": list(self.issues)}


def quality(audio: np.ndarray, sr: int = SR, cfg: Optional[dict] = None) -> Quality:
    """Measure a recording and list what is wrong with it.

    Issue codes: TOO_SHORT (speech under min_speech_s), TOO_QUIET (speech under
    min_rms_dbfs), CLIPPING (more than max_clip_fraction of samples at full
    scale), NOISY (speech less than min_snr_db above the pauses).
    """
    v = (cfg or load_config())["voice"]
    mask = speech_mask(audio, sr)
    rms = _frame_rms(audio, sr)
    speech_s = float(mask.sum() * FRAME_S)
    clip = float(np.mean(np.abs(audio) >= 0.99)) if len(audio) else 0.0
    speech_rms = float(np.sqrt(np.mean(rms[mask] ** 2))) if mask.any() else 0.0
    noise_rms = float(np.sqrt(np.mean(rms[~mask] ** 2))) if (~mask).any() else 0.0
    rms_db = 20 * np.log10(speech_rms) if speech_rms > 0 else None
    snr = 20 * np.log10(speech_rms / max(noise_rms, 1e-6)) if speech_rms > 0 and (~mask).any() else None

    overall = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
    overall_db = 20 * np.log10(overall) if overall > 0 else -120.0

    issues = []
    if speech_s < v["min_speech_s"]:
        issues.append("TOO_SHORT")
    # Little speech found in a clip that is loud overall: noise is drowning the voice.
    drowned = speech_s < v["min_speech_s"] and overall_db >= v["min_rms_dbfs"]
    if (rms_db is None or rms_db < v["min_rms_dbfs"]) and not drowned:
        issues.append("TOO_QUIET")
    if clip > v["max_clip_fraction"]:
        issues.append("CLIPPING")
    if drowned or (snr is not None and snr < v["min_snr_db"]):
        issues.append("NOISY")
    return Quality(len(audio) / sr, speech_s, rms_db, snr, clip, issues)


# ── Embeddings ──────────────────────────────────────────────────────────────

def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2norm(a), l2norm(b)))


class VoiceEncoder:
    """ECAPA-TDNN speaker encoder, loaded on first use."""

    def __init__(self, source: Optional[str] = None, savedir: Optional[str] = None, device: str = "cpu"):
        v = load_config()["voice"]
        self.source = source or v["model"]
        self.savedir = savedir or v["savedir"]
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from speechbrain.inference.speaker import EncoderClassifier
                    from speechbrain.utils.fetching import LocalStrategy
                    self._model = EncoderClassifier.from_hparams(
                        source=self.source,
                        savedir=self.savedir,
                        run_opts={"device": self.device},
                        local_strategy=LocalStrategy.COPY,
                    )
        return self._model

    def embed(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        """192-dim L2-normalised embedding of the speech in `audio` (16 kHz mono float32)."""
        import torch
        if sr != SR:
            raise ValueError(f"expected {SR} Hz audio, got {sr}")
        speech = speech_only(audio, sr)
        if len(speech) < SR // 2:          # too little speech found: use everything
            speech = audio
        with torch.no_grad():
            emb = self._load().encode_batch(torch.from_numpy(np.ascontiguousarray(speech, np.float32)).unsqueeze(0))
        return l2norm(emb.squeeze().cpu().numpy())


_encoder: Optional[VoiceEncoder] = None


def get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


def score(embedding: np.ndarray, enrolled: list[np.ndarray]) -> Optional[float]:
    """Cosine between a clip's embedding and the mean of a person's enrolled embeddings."""
    if not enrolled:
        return None
    centroid = l2norm(np.mean([l2norm(e) for e in enrolled], axis=0))
    return cosine(embedding, centroid)


def random_digits(n: int) -> str:
    """n digits from the OS random source, with no digit repeating the one before it."""
    out = [str(secrets.randbelow(10))]
    while len(out) < n:
        d = str(secrets.randbelow(10))
        if d != out[-1]:            # "4 4" is easy to mishear as one digit
            out.append(d)
    return "".join(out)
