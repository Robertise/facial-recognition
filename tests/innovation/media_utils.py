"""Synthetic clips shaped like browser MediaRecorder output, for the innovation tests."""
import io
from fractions import Fraction
from typing import Optional

import numpy as np
import av


def speechlike(seconds: float, rate: int = 48000, level: float = 0.3, seed: int = 0,
               pause_s: float = 0.5, noise: float = 0.002) -> np.ndarray:
    """Bursts of voiced tone (harmonics with a pitch wobble) separated by quiet pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 20 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voiced = sum(np.sin(k * phase) / k for k in range(1, 6))
    gate = ((t % (0.6 + pause_s)) < 0.6).astype(np.float64) if pause_s > 0 else np.ones_like(t)
    sig = level * voiced / np.max(np.abs(voiced)) * gate + noise * rng.standard_normal(len(t))
    sig[: int(0.3 * rate)] = noise * rng.standard_normal(int(0.3 * rate))  # leading silence
    return sig.astype(np.float32)


def make_webm(seconds: float = 4.0, fps: int = 30, width: int = 640, height: int = 480,
              audio_rate: int = 48000, audio: Optional[np.ndarray] = None, video: bool = True) -> bytes:
    """VP8 video (optional) + Opus audio in webm. Without `audio`, a 440 Hz tone."""
    if audio is None:
        t = np.arange(int(seconds * audio_rate)) / audio_rate
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="webm") as out:
        v = None
        if video:
            v = out.add_stream("libvpx", rate=fps)
            v.width, v.height, v.pix_fmt = width, height, "yuv420p"
        a = out.add_stream("libopus", rate=audio_rate)
        a.layout = "mono"
        if v is not None:
            for i in range(int(seconds * fps)):
                img = np.full((height, width, 3), (i * 7) % 255, np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts, frame.time_base = i, Fraction(1, fps)
                for pkt in v.encode(frame):
                    out.mux(pkt)
        step = 960
        for k in range(0, len(audio) - step + 1, step):
            af = av.AudioFrame.from_ndarray(audio[k:k + step].reshape(1, -1).astype(np.float32), format="flt", layout="mono")
            af.sample_rate, af.pts, af.time_base = audio_rate, k, Fraction(1, audio_rate)
            for pkt in a.encode(af):
                out.mux(pkt)
        for s in ([v] if v is not None else []) + [a]:
            for pkt in s.encode(None):
                out.mux(pkt)
    return buf.getvalue()
