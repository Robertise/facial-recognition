"""
media.py – decode a browser clip (webm with VP8/VP9 + Opus, or any container
FFmpeg reads) into BGR frames with their timestamps and 16 kHz mono audio.

Browser MediaRecorder output has a variable frame rate, so the real rate is
measured from the frame timestamps rather than read from the header.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import av
import numpy as np

AUDIO_RATE = 16000

Source = Union[str, Path, bytes, BinaryIO]


@dataclass
class MediaClip:
    frames: list[np.ndarray] = field(default_factory=list)   # BGR uint8, H x W x 3
    frame_times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # seconds
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # mono float32 in [-1, 1]
    sample_rate: int = AUDIO_RATE
    decode_ms: float = 0.0

    @property
    def fps(self) -> float:
        """Measured frame rate: frames over the span of their timestamps."""
        if len(self.frame_times) < 2:
            return 0.0
        span = float(self.frame_times[-1] - self.frame_times[0])
        return (len(self.frame_times) - 1) / span if span > 0 else 0.0

    @property
    def video_duration_s(self) -> float:
        return float(self.frame_times[-1] - self.frame_times[0]) if len(self.frame_times) > 1 else 0.0

    @property
    def audio_duration_s(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    def summary(self) -> dict:
        h, w = self.frames[0].shape[:2] if self.frames else (0, 0)
        rms = float(np.sqrt(np.mean(self.audio ** 2))) if len(self.audio) else 0.0
        return {
            "video": {"frames": len(self.frames), "fps": round(self.fps, 2),
                      "width": w, "height": h, "duration_s": round(self.video_duration_s, 3)},
            "audio": {"sample_rate": self.sample_rate, "samples": int(len(self.audio)),
                      "duration_s": round(self.audio_duration_s, 3),
                      "rms_dbfs": round(20 * np.log10(rms), 1) if rms > 0 else None},
            "decode_ms": round(self.decode_ms, 1),
        }


def _open(source: Source):
    if isinstance(source, (bytes, bytearray)):
        return av.open(io.BytesIO(source))
    if isinstance(source, (str, Path)):
        return av.open(str(source))
    return av.open(source)


def decode_media(source: Source, with_video: bool = True, with_audio: bool = True,
                 audio_rate: int = AUDIO_RATE) -> MediaClip:
    """Decode every video frame and the first audio track (resampled to mono `audio_rate`)."""
    t0 = time.perf_counter()
    clip = MediaClip(sample_rate=audio_rate)
    frames, times, chunks = [], [], []

    with _open(source) as container:
        vstream = container.streams.video[0] if (with_video and container.streams.video) else None
        astream = container.streams.audio[0] if (with_audio and container.streams.audio) else None
        streams = [s for s in (vstream, astream) if s is not None]
        if not streams:
            return clip
        resampler = av.AudioResampler(format="flt", layout="mono", rate=audio_rate) if astream else None

        for packet in container.demux(*streams):
            for frame in packet.decode():
                if packet.stream is vstream:
                    if frame.time is None:
                        continue
                    frames.append(frame.to_ndarray(format="bgr24"))
                    times.append(float(frame.time))
                else:
                    for out in resampler.resample(frame):
                        chunks.append(out.to_ndarray().reshape(-1))
        if resampler is not None:
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray().reshape(-1))

    if frames:
        order = np.argsort(times, kind="stable")
        clip.frames = [frames[i] for i in order]
        clip.frame_times = np.asarray(times, dtype=np.float64)[order]
    if chunks:
        clip.audio = np.clip(np.concatenate(chunks).astype(np.float32), -1.0, 1.0)
    clip.decode_ms = (time.perf_counter() - t0) * 1000
    return clip
