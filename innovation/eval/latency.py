"""
latency.py – time the voice and ASR models on this machine (E4 groundwork).

    python -m innovation.eval.latency                       # synthetic 4 s clip
    python -m innovation.eval.latency --clip my_voice.webm  # a real recording
    python -m innovation.eval.latency --asr tiny.en base.en --repeats 5

Writes results/innovation/<run>/latency.json and prints a table. The first
call of each model includes its download, so load time is reported apart.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from innovation.eval.runs import new_run, write_json
from innovation.media import decode_media


def synthetic_clip(seconds: float = 4.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 6))
    gate = (t % 1.1) < 0.6
    return (0.3 * voiced / np.max(np.abs(voiced)) * gate).astype(np.float32)


def timed(fn, repeats: int) -> tuple[list[float], object]:
    out, times = None, []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times, out


def summary(times: list[float]) -> dict:
    return {"median_ms": round(statistics.median(times), 1), "min_ms": round(min(times), 1),
            "max_ms": round(max(times), 1), "runs": len(times)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", help="audio or video file to use instead of a synthetic 4 s clip")
    ap.add_argument("--asr", nargs="*", default=["tiny.en", "base.en"], help="faster-whisper models to time")
    ap.add_argument("--no-ecapa", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    audio = decode_media(args.clip, with_video=False).audio if args.clip else synthetic_clip()
    run = new_run("latency", vars(args))
    report = {"clip_s": round(len(audio) / 16000, 2), "clip": args.clip or "synthetic"}

    if not args.no_ecapa:
        from innovation.voice import VoiceEncoder
        enc = VoiceEncoder()
        t0 = time.perf_counter(); enc._load(); load_ms = (time.perf_counter() - t0) * 1000
        enc.embed(audio)  # warm-up
        times, emb = timed(lambda: enc.embed(audio), args.repeats)
        report["ecapa"] = {"load_ms": round(load_ms, 1), "dim": int(len(emb)), **summary(times)}

    for name in args.asr:
        from faster_whisper import WhisperModel
        t0 = time.perf_counter(); model = WhisperModel(name, device="cpu", compute_type="int8")
        load_ms = (time.perf_counter() - t0) * 1000

        def run_asr():
            segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
            return " ".join(s.text.strip() for s in segs)

        run_asr()  # warm-up
        times, text = timed(run_asr, args.repeats)
        report[f"asr_{name}"] = {"load_ms": round(load_ms, 1), "text": text, **summary(times)}

    write_json(run / "latency.json", report)
    print(f"\nclip: {report['clip']} ({report['clip_s']} s)   results: {run}")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k:<16} load {v['load_ms']:>8.0f} ms   median {v['median_ms']:>7.0f} ms   "
                  f"min {v['min_ms']:>7.0f}   max {v['max_ms']:>7.0f}" + (f"   text: {v['text']!r}" if "text" in v else ""))


if __name__ == "__main__":
    main()
