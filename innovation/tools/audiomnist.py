"""
audiomnist.py – fetch part of AudioMNIST (60 people reading single English
digits, MIT licence, github.com/soerenab/AudioMNIST) and build challenge-like
utterances: six digits from one speaker, joined with short pauses.

    python -m innovation.tools.audiomnist --speakers 12 --utterances 12

Writes data/public/audiomnist/raw/<speaker>/<digit>_<speaker>_<rep>.wav and
data/public/audiomnist/utterances/<speaker>/<digits>_<k>.wav (16 kHz mono),
plus utterances.csv (speaker, digits, path). data/ is not committed.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np

BASE = "https://raw.githubusercontent.com/soerenab/AudioMNIST/master/data"
OUT = Path(__file__).resolve().parents[2] / "data" / "public" / "audiomnist"
SR = 16000


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        pcm = np.frombuffer(w.readframes(n), dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width])
    x = pcm.astype(np.float32)
    x = (x - 128) / 128 if width == 1 else x / float(2 ** (8 * width - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:  # AudioMNIST is 48 kHz: plain linear resampling is enough for digits
        t = np.arange(int(len(x) * SR / sr)) * sr / SR
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def fetch(speaker: str, digit: int, rep: int) -> Path | None:
    name = f"{digit}_{speaker}_{rep}.wav"
    dest = OUT / "raw" / speaker / name
    if dest.exists():
        return dest
    try:
        data = urlopen(f"{BASE}/{speaker}/{name}", timeout=30).read()
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speakers", type=int, default=12, help="how many speakers (01, 02, ...)")
    ap.add_argument("--reps", type=int, default=6, help="recordings of each digit to fetch per speaker")
    ap.add_argument("--utterances", type=int, default=12, help="six-digit strings to build per speaker")
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    speakers = [f"{i:02d}" for i in range(1, args.speakers + 1)]
    jobs = [(s, d, r) for s in speakers for d in range(10) for r in range(args.reps)]
    with ThreadPoolExecutor(16) as pool:
        got = list(pool.map(lambda j: fetch(*j), jobs))
    print(f"fetched {sum(p is not None for p in got)} of {len(jobs)} digit recordings")

    rng = random.Random(args.seed)
    rows = []
    for s in speakers:
        clips: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
        for d in range(10):
            for r in range(args.reps):
                p = OUT / "raw" / s / f"{d}_{s}_{r}.wav"
                if p.exists():
                    clips[d].append(read_wav(p.read_bytes()))
        if any(not v for v in clips.values()):
            continue
        for k in range(args.utterances):
            digits = [rng.randrange(10)]
            while len(digits) < args.digits:
                d = rng.randrange(10)
                if d != digits[-1]:
                    digits.append(d)
            parts = [np.zeros(int(0.3 * SR), np.float32)]
            for d in digits:
                parts += [rng.choice(clips[d]), np.zeros(int(rng.uniform(0.08, 0.2) * SR), np.float32)]
            parts.append(np.zeros(int(0.3 * SR), np.float32))
            text = "".join(map(str, digits))
            path = OUT / "utterances" / s / f"{text}_{k}.wav"
            write_wav(path, np.concatenate(parts))
            rows.append({"speaker": s, "digits": text, "path": str(path)})

    with open(OUT / "utterances.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["speaker", "digits", "path"])
        w.writeheader(); w.writerows(rows)
    print(f"built {len(rows)} utterances for {len({r['speaker'] for r in rows})} speakers in {OUT / 'utterances'}")


if __name__ == "__main__":
    main()
