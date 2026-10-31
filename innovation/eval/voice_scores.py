"""
voice_scores.py – how well the enrolled voice separates its owner from others.

Genuine trials are the owner's clips scored against the owner's enrolled
centroid; impostor trials are other speakers' clips scored against the same
centroid. Reports EER, AUC and FAR/FRR at the configured voice threshold.

    # enrolled voice taken from the database (after enrolling in the app)
    python -m innovation.eval.voice_scores --employee SV001 \\
        --genuine data/innovation/voice/SV001_test/ \\
        --impostors data/public/speech_commands/ data/public/voxceleb1_test/

    # or enrol from files, holding them out of the genuine trials
    python -m innovation.eval.voice_scores --enroll-files a.webm b.webm c.webm --genuine more/

Impostor folders: each sub-folder is one speaker (VoxCeleb id10270/...,
Speech Commands speaker hashes are taken from the file name prefix).
Writes scores.csv and summary.json under results/innovation/<run>/.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np

from innovation.config import load_config
from innovation.eval.metrics import auc, eer, rates_at
from innovation.eval.runs import new_run, write_json
from innovation.media import decode_media

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".mp4", ".mkv"}


def list_audio(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT) if p.is_dir() else [p]
    return out


def speaker_of(path: Path, root: Path) -> str:
    """Folder under the root, or for flat Speech Commands folders the hash before '_nohash_'."""
    m = re.match(r"([0-9a-f]{8})_nohash_", path.name)
    if m:
        return m.group(1)
    rel = path.relative_to(root) if root in path.parents else path
    return rel.parts[0] if len(rel.parts) > 1 else path.stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--employee", help="use this person's enrolled voice from the database")
    src.add_argument("--enroll-files", nargs="+", help="enrol from these files instead")
    ap.add_argument("--genuine", nargs="+", required=True, help="owner clips (files or folders)")
    ap.add_argument("--impostors", nargs="+", required=True, help="folders of other speakers")
    ap.add_argument("--max-per-speaker", type=int, default=20)
    ap.add_argument("--max-speakers", type=int, default=200)
    args = ap.parse_args()

    from innovation.voice import VoiceEncoder, score
    enc = VoiceEncoder()
    run = new_run("voice_scores", vars(args))

    def embed_file(f: Path) -> np.ndarray:
        return enc.embed(decode_media(f, with_video=False).audio)

    if args.employee:
        from innovation.db import get_voice_embeddings
        enrolled = get_voice_embeddings(args.employee)
        if not enrolled:
            raise SystemExit(f"{args.employee} has no enrolled voice in the database")
        enrolled_files = set()
    else:
        enrolled_files = {Path(f).resolve() for f in args.enroll_files}
        enrolled = [embed_file(Path(f)) for f in args.enroll_files]

    rows, embed_ms = [], []
    for f in list_audio(args.genuine):
        if f.resolve() in enrolled_files:
            continue
        t0 = time.perf_counter(); e = embed_file(f); embed_ms.append((time.perf_counter() - t0) * 1000)
        rows.append({"kind": "genuine", "speaker": "owner", "file": str(f), "score": score(e, enrolled)})

    per_speaker: dict[str, int] = {}
    for root in map(Path, args.impostors):
        for f in list_audio([str(root)]):
            spk = speaker_of(f, root)
            if per_speaker.get(spk, 0) >= args.max_per_speaker:
                continue
            if spk not in per_speaker and len(per_speaker) >= args.max_speakers:
                continue
            per_speaker[spk] = per_speaker.get(spk, 0) + 1
            t0 = time.perf_counter(); e = embed_file(f); embed_ms.append((time.perf_counter() - t0) * 1000)
            rows.append({"kind": "impostor", "speaker": f"{root.name}/{spk}", "file": str(f), "score": score(e, enrolled)})

    with open(run / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "speaker", "file", "score"])
        w.writeheader(); w.writerows(rows)

    g = np.array([r["score"] for r in rows if r["kind"] == "genuine"])
    i = np.array([r["score"] for r in rows if r["kind"] == "impostor"])
    if len(g) == 0 or len(i) == 0:
        raise SystemExit("need both genuine and impostor clips")
    e_rate, e_thr = eer(g, i)
    thr = load_config()["thresholds"]["voice"]
    summary = {"genuine": len(g), "impostor": len(i), "impostor_speakers": len(per_speaker),
               "genuine_mean": float(g.mean()), "impostor_mean": float(i.mean()), "impostor_max": float(i.max()),
               "eer": e_rate, "eer_threshold": e_thr, "auc": auc(g, i), "at_config_threshold": rates_at(g, i, thr),
               "embed_ms_median": float(np.median(embed_ms))}
    write_json(run / "summary.json", summary)
    print(f"results: {run}")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()
