"""
voice_crossval.py – speaker verification across many people, each taking a
turn as the enrolled "owner": the first `--enroll` utterances enrol them, their
other utterances are genuine trials, every other speaker's utterances are
impostor trials. Stands in for the owner-only table until enough of the
owner's own recordings exist, and gives a threshold estimate from many voices.

    python -m innovation.eval.voice_crossval data/public/audiomnist/utterances.csv

Writes scores.csv, per_speaker.csv and summary.json under results/innovation/<run>/.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict

import numpy as np

from innovation.config import load_config
from innovation.eval.metrics import auc, eer, rates_at
from innovation.eval.runs import new_run, write_json
from innovation.media import decode_media


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="utterances.csv with speaker,digits,path")
    ap.add_argument("--enroll", type=int, default=3)
    args = ap.parse_args()

    from innovation.voice import VoiceEncoder, score
    enc = VoiceEncoder()
    run = new_run("voice_crossval", vars(args))

    by_spk: dict[str, list[dict]] = defaultdict(list)
    with open(args.csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_spk[row["speaker"]].append(row)

    emb, ms = {}, []
    for rows in by_spk.values():
        for r in rows:
            t0 = time.perf_counter()
            emb[r["path"]] = enc.embed(decode_media(r["path"], with_video=False).audio)
            ms.append((time.perf_counter() - t0) * 1000)

    trials, per = [], []
    for spk, rows in sorted(by_spk.items()):
        enrolled = [emb[r["path"]] for r in rows[: args.enroll]]
        g = [score(emb[r["path"]], enrolled) for r in rows[args.enroll:]]
        imp = [score(emb[r["path"]], enrolled) for other, orows in by_spk.items() if other != spk for r in orows]
        trials += [("genuine", spk, s) for s in g] + [("impostor", spk, s) for s in imp]
        e, _ = eer(np.array(g), np.array(imp))
        per.append({"speaker": spk, "genuine": len(g), "impostor": len(imp), "genuine_min": min(g),
                    "impostor_max": max(imp), "eer": e})

    with open(run / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["kind", "owner", "score"]); w.writerows(trials)
    with open(run / "per_speaker.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)

    g = np.array([s for k, _, s in trials if k == "genuine"])
    i = np.array([s for k, _, s in trials if k == "impostor"])
    e_rate, e_thr = eer(g, i)
    thr = load_config()["thresholds"]["voice"]
    summary = {"speakers": len(by_spk), "genuine": int(len(g)), "impostor": int(len(i)),
               "pooled_eer": e_rate, "eer_threshold": e_thr, "auc": auc(g, i),
               "at_config_threshold": rates_at(g, i, thr),
               "genuine_mean": float(g.mean()), "genuine_p05": float(np.percentile(g, 5)),
               "impostor_mean": float(i.mean()), "impostor_p99": float(np.percentile(i, 99)),
               "per_speaker_eer_mean": float(np.mean([p["eer"] for p in per])),
               "per_speaker_eer_max": float(np.max([p["eer"] for p in per])),
               "embed_ms_median": float(np.median(ms))}
    write_json(run / "summary.json", summary)
    print(f"results: {run}")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
