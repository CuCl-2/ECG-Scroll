"""Signal-grounded tool-agent reference for ischemia (ST-episode) detection on edb.

Analogue of heuristic.py (the AF R-R-variability tool-agent), but the measured feature is
ST-segment deviation instead of R-R irregularity. Given budget B it splits the window into B
segments, measures ST level (J+80ms vs PR baseline) per segment in BOTH leads, converts to a
per-lead deviation from that lead's median-over-window (removing constant offsets/wander), flags
segments whose |deviation| exceeds a threshold in either lead, merges consecutive flagged
segments, and reports them as predicted ischemic intervals. Scored by temporal-F1 (same verifier
as AF episode detection). Accuracy should rise with B as segment boundaries localize episodes.
No GPU / no VLM.
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import find_peaks

from ecgscroll import render, verify, paths

DATA = paths.task_path("ecgscroll_edb.jsonl")


def st_level(record, pn_dir, t0, t1, lead):
    """Mean ST level (mV) at J+80ms (~R+120ms) relative to a PR baseline (~R-40ms) over beats."""
    sig, names, fs = render.read_window(record, pn_dir, t0, t1)
    if sig is None or sig.shape[0] < int(0.5 * fs) or lead >= sig.shape[1]:
        return 0.0
    x = sig[:, lead].astype(float)
    xn = x - np.nanmedian(x)
    amp = np.nanpercentile(np.abs(xn), 99) + 1e-6
    peaks, _ = find_peaks(xn, distance=int(0.4 * fs), height=0.4 * amp)  # R waves, <=150 bpm
    b, st = int(0.04 * fs), int(0.12 * fs)
    devs = [x[r + st] - x[r - b] for r in peaks if r - b >= 0 and r + st < len(x)]
    return float(np.median(devs)) if devs else 0.0


def detect(sample, budget, thr=0.10):
    """Scan window in `budget` segments x 2 leads; flag segments deviating from the per-lead
    baseline by > thr in either lead; merge consecutive flags into predicted ST intervals."""
    t0, t1 = sample["t0"], sample["t1"]
    B = max(int(budget), 1)
    edges = np.linspace(t0, t1, B + 1)
    rec, pn = sample["record"], sample["pn_dir"]
    per_lead = []
    for lead in (0, 1):
        lv = [st_level(rec, pn, edges[i], edges[i + 1], lead) for i in range(B)]
        base = float(np.median(lv))
        per_lead.append([abs(v - base) for v in lv])   # deviation from this lead's own baseline
    dev = [max(per_lead[0][i], per_lead[1][i]) for i in range(B)]  # strongest lead per segment
    flags = [d > thr for d in dev]
    ivs, i = [], 0
    while i < B:
        if flags[i]:
            j = i
            while j + 1 < B and flags[j + 1]:
                j += 1
            ivs.append([float(edges[i]), float(edges[j + 1])])
            i = j + 1
        else:
            i += 1
    return {"intervals": ivs}


def build_pool(n_pos, n_neg, only_records=None):
    """Deterministic balanced pool over windows with locally-available signal."""
    samples = [json.loads(l) for l in open(DATA)]
    if only_records:
        rs = set(only_records)
        samples = [s for s in samples if s["record"] in rs]
    samples = [s for s in samples
               if render.read_window(s["record"], s["pn_dir"], s["t0"],
                                      min(s["t0"] + 1, s["t1"]))[0] is not None]
    pos = [s for s in samples if s["answer"]["intervals"]]
    neg = [s for s in samples if not s["answer"]["intervals"]]
    pos = pos[:: max(1, len(pos) // n_pos)][:n_pos]
    neg = neg[:: max(1, len(neg) // n_neg)][:n_neg]
    return pos, neg


def run(pool, budgets, thr=0.10):
    n_pos = sum(1 for s in pool if s["answer"]["intervals"])
    rows = []
    for B in budgets:
        sc_all, sc_pos = [], []
        for s in pool:
            r = verify.score(s["verifier"], detect(s, B, thr), s["answer"])
            sc_all.append(r)
            if s["answer"]["intervals"]:
                sc_pos.append(r)
        rows.append({"budget": int(B), "score_all": float(np.mean(sc_all)),
                     "score_pos": float(np.mean(sc_pos)) if sc_pos else 0.0, "mean_obs": float(B)})
        print(f"  B={B}: score_pos={rows[-1]['score_pos']:.3f} score_all={rows[-1]['score_all']:.3f}",
              flush=True)
    return rows


def empty_baseline(pool):
    """predict-empty floor: temporal_f1 of an empty prediction (1.0 on ST-free windows)."""
    return float(np.mean([verify.score(s["verifier"], {"intervals": []}, s["answer"]) for s in pool]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="2,4,6,8,12,20")
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--n_pos", type=int, default=40)
    ap.add_argument("--n_neg", type=int, default=40)
    ap.add_argument("--out", default="results_ischemia_heuristic.json")
    args = ap.parse_args()

    pos, neg = build_pool(args.n_pos, args.n_neg)
    pool = pos + neg
    print(f"pool: {len(pos)} ST-positive + {len(neg)} negative windows")
    floor = empty_baseline(pool)
    print(f"predict-empty floor: {floor:.3f}")
    rows = run(pool, [int(x) for x in args.budgets.split(",")], args.thr)
    out = {"n_pos": len(pos), "n_neg": len(neg), "predict_empty": floor,
           "budgets": rows, "thr": args.thr}
    os.makedirs(paths.RESULTS, exist_ok=True)
    json.dump(out, open(paths.result_path(args.out), "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
