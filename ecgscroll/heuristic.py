"""Heuristic tool-agent reference for ECG-Scroll episode detection.

A scripted policy that USES the signal-grounded caliper as intended: given an
observation budget B, it splits the window into B segments, measures R-R variability
(rr_cv) in each via the caliper, flags high-variability segments as atrial fibrillation,
and merges adjacent flags into intervals. This is not an LLM -- it is an oracle-tool
reference showing the benchmark is solvable when the tools + planning are used, and
that accuracy RISES with observation budget (unlike the flat zero-shot VLM curve).
"""
import os, json, argparse
import numpy as np
from scipy.signal import find_peaks
from ecgscroll import render, verify, paths

# Set HTTP_PROXY / HTTPS_PROXY in your environment if PhysioNet needs a proxy.

DATA = paths.task_path("ecgscroll_afdb.jsonl")


def build_pool(n_af=30, n_neg=30, only_records=None):
    """Deterministic balanced episode_detection pool over locally-available afdb windows.

    Self-contained (was previously imported from the now-deprecated run_eval.py): reads the
    afdb task JSONL, keeps only episode_detection windows whose record signal is on disk
    (local-first, no PhysioNet streaming), and returns balanced positive/negative splits.
    """
    samples = [json.loads(l) for l in open(DATA)]
    samples = [s for s in samples if s["task"] == "episode_detection"
               and render.has_local(s["record"], s["pn_dir"])]
    if only_records:
        rs = set(only_records)
        samples = [s for s in samples if s["record"] in rs]
    af = [s for s in samples if s["answer"]["intervals"]]
    neg = [s for s in samples if not s["answer"]["intervals"]]
    af = af[:: max(1, len(af) // n_af)][:n_af]
    neg = neg[:: max(1, len(neg) // n_neg)][:n_neg]
    recs = sorted({s["record"] for s in af + neg})
    return af, neg, recs


def rr_cv(record, pn_dir, t0, t1, lead=0):
    sig, names, fs = render.read_window(record, pn_dir, t0, t1, )
    x = sig[:, lead]
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)
    peaks, _ = find_peaks(x, distance=int(0.25 * fs), height=1.0)
    if len(peaks) < 4:
        return 0.0
    rr = np.diff(peaks) / fs
    return float(np.std(rr) / (np.mean(rr) + 1e-6))


def detect(sample, budget, thr=0.20):
    """Split window into `budget` segments, caliper each, flag+merge AF intervals."""
    t0, t1 = sample["t0"], sample["t1"]
    B = max(budget, 1)
    edges = np.linspace(t0, t1, B + 1)
    flags = [rr_cv(sample["record"], sample["pn_dir"], edges[i], edges[i + 1]) > thr
             for i in range(B)]
    intervals, i = [], 0
    while i < B:
        if flags[i]:
            j = i
            while j + 1 < B and flags[j + 1]:
                j += 1
            intervals.append([float(edges[i]), float(edges[j + 1])])
            i = j + 1
        else:
            i += 1
    return {"intervals": intervals}


def run(pool, budgets, thr=0.20):
    n_af = sum(1 for s in pool if s["answer"]["intervals"])
    rows = []
    for B in budgets:
        sc_all, sc_af = [], []
        for k, s in enumerate(pool):
            pred = detect(s, B, thr)
            r = verify.score(s["verifier"], pred, s["answer"])
            sc_all.append(r)
            if s["answer"]["intervals"]:
                sc_af.append(r)
        rows.append({"budget": B, "score_all": float(np.mean(sc_all)),
                     "score_af": float(np.mean(sc_af)), "mean_obs": float(B)})
        print(f"  budget={B:>2}: score_all={rows[-1]['score_all']:.3f} "
              f"score_af={rows[-1]['score_af']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="2,4,6,8,12,20")
    ap.add_argument("--thr", type=float, default=0.20)
    args = ap.parse_args()
    af, neg, recs = build_pool()
    pool = af + neg
    print(f"heuristic tool-agent on {len(af)} AF + {len(neg)} non-AF windows "
          f"(records {sorted(recs)}), thr={args.thr}", flush=True)
    budgets = [int(x) for x in args.budgets.split(",")]
    rows = run(pool, budgets, args.thr)
    os.makedirs(paths.RESULTS, exist_ok=True)
    out = paths.result_path("results_heuristic.json")
    json.dump({"budgets": rows, "thr": args.thr}, open(out, "w"), indent=2)
    print("saved", out, flush=True)
