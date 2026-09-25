"""Streaming tool-agent reference + driver for long-horizon ECG-Scroll (ecgscroll_lh.jsonl).

This is the ONLINE/CAUSAL counterpart of heuristic.py. Instead of splitting a fixed window
into B segments up front, it walks the recording chunk by chunk exactly as the streaming env
delivers it (cursor moves left-to-right, the future is never observed). In each freshly-arrived
chunk it calls the signal-grounded caliper once, and:

  * AF (episode_detection / burden): flags the chunk when R-R coefficient-of-variation exceeds
    a threshold (irregularity = atrial fibrillation cue);
  * ischemia (ischemia_detection): flags when |ST deviation| in either lead exceeds a threshold.

A flagged chunk is immediately written to the findings ledger with its cursor timestamp, so
detection LATENCY (onset -> first record) is measured. At end of stream, contiguous flagged
chunks are merged into predicted intervals (episode/ischemia) or aggregated into a burden
scalar, and scored by the SAME rule-based verifier as the batch tasks. No GPU / no VLM;
runs fully offline on CPU from locally-cached signals.

Usage:
    python -m ecgscroll.run_stream --data ecgscroll_lh.jsonl --n 20 --chunk_s 30
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import find_peaks

from ecgscroll import render, verify, paths, tools
from ecgscroll.env import EcgScrollEnv

# Downloading records from PhysioNet may require a proxy on some networks;
# set HTTP_PROXY / HTTPS_PROXY in your environment if needed (no default here).


def _rr_cv(meas):
    v = meas.get("rr_cv")
    return float(v) if v is not None else 0.0


def _abs_st(meas):
    """Max |ST deviation| across the two leads reported by the caliper."""
    vals = [abs(meas.get(f"st_mv_lead{L}", 0.0) or 0.0) for L in (0, 1)]
    return max(vals) if vals else 0.0


# tasks that flag on RR irregularity (AF cue), ST level (ischemia), or ectopy (PVC)
_AF_TASKS = {"episode_detection", "burden_quantification", "change_detection"}
_ST_TASKS = {"ischemia_detection"}
_PVC_TASKS = {"rare_event_search"}  # + pvc-burden handled by kind below


def _ectopy_frac(sample, c0, c1, fs, lead):
    """Fraction of wide/ectopic beats in [c0,c1] via the signal-grounded morphology tool."""
    m = tools.get_beat_morphology(sample["record"], sample["pn_dir"], c0, c1, fs, lead, "cooked")
    return float(m.get("wide_beat_frac", 0.0) or 0.0), int(m.get("ectopic_count", 0) or 0)


def stream_agent(sample, chunk_s=30.0, af_thr=0.20, st_thr=0.10, pvc_thr=0.15, dx=False):
    """Run the causal tool-agent over one recording; return (pred_answer, env).

    Each freshly-arrived chunk is measured (caliper reads only elapsed signal <= cursor) and
    flagged by the task-appropriate cue: RR irregularity (AF / change), ST level (ischemia),
    or ectopic-beat fraction (rare-event / PVC burden). With dx=True the chunk is instead
    flagged on the diagnostic tool's ready-made label (the +dx arm). Flagged chunks are written
    online (so latency is measured) and merged into the task's answer schema at end of stream."""
    task = sample["task"]
    kind = sample.get("answer", {}).get("kind", "")
    is_st = task in _ST_TASKS
    is_pvc = task in _PVC_TASKS or kind == "pvc_burden"
    fs = render.get_signal(sample["record"], sample["pn_dir"])[1]
    lead = sample.get("lead", 0) or 0
    feature = "st_level" if is_st else ("ectopy" if is_pvc else "rr_irregularity")
    env = EcgScrollEnv(sample, budget=10**9, streaming=True, chunk_s=chunk_s,
                       render_tiles=False)
    env.reset()

    flagged = []              # (t0, t1) chunks flagged as event
    strengths = []            # per-flagged-chunk cue strength (for rare-event: pick the peak)
    etype = "ISCH" if is_st else ("PVC" if is_pvc else "AFIB")
    rec, pn = sample["record"], sample["pn_dir"]
    while True:
        c1 = env.cursor
        c0 = max(sample["t0"], c1 - chunk_s)
        _, _, _, info = env.step({"op": "caliper", "t0": c0, "t1": c1,
                                  "lead": lead})
        meas = info.get("measurement", {})
        if dx:  # +dx arm: flag on the diagnostic tool's ready-made label
            fn = tools.diagnostic_for(task, feature)
            d = fn(rec, pn, c0, c1)
            if is_st:
                hit = bool(d.get("ischemic")); strength = abs(d.get("st_mv", 0.0) or 0.0)
            elif is_pvc:
                strength = len(d.get("pvc_times_s", [])); hit = strength > 0
            else:
                strength = float(d.get("af_frac", 0.0) or 0.0); hit = d.get("dominant_rhythm") == "AF"
        elif is_st:
            strength = _abs_st(meas); hit = strength > st_thr
        elif is_pvc:
            frac, _ = _ectopy_frac(sample, c0, c1, fs, lead); strength = frac; hit = frac > pvc_thr
        else:
            strength = _rr_cv(meas); hit = strength > af_thr
        if hit:
            flagged.append((c0, c1)); strengths.append(strength)
            env.step({"op": "write", "entry": {"type": etype,
                                               "interval": [round(c0, 2), round(c1, 2)]}})
        if env.cursor >= sample["t1"]:
            break
        env.step({"op": "advance", "n": 1})

    # merge contiguous flagged chunks -> intervals
    merged = []
    for a, b in flagged:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    if task == "burden_quantification":
        total = sample["t1"] - sample["t0"]
        flagged_s = sum(b - a for a, b in merged)
        pred = {"value": round(flagged_s / total, 4) if total > 0 else 0.0,
                "kind": kind or "af_burden"}
    elif task == "change_detection":
        # change-point = onset of the FIRST flagged (AF-like) chunk run
        pred = {"change_point": round(merged[0][0], 2) if merged else 0.0,
                "kind": kind or "af_onset"}
    elif task == "rare_event_search":
        # single rare event = the strongest-cue flagged chunk (peak ectopy)
        if flagged:
            i = int(np.argmax(strengths))
            pred = {"interval": [round(flagged[i][0], 2), round(flagged[i][1], 2)]}
        else:
            pred = {"interval": [0.0, 0.0]}
    else:  # episode_detection / ischemia_detection
        pred = {"intervals": [[round(a, 2), round(b, 2)] for a, b in merged]}
    return pred, env


def run(pool, chunk_s=30.0, af_thr=0.20, st_thr=0.10, pvc_thr=0.15, dx=False):
    rows = []
    for k, s in enumerate(pool):
        pred, env = stream_agent(s, chunk_s, af_thr, st_thr, pvc_thr, dx)
        score = verify.score(s["verifier"], pred, s["answer"])
        row = {"id": s["id"], "task": s["task"], "pn_dir": s["pn_dir"],
               "duration_s": round(s["t1"] - s["t0"], 1), "score": round(score, 4)}
        if "intervals" in s["answer"]:
            row["latency"] = env.latency_report()
        if s["task"] == "rare_event_search":
            row["pred_interval"] = pred.get("interval")
            row["gt_interval"] = s["answer"].get("interval")
        rows.append(row)
        lat = row.get("latency", {})
        print(f"  [{k+1}/{len(pool)}] {s['id']:>22} {s['task']:<22} "
              f"score={score:.3f} "
              f"det={lat.get('detected','-')}/{lat.get('n_events','-')} "
              f"lat={lat.get('mean_latency_s','-')}s", flush=True)
    return rows


def load_pool(data, n, task=None, only_local=True):
    samples = [json.loads(l) for l in open(paths.task_path(data))]
    if task:
        samples = [s for s in samples if s["task"] == task]
    if only_local:
        samples = [s for s in samples if render.has_local(s["record"], s["pn_dir"])]
    # deterministic stride sample for a representative, quick subset
    if n and n < len(samples):
        samples = samples[:: max(1, len(samples) // n)][:n]
    return samples


def _summary(rows):
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    out = {}
    for t, rs in by_task.items():
        lats = [r["latency"]["mean_latency_s"] for r in rs if "latency" in r]
        det = [r["latency"]["detected"] for r in rs if "latency" in r]
        nev = [r["latency"]["n_events"] for r in rs if "latency" in r]
        out[t] = {"n": len(rs), "mean_score": _mean([r["score"] for r in rs]),
                  "mean_latency_s": _mean(lats),
                  "recall_events": (round(sum(det) / sum(nev), 4) if sum(nev) else None)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ecgscroll_lh.jsonl")
    ap.add_argument("--task", default=None, help="restrict to one task (default: all in file)")
    ap.add_argument("--n", type=int, default=20, help="number of recordings to sample")
    ap.add_argument("--chunk_s", type=float, default=30.0)
    ap.add_argument("--af_thr", type=float, default=0.20)
    ap.add_argument("--st_thr", type=float, default=0.10)
    ap.add_argument("--pvc_thr", type=float, default=0.15)
    ap.add_argument("--dx", type=int, default=0, help="1 = +dx arm (flag on diagnostic label)")
    ap.add_argument("--out", default="results_stream_tool.json")
    args = ap.parse_args()

    pool = load_pool(args.data, args.n, args.task)
    print(f"streaming tool-agent on {len(pool)} recordings from {args.data} "
          f"(chunk_s={args.chunk_s}s, dx={bool(args.dx)})", flush=True)
    rows = run(pool, args.chunk_s, args.af_thr, args.st_thr, args.pvc_thr, bool(args.dx))
    summ = _summary(rows)
    print("\nsummary by task:")
    for t, d in summ.items():
        print(f"  {t:<22} n={d['n']:<3} mean_score={d['mean_score']} "
              f"event_recall={d['recall_events']} mean_latency={d['mean_latency_s']}s")

    os.makedirs(paths.RESULTS, exist_ok=True)
    out = paths.result_path(args.out)
    json.dump({"data": args.data, "chunk_s": args.chunk_s, "af_thr": args.af_thr,
               "st_thr": args.st_thr, "summary": summ, "rows": rows},
              open(out, "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
