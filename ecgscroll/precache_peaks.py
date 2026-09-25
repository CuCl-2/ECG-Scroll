"""Pre-detect and cache R-peaks for every long-horizon record, so experiments read them
instantly instead of re-running XQRS (~1 min per 24h record) on the fly.

Writes data/cache/peaks_<pn_dir>_<record>_lead<L>.npy (peak times in seconds).
tools.py loads these if present; otherwise it detects on demand and caches in memory.

Usage:
    python -m ecgscroll.precache_peaks                 # all records in ecgscroll_lh.jsonl
    python -m ecgscroll.precache_peaks --data ecgscroll_edb.jsonl
"""
import argparse
import json
import os
import time

import numpy as np
from wfdb import processing

from ecgscroll import render, paths

CACHE = os.path.join(paths.DATA, "cache")


def cache_path(record, pn_dir, lead=0):
    return os.path.join(CACHE, f"peaks_{pn_dir}_{record}_lead{lead}.npy")


def detect_record(record, pn_dir, lead=0):
    sig, _, f = render.read_window(record, pn_dir, 0, 10**9, 250)
    x = sig[:, lead if lead >= 0 else 0].astype("float64")
    xq = processing.XQRS(sig=x, fs=f)
    xq.detect(verbose=False)
    return np.asarray(xq.qrs_inds) / f, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ecgscroll_lh.jsonl")
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    seen = set()
    recs = []
    for line in open(paths.task_path(args.data)):
        o = json.loads(line)
        key = (o["record"], o["pn_dir"])
        if key not in seen and render.has_local(o["record"], o["pn_dir"]):
            seen.add(key)
            recs.append(key)
    print(f"{len(recs)} unique local records in {args.data}", flush=True)

    for i, (rec, pn) in enumerate(recs):
        cp = cache_path(rec, pn, args.lead)
        if os.path.exists(cp) and not args.force:
            print(f"  [{i+1}/{len(recs)}] {pn}/{rec}: cached, skip", flush=True)
            continue
        t0 = time.time()
        peaks, f = detect_record(rec, pn, args.lead)
        np.save(cp, peaks)
        print(f"  [{i+1}/{len(recs)}] {pn}/{rec}: {len(peaks)} peaks, fs={f}, "
              f"{time.time()-t0:.1f}s -> {os.path.basename(cp)}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
