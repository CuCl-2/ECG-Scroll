#!/usr/bin/env python
"""Full 6-task x 2-tier serial runner for the SKIM agent (run_llm_skim) on one server.
Each task gets a skim_max that forces appropriately-fine scanning (small for point/short events,
larger for long records). Writes exp_skim_<mtag>_<task>_<sig|dx>.json.

Usage: python scripts/run_skim_full.py <served> <mtag> [base_url] [workers]
"""
import json, os, subprocess, sys, time

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of scripts/)
PY = os.environ.get("ECGSCROLL_PY", sys.executable)  # override with ECGSCROLL_PY if needed
RESULTS = os.path.join(BENCH, "data/results")

served   = sys.argv[1] if len(sys.argv) > 1 else "qwen3-30b"
mtag     = sys.argv[2] if len(sys.argv) > 2 else "30b"
base_url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000/v1"
workers  = sys.argv[4] if len(sys.argv) > 4 else "8"

# (tname, data, task, n, skim_max) — skim_max tuned per task to force fine-enough scanning
_TASKS = [
    ("episode",   "ecgscroll_lh.jsonl",       "episode_detection",     30, "300"),
    ("afburden",  "ecgscroll_lh.jsonl",       "burden_quantification", 30, "300"),
    ("change",    "ecgscroll_lh_multi.jsonl", "change_detection",      23, "300"),
    ("ischemia",  "ecgscroll_lh_multi.jsonl", "ischemia_detection",    30, "120"),
    ("pvcburden", "ecgscroll_lh_multi.jsonl", "burden_quantification", 30, "120"),
    ("rare",      "ecgscroll_lh_multi.jsonl", "rare_event_search",     13, "60"),
]

JOBS = []
for tname, data, task, n, sm in _TASKS:
    for tier, sfx in (("signal", "sig"), ("signal+dx", "dx")):
        JOBS.append((f"skim_{mtag}_{tname}_{sfx}", data, task, n, sm, tier))

_filter = sys.argv[5] if len(sys.argv) > 5 else None
if _filter:
    JOBS = [j for j in JOBS if _filter in j[0]]

print(f"run_skim_full: served={served} mtag={mtag} jobs={len(JOBS)}", flush=True)
for tag, data, task, n, sm, tier in JOBS:
    out = f"exp_{tag}.json"
    cmd = [PY, "-m", "ecgscroll.run_llm_skim", "--data", data, "--task", task,
           "--n", str(n), "--served", served, "--base_url", base_url,
           "--tool_tier", tier, "--skim_max", sm, "--workers", workers, "--out", out]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=BENCH, no_proxy="localhost,127.0.0.1,::1", http_proxy="", https_proxy="")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=BENCH, env=env, capture_output=True, text=True, timeout=7200)
        dt = time.time() - t0
        fp = os.path.join(RESULTS, out)
        summ = json.dumps(json.load(open(fp))["summary"]) if (p.returncode == 0 and os.path.exists(fp)) \
            else f"FAIL rc={p.returncode}: {p.stderr[-300:]}"
    except subprocess.TimeoutExpired:
        dt = time.time() - t0; summ = "TIMEOUT"
    print(f"[{dt:6.0f}s] {tag} (sm={sm}): {summ}", flush=True)

print(f"ALL SKIM JOBS DONE for {mtag}", flush=True)
