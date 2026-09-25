"""Assemble ECG-Scroll task samples (JSONL) from parsed afdb ground truth.

Each recording is chunked into fixed windows; tasks are emitted with objective
ground truth and verifier parameters. Tiles are NOT pre-rendered here -- the agent
environment renders them lazily on navigation. Deterministic (no randomness).

Task schema (one JSON object per line):
  {id, task, record, pn_dir, lead, t0, t1, answer:{...}, verifier:{type, params}}
"""
import json
import os
from ecgscroll import paths
from ecgscroll.parse_afdb import usable_records, get_rhythm_segments, af_burden

PN_DIR = "afdb"
WIN = 600.0          # window length (s) for episode-detection tasks
CTX = 300.0          # +/- context (s) around a change-point
TAU = 0.5            # IoU match threshold
BURDEN_FLOOR = 0.02  # relative-error score floor for burden
DELTA = 10.0         # change-point tolerance (s)
RARE_MAX_AF_S = 120.0  # rare-event: total AF in record below this and few episodes


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def af_intervals(record):
    segs, fs, n = get_rhythm_segments(record, PN_DIR)
    af = [(g["start_t"], g["end_t"]) for g in segs if "(" + g["rhythm"] in {"(AFIB", "(AFL"}]
    return af, n / fs


def build():
    samples = []
    stats = {"episode": 0, "episode_pos": 0, "burden": 0, "change": 0, "rare": 0}
    for rec in usable_records(PN_DIR):
        af, total = af_intervals(rec)

        # ---- burden: one per record (long-horizon; needs whole-record coverage) ----
        b, _, _ = af_burden(rec, PN_DIR)
        samples.append({
            "id": f"{rec}-burden", "task": "burden_quantification",
            "record": rec, "pn_dir": PN_DIR, "lead": 0, "t0": 0.0, "t1": total,
            "answer": {"value": round(b, 4), "kind": "af_burden"},
            "verifier": {"type": "rel_error", "params": {"floor": BURDEN_FLOOR}},
        })
        stats["burden"] += 1

        # ---- episode detection: per WIN window (keep windows overlapping AF) ----
        n_win = int(total // WIN) + 1
        for w in range(n_win):
            wt0, wt1 = w * WIN, min((w + 1) * WIN, total)
            if wt1 - wt0 < 30:
                continue
            ints = [[round(max(a, wt0), 2), round(min(bnd, wt1), 2)]
                    for a, bnd in af if _overlap(a, bnd, wt0, wt1) > 0]
            stats["episode"] += 1
            if ints:
                stats["episode_pos"] += 1
            samples.append({
                "id": f"{rec}-ep-{w}", "task": "episode_detection",
                "record": rec, "pn_dir": PN_DIR, "lead": 0, "t0": wt0, "t1": wt1,
                "answer": {"intervals": ints},
                "verifier": {"type": "temporal_f1", "params": {"tau": TAU}},
            })

        # ---- change detection: one per AF onset (windowed context) ----
        segs, fs, n = get_rhythm_segments(rec, PN_DIR)
        for i in range(1, len(segs)):
            prev_af = "(" + segs[i - 1]["rhythm"] in {"(AFIB", "(AFL"}
            cur_af = "(" + segs[i]["rhythm"] in {"(AFIB", "(AFL"}
            if cur_af and not prev_af:
                t = segs[i]["start_t"]
                samples.append({
                    "id": f"{rec}-cp-{i}", "task": "change_detection",
                    "record": rec, "pn_dir": PN_DIR, "lead": 0,
                    "t0": max(0.0, t - CTX), "t1": min(total, t + CTX),
                    "answer": {"change_point": round(t, 2), "kind": "af_onset"},
                    "verifier": {"type": "cp_tolerance", "params": {"delta": DELTA}},
                })
                stats["change"] += 1

        # ---- rare-event search: records with little, short AF ----
        total_af = sum(bnd - a for a, bnd in af)
        if 0 < total_af <= RARE_MAX_AF_S and len(af) <= 3:
            a, bnd = min(af, key=lambda p: p[1] - p[0])  # shortest episode
            samples.append({
                "id": f"{rec}-rare", "task": "rare_event_search",
                "record": rec, "pn_dir": PN_DIR, "lead": 0, "t0": 0.0, "t1": total,
                "answer": {"interval": [round(a, 2), round(bnd, 2)]},
                "verifier": {"type": "hit_iou", "params": {"tau": TAU}},
            })
            stats["rare"] += 1

    return samples, stats


if __name__ == "__main__":
    os.makedirs(paths.TASKS, exist_ok=True)
    out = paths.task_path("ecgscroll_afdb.jsonl")
    samples, stats = build()
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    by_task = {}
    for s in samples:
        by_task[s["task"]] = by_task.get(s["task"], 0) + 1
    print(f"wrote {len(samples)} samples -> {out}")
    for k, v in sorted(by_task.items()):
        print(f"  {k:>22}: {v}")
    print(f"  episode windows with AF: {stats['episode_pos']}/{stats['episode']}")
