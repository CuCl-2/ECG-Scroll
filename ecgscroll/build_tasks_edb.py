"""Assemble ECG-Scroll ischemia (ST-episode) detection tasks from edb ground truth.

Framed exactly like AF episode detection, but the target events are ischemic ST-segment
episodes (a different signal modality: ST-amplitude deviation, not rhythm). Each 2-hour edb
recording is chunked into fixed windows; the ground truth for a window is the set of ST-episode
intervals (union across both leads) overlapping it. Scored by the same objective temporal-F1
verifier (IoU-matched intervals) used for AF -- so the VLM-vs-tool-agent comparison is
apples-to-apples across databases and modalities.

Task schema (one JSON object per line):
  {id, task, record, pn_dir, lead, t0, t1, answer:{intervals:[[s,e],...]}, verifier:{temporal_f1}}
`lead` is -1: the window is rendered with BOTH leads (ST change may appear in either).
"""
import json
import os
from ecgscroll import paths
from ecgscroll.parse_edb import st_episodes, record_duration, usable_records

PN_DIR = "edb"
WIN = 600.0     # window length (s), matching the AF episode-detection task
TAU = 0.5       # IoU match threshold


def _merge(intervals):
    """Union overlapping [t0,t1] intervals -> sorted disjoint list."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def build(require_signal=False):
    samples = []
    stats = {"windows": 0, "positive": 0, "records": 0}
    for rec in usable_records(PN_DIR, require_signal=require_signal):
        total = record_duration(rec, PN_DIR)
        eps = st_episodes(rec, PN_DIR)
        if not eps:
            continue
        stats["records"] += 1
        st_ivs = _merge([[e["start_t"], e["end_t"]] for e in eps])  # union across leads
        n_win = int(total // WIN) + 1
        for w in range(n_win):
            wt0, wt1 = w * WIN, min((w + 1) * WIN, total)
            if wt1 - wt0 < 30:
                continue
            ivs = [[round(max(a, wt0), 2), round(min(b, wt1), 2)]
                   for a, b in st_ivs if _overlap(a, b, wt0, wt1) > 0]
            stats["windows"] += 1
            if ivs:
                stats["positive"] += 1
            samples.append({
                "id": f"{rec}-isch-{w}",
                "task": "ischemia_detection",
                "record": rec, "pn_dir": PN_DIR, "lead": -1,
                "t0": wt0, "t1": wt1,
                "answer": {"intervals": ivs},
                "verifier": {"type": "temporal_f1", "params": {"tau": TAU}},
            })
    return samples, stats


if __name__ == "__main__":
    os.makedirs(paths.TASKS, exist_ok=True)
    out = paths.task_path("ecgscroll_edb.jsonl")
    samples, stats = build()
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} ischemia-detection windows -> {out}")
    print(f"  records: {stats['records']}, windows: {stats['windows']}, "
          f"ST-positive: {stats['positive']}")
