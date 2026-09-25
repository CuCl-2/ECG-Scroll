"""Build ECG-Scroll tasks from MIT-BIH Arrhythmia (mitdb), adding a beat-level event type
(ventricular ectopic / PVC) alongside the rhythm (AF) and amplitude (ischemia) events.

Two tasks, both with objective rule-based verifiers:
  * rare_event_search --- locate a single isolated ventricular ectopic beat (PVC) inside a
    ~600 s window that contains exactly one such beat. Answer is a short interval; scored by
    hit_iou. This is the canonical "one short event in a long trace" task.
  * burden_quantification --- report the ventricular-ectopic burden (#PVC / #beats) for the
    whole recording. Scalar; scored by rel_error (kind=pvc_burden).

mitdb samples carry an explicit `prompt` (PVC-specific), consumed by EcgScrollEnv, so the
AF/ischemia prompts are untouched. Writes data/ecgscroll_mitdb.jsonl.
"""
import json
import os

from ecgscroll import paths
from ecgscroll.parse_mitdb import (usable_records, beat_times, pvc_times,
                                    pvc_burden, record_seconds, VENTRICULAR)

PN_DIR = "mitdb"
WIN = 600.0        # rare-event window length (s)
PAD = 0.5          # +/- seconds around the ectopic beat = target interval
TAU = 0.5          # IoU threshold for hit_iou
BURDEN_FLOOR = 0.02

RARE_PROMPT = ("Find the single ventricular ectopic beat (PVC) in this window and report its "
               "time as a short interval [start_s, end_s].")
BURDEN_PROMPT = ("Report the ventricular-ectopic (PVC) burden --- the fraction of heartbeats "
                 "that are premature ventricular contractions --- for the whole recording.")


def build():
    samples, stats = [], {"rare": 0, "burden": 0}
    for rec in usable_records():
        total = record_seconds(rec)
        beats = beat_times(rec)
        vt = pvc_times(rec)

        # ---- burden: ventricular-ectopic fraction, one per record ----
        samples.append({
            "id": f"{rec}-pvcburden", "task": "burden_quantification",
            "record": rec, "pn_dir": PN_DIR, "lead": 0, "t0": 0.0, "t1": round(total, 2),
            "answer": {"value": round(pvc_burden(rec), 4), "kind": "pvc_burden"},
            "verifier": {"type": "rel_error", "params": {"floor": BURDEN_FLOOR}},
            "prompt": BURDEN_PROMPT,
        })
        stats["burden"] += 1

        # ---- rare-event: 600 s windows containing exactly one ventricular ectopic beat ----
        for t in vt:
            t0 = max(0.0, min(t - WIN / 2, total - WIN))
            t1 = min(total, t0 + WIN)
            n_in = sum(1 for u in vt if t0 <= u < t1)
            if n_in != 1:
                continue
            samples.append({
                "id": f"{rec}-rare-{int(t)}", "task": "rare_event_search",
                "record": rec, "pn_dir": PN_DIR, "lead": 0,
                "t0": round(t0, 2), "t1": round(t1, 2),
                "answer": {"interval": [round(t - PAD, 2), round(t + PAD, 2)]},
                "verifier": {"type": "hit_iou", "params": {"tau": TAU}},
                "prompt": RARE_PROMPT,
            })
            stats["rare"] += 1
    return samples, stats


if __name__ == "__main__":
    os.makedirs(paths.TASKS, exist_ok=True)
    out = paths.task_path("ecgscroll_mitdb.jsonl")
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
