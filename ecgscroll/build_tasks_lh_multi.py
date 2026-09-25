"""Whole-recording (long-horizon streaming) task instances for the NON-AF event types:
ischemia (edb), ventricular ectopy / PVC (mitdb + ltdb), and AF change-point (afdb).

Complements build_tasks_lh.py (which covers AF episode+burden on afdb/ltafdb). Same schema:
one instance = the entire recording over [0,T], scored by the same rule-based verifiers and,
in streaming, the detection-latency metric. Local-first (reads data/raw/*_local/).

Emits -> data/tasks/ecgscroll_lh_multi.jsonl:
  * ischemia_detection   (edb)          answer.intervals = union of ST episodes;   temporal_f1
  * pvc_burden           (mitdb, ltdb)  answer.value = #PVC / #beats;              rel_error
  * rare_event_search    (mitdb, ltdb)  answer.interval = the single isolated PVC; hit_iou
                                        (only for records with a genuinely rare PVC count)
  * change_detection     (afdb)         answer.change_point = first AF onset (s);  cp_tolerance
"""
import json
import os

from ecgscroll import paths, parse_edb, parse_mitdb, parse_ltdb
from ecgscroll.build_tasks_lh import af_intervals, usable_records as af_usable

TAU = 0.5
ISCH_TAU = 0.75       # ischemia ST-episode localization: stricter IoU than episode detection.
                      # ST episodes carry clinically meaningful boundaries, so we require a tighter
                      # temporal overlap (0.75) than the AF episode task's 0.5.
BURDEN_FLOOR = 0.02
DELTA = 10.0
RARE_DELTA = 15.0     # rare-event localization tolerance (half a 30s chunk); IoU is unreachable
                      # for a ~1s point event under a 30s streaming protocol, so score by
                      # midpoint tolerance instead (see verify.hit_tolerance).
PAD = 0.5             # +/- s around an isolated PVC = target interval
RARE_MAX_PVC = 30      # a record is a "rare-event" instance if it has <= this many PVCs


def _merge(intervals):
    ivs = sorted([list(v) for v in intervals])
    if not ivs:
        return []
    out = [ivs[0][:]]
    for a, b in ivs[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def build():
    samples, stats = [], {}

    # ---- ischemia detection: whole edb recording ----
    n = 0
    for rec in parse_edb.usable_records(require_signal=True):
        try:
            eps = parse_edb.st_episodes(rec)
            total = parse_edb.record_duration(rec)
        except Exception:
            continue
        if total <= 0:
            continue
        ivs = _merge([[e["start_t"], e["end_t"]] for e in eps])
        samples.append({
            "id": f"edb-{rec}-isch-full", "task": "ischemia_detection",
            "record": rec, "pn_dir": "edb", "lead": -1,
            "t0": 0.0, "t1": round(total, 2), "horizon": "full_record",
            "answer": {"intervals": [[round(a, 2), round(b, 2)] for a, b in ivs]},
            "verifier": {"type": "temporal_f1", "params": {"tau": ISCH_TAU}},
            "meta": {"duration_s": round(total, 1), "n_episodes": len(ivs)},
        })
        n += 1
    stats["edb_ischemia"] = n

    # ---- PVC burden + rare-event: whole mitdb / ltdb recording ----
    for pn, mod, usable in (("mitdb", parse_mitdb, parse_mitdb.usable_records()),
                            ("ltdb", parse_ltdb, parse_ltdb.usable_records())):
        nb = nr = 0
        for rec in usable:
            try:
                total = mod.record_seconds(rec)
                pvt = mod.pvc_times(rec)
                burden = mod.pvc_burden(rec)
            except Exception:
                continue
            if total <= 0:
                continue
            samples.append({
                "id": f"{pn}-{rec}-pvcburden-full", "task": "burden_quantification",
                "record": rec, "pn_dir": pn, "lead": 0,
                "t0": 0.0, "t1": round(total, 2), "horizon": "full_record",
                "answer": {"value": round(burden, 4), "kind": "pvc_burden"},
                "verifier": {"type": "rel_error", "params": {"floor": BURDEN_FLOOR}},
                "meta": {"duration_s": round(total, 1), "n_pvc": len(pvt)},
                "prompt": ("Report the ventricular-ectopic (PVC) burden -- the fraction of "
                           "heartbeats that are premature ventricular contractions -- for the "
                           "whole recording."),
            })
            nb += 1
            # rare-event: only when PVCs are genuinely sparse (one short target to localize)
            if 0 < len(pvt) <= RARE_MAX_PVC:
                t = pvt[0]
                samples.append({
                    "id": f"{pn}-{rec}-rare-full", "task": "rare_event_search",
                    "record": rec, "pn_dir": pn, "lead": 0,
                    "t0": 0.0, "t1": round(total, 2), "horizon": "full_record",
                    "answer": {"interval": [round(t - PAD, 2), round(t + PAD, 2)]},
                    "verifier": {"type": "hit_tolerance", "params": {"delta": RARE_DELTA}},
                    "meta": {"duration_s": round(total, 1), "n_pvc": len(pvt)},
                    "prompt": ("Find the single ventricular ectopic beat (PVC) in this recording "
                               "and report its time as a short interval [start_s, end_s]."),
                })
                nr += 1
        stats[f"{pn}_pvcburden"] = nb
        stats[f"{pn}_rare"] = nr

    # ---- change detection: first AF onset in each afdb recording that has AF ----
    nc = 0
    for rec in af_usable("afdb"):
        try:
            af, total = af_intervals("afdb", rec)
        except Exception:
            continue
        if not af or total <= 0:
            continue
        onset = round(min(a for a, _ in af), 2)
        samples.append({
            "id": f"afdb-{rec}-change-full", "task": "change_detection",
            "record": rec, "pn_dir": "afdb", "lead": 0,
            "t0": 0.0, "t1": round(total, 2), "horizon": "full_record",
            "answer": {"change_point": onset, "kind": "af_onset"},
            "verifier": {"type": "cp_tolerance", "params": {"delta": DELTA}},
            "meta": {"duration_s": round(total, 1)},
        })
        nc += 1
    stats["afdb_change"] = nc
    return samples, stats


if __name__ == "__main__":
    os.makedirs(paths.TASKS, exist_ok=True)
    out = paths.task_path("ecgscroll_lh_multi.jsonl")
    samples, stats = build()
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} multi-task long-horizon samples -> {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    by = {}
    for s in samples:
        by[s["task"]] = by.get(s["task"], 0) + 1
    print("by task:", by)
