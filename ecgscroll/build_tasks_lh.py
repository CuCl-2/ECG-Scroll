"""Assemble ECG-Scroll *long-horizon* task instances (JSONL).

Unlike build_tasks.py (which chunks each recording into fixed 600 s windows), here the
task instance IS the whole recording R over [0, T] -- exactly the POMDP formalization in
the paper (instance = recording R, agent scrolls under a budget B that is far smaller than
the number of tiles needed to cover T). No windowing: one instance per recording per task.

Sources (all rhythm-annotated, same '(AFIB'/'(AFL' segment convention):
  * afdb   -- MIT-BIH AFib, 23 usable recordings, ~10 h each
  * ltafdb -- Long-Term AF DB, 84 recordings, ~24 h each

Tasks emitted (objective, rule-based verifiers -- identical to the windowed benchmark):
  * episode_detection    -- answer = every AF interval in the recording; temporal_f1 (IoU>=tau)
  * burden_quantification-- answer = AF burden (time_in_AF / T) for the recording; rel_error

Local-first: reads .hea/.atr/.dat straight from data/raw/<db>_local/ (offline cephfs cache),
never streaming from PhysioNet -- so it runs in the HF_HUB_OFFLINE evaluation environment.

Task schema (one JSON object per line):
  {id, task, record, pn_dir, lead, t0, t1, horizon, answer:{...}, verifier:{type, params},
   meta:{duration_s, n_episodes}}
"""
import glob
import json
import os

import wfdb

from ecgscroll import paths

AF_LABELS = {"(AFIB", "(AFL"}
TAU = 0.5            # IoU match threshold (matches windowed episode_detection)
BURDEN_FLOOR = 0.02  # relative-error score floor (matches windowed burden)
SOURCES = ("afdb", "ltafdb")


def _rec_path(pn_dir, record):
    return os.path.join(paths.raw_path(pn_dir), record)


def usable_records(pn_dir):
    """Records with .hea, .atr and .dat all present locally (offline-usable).

    Names containing whitespace are download artifacts (a botched multi-record wget
    lands a single file whose name is a space-joined list of record IDs) and are skipped.
    """
    heas = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(os.path.join(paths.raw_path(pn_dir), "*.hea")))
    return [r for r in heas
            if " " not in r
            and os.path.exists(_rec_path(pn_dir, r) + ".atr")
            and os.path.exists(_rec_path(pn_dir, r) + ".dat")]


def _rhythm_segments(pn_dir, record):
    """Rhythm segments from local .atr. Returns (segments, fs, sig_len).

    Both afdb and ltafdb mark rhythm with aux_note strings starting with '(' at the '+'
    symbol; each marks the START of a segment running until the next such marker (or EOR).
    Beat symbols and stray non-rhythm aux markers (e.g. '\\x01 Aux') are skipped.
    """
    lp = _rec_path(pn_dir, record)
    ann = wfdb.rdann(lp, "atr")
    hdr = wfdb.rdheader(lp)
    fs, n = ann.fs, hdr.sig_len
    marks = []
    for s, aux in zip(ann.sample, ann.aux_note):
        a = (aux or "").replace("\x00", "").strip()
        if a.startswith("("):
            marks.append((int(s), a))
    segs = []
    for i, (s, aux) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n
        segs.append({"rhythm": aux, "start_t": s / fs, "end_t": end / fs})
    return segs, fs, n


def _merge(intervals):
    """Union overlapping/adjacent [t0,t1] intervals -> sorted disjoint list."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def af_intervals(pn_dir, record):
    """(merged AF intervals in seconds, total duration s)."""
    segs, fs, n = _rhythm_segments(pn_dir, record)
    total = n / fs if fs else 0.0
    af = _merge([[g["start_t"], g["end_t"]] for g in segs if g["rhythm"] in AF_LABELS])
    return af, total


def build(sources=SOURCES):
    samples = []
    stats = {}
    for pn_dir in sources:
        recs = usable_records(pn_dir)
        db_stat = {"episode": 0, "episode_pos": 0, "burden": 0, "hours": 0.0, "skipped": 0}
        for rec in recs:
            try:
                af, total = af_intervals(pn_dir, rec)
            except Exception as e:  # unreadable/corrupt local file -- skip, don't abort
                print(f"  !! {pn_dir}/{rec}: {e} -- skipped")
                db_stat["skipped"] += 1
                continue
            if total <= 0:
                continue
            db_stat["hours"] += total / 3600.0
            af_s = sum(b - a for a, b in af)
            ivs = [[round(a, 2), round(b, 2)] for a, b in af]

            # ---- episode detection over the whole recording ----
            samples.append({
                "id": f"{pn_dir}-{rec}-ep-full", "task": "episode_detection",
                "record": rec, "pn_dir": pn_dir, "lead": 0,
                "t0": 0.0, "t1": round(total, 2), "horizon": "full_record",
                "answer": {"intervals": ivs},
                "verifier": {"type": "temporal_f1", "params": {"tau": TAU}},
                "meta": {"duration_s": round(total, 1), "n_episodes": len(ivs)},
            })
            db_stat["episode"] += 1
            if ivs:
                db_stat["episode_pos"] += 1

            # ---- burden over the whole recording ----
            samples.append({
                "id": f"{pn_dir}-{rec}-burden-full", "task": "burden_quantification",
                "record": rec, "pn_dir": pn_dir, "lead": 0,
                "t0": 0.0, "t1": round(total, 2), "horizon": "full_record",
                "answer": {"value": round(af_s / total, 4), "kind": "af_burden"},
                "verifier": {"type": "rel_error", "params": {"floor": BURDEN_FLOOR}},
                "meta": {"duration_s": round(total, 1), "n_episodes": len(ivs)},
            })
            db_stat["burden"] += 1
        stats[pn_dir] = db_stat
    return samples, stats


if __name__ == "__main__":
    os.makedirs(paths.TASKS, exist_ok=True)
    out = paths.task_path("ecgscroll_lh.jsonl")
    samples, stats = build()
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} long-horizon samples -> {out}")
    for db, st in stats.items():
        print(f"  {db:>7}: {st['episode']} recs, {st['episode_pos']} AF-positive, "
              f"{st['burden']} burden, {st['hours']:.0f}h total")
