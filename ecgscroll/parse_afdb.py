"""ECG-Scroll data pipeline: parse MIT-BIH AFib (afdb) rhythm annotations.

afdb .atr rhythm annotations use symbol '+' with aux_note in
{'(N','(AFIB','(AFL','(J', ...}. Each annotation marks the START of a rhythm
segment that runs until the next annotation (or end of record).

From these we derive ground truth for three ECG-Scroll tasks:
  - episode detection/localization  -> AF episode intervals (seconds)
  - burden quantification           -> AF burden = time_in_AF / total_time
  - change detection                -> rhythm transition change-points
"""
import os

import wfdb

from ecgscroll import paths

PN_DIR = "afdb"
AF_LABELS = {"(AFIB", "(AFL"}  # atrial fibrillation / flutter


def _local(record, pn_dir):
    return paths.raw_path(pn_dir, record)


def _rdann(record, pn_dir=PN_DIR):
    """Read annotations, preferring the local cache in data/{pn_dir}_local/ (no network)."""
    lp = _local(record, pn_dir)
    if os.path.exists(lp + ".atr"):
        return wfdb.rdann(lp, "atr")
    return wfdb.rdann(record, "atr", pn_dir=pn_dir)


def _rdheader(record, pn_dir=PN_DIR):
    lp = _local(record, pn_dir)
    if os.path.exists(lp + ".hea"):
        return wfdb.rdheader(lp)
    return wfdb.rdheader(record, pn_dir=pn_dir)


def get_rhythm_segments(record, pn_dir=PN_DIR):
    """Return list of dicts: {rhythm, start_s, end_s, start_t, end_t} in samples & seconds."""
    ann = _rdann(record, pn_dir)
    hdr = _rdheader(record, pn_dir)
    fs, n = ann.fs, hdr.sig_len
    marks = [(s, a) for s, a in zip(ann.sample, ann.aux_note) if a]
    segs = []
    for i, (s, aux) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n
        segs.append({
            "rhythm": aux.lstrip("("),
            "start_s": int(s), "end_s": int(end),
            "start_t": s / fs, "end_t": end / fs,
        })
    return segs, fs, n


def af_episodes(record, pn_dir=PN_DIR):
    """AF episode intervals [(t0,t1), ...] in seconds."""
    segs, fs, n = get_rhythm_segments(record, pn_dir)
    return [(g["start_t"], g["end_t"]) for g in segs if "(" + g["rhythm"] in AF_LABELS]


def af_burden(record, pn_dir=PN_DIR):
    """Fraction of total recording time spent in AF."""
    segs, fs, n = get_rhythm_segments(record, pn_dir)
    total = n / fs
    if total == 0:
        return None, 0.0, 0.0  # signal-less record (afdb: 00735, 03665)
    af = sum(g["end_t"] - g["start_t"] for g in segs if "(" + g["rhythm"] in AF_LABELS)
    return af / total, af, total


def usable_records(pn_dir=PN_DIR):
    """afdb records that actually have signal files (excludes annotation-only 00735, 03665)."""
    out = []
    for r in wfdb.get_record_list(pn_dir):
        if wfdb.rdheader(r, pn_dir=pn_dir).sig_len > 0:
            out.append(r)
    return out


def change_points(record, pn_dir=PN_DIR):
    """Rhythm transitions as (time_s, kind) where kind in {af_onset, af_offset, other}."""
    segs, fs, n = get_rhythm_segments(record, pn_dir)
    cps = []
    for i in range(1, len(segs)):
        prev_af = "(" + segs[i - 1]["rhythm"] in AF_LABELS
        cur_af = "(" + segs[i]["rhythm"] in AF_LABELS
        t = segs[i]["start_t"]
        if cur_af and not prev_af:
            cps.append((t, "af_onset"))
        elif prev_af and not cur_af:
            cps.append((t, "af_offset"))
        else:
            cps.append((t, "other"))
    return cps


