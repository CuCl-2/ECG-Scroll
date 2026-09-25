"""ECG-Scroll data pipeline: parse Long-Term AF Database (ltafdb) rhythm annotations.

ltafdb uses the same rhythm-segment convention as afdb: the .atr aux_note strings
'(AFIB' / '(N' / '(VT' / ... at symbol '+' mark the START of a rhythm segment that runs
until the next such annotation (or end of record). Beat symbols ('N','V','A',...) and
non-rhythm markers (e.g. '"'/'\x01 Aux') are ignored here -- we only need rhythm.

Records: 84 recordings, ~24h each at fs=128 Hz. Only AF ((AFIB) is treated as positive;
(AFL is not observed here (unlike afdb), and (VT is a ventricular co-annotation, not AF.

Local-first: reads .hea/.atr from data/raw/ltafdb_local/ (offline cephfs cache).
"""
import glob
import os
import wfdb

from ecgscroll import paths

PN_DIR = "ltafdb"
AF_LABELS = {"(AFIB", "(AFL"}
LOCAL = paths.raw_path(PN_DIR)


def _rec_path(record):
    return os.path.join(LOCAL, record)


def _rdann(record):
    return wfdb.rdann(_rec_path(record), "atr")


def _rdheader(record):
    return wfdb.rdheader(_rec_path(record))


def get_rhythm_segments(record):
    """Return (segments, fs, sig_len) with segments = list of dicts:
    {rhythm, start_s, end_s, start_t, end_t}.

    Only aux_notes starting with '(' are treated as rhythm markers (there are also
    beat-symbol annotations and a stray '\\x01 Aux' marker that we must skip).
    """
    ann = _rdann(record)
    hdr = _rdheader(record)
    fs, n = ann.fs, hdr.sig_len
    marks = []
    for s, aux in zip(ann.sample, ann.aux_note):
        a = (aux or "").replace("\x00", "").strip()
        if a.startswith("("):
            marks.append((int(s), a))
    segs = []
    for i, (s, aux) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n
        segs.append({
            "rhythm": aux.lstrip("("),
            "start_s": s, "end_s": end,
            "start_t": s / fs, "end_t": end / fs,
        })
    return segs, fs, n


def af_episodes(record):
    """AF episode intervals [(t0,t1), ...] in seconds (merged across adjacent (AFIB segments)."""
    segs, _, _ = get_rhythm_segments(record)
    raw = [(g["start_t"], g["end_t"]) for g in segs if "(" + g["rhythm"] in AF_LABELS]
    if not raw:
        return []
    raw.sort()
    out = [list(raw[0])]
    for a, b in raw[1:]:
        if a <= out[-1][1] + 1e-6:  # adjacent/overlapping -> merge
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def af_burden(record):
    """(burden, af_seconds, total_seconds).  burden = time_in_AF / total."""
    segs, fs, n = get_rhythm_segments(record)
    total = n / fs
    if total == 0:
        return None, 0.0, 0.0
    af = sum(g["end_t"] - g["start_t"] for g in segs if "(" + g["rhythm"] in AF_LABELS)
    return af / total, af, total


def record_seconds(record):
    hdr = _rdheader(record)
    return hdr.sig_len / hdr.fs


def usable_records():
    """Records with .hea, .dat, and .atr all present locally."""
    heas = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(LOCAL, "*.hea")))
    return [r for r in heas
            if os.path.exists(_rec_path(r) + ".dat")
            and os.path.exists(_rec_path(r) + ".atr")]


if __name__ == "__main__":
    recs = usable_records()
    print(f"{len(recs)} local ltafdb records")
    for rec in recs[:8]:
        b, af_s, tot = af_burden(rec)
        eps = af_episodes(rec)
        b_str = f"{b:.3f}" if b is not None else "n/a"
        print(f"  {rec}: {tot/3600:.1f}h, {len(eps)} AF episodes, AF={af_s/60:.1f}min, "
              f"burden={b_str}")
