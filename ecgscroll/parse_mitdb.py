"""Parse MIT-BIH Arrhythmia (mitdb) beat annotations into ventricular-ectopic (PVC) events.

mitdb records are ~30 min, 2-lead, sampled at 360 Hz, with a beat-level .atr annotation per
QRS. Each beat carries a symbol; ventricular ectopic beats are 'V' (PVC) and 'E' (ventricular
escape). We expose:
  - beat_times(rec):   list of (time_s, symbol) for every annotated beat
  - pvc_times(rec):    times (s) of ventricular ectopic beats
  - pvc_burden(rec):   #ventricular-ectopic / #total beats  (scalar in [0,1])
  - record_seconds(rec)
Local-first: reads from data/mitdb_local/. Verified against wfdb.
"""
import os

import wfdb

from ecgscroll import paths

LOCAL = paths.raw_path("mitdb")
VENTRICULAR = {"V", "E"}  # PVC (premature ventricular contraction) and ventricular escape
# non-beat annotation symbols to ignore when counting beats
NON_BEAT = {"+", "~", "|", "!", "[", "]", '"', "x", "(", ")"}


def _ann(rec):
    return wfdb.rdann(os.path.join(LOCAL, rec), "atr")


def record_seconds(rec):
    hdr = wfdb.rdheader(os.path.join(LOCAL, rec))
    return hdr.sig_len / hdr.fs


def beat_times(rec):
    ann = _ann(rec)
    fs = ann.fs or 360
    out = []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in NON_BEAT:
            continue
        out.append((s / fs, sym))
    return out


def pvc_times(rec):
    return [t for t, sym in beat_times(rec) if sym in VENTRICULAR]


def pvc_burden(rec):
    beats = beat_times(rec)
    if not beats:
        return 0.0
    nv = sum(1 for _, sym in beats if sym in VENTRICULAR)
    return nv / len(beats)


def usable_records():
    import glob
    recs = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(os.path.join(LOCAL, "*.hea")))
    # only records whose signal (.dat) and beat annotation (.atr) are both present
    return [r for r in recs if os.path.exists(os.path.join(LOCAL, r + ".dat"))
            and os.path.exists(os.path.join(LOCAL, r + ".atr"))]


if __name__ == "__main__":
    recs = usable_records()
    print(f"{len(recs)} local mitdb records")
    for rec in recs[:8]:
        pv = pvc_times(rec)
        print(f"  {rec}: {record_seconds(rec):.0f}s, {len(pv)} PVC/vent-ectopic, "
              f"burden={pvc_burden(rec):.3f}")
