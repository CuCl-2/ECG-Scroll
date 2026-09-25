"""ECG-Scroll data pipeline: parse Long-Term ST DB (ltdb) beat annotations.

Note on the name: PhysioNet 'ltdb' is the Long-Term ST Database. Its .atr files are
beat-level (per-QRS symbols); rhythm aux_notes are absent (verified: aux_note is empty
across records). We therefore treat it exactly like mitdb -- expose beat symbols and
derive ventricular-ectopic (PVC + escape) times/burden for rare-event and burden tasks.

Records: 7 recordings, ~21-24h each at fs=128 Hz. Symbols seen include
{'N','V','F','S','J'} (normal, PVC, fusion, supraventricular, junctional). We count
ventricular ectopic beats = {'V','E'} to match parse_mitdb.py's PVC definition.

Local-first: reads from data/raw/ltdb_local/.
"""
import glob
import os
import wfdb

from ecgscroll import paths

PN_DIR = "ltdb"
LOCAL = paths.raw_path(PN_DIR)
VENTRICULAR = {"V", "E"}  # match parse_mitdb.py
NON_BEAT = {"+", "~", "|", "!", "[", "]", '"', "x", "(", ")"}


def _rec_path(rec):
    return os.path.join(LOCAL, rec)


def _ann(rec):
    return wfdb.rdann(_rec_path(rec), "atr")


def record_seconds(rec):
    hdr = wfdb.rdheader(_rec_path(rec))
    return hdr.sig_len / hdr.fs


def beat_times(rec):
    ann = _ann(rec)
    fs = ann.fs or 128
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
    """Records with .hea, .dat, .atr all present locally."""
    recs = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(LOCAL, "*.hea")))
    return [r for r in recs if os.path.exists(_rec_path(r) + ".dat")
            and os.path.exists(_rec_path(r) + ".atr")]


if __name__ == "__main__":
    recs = usable_records()
    print(f"{len(recs)} local ltdb records")
    for rec in recs:
        pv = pvc_times(rec)
        print(f"  {rec}: {record_seconds(rec)/3600:.1f}h, {len(pv)} PVC/vent-ectopic, "
              f"burden={pvc_burden(rec):.4f}")
