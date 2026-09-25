"""ECG-Scroll data pipeline: parse European ST-T Database (edb) ST-episode annotations.

edb .atr annotations mark ischemic ST episodes with symbol 's' and aux_note strings
(the standard European ST-T protocol):
  '(STx±'      -> onset of an ST episode in signal x (0-based), sign ± (+ elevation / - depression)
  'ASTx±VVV'   -> extremum marker; VVV = peak ST deviation magnitude in microvolts
  'STx±)'      -> offset of the ST episode
(There are analogous T-wave episodes '(Tx±' ... which we ignore; ischemia = ST segment.)

From matched onset/offset pairs we derive ground truth for the ischemia-localization task:
  each episode -> {lead, sign, start_t, end_t, st_mv}  where st_mv = VVV / 1000 (mV),
  and lead is the 0-based signal index (confirmed: annotation digits span {0,1}).
"""
import os
import re
import wfdb

from ecgscroll import paths

PN_DIR = "edb"
# matches '(ST0-', 'AST1+200', 'ST0-)', etc.
_ST_RE = re.compile(r"^\(?A?ST(\d)([+-])(\d+)?\)?$")


def _local(record, pn_dir):
    return paths.raw_path(pn_dir, record)


def _rdann(record, pn_dir=PN_DIR):
    """Read annotations, preferring the local cache in data/{pn_dir}_local/."""
    lp = _local(record, pn_dir)
    if os.path.exists(lp + ".atr"):
        return wfdb.rdann(lp, "atr")
    return wfdb.rdann(record, "atr", pn_dir=pn_dir)


def _rdheader(record, pn_dir=PN_DIR):
    lp = _local(record, pn_dir)
    if os.path.exists(lp + ".hea"):
        return wfdb.rdheader(lp)
    return wfdb.rdheader(record, pn_dir=pn_dir)


def st_episodes(record, pn_dir=PN_DIR):
    """Return matched ST episodes as list of dicts:
    {lead:int, sign:str, start_t:float, end_t:float, st_mv:float}.

    Each '(STx±' opens an episode for the (signal, sign) key; the enclosed 'ASTx±VVV'
    gives the peak deviation (kept as max magnitude); 'STx±)' closes it.
    """
    ann = _rdann(record, pn_dir)
    fs = ann.fs
    open_ep = {}   # (sig, sign) -> {start_t, peak_uv}
    eps = []
    for s, aux in zip(ann.sample, ann.aux_note):
        a = (aux or "").replace("\x00", "").strip()
        if not a:
            continue
        m = _ST_RE.match(a)
        if not m:
            continue
        sig, sign, val = int(m.group(1)), m.group(2), m.group(3)
        key = (sig, sign)
        if a.startswith("("):                       # onset '(STx±'
            open_ep[key] = {"start_t": s / fs, "peak_uv": None}
        elif a.startswith("A"):                     # extremum 'ASTx±VVV'
            if key in open_ep and val is not None:
                v = int(val)
                cur = open_ep[key]["peak_uv"]
                open_ep[key]["peak_uv"] = v if cur is None else max(cur, v)
        elif a.endswith(")"):                       # offset 'STx±)'
            st = open_ep.pop(key, None)
            if st is not None:
                eps.append({
                    "lead": sig,                    # 0-based signal index
                    "sign": sign,
                    "start_t": round(st["start_t"], 2),
                    "end_t": round(s / fs, 2),
                    "st_mv": round((st["peak_uv"] or 0) / 1000.0, 3),
                })
    return eps


def record_duration(record, pn_dir=PN_DIR):
    """Total recording length in seconds (from header)."""
    hdr = _rdheader(record, pn_dir)
    return hdr.sig_len / hdr.fs


def has_signal(record, pn_dir=PN_DIR):
    """True if a local .dat exists (needed for rendering / tool measurement)."""
    return os.path.exists(_local(record, pn_dir) + ".dat")


def usable_records(pn_dir=PN_DIR, require_signal=False):
    """edb records that contain at least one ST episode (optionally requiring local signal)."""
    import glob
    atrs = sorted(glob.glob(os.path.join(paths.raw_path(pn_dir), "*.atr")))
    out = []
    for f in atrs:
        rec = os.path.basename(f)[:-4]
        if require_signal and not has_signal(rec, pn_dir):
            continue
        if st_episodes(rec, pn_dir):
            out.append(rec)
    return out


if __name__ == "__main__":
    import sys
    rec = sys.argv[1] if len(sys.argv) > 1 else "e0103"
    eps = st_episodes(rec)
    print(f"{rec}: dur={record_duration(rec):.0f}s, {len(eps)} ST episodes")
    for e in eps:
        print(f"  lead={e['lead']} {e['sign']}  [{e['start_t']:.1f},{e['end_t']:.1f}]s  "
              f"peak |ST|={e['st_mv']:.3f} mV")
