"""ECG-Scroll signal-grounded measurement tools (shared by rule- and LLM-agents).

Every tool reads ONLY the retained signal over an elapsed window [t0,t1] (the caller is
responsible for causal clamping to <= cursor). Tools return compact, JSON-serializable dicts
rendered into the text observation for a language-model agent -- pixels are never shown, so
these measurements are the agent's only "eyes".

R-peaks come from wfdb's XQRS detector (Pan-Tompkins family), run ONCE per record and cached;
tools slice the cached peak train. This is a real signal detector (not the ground-truth beat
annotations, and not a hand-rolled find_peaks): defensible, accurate (validated within ~1-2%
of annotated beats), and fast after the one-time detection. Measurement windows are still
causal -- a tool only ever uses peaks/samples at times <= the window it is given.

Two information levels (env `tool_mode`):
  * "cooked": clinician-style derived quantities (rr_irregularity, st deviation, ...).
  * "raw":    primitive per-beat measurements (rr intervals, st samples, qrs widths).
No tool returns a diagnostic LABEL -- only measurements. The agent turns measurements into a
verifiable answer.
"""
import os

import numpy as np
import wfdb
from wfdb import processing

from ecgscroll import render, paths

MAX_LIST = 60  # cap on raw list lengths returned to the LLM (keep the observation small)

# per-record cache: (record, pn_dir, lead) -> (peak_times_s sorted np.array, fs)
_PEAKS = {}
_CACHE_DIR = os.path.join(paths.DATA, "cache")


def _peaks(record, pn_dir, fs, lead=0):
    """Full-record R-peak train (seconds): from disk cache if present (precache_peaks.py),
    else detect once with XQRS and cache in memory."""
    key = (record, pn_dir, lead)
    if key not in _PEAKS:
        disk = os.path.join(_CACHE_DIR, f"peaks_{pn_dir}_{record}_lead{lead}.npy")
        if os.path.exists(disk):
            _, _, f = render.read_window(record, pn_dir, 0, 1, fs)  # just to get fs
            _PEAKS[key] = (np.load(disk), f)
        else:
            sig, _, f = render.read_window(record, pn_dir, 0, 10**9, fs)  # whole record (local)
            x = sig[:, lead if lead is not None and lead >= 0 else 0].astype("float64")
            xq = processing.XQRS(sig=x, fs=f)
            xq.detect(verbose=False)
            _PEAKS[key] = (np.asarray(xq.qrs_inds) / f, f)
    return _PEAKS[key]


def _peaks_in(record, pn_dir, fs, t0, t1, lead=0):
    pk, f = _peaks(record, pn_dir, fs, lead)
    return pk[(pk >= t0) & (pk < t1)], f


def _read(record, pn_dir, t0, t1, fs):
    sig, names, f = render.read_window(record, pn_dir, t0, t1, fs)
    return sig, f


# ---------------------------------------------------------------- rhythm (AF / change) ----
def _clean_rr(rr_ms):
    """Physiologically clean an RR series before computing variability.

    XQRS (like any R-peak detector) occasionally misses a beat -- merging two RR into one
    that is ~2x too long -- or fires twice, splitting one RR into two that are too short.
    These detector artifacts inflate RR variance and make a REGULAR rhythm look highly
    irregular (a clean 720-816 ms run scored CV=0.94 before cleaning), swamping the true
    scatter of atrial fibrillation. We drop (i) non-physiological RR (<300 or >2000 ms) and
    (ii) isolated spikes that deviate from their LOCAL neighborhood -- an artifact makes one
    RR jump then revert, whereas real AF scatter is sustained, so a local (not global) test
    removes artifacts without flattening genuine AF irregularity. Returns the cleaned array
    (may be shorter); callers should check len>=4 before using."""
    rr = np.asarray(rr_ms, dtype="float64")
    rr = rr[(rr >= 300.0) & (rr <= 2000.0)]
    if len(rr) < 4:
        return rr
    keep = np.ones(len(rr), dtype=bool)
    for i in range(len(rr)):
        lo, hi = max(0, i - 2), min(len(rr), i + 3)
        neigh = np.concatenate([rr[lo:i], rr[i + 1:hi]])
        local = np.median(neigh) if len(neigh) else np.median(rr)
        if rr[i] > 1.8 * local or rr[i] < 0.5 * local:
            keep[i] = False
    return rr[keep]


def get_rhythm(record, pn_dir, t0, t1, fs, lead=0, mode="cooked"):
    pk, f = _peaks_in(record, pn_dir, fs, t0, t1, lead)
    if len(pk) < 4:
        return {"n_beats": int(len(pk)), "note": "too few beats to measure rhythm"}
    rr_ms = np.diff(pk) * 1000.0
    if mode == "raw":
        return {"rr_intervals_ms": [int(v) for v in rr_ms[:MAX_LIST]], "n_beats": int(len(pk))}
    rr_c = _clean_rr(rr_ms)
    if len(rr_c) < 4:  # too little clean signal to characterize rhythm
        return {"n_beats": int(len(pk)), "note": "too few clean beats to measure rhythm"}
    return {"hr_bpm": round(60000.0 / float(np.mean(rr_c)), 1),
            "rr_mean_ms": round(float(np.mean(rr_c)), 1),
            "rr_sd_ms": round(float(np.std(rr_c)), 1),
            "rr_irregularity": round(float(np.std(rr_c) / (np.mean(rr_c) + 1e-6)), 3),
            "n_beats": int(len(pk))}


# ------------------------------------------------------------------------ ST (ischemia) ----
def _st_per_lead(x, peaks_idx, fs):
    """ST level (mV) at J+80ms vs PR baseline, averaged over beats whose R index is given."""
    b, st = int(0.04 * fs), int(0.12 * fs)
    devs = [x[r + st] - x[r - b] for r in peaks_idx if r - b >= 0 and r + st < len(x)]
    return devs


def get_st_level(record, pn_dir, t0, t1, fs, mode="cooked", lead=0, **_):
    sig, f = _read(record, pn_dir, t0, t1, fs)
    if sig is None or sig.shape[0] < int(0.5 * f):
        return {"note": "window too short / no signal"}
    pk, _ = _peaks_in(record, pn_dir, fs, t0, t1, lead)
    idx = ((pk - t0) * f).astype(int)  # peak indices within this window
    out = {}
    for L in range(sig.shape[1]):
        devs = _st_per_lead(sig[:, L].astype(float), idx, f)
        if mode == "raw":
            out[f"st_samples_lead{L}_mv"] = [round(float(v), 3) for v in devs[:MAX_LIST]]
        else:
            out[f"st_mv_lead{L}"] = round(float(np.mean(devs)), 3) if devs else 0.0
    if mode != "raw":
        peak_lead = max(("st_mv_lead0", "st_mv_lead1"), key=lambda k: abs(out.get(k, 0.0)))
        out["sign"] = "elevation" if out.get(peak_lead, 0.0) > 0 else "depression"
    return out


# ------------------------------------------------------------------- beat morphology (PVC) ----
def get_beat_morphology(record, pn_dir, t0, t1, fs, lead=0, mode="cooked", **_):
    sig, f = _read(record, pn_dir, t0, t1, fs)
    if sig is None or sig.shape[0] < int(0.5 * f):
        return {"note": "window too short / no signal"}
    pk, _ = _peaks_in(record, pn_dir, fs, t0, t1, lead)
    if len(pk) < 3:
        return {"n_beats": int(len(pk)), "note": "too few beats"}
    x = sig[:, lead if lead is not None and lead >= 0 else 0]
    xn = np.abs(x - np.nanmedian(x))
    thr = 0.3 * np.nanpercentile(xn, 99)
    half = int(0.12 * f)
    widths_ms, times = [], []
    for tsec in pk:
        p = int((tsec - t0) * f)
        a, b = max(0, p - half), min(len(x), p + half)
        widths_ms.append(float(np.sum(xn[a:b] > thr) / f * 1000.0))
        times.append(round(float(tsec), 2))
    widths_ms = np.array(widths_ms)
    if mode == "raw":
        return {"qrs_widths_ms": [round(w, 1) for w in widths_ms[:MAX_LIST]],
                "beat_times_s": times[:MAX_LIST], "n_beats": len(pk)}
    wide = widths_ms > 120.0
    return {"mean_qrs_ms": round(float(np.mean(widths_ms)), 1),
            "wide_beat_frac": round(float(np.mean(wide)), 3),
            "ectopic_count": int(np.sum(wide)),
            "ectopic_times_s": [times[i] for i in range(len(times)) if wide[i]][:MAX_LIST],
            "n_beats": len(pk)}


# ------------------------------------------------------------------- P wave (flutter/conduction) ----
def get_p_wave(record, pn_dir, t0, t1, fs, lead=0, mode="cooked", **_):
    """P-wave measurement over [t0,t1]: locate the atrial deflection in the PR segment just
    before each R-peak. p_present_frac = fraction of beats with a detectable P wave; pr_mean_ms =
    mean P-peak -> R delay; p_amp = mean P deflection (mV, relative to the local PR baseline);
    pp_regularity = CV of the P-P interval (a flutter/atrial-rhythm cue). Numbers only, no label;
    causal (uses only the retained window). Not a ground-truth annotation reader -- P onset is
    approximated from the waveform, like a caliper on the strip."""
    sig, f = _read(record, pn_dir, t0, t1, fs)
    if sig is None or sig.shape[0] < int(0.5 * f):
        return {"note": "window too short / no signal"}
    pk, _ = _peaks_in(record, pn_dir, fs, t0, t1, lead)
    if len(pk) < 3:
        return {"n_beats": int(len(pk)), "note": "too few beats"}
    x = sig[:, lead if lead is not None and lead >= 0 else 0].astype("float64")
    lo, hi = int(0.25 * f), int(0.05 * f)  # P-search window: 250..50 ms before R
    base_w = int(0.02 * f)
    pr_ms, p_amps, p_times = [], [], []
    for tsec in pk:
        r = int((tsec - t0) * f)
        a, b = r - lo, r - hi
        if a < base_w or b <= a:
            continue
        seg = x[a:b]
        baseline = float(np.median(x[max(0, a - base_w):a])) if a - base_w >= 0 else float(np.median(seg))
        dev = seg - baseline
        j = int(np.argmax(np.abs(dev)))
        amp = float(dev[j])
        p_amps.append(amp)
        p_times.append((a + j) / f)                 # P-peak time within the window (s, window-local)
        pr_ms.append((r - (a + j)) / f * 1000.0)
    if not p_amps:
        return {"n_beats": int(len(pk)), "note": "no PR segment measurable"}
    amp_arr = np.abs(np.array(p_amps))
    present = amp_arr > 0.05                          # ~0.05 mV: a discernible atrial deflection
    if mode == "raw":
        return {"pr_intervals_ms": [round(v, 1) for v in pr_ms[:MAX_LIST]],
                "p_amps_mv": [round(v, 3) for v in p_amps[:MAX_LIST]], "n_beats": int(len(pk))}
    pp = np.diff(np.array(p_times))
    pp_cv = float(np.std(pp) / (np.mean(pp) + 1e-6)) if len(pp) >= 2 else 0.0
    return {"p_present_frac": round(float(np.mean(present)), 3),
            "pr_mean_ms": round(float(np.mean([m for m, ok in zip(pr_ms, present) if ok])), 1)
            if present.any() else 0.0,
            "p_amp_mv": round(float(np.mean(amp_arr)), 3),
            "pp_regularity": round(pp_cv, 3), "n_beats": int(len(pk))}


# ------------------------------------------------------------------- T wave (ischemia/K+) ----
def get_t_wave(record, pn_dir, t0, t1, fs, lead=0, mode="cooked", **_):
    """T-wave measurement over [t0,t1]: peak repolarization deflection in the ST-T window after
    each R-peak, per lead. Returns t_amp (mV, signed, vs. the PR baseline) and polarity
    (positive/negative/flat) per lead. Numbers only, causal, waveform-derived (no label)."""
    sig, f = _read(record, pn_dir, t0, t1, fs)
    if sig is None or sig.shape[0] < int(0.5 * f):
        return {"note": "window too short / no signal"}
    pk, _ = _peaks_in(record, pn_dir, fs, t0, t1, lead)
    if len(pk) < 3:
        return {"n_beats": int(len(pk)), "note": "too few beats"}
    a0, a1 = int(0.15 * f), int(0.40 * f)            # T-search window: 150..400 ms after R
    base_b = int(0.04 * f)                            # PR baseline: 40 ms before R
    out = {}
    for L in range(sig.shape[1]):
        x = sig[:, L].astype("float64")
        amps = []
        for tsec in pk:
            r = int((tsec - t0) * f)
            lo, hi = r + a0, r + a1
            if r - base_b < 0 or hi >= len(x):
                continue
            baseline = float(x[r - base_b])
            dev = x[lo:hi] - baseline
            amps.append(float(dev[int(np.argmax(np.abs(dev)))]))
        if not amps:
            out[f"t_amp_lead{L}_mv"] = 0.0
            continue
        if mode == "raw":
            out[f"t_samples_lead{L}_mv"] = [round(v, 3) for v in amps[:MAX_LIST]]
        else:
            m = float(np.mean(amps))
            out[f"t_amp_lead{L}_mv"] = round(m, 3)
            out[f"polarity_lead{L}"] = ("positive" if m > 0.05 else
                                        "negative" if m < -0.05 else "flat")
    if mode == "raw":
        out["n_beats"] = int(len(pk))
    return out


# ------------------------------------------------------------------- QT interval (long/short QT) ----
def get_qt(record, pn_dir, t0, t1, fs, lead=0, mode="cooked", **_):
    """QT-interval measurement over [t0,t1]: Q-onset (~50 ms before R) to T-end, where T-end is
    approximated as the point where the ST-T deflection energy falls back toward the PR baseline.
    Returns qt_mean_ms and qtc_ms (Bazett: QT/sqrt(RR)). Numbers only, causal, waveform-derived."""
    sig, f = _read(record, pn_dir, t0, t1, fs)
    if sig is None or sig.shape[0] < int(0.5 * f):
        return {"note": "window too short / no signal"}
    pk, _ = _peaks_in(record, pn_dir, fs, t0, t1, lead)
    if len(pk) < 3:
        return {"n_beats": int(len(pk)), "note": "too few beats"}
    x = sig[:, lead if lead is not None and lead >= 0 else 0].astype("float64")
    q_off = int(0.05 * f)                             # Q onset ~ 50 ms before R
    t_lo, t_hi = int(0.15 * f), int(0.50 * f)         # T-end search: 150..500 ms after R
    base_b = int(0.04 * f)
    rr_ms = np.diff(pk) * 1000.0
    qt_ms = []
    for i, tsec in enumerate(pk):
        r = int((tsec - t0) * f)
        if r - q_off < 0 or r + t_hi >= len(x):
            continue
        baseline = float(x[r - base_b])
        seg = np.abs(x[r + t_lo:r + t_hi] - baseline)
        if len(seg) < 4:
            continue
        thr = 0.15 * float(np.max(seg))               # T-end = deflection falls below 15% of its peak
        tpk = int(np.argmax(seg))
        end = tpk
        while end < len(seg) and seg[end] > thr:
            end += 1
        t_end = r + t_lo + end
        qt_ms.append((t_end - (r - q_off)) / f * 1000.0)
    qt_ms = [v for v in qt_ms if 200.0 <= v <= 700.0]  # physiological QT range
    if len(qt_ms) < 2:
        return {"n_beats": int(len(pk)), "note": "too few clean beats to measure QT"}
    if mode == "raw":
        return {"qt_intervals_ms": [round(v, 1) for v in qt_ms[:MAX_LIST]], "n_beats": int(len(pk))}
    qt_mean = float(np.mean(qt_ms))
    rr_s = float(np.mean(rr_ms)) / 1000.0 if len(rr_ms) else 0.8
    qtc = qt_mean / (np.sqrt(rr_s) + 1e-6)             # Bazett
    return {"qt_mean_ms": round(qt_mean, 1), "qtc_ms": round(qtc, 1),
            "rr_mean_ms": round(float(np.mean(rr_ms)), 1) if len(rr_ms) else 0.0,
            "n_beats": int(len(pk))}


# ------------------------------------------------------------------- feature_profile ----
_FEATURE_FN = {
    "rr_irregularity": lambda rec, pn, a, b, fs, lead: get_rhythm(rec, pn, a, b, fs, lead).get("rr_irregularity", 0.0),
    "st_level": lambda rec, pn, a, b, fs, lead: max(abs(get_st_level(rec, pn, a, b, fs).get("st_mv_lead0", 0.0)),
                                                    abs(get_st_level(rec, pn, a, b, fs).get("st_mv_lead1", 0.0))),
    "ectopy": lambda rec, pn, a, b, fs, lead: _ectopy_cue(rec, pn, a, b, fs, lead),
}


def _ectopy_cue(rec, pn, a, b, fs, lead):
    """SKIM cue for ventricular ectopy. PVCs are SPARSE single beats, so wide_beat_frac (ectopic
    beats / all beats) dilutes to ~0 over a coarse window and hides them from the profile peak.
    Instead report a saturating function of the ectopic COUNT: >=1 wide beat in a bin already
    lifts the cue above the nudge threshold, so even a lone PVC surfaces as an elevated peak the
    agent can zoom into, while the value still grows with denser ectopy."""
    m = get_beat_morphology(rec, pn, a, b, fs, lead)
    n = int(m.get("ectopic_count", 0) or 0)
    if n <= 0:
        return 0.0
    # 1 PVC -> 0.2 (just over the 0.15 nudge threshold); saturates toward 1.0 as count grows.
    return round(min(1.0, 0.2 + 0.1 * (n - 1)), 3)

PROFILE_MIN_STRIDE = 10.0
PROFILE_MAX_SEGMENTS = 120


def feature_profile(record, pn_dir, t0, t1, fs, stride, feature, lead=0):
    """Coarse feature profile over [t0,t1]: split into `stride`-second segments and report the
    feature per segment. Internal helper used to build the SKIM view; distinct from the agent's
    skim() action. (Not exposed directly in the streaming action set.)"""
    if feature not in _FEATURE_FN:
        return {"error": f"unknown feature {feature}; use one of {list(_FEATURE_FN)}"}
    stride = max(PROFILE_MIN_STRIDE, float(stride))
    n = int((t1 - t0) // stride)
    if n > PROFILE_MAX_SEGMENTS:
        stride = (t1 - t0) / PROFILE_MAX_SEGMENTS
        n = PROFILE_MAX_SEGMENTS
    fn = _FEATURE_FN[feature]
    edges, vals = [round(t0, 1)], []
    for i in range(max(1, n)):
        a = t0 + i * stride
        b = min(t1, a + stride)
        vals.append(round(float(fn(record, pn_dir, a, b, fs, lead)), 3))
        edges.append(round(b, 1))
    return {"feature": feature, "stride_s": round(stride, 1), "edges_s": edges, "values": vals}


# ------------------------------------------------------------------------------ profile ----
# The core primitive of the adaptive-chunk env: observe a span [t0,t1] at a chosen granularity.
# Internally the span is split into a fixed number of fine sub-bins; we report the PEAK sub-bin
# feature (and where it is), NOT the mean. So a large chunk skimming a quiet region reports low,
# but a large chunk containing a short buried episode still reports a high peak -> the agent sees
# "something in here" and can zoom in. This is what makes coarse observation lossy-but-not-blind.
PROFILE_BINS = 6         # sub-bins per observe() call
PROFILE_MIN_BIN = 4.0    # s: don't sub-bin finer than this (RR needs a few beats)


def profile(record, pn_dir, t0, t1, fs, feature="rr_irregularity", lead=0):
    if feature not in _FEATURE_FN:
        return {"error": f"unknown feature {feature}; use one of {list(_FEATURE_FN)}"}
    span = t1 - t0
    nbins = max(1, min(PROFILE_BINS, int(span // PROFILE_MIN_BIN)))
    fn = _FEATURE_FN[feature]
    binw = span / nbins
    bins = []
    for i in range(nbins):
        a = t0 + i * binw
        b = min(t1, a + binw)
        bins.append((round(a, 1), round(b, 1), round(float(fn(record, pn_dir, a, b, fs, lead)), 3)))
    vals = [v for _, _, v in bins]
    peak_i = int(np.argmax(vals)) if vals else 0
    return {"feature": feature, "span_s": [round(t0, 1), round(t1, 1)],
            "peak_value": vals[peak_i] if vals else 0.0,
            "peak_range_s": [bins[peak_i][0], bins[peak_i][1]] if bins else [t0, t1],
            "mean_value": round(float(np.mean(vals)), 3) if vals else 0.0,
            "bins": [{"range_s": [a, b], "value": v} for a, b, v in bins]}


# ============================================================================================
#  DIAGNOSTIC TOOLS (return a ready-made clinical label, not a measurement).
#
#  Unlike the signal tools above -- which return numbers and leave the agent to conclude a
#  diagnosis -- these return the conclusion itself (rhythm class, per-beat type, ischemia flag),
#  read from the time-stamped ground-truth annotations and clamped causally to the elapsed
#  window [t0,t1] (never the future).  They are the "+dx" arm of the with/without ablation:
#  the question is whether a ready label helps online localization/latency or is leaned on in
#  place of signal reasoning.  Exposed to the agent ONLY when tool_tier == "signal+dx".
# ============================================================================================

def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


# per-record ground-truth caches (diagnostic tools re-read the same annotations every chunk;
# cache the parsed episodes/beats once per record to keep the streaming rule-agent fast).
_GT_AF, _GT_BEATS, _GT_ST = {}, {}, {}


def _af_eps(record, pn_dir):
    key = (pn_dir, record)
    if key not in _GT_AF:
        from ecgscroll import parse_afdb, parse_ltafdb
        try:
            _GT_AF[key] = (parse_ltafdb.af_episodes(record) if pn_dir == "ltafdb"
                           else parse_afdb.af_episodes(record, pn_dir))
        except Exception:
            _GT_AF[key] = None
    return _GT_AF[key]


def _beat_ann(record, pn_dir):
    key = (pn_dir, record)
    if key not in _GT_BEATS:
        from ecgscroll import parse_mitdb, parse_ltdb
        mod = parse_ltdb if pn_dir == "ltdb" else parse_mitdb
        try:
            _GT_BEATS[key] = mod.beat_times(record)
        except Exception:
            _GT_BEATS[key] = None
    return _GT_BEATS[key]


def _st_eps(record, pn_dir):
    key = (pn_dir, record)
    if key not in _GT_ST:
        from ecgscroll import parse_edb
        try:
            _GT_ST[key] = parse_edb.st_episodes(record, pn_dir)
        except Exception:
            _GT_ST[key] = None
    return _GT_ST[key]


def classify_rhythm(record, pn_dir, t0, t1, **_):
    """Dominant rhythm over [t0,t1] from the AF ground truth (afdb/ltafdb)."""
    eps = _af_eps(record, pn_dir)
    if eps is None:
        return {"note": "no rhythm ground truth for this record"}
    span = max(1e-6, t1 - t0)
    af = sum(_overlap(t0, t1, a, b) for a, b in eps)
    frac = float(af) / span
    return {"dominant_rhythm": "AF" if frac >= 0.5 else "sinus",
            "af_frac": round(frac, 3), "confidence": 0.99}


def classify_beats(record, pn_dir, t0, t1, **_):
    """Per-beat type counts and PVC times over [t0,t1] from beat annotations (mitdb/ltdb)."""
    bts = _beat_ann(record, pn_dir)
    if bts is None:
        return {"note": "no beat-level ground truth for this record"}
    types, pvc = {}, []
    for tsec, sym in bts:
        if t0 <= tsec < t1:
            types[sym] = types.get(sym, 0) + 1
            if sym in ("V", "E"):
                pvc.append(round(float(tsec), 2))
    return {"beat_types": types, "pvc_times_s": pvc[:MAX_LIST],
            "n_beats": sum(types.values()), "confidence": 0.99}


def flag_ischemia(record, pn_dir, t0, t1, **_):
    """Ischemic ST-episodes overlapping [t0,t1] from the European ST-T ground truth (edb)."""
    eps = _st_eps(record, pn_dir)
    if eps is None:
        return {"note": "no ischemia ground truth for this record"}
    hit = [e for e in eps if _overlap(t0, t1, e["start_t"], e["end_t"]) > 0]
    if not hit:
        return {"ischemic": False, "episodes": [], "confidence": 0.99}
    strongest = max(hit, key=lambda e: abs(e.get("st_mv", 0.0)))
    return {"ischemic": True, "lead": int(strongest.get("lead")),
            "st_mv": round(float(strongest.get("st_mv", 0.0)), 3),
            "sign": "elevation" if strongest.get("sign") == "+" else "depression",
            "episodes": [[round(float(max(t0, e["start_t"])), 2), round(float(min(t1, e["end_t"])), 2)]
                         for e in hit][:MAX_LIST],
            "confidence": 0.99}


# task -> the signal tool measure() should dispatch to, and the diagnostic tool for +dx.
_SIGNAL_FOR_FEATURE = {"rr_irregularity": "rate_rhythm", "st_level": "st", "ectopy": "qrs"}
_DIAGNOSTIC_FOR_TASK = {
    "episode_detection": classify_rhythm, "change_detection": classify_rhythm,
    "burden_quantification": classify_rhythm,   # AF burden; PVC burden overridden by feature below
    "ischemia_detection": flag_ischemia,
    "rare_event_search": classify_beats,
}


def diagnostic_for(task, feature):
    """Pick the diagnostic tool for a task/feature (PVC burden & ectopy -> beat classifier)."""
    if feature == "ectopy":
        return classify_beats
    if feature == "st_level":
        return flag_ischemia
    return _DIAGNOSTIC_FOR_TASK.get(task, classify_rhythm)


# ============================================================================================
#  TOOL REGISTRY — the model-facing menu that `measure(tool, t0, t1)` dispatches over.
#
#  measure is a single BROAD action; the LLM picks which concrete tool it invokes this call.
#  Names match Table 2 (clinician's reading order). Signal tools return numbers only; diagnostic
#  tools return a ready-made label and are exposed only in the "signal+dx" tier. Every entry has
#  the SAME call signature (rec, pn, a, b, fs, lead, mode) so stream.py can dispatch uniformly;
#  the lambdas adapt the underlying functions' (older, heterogeneous) argument orders. This is the
#  single source of truth -- stream.py and run_llm_skim.py both read it.
# ============================================================================================

SIGNAL_TOOLS = {
    "get_rate_rhythm": lambda rec, pn, a, b, fs, lead, mode: get_rhythm(rec, pn, a, b, fs, lead, mode),
    "get_p_wave":      lambda rec, pn, a, b, fs, lead, mode: get_p_wave(rec, pn, a, b, fs, lead, mode),
    "get_qrs":         lambda rec, pn, a, b, fs, lead, mode: get_beat_morphology(rec, pn, a, b, fs, lead, mode),
    "get_st":          lambda rec, pn, a, b, fs, lead, mode: get_st_level(rec, pn, a, b, fs, mode, lead),
    "get_t_wave":      lambda rec, pn, a, b, fs, lead, mode: get_t_wave(rec, pn, a, b, fs, lead, mode),
    "get_qt":          lambda rec, pn, a, b, fs, lead, mode: get_qt(rec, pn, a, b, fs, lead, mode),
}

DIAGNOSTIC_TOOLS = {  # exposed only when tool_tier == "signal+dx"
    "classify_rhythm": classify_rhythm,
    "classify_beats":  classify_beats,
    "flag_ischemia":   flag_ischemia,
}

# task feature -> default signal tool, used when the model omits `tool` or names an unknown/
# disallowed one. Keeps the fallback measurement identical to the old auto-dispatch behaviour
# (so results stay comparable) and prevents a crash on a malformed tool call.
DEFAULT_SIGNAL_TOOL = {"rr_irregularity": "get_rate_rhythm", "st_level": "get_st", "ectopy": "get_qrs"}

