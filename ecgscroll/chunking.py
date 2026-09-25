"""Shared chunk-level scanning + answer assembly for ECG-Scroll streaming agents.

Both the Rule Agent (run_stream.py) and the LLM agent (run_llm_fc.py) walk a recording
chunk by chunk exactly as the streaming env delivers it (monotonic cursor, no future signal),
measure each freshly-arrived chunk, then decide which chunks belong to an event. Factoring the
per-chunk measurement and the end-of-stream answer assembly here guarantees the two agents run
the IDENTICAL protocol -- the only difference is *who decides* a chunk is an event: a fixed
threshold (Rule) or a language model (LLM). Keeping this shared is what makes the +dx / --dx and
Rule-vs-LLM comparisons fair.
"""
import numpy as np

from ecgscroll import render, tools
from ecgscroll.env import EcgScrollEnv

# task families by the signal cue they flag on
_AF_TASKS = {"episode_detection", "burden_quantification", "change_detection"}
_ST_TASKS = {"ischemia_detection"}
_PVC_TASKS = {"rare_event_search"}


def feature_for(sample):
    """The signal feature this sample flags on (rr_irregularity / st_level / ectopy)."""
    task = sample["task"]
    kind = sample.get("answer", {}).get("kind", "")
    if task in _ST_TASKS:
        return "st_level"
    if task in _PVC_TASKS or kind == "pvc_burden":
        return "ectopy"
    return "rr_irregularity"


def etype_for(sample):
    f = feature_for(sample)
    return {"st_level": "ISCH", "ectopy": "PVC"}.get(f, "AFIB")


def _abs_st(meas):
    return max(abs(meas.get(f"st_mv_lead{L}", 0.0) or 0.0) for L in (0, 1))


def scan_chunks(sample, chunk_s=30.0, dx=False):
    """Walk the whole recording chunk by chunk (causal) and return, per chunk, a feature dict.

    Returns (env, chunks) where chunks is a list of dicts:
      {i, t0, t1, ...cue fields...}
    --dx (signal) cue fields: rr_cv / st_mv / wide_beat_frac + strength.
    +dx (diagnostic) cue fields: the label dict from the diagnostic tool + strength + a boolean
    `dx_hit` (what a label-following policy would flag).

    The env is returned un-finished so the caller can `write` flagged chunks (for latency) and
    then assemble the answer. No decision is made here -- only measurement.
    """
    task = sample["task"]
    feature = feature_for(sample)
    is_st, is_pvc = feature == "st_level", feature == "ectopy"
    fs = render.get_signal(sample["record"], sample["pn_dir"])[1]
    lead = sample.get("lead", 0) or 0
    rec, pn = sample["record"], sample["pn_dir"]
    env = EcgScrollEnv(sample, budget=10**9, streaming=True, chunk_s=chunk_s, render_tiles=False)
    env.reset()

    chunks = []
    i = 0
    while True:
        c1 = env.cursor
        c0 = max(sample["t0"], c1 - chunk_s)
        _, _, _, info = env.step({"op": "caliper", "t0": c0, "t1": c1, "lead": lead})
        meas = info.get("measurement", {})
        row = {"i": i, "t0": round(c0, 1), "t1": round(c1, 1)}
        if dx:
            fn = tools.diagnostic_for(task, feature)
            d = fn(rec, pn, c0, c1)
            row["label"] = d
            if is_st:
                row["strength"] = abs(d.get("st_mv", 0.0) or 0.0)
                row["dx_hit"] = bool(d.get("ischemic"))
            elif is_pvc:
                row["strength"] = len(d.get("pvc_times_s", []) or [])
                row["dx_hit"] = row["strength"] > 0
            else:
                row["strength"] = float(d.get("af_frac", 0.0) or 0.0)
                row["dx_hit"] = d.get("dominant_rhythm") == "AF"
        else:
            if is_st:
                row["st_mv"] = round(_abs_st(meas), 3)
                row["strength"] = row["st_mv"]
            elif is_pvc:
                m = tools.get_beat_morphology(rec, pn, c0, c1, fs, lead, "cooked")
                row["wide_beat_frac"] = round(float(m.get("wide_beat_frac", 0.0) or 0.0), 3)
                row["ectopic_count"] = int(m.get("ectopic_count", 0) or 0)
                row["strength"] = row["wide_beat_frac"]
            else:
                v = meas.get("rr_cv")
                row["rr_cv"] = round(float(v), 3) if v is not None else None
                row["hr_bpm"] = meas.get("hr_bpm")
                row["strength"] = float(v) if v is not None else 0.0
        chunks.append(row)
        if env.cursor >= sample["t1"]:
            break
        env.step({"op": "advance", "n": 1})
        i += 1
    return env, chunks


def assemble_answer(sample, flagged, strengths):
    """Build the task's answer schema from the flagged chunks (list of (t0,t1)) + cue strengths.

    Mirrors the Rule Agent's end-of-stream assembly: merge contiguous flagged chunks into
    intervals for episode/ischemia, take the union fraction for burden, the first run onset for
    change, and the single strongest chunk for rare-event."""
    task = sample["task"]
    kind = sample.get("answer", {}).get("kind", "")
    merged = []
    for a, b in flagged:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if task == "burden_quantification":
        total = sample["t1"] - sample["t0"]
        flagged_s = sum(b - a for a, b in merged)
        return {"value": round(flagged_s / total, 4) if total > 0 else 0.0,
                "kind": kind or "af_burden"}
    if task == "change_detection":
        return {"change_point": round(merged[0][0], 2) if merged else 0.0,
                "kind": kind or "af_onset"}
    if task == "rare_event_search":
        if flagged:
            j = int(np.argmax(strengths))
            return {"interval": [round(flagged[j][0], 2), round(flagged[j][1], 2)]}
        return {"interval": [0.0, 0.0]}
    return {"intervals": [[round(a, 2), round(b, 2)] for a, b in merged]}
