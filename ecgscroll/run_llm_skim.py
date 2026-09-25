"""LLM streaming agent — SKIM version: the LLM plans its own scan granularity.

This is the "planning" agent (distinct from run_llm_fc.py, the harness-driven chunk-judge agent).
Here the model itself drives the causal cursor via skim(duration): it chooses how big the next
slice is, coarse over quiet stretches and fine over suspicious ones, then measure(tool,t0,t1) to
confirm and write() to record. measure is a broad action: the model also picks WHICH tool it runs
(signal tools always; diagnostic tools in the signal+dx tier). This exercises the Planning
competency directly. The prompt teaches a good scan strategy (coarse-scan -> zoom on an elevated
peak -> confirm -> write).

Uses the same StreamSession engine (ecgscroll.stream) as before; only the driver + prompts live
here. Start the vLLM server with tool-call parsing:
  python -m vllm.entrypoints.openai.api_server --model <snap> --served-model-name qwen3-30b \
     --enable-auto-tool-choice --tool-call-parser hermes --port 8000
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

from ecgscroll import paths, render
from ecgscroll.stream import StreamSession

# ---- shared skim/measure explanation, prepended to every task prompt --------------------------
_SKIM_DOC = """You are an expert cardiologist reading a long ambulatory ECG that streams in ONLINE, \
in time order. You cannot see the waveform or the future; you read it only by calling tools on the \
already-elapsed signal, and you must plan your own scan.

Your core lever is skim(duration): YOU choose how big the next slice is.
- Over a long quiet stretch, skim BIG (300-1800 s) to cover ground fast.
- When something looks suspicious, skim SMALL (5-60 s) to look closely.
Each skim returns a lossy profile of the newly-elapsed span: peak_value (the most abnormal \
sub-region's cue) and peak_range_s (where it is). A big slice is low-resolution, so a short buried \
event shows up only as an ELEVATED PEAK -- when you see one, do NOT skim forward; instead skim (or \
measure) that peak_range again at fine resolution to pin it, then record it.

measure(tool, t0, t1) is a single action that runs ONE tool of YOUR choice over an elapsed \
sub-window, without advancing time. YOU pick which tool. The signal tools return numbers only: \
get_rate_rhythm (heart rate + RR irregularity, the AF cue), get_p_wave, get_qrs (beat morphology / \
ectopy), get_st (ST-segment level, the ischemia cue), get_t_wave, get_qt. Choose the tool whose \
measurement answers the current task, then reason from the numbers to a conclusion.

The scan strategy that works:
  skim BIG  ->  peak elevated?  ->  skim/measure the peak_range SMALL to localize the event  -> \
confirm it  ->  now EXTEND the boundary: probe the windows just BEFORE and just AFTER with \
measure to find where the event actually starts and ends (events are often much longer \
than the little window you first hit)  ->  write(start,end) the WHOLE event span  ->  continue.
Do NOT write a tiny 5-10 s stub when the event clearly continues past it -- extend to its true \
edges first, because your interval is scored against the true one by IoU. When a stretch is \
confirmed sinus/normal, move on. An event you never write scores zero. When the cursor reaches \
total, call finish().
"""

_TASK_TAIL = {
    "episode_detection":
        "TASK: find every atrial-fibrillation EPISODE and write each as a tight interval. "
        "Use measure(tool=get_rate_rhythm,...): AF cue is high rr_irregularity (>~0.15 vs ~0.02-0.08 "
        "in sinus) with elevated/irregular rate. "
        "Aim for tight boundaries -- the interval you write is scored against the true one by IoU.",
    "burden_quantification":
        "TASK: quantify total AF BURDEN = fraction of the whole recording spent in AF. Cover the "
        "WHOLE record by skimming, and write(start,end) for EVERY AF stretch you confirm; the burden "
        "is the union of your written intervals over total duration. Missing AF underestimates it, "
        "writing non-AF overestimates it, so favor coverage. Confirm with measure(tool=get_rate_rhythm,"
        "...): AF cue is high rr_irregularity (>~0.15).",
    "change_detection":
        "TASK: report WHEN atrial fibrillation first begins (sinus->AF onset). Skim forward; the "
        "moment you see AF-like irregularity, measure(tool=get_rate_rhythm,...) smaller to pin the "
        "onset, then write(start,end) for that first AF stretch (its start is the change-point). "
        "Report it early and precisely, then finish.",
    "ischemia_detection":
        "TASK: find every ischemic ST EPISODE. Ischemia cue: sustained |ST deviation| >~0.1 mV in "
        "either lead. skim BIG -> on an elevated peak, measure(tool=get_st,...) to confirm the "
        "per-lead ST level -> write(start,end) for the ischemic interval -> continue.",
    "rare_event_search":
        "TASK: locate the SINGLE short target beat (one ectopic/PVC) in a long otherwise-normal "
        "record, and write it once. Skim to find the one abnormal region (elevated wide-QRS peak), "
        "measure(tool=get_qrs,...) it small to pin the beat tightly, write(start,end) once, then finish.",
    "pvc_burden":
        "TASK: quantify total VENTRICULAR-ECTOPY (PVC) BURDEN = fraction of the whole recording "
        "spent in ventricular ectopic beats. Cover the WHOLE record by skimming, and write(start,end) "
        "for EVERY stretch containing ventricular ectopy you confirm; the burden is the union of your "
        "written intervals over total duration. Confirm with measure(tool=get_qrs,...): the ectopy cue "
        "is wide QRS (mean_qrs_ms and wide_beat_frac elevated, ectopic_count>0). This is a BEAT-"
        "MORPHOLOGY task, NOT a rhythm/AF task -- do not use rr_irregularity here.",
}

_DX_TAIL = {
    "episode_detection":
        " In this tier measure ALSO offers a diagnostic tool: measure(tool=classify_rhythm,t0,t1) "
        "returns a ready-made rhythm label {dominant_rhythm:'AF'/'sinus', af_frac} for the window. "
        "When you have narrowed to a tight suspicious window, call it to confirm; if it says AF "
        "(af_frac>=0.5), write it.",
    "burden_quantification":
        " In this tier measure ALSO offers measure(tool=classify_rhythm,t0,t1): a ready-made label "
        "{dominant_rhythm, af_frac}. Use it to confirm each AF stretch before writing (write windows "
        "it calls AF).",
    "change_detection":
        " In this tier measure ALSO offers measure(tool=classify_rhythm,t0,t1): a ready-made rhythm "
        "label. The first window it calls AF marks the onset -- write it.",
    "ischemia_detection":
        " In this tier measure ALSO offers measure(tool=flag_ischemia,t0,t1): a ready-made ischemia "
        "flag {ischemic:bool, lead, st_mv}. When you narrow to a suspicious window, call it; if "
        "ischemic, write that interval.",
    "rare_event_search":
        " In this tier measure ALSO offers measure(tool=classify_beats,t0,t1): returns beat types and "
        "pvc_times_s for the window. Use it to confirm the target beat, then write tightly around the "
        "returned pvc time.",
    "pvc_burden":
        " In this tier measure ALSO offers measure(tool=classify_beats,t0,t1): returns per-beat types "
        "and pvc_times_s for the window. Use it to confirm ventricular ectopy (V/E beats) before "
        "writing each PVC stretch. Do NOT use classify_rhythm here -- that is an AF label and will not "
        "flag PVCs.",
}


def _prompt_key(task, kind=""):
    """burden_quantification is shared by AF burden and PVC burden; disambiguate by answer kind so
    each gets the right tool-selection guidance (mirrors StreamSession.feature)."""
    if task == "burden_quantification" and kind == "pvc_burden":
        return "pvc_burden"
    return task


def system_for(task, tier="signal", kind=""):
    key = _prompt_key(task, kind)
    tail = _TASK_TAIL.get(key, _TASK_TAIL["episode_detection"])
    if tier == "signal+dx":
        tail += _DX_TAIL.get(key, _DX_TAIL["episode_detection"])
    return _SKIM_DOC + "\n" + tail + "\n\nCall exactly one tool per turn."


# measure is ONE broad action; the model picks which tool it runs via the `tool` enum. Signal
# tools are always available; diagnostic tools are added to the enum only in the signal+dx tier.
_SIGNAL_TOOL_NAMES = ["get_rate_rhythm", "get_p_wave", "get_qrs", "get_st", "get_t_wave", "get_qt"]
_DX_TOOL_NAMES = ["classify_rhythm", "classify_beats", "flag_ischemia"]


def _measure_tool(tier):
    names = _SIGNAL_TOOL_NAMES + (_DX_TOOL_NAMES if tier == "signal+dx" else [])
    desc = ("Run one measurement tool over an elapsed sub-window [t0,t1] without advancing time; "
            "use to confirm/pin an event. Choose `tool`: signal tools return numbers "
            "(get_rate_rhythm=HR/RR irregularity, get_p_wave, get_qrs=beat morphology/ectopy, "
            "get_st=ST level, get_t_wave, get_qt).")
    if tier == "signal+dx":
        desc += (" Diagnostic tools return a ready-made label (classify_rhythm={dominant_rhythm,"
                 "af_frac}, classify_beats={beat types, pvc_times_s}, flag_ischemia={ischemic,st_mv}).")
    return {"type": "function", "function": {
        "name": "measure", "description": desc,
        "parameters": {"type": "object", "properties": {
            "tool": {"type": "string", "enum": names, "description": "which tool to run"},
            "t0": {"type": "number"}, "t1": {"type": "number"}},
            "required": ["tool", "t0", "t1"]}}}


_SKIM_TOOL = {"type": "function", "function": {
    "name": "skim", "description": "Advance the cursor by `duration` seconds and return that "
    "span's profile (peak_value, peak_range_s, mean). Choose duration: big (300-1800) to cover "
    "quiet ground, small (5-60) to look closely at a suspicious region.",
    "parameters": {"type": "object", "properties": {
        "duration": {"type": "number", "description": "seconds to skim next (5-1800)"}},
        "required": ["duration"]}}}
_WRITE_TOOL = {"type": "function", "function": {
    "name": "write", "description": "Record a confirmed event interval to memory. REQUIRED to score.",
    "parameters": {"type": "object", "properties": {
        "start": {"type": "number"}, "end": {"type": "number"}}, "required": ["start", "end"]}}}
_FINISH_TOOL = {"type": "function", "function": {
    "name": "finish", "description": "End the episode; the answer is built from written intervals.",
    "parameters": {"type": "object", "properties": {}}}}


def tools_for(tier):
    return [_SKIM_TOOL, _measure_tool(tier), _WRITE_TOOL, _FINISH_TOOL]


AF_HINT_THR = 0.15
ST_HINT_THR = 0.10  # |ST deviation| (mV) cue for ischemia in the signal tier (matches the prompt)


def _abs_st(m):
    """Max |ST deviation| (mV) across the two leads from a get_st measurement (0 if absent)."""
    return max(abs(m.get("st_mv_lead0", 0.0) or 0.0), abs(m.get("st_mv_lead1", 0.0) or 0.0))

# recommended measure `tool` per task, for the observation-line nudges. In the signal+dx tier we
# nudge toward the diagnostic tool (ready label); in signal-only, toward the matching signal tool.
_SIG_TOOL_FOR_TASK = {"episode_detection": "get_rate_rhythm", "burden_quantification": "get_rate_rhythm",
                      "change_detection": "get_rate_rhythm", "ischemia_detection": "get_st",
                      "rare_event_search": "get_qrs", "pvc_burden": "get_qrs"}
_DX_TOOL_FOR_TASK = {"episode_detection": "classify_rhythm", "burden_quantification": "classify_rhythm",
                     "change_detection": "classify_rhythm", "ischemia_detection": "flag_ischemia",
                     "rare_event_search": "classify_beats", "pvc_burden": "classify_beats"}


def state_text(o, last, tier="signal", task="episode_detection", kind=""):
    """Compact observation text. Reports what the last action returned and, on an elevated peak,
    gives a light nudge toward the right NEXT move (zoom if coarse, confirm+write if already tight).
    Deliberately advisory -- the model still chooses durations, windows, and tools itself (planning)."""
    dx = (tier == "signal+dx")
    key = _prompt_key(task, kind)
    rec_tool = (_DX_TOOL_FOR_TASK if dx else _SIG_TOOL_FOR_TASK).get(key, "get_rate_rhythm")
    def _m(a, b):  # a suggested measure call with the recommended tool
        return f"measure(tool={rec_tool},t0={a},t1={b})"
    coverage = (task == "burden_quantification")
    n_written = len([e for e in o["ledger"] if "interval" in e])
    L = [f"now={o['now_s']}s / total={o['total_s']}s (stream_end={o['stream_end']}); "
         f"events written so far: {n_written}"]
    nudge = None
    if last:
        if "profile" in last:
            p = last["profile"]
            L.append(f"skimmed {last['skimmed_s']}: peak_value={p['peak_value']} at "
                     f"{p['peak_range_s']}, mean={p['mean_value']}")
            pv, pr = p.get("peak_value", 0.0), p.get("peak_range_s")
            sk = last.get("skimmed_s")
            if pv >= AF_HINT_THR and sk:
                # measure the WHOLE just-skimmed window, not only the lossy peak sub-range --
                # otherwise a target that is not the single strongest sub-bin is silently missed.
                nudge = (f"Something may be in the window {sk} you just skimmed (peak {pv} at {pr}). "
                         f"{_m(sk[0], sk[1])} to check the WHOLE window; if it flags an "
                         f"event, write that event's interval before moving on.")
        if "measurement" in last:
            m = last["measurement"]
            tool_used = last.get("tool", "")
            L.append(f"measured {last.get('measured_s')} with {tool_used}: {json.dumps(m)}")
            ms = last.get("measured_s")
            # the measurement may be a signal reading (rr_irregularity, ...) or a diagnostic label
            # (af_frac / ischemic / pvc_times_s) -- both now arrive under "measurement". Route the
            # nudge on whichever fields are present, preferring the concrete diagnostic cues.
            irr = m.get("rr_irregularity")
            frac = m.get("af_frac")
            pvc = m.get("pvc_times_s") or []
            if ms and pvc:
                if key == "pvc_burden":
                    # PVC BURDEN is a per-beat COUNTING task: write one tight (~2 s) interval around
                    # EACH returned pvc_time. Never write a wide span -- a single [0,T] write makes
                    # burden = 100%. Burden = union of these tight intervals over total duration.
                    calls = ", ".join(f"write(start={round(t-1,2)},end={round(t+1,2)})" for t in pvc[:6])
                    nudge = (f"{len(pvc)} ventricular ectopic beat(s) here (pvc_times_s={pvc}). This is a "
                             f"per-beat COUNT task: write a TIGHT ~2s interval around EACH pvc_time -- "
                             f"e.g. {calls}. Do NOT write one wide interval spanning the whole window/"
                             f"record; each PVC is ~1 beat, so keep every interval ~2s.")
                else:
                    # rare_event_search: a single target beat -> write once, tightly, and move on.
                    t = pvc[0]
                    nudge = (f"TARGET FOUND: a ventricular ectopic beat at t={t}s (pvc_times_s={pvc}). "
                             f"Call write(start={round(t-1,2)},end={round(t+1,2)}) NOW -- this is the "
                             f"rare event; do not skim past it.")
            elif ms and m.get("ischemic") is True:
                nudge = (f"ISCHEMIA POSITIVE over {ms}. EXTEND: {rec_tool} the windows just before "
                         f"{ms[0]}s and after {ms[1]}s while still ischemic, then write the full span.")
            elif ms and key == "ischemia_detection" and _abs_st(m) >= ST_HINT_THR:
                # signal-tier ischemia: get_st returns per-lead st_mv (numbers, no label). A sustained
                # |ST| >~0.1 mV is the ischemia cue -> confirm, EXTEND the boundary, write that span.
                st = _abs_st(m)
                nudge = (f"CONFIRMED ischemia-like (|ST|={st} mV >~{ST_HINT_THR}) over {ms}. Now EXTEND: "
                         f"measure(tool=get_st,...) the windows just before {ms[0]}s and after {ms[1]}s; "
                         f"keep extending while |ST| stays elevated, then write(start,end) the whole "
                         f"ischemic span. Do NOT write a whole-record interval; write the elevated span.")
            elif ms and frac is not None and frac >= 0.5:
                nudge = (f"CONFIRMED AF (af_frac={frac}) over {ms}. Now EXTEND the boundary: "
                         f"measure the windows just before {ms[0]}s and just after {ms[1]}s; keep "
                         f"extending while they are still AF, then write(start,end) the whole AF "
                         f"span. Do not write just this {int(ms[1]-ms[0])}s window if AF continues.")
            elif ms and frac is not None and frac >= 0.10:
                nudge = (f"PARTIAL AF over {ms} (af_frac={frac}): AF covers only part of this "
                         f"window. measure smaller/adjacent sub-windows to find the exact AF span, "
                         f"then write that span.")
            elif ms and irr is not None and irr >= AF_HINT_THR:
                nudge = (f"CONFIRMED event-like (rr_irregularity {irr}) over {ms}. Now EXTEND: "
                         f"measure the adjacent windows before {ms[0]}s and after {ms[1]}s to find "
                         f"where it starts/ends, then write(start,end) the FULL span (an unrecorded "
                         f"event scores zero; a too-short stub scores near zero by IoU).")
            elif ms and key == "pvc_burden" and (m.get("ectopic_times_s") or m.get("ectopic_count")):
                # signal-tier PVC burden: get_qrs returns wide-QRS beat times -> write a tight ~2s
                # interval around EACH, same per-beat counting logic as the +dx branch above.
                et = m.get("ectopic_times_s") or []
                if et:
                    calls = ", ".join(f"write(start={round(t-1,2)},end={round(t+1,2)})" for t in et[:6])
                    nudge = (f"{m.get('ectopic_count', len(et))} wide/ectopic beat(s) here "
                             f"(ectopic_times_s={et[:6]}). Write a TIGHT ~2s interval around EACH -- "
                             f"e.g. {calls}. Do NOT write one wide span; each PVC is ~1 beat.")
        # COMMIT-DEBT escalation (single, task-agnostic). The dominant failure is the model
        # measuring an event over and over yet never issuing write() -- so an unrecorded event
        # scores zero. When this turn confirmed an event (a confirmation nudge fired) but the
        # ledger is STILL empty, stop merely suggesting and issue a hard, concretely-parameterized
        # write command using the window just measured. Decision authority stays with the model
        # (it may ignore it); we only lower the execution friction and surface the debt.
        if nudge is not None and n_written == 0 and last.get("measurement") is not None:
            ms = last.get("measured_s")
            if ms and not pvc and not (key == "pvc_burden"):
                # interval/burden tasks (AF, ischemia): commit the confirmed window NOW, then extend.
                nudge = (f"You have confirmed an event but written 0 so far -- an unrecorded event "
                         f"scores ZERO. Call write(start={ms[0]},end={ms[1]}) NOW to commit this "
                         f"window, THEN extend its boundary and continue. Do not measure again "
                         f"before you have written it.")
        if "written" in last:
            L.append(f"recorded {last['written'].get('interval')} (good -- keep going)")
        if "error" in last:
            L.append(f"error: {last['error']}")
    if o["stream_end"]:
        if n_written > 0:
            L.append("STREAM ENDED and you have written findings -- call finish() NOW.")
        elif coverage:
            # burden tasks: if nothing was confirmed, the burden is ~0 -- do NOT invent a write
            # (a spurious wide write forces burden toward 100%). Just finish.
            L.append("STREAM ENDED with no confirmed events -- burden is ~0; call finish() NOW "
                     "(do not write a catch-all interval).")
        elif task == "rare_event_search":
            # single-target: write ONE tight (~2s) interval at your best-guess beat time. A wide
            # span is rejected (its midpoint lands at the record centre and misses the target).
            L.append("STREAM ENDED -- write ONE tight ~2s interval around your best-guess target "
                     "time, then finish(). Do NOT write a wide/whole-record interval; it will be "
                     "rejected and cannot localize the beat.")
        else:
            L.append("STREAM ENDED -- write your single best interval ONCE, then finish().")
    elif nudge:
        L.append(nudge)
    return " | ".join(L)


def _apply(ss, name, args):
    if name == "skim":
        return ss.step({"op": "skim", "duration": float(args.get("duration", 300))})
    if name == "measure":
        return ss.step({"op": "measure", "tool": args.get("tool"),
                        "t0": float(args.get("t0", 0)), "t1": float(args.get("t1", 0))})
    if name == "write":
        return ss.step({"op": "write", "entry": {"type": "EVENT",
                        "interval": [float(args.get("start", 0)), float(args.get("end", 0))]}})
    if name == "finish":
        return ss.step({"op": "answer", "answer": None})
    return ss.step({"op": "skim", "duration": 300})


def budget_for(sample, floor, cap=3000, skim_max=1800.0, k=1.5):
    """Action budget scaled to record length AND the skim cap, so the agent can cover the whole
    record even at fine granularity (like the Rule Agent's unbounded per-chunk walk). If skims are
    capped small (fine scanning), you need ~duration/step steps to reach the end; k>1 leaves room
    for the measure/write around each skim. cap is a runtime backstop, raised to 3000."""
    dur = float(sample["t1"] - sample["t0"])
    step = min(skim_max, 300.0)          # effective coverage stride the agent is nudged toward
    need = int((dur / step) * k)
    return max(floor, min(cap, need))


def run_one(sample, client, served, tier="signal", max_rounds=60, temperature=0.3,
            skim_max=1800.0, cap=3000, allow_rewind=True):
    rounds = budget_for(sample, max_rounds, cap=cap, skim_max=skim_max)
    ss = StreamSession(sample, allow_rewind=allow_rewind, tool_mode="cooked",
                       max_actions=rounds + 5, skim_max=skim_max, tool_tier=tier)
    kind = sample.get("answer", {}).get("kind", "")
    sys_prompt = system_for(sample["task"], tier, kind)
    tools = tools_for(tier)
    o = ss.reset()
    last = None
    trajectory = []   # full per-round record: what the model requested + what the env returned
    for _ in range(rounds):
        obs_text = state_text(o, last, tier, sample["task"], kind)
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": obs_text}]
        _xb = ({"thinking": {"type": "disabled"}} if "deepseek" in served.lower()
               else {"chat_template_kwargs": {"enable_thinking": False}})
        try:
            r = client.chat.completions.create(model=served, messages=msgs, tools=tools,
                                               tool_choice="required", temperature=temperature,
                                               max_tokens=256, extra_body=_xb)
            tc = r.choices[0].message.tool_calls
            if tc:
                name = tc[0].function.name
                args = json.loads(tc[0].function.arguments or "{}")
            else:
                name, args = "skim", {"duration": 300}
        except Exception as e:
            last = {"error": str(e)[:100]}
            trajectory.append({"obs": obs_text, "req_name": None, "req_args": None,
                               "api_error": str(e)[:200], "info": last})
            o, _, d, _ = ss.step({"op": "skim", "duration": 300})
            if d:
                break
            continue
        o, _, d, last = _apply(ss, name, args)
        # req_* = the model's raw decision (incl. its chosen tool); info = what the env actually did
        # (actual tool after any fallback, measurement/profile returned, rewind, error). Keeping both
        # lets us analyze tool-selection: requested vs allowed vs fallen-back.
        trajectory.append({"obs": obs_text, "req_name": name, "req_args": args, "info": last})
        if d:
            break
    if not ss.done:
        ss.done = True
        ss._score(None)
    pred = getattr(ss, "pred", None)
    row = {"id": ss.s["id"], "task": ss.s["task"], "pn_dir": ss.s["pn_dir"],
           "duration_s": round(ss.T - ss.t0, 1), "score": round(ss.reward or 0.0, 4),
           "skims": ss.skims, "measures": ss.measures, "writes": len(ss.ledger),
           "actions": ss.actions, "pred": pred, "gold": ss.s["answer"],
           "verifier": ss.s["verifier"], "trajectory": trajectory}
    lat = ss.latency_report()
    if lat:
        row["latency"] = lat
    if ss.s["task"] == "rare_event_search":
        row["pred_interval"] = (pred or {}).get("interval")
        row["gt_interval"] = ss.s["answer"].get("interval")
    return row


def load_pool(data, n, task):
    S = [json.loads(l) for l in open(paths.task_path(data))]
    if task:
        S = [s for s in S if s["task"] == task]
    S = [s for s in S if render.has_local(s["record"], s["pn_dir"])]
    if n and n < len(S):
        S = S[:: max(1, len(S) // n)][:n]
    return S


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    out = {}
    for t, rs in by.items():
        det = [r["latency"]["detected"] for r in rs if "latency" in r]
        nev = [r["latency"]["n_events"] for r in rs if "latency" in r]
        lat = [r["latency"]["mean_latency_s"] for r in rs
               if "latency" in r and r["latency"]["mean_latency_s"] is not None]
        out[t] = {"n": len(rs), "mean_score": round(float(np.mean([r["score"] for r in rs])), 4),
                  "event_recall": round(sum(det) / sum(nev), 4) if sum(nev) else None,
                  "mean_latency_s": round(float(np.mean(lat)), 1) if lat else None,
                  "mean_writes": round(float(np.mean([r["writes"] for r in rs])), 1),
                  "mean_skims": round(float(np.mean([r["skims"] for r in rs])), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ecgscroll_lh.jsonl")
    ap.add_argument("--task", default="episode_detection")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--served", default="qwen3-30b")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--tool_tier", choices=["signal", "signal+dx"], default="signal")
    ap.add_argument("--max_rounds", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--skim_max", type=float, default=1800.0)
    ap.add_argument("--cap", type=int, default=3000, help="max action budget per record")
    ap.add_argument("--rewind", type=int, default=1, help="1=allow rewind (default), 0=forbid")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results_skim.json")
    args = ap.parse_args()

    pool = load_pool(args.data, args.n, args.task)
    print(f"SKIM agent: {len(pool)} recs | served={args.served} | tier={args.tool_tier}", flush=True)
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda s: run_one(s, client, args.served, args.tool_tier,
                                             args.max_rounds, args.temperature, args.skim_max,
                                             args.cap, bool(args.rewind)),
                           pool))
    summ = summarize(rows)
    print("summary:")
    for t, d in summ.items():
        print(f"  {t}: {d}", flush=True)
    os.makedirs(paths.RESULTS, exist_ok=True)
    json.dump({"meta": {"served": args.served, "tool_tier": args.tool_tier, "agent": "skim",
                        "n": len(pool)}, "summary": summ, "rows": rows},
              open(paths.result_path(args.out), "w"), indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
