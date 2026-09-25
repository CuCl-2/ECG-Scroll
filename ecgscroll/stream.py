"""ECG-Scroll streaming session — adaptive-granularity reading (the current design).

A recording streams in causally. The agent's core action is SKIM(duration): it chooses how
big the next skim window is. Big windows skim quiet regions cheaply at low resolution; small
windows still skim (not measure) but at finer resolution. Every time point is skimmed (nothing
is skipped) -- but a large window returns a LOSSY view (peak + coarse profile), so short buried
events show up as an elevated peak the agent must MEASURE precisely to localize. This makes the
choice of granularity the central planning problem, and it is fully causal (never sees past
`cursor`). SKIM is the clinician's fast page-through; MEASURE is stopping to read exact values.

Actions:
  {"op":"skim","duration":D}               advance the cursor by D seconds and return that span's
                                           profile at adaptive resolution (peak + sub-bins + rhythm)
  {"op":"measure","tool":T,"t0":..,"t1":..} run a chosen tool over an elapsed sub-window. `tool` is
                                           a broad menu: signal tools (numbers only) always, plus
                                           diagnostic tools (ready-made labels) in the signal+dx tier
  {"op":"write","entry":{type,interval}}   record a confirmed event to memory (cursor-timestamped)
  {"op":"answer","answer":{..}}            finish; if omitted, memory is auto-summarized at stream end

`measure` on already-elapsed signal before the current skim window is a costed "rewind"
(or forbidden if allow_rewind=False). Detection latency = onset -> first matching write.
"""
import numpy as np

from ecgscroll import tools, verify, render

_TASK_PROMPT = {
    "episode_detection": "Report ALL atrial-fibrillation episodes in this recording as time "
                         "intervals [start_s, end_s].",
    "burden_quantification": "Report the atrial-fibrillation burden: fraction of the whole "
                             "recording spent in AF (a single number in [0,1]).",
    "ischemia_detection": "Report ALL ischemic ST-episode intervals [start_s, end_s].",
    "rare_event_search": "Find the single short target event; report its interval [start_s, end_s].",
    "change_detection": "Report the time (s) at which atrial fibrillation begins.",
}
_FEATURE = {"episode_detection": "rr_irregularity", "burden_quantification": "rr_irregularity",
            "change_detection": "rr_irregularity", "ischemia_detection": "st_level",
            "rare_event_search": "ectopy"}

MIN_OBS, MAX_OBS = 5.0, 1800.0  # skim() duration bounds (s): 5s fine .. 30min coarse
POINT_MAX_WRITE_S = 30.0  # max width of a pointwise write (PVC / rare-event: a single beat ~1s)


class StreamSession:
    def __init__(self, sample, allow_rewind=True, tool_mode="cooked", max_actions=300,
                 skim_max=MAX_OBS, tool_tier="signal"):
        self.s = sample
        self.allow_rewind = allow_rewind
        self.tool_mode = tool_mode
        self.tool_tier = tool_tier  # "signal" (default) | "signal+dx" (diagnostic tools exposed)
        self.max_actions = max_actions
        self.skim_max = skim_max  # cap on skim() duration (granularity ablation)
        self.feature = _FEATURE.get(sample["task"], "rr_irregularity")
        # burden_quantification covers both AF burden (rhythm) and PVC burden (ectopy);
        # disambiguate by the answer kind so measure() dispatches to the right signal tool.
        if sample.get("answer", {}).get("kind") == "pvc_burden":
            self.feature = "ectopy"
        _, self.fs = render.get_signal(sample["record"], sample["pn_dir"])
        self.lead = sample.get("lead", 0)
        if self.lead is None or self.lead < 0:
            self.lead = 0
        self.reset()

    def reset(self):
        self.t0 = self.s["t0"]
        self.T = self.s["t1"]
        self.cursor = self.t0            # nothing skimmed yet; skim() moves it forward
        self.win_start = self.t0         # start of the most recent skim window
        self.ledger = []
        self.actions = self.skims = self.measures = self.rewinds = 0
        self.done = False
        self.reward = None
        self.trajectory = []
        return self.observe_state(None)

    def observe_state(self, last):
        o = {"task": self.s["task"], "prompt": self.s.get("prompt") or _TASK_PROMPT[self.s["task"]],
             "feature": self.feature, "now_s": round(self.cursor, 1), "total_s": round(self.T, 1),
             "start_s": round(self.t0, 1), "actions_used": self.actions,
             "rewind_allowed": self.allow_rewind, "stream_end": self.cursor >= self.T,
             "ledger": list(self.ledger)}
        if last:
            o["last"] = last
        return o

    def step(self, action):
        if self.done:
            raise RuntimeError("session finished; call reset()")
        op = action.get("op")
        info = {"op": op}
        self.actions += 1

        if op == "skim":
            d = float(action.get("duration", 60.0))
            d = max(MIN_OBS, min(self.skim_max, d))
            a, b = self.cursor, min(self.T, self.cursor + d)
            self.win_start = a
            self.cursor = b
            self.skims += 1
            prof = tools.profile(self.s["record"], self.s["pn_dir"], a, b, self.fs,
                                 feature=self.feature, lead=self.lead)
            info["skimmed_s"] = [round(a, 1), round(b, 1)]
            info["profile"] = prof

        elif op == "measure":
            a = float(action.get("t0", self.win_start))
            b = min(float(action.get("t1", self.cursor)), self.cursor)  # causal
            a = max(self.t0, min(a, b - 1.0))
            if a < self.win_start - 1e-6:  # looking before the current skim window = rewind
                if not self.allow_rewind:
                    info["error"] = "rewind_forbidden: past signal not re-observable; use memory"
                    return self._ret(info)
                self.rewinds += 1
                info["rewind"] = True
            # measure is a BROAD action: the model names which tool to run this call. Signal tools
            # are always available; diagnostic tools only in the signal+dx tier. An omitted/unknown/
            # disallowed name falls back to this task's default signal tool (keeps parity with the
            # old auto-dispatch and never crashes on a malformed call).
            tool = action.get("tool")
            rec, pn = self.s["record"], self.s["pn_dir"]
            if tool in tools.DIAGNOSTIC_TOOLS:
                if self.tool_tier != "signal+dx":
                    info["error"] = "diagnostic tools disabled in signal-only tier"
                    return self._ret(info)
                self.measures += 1
                info["tool"] = tool
                info["measured_s"] = [round(a, 1), round(b, 1)]
                info["measurement"] = tools.DIAGNOSTIC_TOOLS[tool](rec, pn, a, b)
            else:
                if tool not in tools.SIGNAL_TOOLS:
                    tool = tools.DEFAULT_SIGNAL_TOOL.get(self.feature, "get_rate_rhythm")
                self.measures += 1
                info["tool"] = tool
                info["measured_s"] = [round(a, 1), round(b, 1)]
                info["measurement"] = tools.SIGNAL_TOOLS[tool](rec, pn, a, b, self.fs,
                                                               self.lead, self.tool_mode)

        elif op == "write":
            e = dict(action.get("entry", {}))
            iv = e.get("interval")
            # Per-beat / single-target tasks (PVC burden, rare-event search) write ONE ectopic beat
            # (~1 s) at a time. A span covering a large fraction of the record is a degenerate "flag
            # everything" write: it forces PVC burden -> 100% and puts a rare-event midpoint at the
            # record centre (guaranteed miss). Reject it with feedback instead of poisoning the
            # answer; the agent must localize the beat and write a tight interval around it.
            is_pointwise = (self.s.get("answer", {}).get("kind") == "pvc_burden"
                            or self.s["task"] == "rare_event_search")
            if is_pointwise and isinstance(iv, (list, tuple)) and len(iv) == 2:
                span = float(iv[1]) - float(iv[0])
                if span > POINT_MAX_WRITE_S:
                    info["error"] = (f"write rejected: interval {round(span,1)}s is too wide "
                                     f"(> {POINT_MAX_WRITE_S:.0f}s). Each target is a single beat; measure "
                                     f"to find the exact time and write a tight ~2s interval around it.")
                    return self._ret(info)
            e["t_recorded"] = round(self.cursor, 2)
            self.ledger.append(e)
            info["written"] = e

        elif op == "answer":
            self.done = True
            info["score"] = self._score(action.get("answer"))
            return self._ret(info)

        else:
            info["error"] = f"unknown op {op}"

        if self.actions >= self.max_actions:
            self.done = True
            info["forced_answer"] = True
            info["score"] = self._score(None)
        return self._ret(info)

    def _ret(self, info):
        self.trajectory.append(info)
        return self.observe_state(info), (self.reward or 0.0), self.done, info

    # ---- scoring: explicit answer, or auto-summary from memory ledger ----
    def _memory_answer(self):
        task = self.s["task"]
        ivs = [e["interval"] for e in self.ledger if "interval" in e and len(e["interval"]) == 2]
        if task in ("episode_detection", "ischemia_detection"):
            return {"intervals": _merge(ivs)}
        if task == "rare_event_search":
            return {"interval": ivs[0] if ivs else [0.0, 0.0]}
        if task == "burden_quantification":
            flagged = sum(b - a for a, b in _merge(ivs))
            tot = self.T - self.t0
            return {"value": round(flagged / tot, 4) if tot > 0 else 0.0,
                    "kind": self.s["answer"].get("kind", "af_burden")}
        if task == "change_detection":
            return {"change_point": ivs[0][0] if ivs else 0.0, "kind": "af_onset"}
        return {}

    def _score(self, answer):
        if answer is None or (isinstance(answer, dict) and not answer):
            pred = self._memory_answer()
        else:
            from ecgscroll.env import normalize_answer
            pred = normalize_answer(self.s["task"], answer)
        self.reward = verify.score(self.s["verifier"], pred, self.s["answer"])
        self.pred = pred
        return self.reward

    def latency_report(self, tau=0.5):
        gt = self.s.get("answer", {}).get("intervals")
        if gt is None:
            return None
        writes = [{"interval": e["interval"], "t_recorded": e["t_recorded"]}
                  for e in self.ledger if "interval" in e and "t_recorded" in e]
        return verify.detection_latency(gt, writes, tau)


def _merge(intervals):
    ivs = sorted([list(v) for v in intervals if len(v) == 2])
    if not ivs:
        return []
    out = [ivs[0][:]]
    for a, b in ivs[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


if __name__ == "__main__":
    import json
    from ecgscroll import paths
    inst = [json.loads(l) for l in open(paths.task_path("ecgscroll_lh.jsonl"))
            if '"afdb-04015-ep-full"' in l][0]
    print("instance:", inst["id"], "| GT episodes:", len(inst["answer"]["intervals"]))
    ss = StreamSession(inst)
    o = ss.reset()
    # scripted adaptive agent: coarse-skim; on an elevated peak, MEASURE that sub-range
    # finely to pin tight AF boundaries, then write. Demonstrates the harness is solvable when
    # granularity is used adaptively.
    steps = 0
    while not ss.done and steps < 400:
        steps += 1
        o, r, d, info = ss.step({"op": "skim", "duration": 300})
        prof = info.get("profile", {})
        if o["stream_end"]:
            break
        if prof.get("peak_value", 0) > 0.15:            # suspicious block -> measure finely
            za, zb = prof["peak_range_s"]
            # measure 10s sub-windows in the peak range to pin tight AF boundaries
            t = za
            run_start = None
            while t < zb and not ss.done:
                _, _, _, mi = ss.step({"op": "measure", "t0": t, "t1": min(zb, t + 10)})
                irr = mi.get("measurement", {}).get("rr_irregularity", 0)
                if irr > 0.15 and run_start is None:
                    run_start = t
                elif irr <= 0.15 and run_start is not None:
                    ss.step({"op": "write", "entry": {"type": "AFIB", "interval": [run_start, t]}})
                    run_start = None
                t += 10
            if run_start is not None:
                ss.step({"op": "write", "entry": {"type": "AFIB", "interval": [run_start, zb]}})
    if not ss.done:
        _, r, _, _ = ss.step({"op": "answer", "answer": None})
    print(f"scripted-adaptive reward={ss.reward:.3f} writes={len(ss.ledger)} "
          f"skims={ss.skims} measures={ss.measures}")
    print("latency:", ss.latency_report())
