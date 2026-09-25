"""ECG-Scroll agent environment (POMDP over a scrollable long recording).

The agent observes one tile at a time and issues navigation / tool / memory / answer
actions until it answers or exhausts its observation budget. Tiles are rendered lazily.
Every answer is scored by an objective verifier (see verify.py).

Action dicts (op-tagged):
  {"op":"zoom","t0":..,"t1":..}      reveal a detail tile of [t0,t1]  (costs 1 observation)
  {"op":"goto","t":..,"span":..}     center a `span`-second window at t (costs 1 observation)
  {"op":"next"} / {"op":"prev"}      shift viewport by current span    (costs 1 observation)
  {"op":"caliper","t0":..,"t1":..,"lead":..}   signal-grounded measurement (tool call)
  {"op":"write","entry":{...}}       append to findings ledger
  {"op":"answer","answer":{...}}     terminate and score
"""
import os
import numpy as np
from scipy.signal import find_peaks
from ecgscroll import render, verify, paths
import wfdb

# Downloading records from PhysioNet may require a proxy on some networks;
# set HTTP_PROXY / HTTPS_PROXY in your environment if needed (no default here).

_TASK_PROMPT = {
    "episode_detection": "Report all atrial-fibrillation episodes in this window as "
                         "intervals [start_s, end_s].",
    "burden_quantification": "Report the atrial-fibrillation burden (fraction of time in AF) "
                            "for the whole recording.",
    "change_detection": "Report the time (seconds) at which atrial fibrillation begins.",
    "rare_event_search": "Find the single short atrial-fibrillation episode and report its "
                        "interval [start_s, end_s].",
    "ischemia_detection": "Report all ischemic ST-segment episodes in this two-lead window "
                          "as intervals [start_s, end_s] (ST deviation may appear in either "
                          "lead; empty list if none).",
}


def _to_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize_answer(task, ans):
    """Coerce the model's free-form answer into the schema the verifier expects.
    Handles common variants: bare list of intervals, {"intervals":..}, {"interval":..},
    bare number, {"value":..}/{"change_point":..}, strings, etc."""
    def as_intervals(a):
        if isinstance(a, dict):
            a = a.get("intervals", a.get("interval", a.get("episodes", [])))
        if isinstance(a, (int, float)):
            return []
        if isinstance(a, list):
            if a and isinstance(a[0], (int, float)) and len(a) == 2:
                return [[float(a[0]), float(a[1])]]  # a single [t0,t1]
            out = []
            for it in a:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    n0, n1 = _to_num(it[0]), _to_num(it[1])
                    if n0 is not None and n1 is not None:
                        out.append([n0, n1])
            return out
        return []

    if ans is None:
        ans = {}
    if task in ("episode_detection", "ischemia_detection"):
        return {"intervals": as_intervals(ans)}
    if task == "rare_event_search":
        ivs = as_intervals(ans)
        return {"interval": ivs[0] if ivs else [0.0, 0.0]}
    if task == "burden_quantification":
        v = ans.get("value") if isinstance(ans, dict) else ans
        v = _to_num(v)
        return {"value": v if v is not None else 0.0, "kind": "af_burden"}
    if task == "change_detection":
        t = ans.get("change_point") if isinstance(ans, dict) else ans
        t = _to_num(t)
        return {"change_point": t if t is not None else 0.0, "kind": "af_onset"}
    if task == "ischemia_localization":
        d = ans if isinstance(ans, dict) else {}
        lead = d.get("lead")
        lead = int(lead) if _to_num(lead) is not None else 0
        ivs = as_intervals(d.get("interval", d.get("intervals", [])))
        st = _to_num(d.get("st_mv", d.get("st", 0.0)))
        return {"lead": lead, "interval": ivs[0] if ivs else [0.0, 0.0],
                "st_mv": abs(st) if st is not None else 0.0}
    return ans if isinstance(ans, dict) else {}


class EcgScrollEnv:
    """POMDP/streaming environment over a long ECG recording.

    Two modes:
      * streaming=False (legacy): the whole recording [t0,t1] is navigable at once; `budget`
        caps the number of observation actions. Kept for the windowed tasks and the scripted
        tool-agents that scan a fixed window left-to-right.
      * streaming=True (online/causal): the recording arrives chunk by chunk. A monotonic
        `cursor` marks "now"; the agent can never observe signal past the cursor (the future
        has not happened yet). `advance` moves time forward by one chunk. Re-reading already-
        elapsed signal (t < cursor) is allowed only if allow_rewind=True and costs budget --
        pressuring the agent to commit evidence to memory instead of re-scanning. Every
        `write`/`answer` is timestamped with the cursor so detection LATENCY (how long after an
        event's onset the agent first records it) is measurable -- a metric batch eval cannot give.
    """

    def __init__(self, sample: dict, budget: int = 12, tile_dir: str = "/tmp/ecgscroll_tiles",
                 default_span: float = 60.0, streaming: bool = False,
                 chunk_s: float = 30.0, allow_rewind: bool = True, render_tiles: bool = True):
        self.s = sample
        self.budget = budget
        self.default_span = default_span
        self.tile_dir = tile_dir
        self.streaming = streaming
        self.chunk_s = chunk_s
        self.allow_rewind = allow_rewind
        self.render_tiles = render_tiles  # headless signal-tool agents can skip PNG rendering
        os.makedirs(tile_dir, exist_ok=True)
        _, self.fs = render.get_signal(sample["record"], sample["pn_dir"])
        self.reset()

    def reset(self):
        self.obs_used = 0
        self.tool_calls = 0
        self.rewinds = 0
        self.ledger = []
        self.done = False
        self.reward = None
        self.trajectory = []
        if self.streaming:
            # time starts at t0; the first chunk [t0, t0+chunk_s] has just arrived.
            self.cursor = min(self.s["t1"], self.s["t0"] + self.chunk_s)
            self.view = (self.s["t0"], self.cursor)
        else:
            self.cursor = self.s["t1"]  # everything visible up front
            self.view = (self.s["t0"], min(self.s["t1"], self.s["t0"] + self._init_span()))
        tile = self._render(self.view, overview=True)
        return self._obs(tile)

    def _init_span(self):
        full = self.s["t1"] - self.s["t0"]
        return min(full, max(self.default_span, full / 12))

    def _render(self, view, overview=False):
        if not self.render_tiles:
            return None  # headless mode: signal-level tool agents don't need image tiles
        t0, t1 = view
        fn = f"{self.s['record']}_{int(t0)}_{int(t1)}_{'ov' if overview else 'zm'}.png"
        path = os.path.join(self.tile_dir, fn)
        if self.s["lead"] == -1:  # ischemia: show both leads, agent must pick
            return render.render_tile_multi(self.s["record"], self.s["pn_dir"], t0, t1,
                                            leads=(0, 1), out_path=path, fs=self.fs)
        return render.render_tile(self.s["record"], self.s["pn_dir"], t0, t1,
                                  lead=self.s["lead"], out_path=path, fs=self.fs)

    def _clamp_future(self, t0, t1):
        """In streaming mode, no view may extend past `cursor` (the future is unobserved).
        Also enforce a minimum non-degenerate span so the renderer never sees an empty slice."""
        if self.streaming:
            t1 = min(t1, self.cursor)
        min_span = min(2.0, self.s["t1"] - self.s["t0"])
        if t1 - t0 < min_span:
            t0 = max(self.s["t0"], t1 - min_span)
        return t0, t1

    def _obs(self, tile_path):
        o = {
            "task": self.s["task"],
            "prompt": self.s.get("prompt") or _TASK_PROMPT.get(self.s["task"], ""),
            "tile_path": tile_path,
            "viewport": list(self.view),
            "recording_span": [self.s["t0"], self.s["t1"]],
            "ledger": list(self.ledger),
            "budget_left": self.budget - self.obs_used,
        }
        if self.streaming:
            o["cursor"] = self.cursor            # "now": signal is observable only up to here
            o["stream_end"] = self.cursor >= self.s["t1"]
        return o

    def _caliper(self, t0, t1, lead):
        """Signal-grounded measurement over [t0,t1]: HR, mean/std RR, irregularity, and
        ST-segment deviation (mV) per lead (J+80ms vs PR baseline)."""
        sig, names, fs = render.read_window(self.s["record"], self.s["pn_dir"], t0, t1, self.fs)
        meas = {}
        ml = 0 if (lead is None or lead < 0) else lead
        x = sig[:, ml]
        xn = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)
        peaks, _ = find_peaks(xn, distance=int(0.25 * fs), height=1.0)  # >=240 bpm cap
        if len(peaks) >= 3:
            rr = np.diff(peaks) / fs
            meas.update({
                "n_beats": int(len(peaks)),
                "hr_bpm": round(60.0 / np.mean(rr), 1),
                "rr_mean_s": round(float(np.mean(rr)), 3),
                "rr_std_s": round(float(np.std(rr)), 3),
                "rr_cv": round(float(np.std(rr) / (np.mean(rr) + 1e-6)), 3),  # AF cue: high CV
            })
        else:
            meas.update({"n_beats": int(len(peaks)), "hr_bpm": None, "rr_cv": None})
        # ST deviation per lead (ischemia cue: |ST| >= 0.1 mV)
        for L in range(sig.shape[1]):
            meas[f"st_mv_lead{L}"] = self._st_dev(sig[:, L], fs)
        return meas

    @staticmethod
    def _st_dev(x, fs):
        """Mean ST deviation (mV) at J+80ms (~R+120ms) vs PR baseline (~R-40ms)."""
        x = x.astype(float)
        xn = x - np.nanmedian(x)
        amp = np.nanpercentile(np.abs(xn), 99) + 1e-6
        peaks, _ = find_peaks(xn, distance=int(0.4 * fs), height=0.4 * amp)
        b, st = int(0.04 * fs), int(0.12 * fs)
        devs = [x[r + st] - x[r - b] for r in peaks if r - b >= 0 and r + st < len(x)]
        return round(float(np.mean(devs)), 3) if devs else 0.0

    def _advance(self, n_chunks=1):
        """Stream in the next chunk(s): move `cursor` forward by n_chunks * chunk_s.
        Returns True if the recording just ended."""
        self.cursor = min(self.s["t1"], self.cursor + n_chunks * self.chunk_s)
        # default view: the newly-arrived slice ending at the (new) present
        self.view = (max(self.s["t0"], self.cursor - self.chunk_s), self.cursor)
        return self.cursor >= self.s["t1"]

    def _write(self, entry):
        """Append a findings-ledger entry, stamping it with the current cursor (for latency)."""
        e = dict(entry)
        if self.streaming:
            e.setdefault("t_recorded", round(self.cursor, 2))
        self.ledger.append(e)

    def step(self, action: dict):
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        op = action.get("op")
        info = {"op": op}
        obs_tile = None

        if op == "advance":  # streaming only: let time pass, stream in the next chunk
            if not self.streaming:
                info["error"] = "advance is a streaming-only op"
            else:
                ended = self._advance(int(action.get("n", 1)))
                info["cursor"] = self.cursor
                info["stream_end"] = ended
                obs_tile = self._render(self.view, overview=(self.view[1] - self.view[0] > 12))

        elif op in ("zoom", "goto", "next", "prev"):
            if self.obs_used >= self.budget:
                self.done = True
                self.reward = 0.0
                info["error"] = "budget_exhausted"
                return self._obs(self._render(self.view)), 0.0, True, info
            lo, hi = self.s["t0"], (self.cursor if self.streaming else self.s["t1"])
            if op == "zoom":
                self.view = (max(lo, action["t0"]), min(hi, action["t1"]))
            elif op == "goto":
                span = action.get("span", self.default_span)
                c = action["t"]
                self.view = (max(lo, c - span / 2), min(hi, c + span / 2))
            elif op in ("next", "prev"):
                span = self.view[1] - self.view[0]
                shift = span if op == "next" else -span
                nt0 = min(max(lo, self.view[0] + shift), max(lo, hi - span))
                self.view = (nt0, min(hi, nt0 + span))
            self.view = self._clamp_future(*self.view)
            self.obs_used += 1
            # streaming: revisiting elapsed signal (strictly before the current chunk) is a
            # "rewind" -- allowed only if permitted, and it costs an extra budget unit.
            if self.streaming and self.view[0] < self.cursor - self.chunk_s - 1e-6:
                if not self.allow_rewind:
                    self.done = True
                    self.reward = 0.0
                    info["error"] = "rewind_forbidden"
                    return self._obs(self._render(self.view)), 0.0, True, info
                self.rewinds += 1
                self.obs_used += 1  # rewinding past signal is more expensive than moving on
                info["rewind"] = True
            obs_tile = self._render(self.view, overview=(self.view[1] - self.view[0] > 12))
            info["viewport"] = list(self.view)

        elif op == "caliper":
            self.tool_calls += 1
            t0, t1 = self._clamp_future(action["t0"], action["t1"])
            info["measurement"] = self._caliper(t0, t1, action.get("lead", self.s["lead"]))
            obs_tile = self._render(self.view, overview=(self.view[1] - self.view[0] > 12))

        elif op == "write":
            self._write(action["entry"])
            obs_tile = self._render(self.view, overview=(self.view[1] - self.view[0] > 12))

        elif op == "answer":
            self.done = True
            pred = normalize_answer(self.s["task"], action.get("answer"))
            self.reward = verify.score(self.s["verifier"], pred, self.s["answer"])
            info["score"] = self.reward
            self.trajectory.append(info)
            return self._obs(self._render(self.view)), self.reward, True, info
        else:
            info["error"] = f"unknown op {op}"

        self.trajectory.append(info)
        return self._obs(obs_tile), 0.0, self.done, info

    def latency_report(self, tol_iou=0.5):
        """Detection-latency summary for streaming episodes. Requires interval ground truth
        and a findings ledger whose entries carry `interval` + `t_recorded` (stamped by _write).

        For each GT episode [a,b], latency = (first cursor time at which a ledger entry's
        interval overlaps it) - a. Events never recorded count as `missed`. Pure rule-based."""
        gt = self.s.get("answer", {}).get("intervals", [])
        writes = [e for e in self.ledger if "interval" in e and "t_recorded" in e]
        lats, missed = [], 0
        for a, b in gt:
            hit_t = None
            for e in writes:
                ea, eb = e["interval"]
                inter = max(0.0, min(b, eb) - max(a, ea))
                union = (b - a) + (eb - ea) - inter
                if union > 0 and inter / union >= tol_iou and e["t_recorded"] >= a:
                    hit_t = e["t_recorded"] if hit_t is None else min(hit_t, e["t_recorded"])
            if hit_t is None:
                missed += 1
            else:
                lats.append(max(0.0, hit_t - a))
        return {"n_events": len(gt), "detected": len(lats), "missed": missed,
                "mean_latency_s": round(float(np.mean(lats)), 2) if lats else None,
                "latencies_s": [round(x, 2) for x in lats]}


if __name__ == "__main__":
    import json
    path = paths.task_path("ecgscroll_afdb.jsonl")
    # find an episode_detection sample with AF for a scripted-agent test
    samp = None
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o["task"] == "episode_detection" and o["answer"]["intervals"]:
                samp = o; break
    print("sample:", samp["id"], "GT intervals:", samp["answer"]["intervals"])
    env = EcgScrollEnv(samp, budget=8)
    o = env.reset()
    print("reset ok. tile:", os.path.basename(o["tile_path"]), "budget:", o["budget_left"])
    # scripted "oracle-ish" agent: zoom to first GT interval, caliper it, answer GT
    gt = samp["answer"]["intervals"]
    o, r, d, i = env.step({"op": "zoom", "t0": gt[0][0] - 2, "t1": gt[0][1] + 2})
    print("zoom ->", i.get("viewport"), "budget:", o["budget_left"])
    o, r, d, i = env.step({"op": "caliper", "t0": gt[0][0], "t1": gt[0][1],
                           "lead": samp["lead"]})
    print("caliper ->", i["measurement"])
    o, r, d, i = env.step({"op": "write", "entry": {"type": "AFIB", "interval": gt[0]}})
    # correct answer
    o, r, d, i = env.step({"op": "answer", "answer": {"intervals": gt}})
    print(f"CORRECT answer -> reward={r:.3f} (expect 1.0), obs_used={env.obs_used}, tools={env.tool_calls}")
    # wrong answer on a fresh env
    env2 = EcgScrollEnv(samp, budget=8); env2.reset()
    _, r2, _, _ = env2.step({"op": "answer", "answer": {"intervals": [[0, 1]]}})
    print(f"WRONG answer -> reward={r2:.3f} (expect 0.0)")
    # budget exhaustion
    env3 = EcgScrollEnv(samp, budget=2); env3.reset()
    env3.step({"op": "next"}); _, r3, d3, i3 = env3.step({"op": "next"})
    _, r3, d3, i3 = env3.step({"op": "next"})
    print(f"budget exhaustion -> done={d3}, reward={r3:.3f}, err={i3.get('error')}")

    # ---- streaming-mode self-tests ----
    print("\n=== streaming mode ===")
    env4 = EcgScrollEnv(samp, budget=1000, streaming=True, chunk_s=30.0)
    o = env4.reset()
    print(f"reset: cursor={o['cursor']:.0f}s, view={[round(x) for x in o['viewport']]}, "
          f"stream_end={o['stream_end']}")
    # (a) cannot observe the future: request a view well past the cursor
    o, r, d, i = env4.step({"op": "goto", "t": o["cursor"] + 10000, "span": 60})
    print(f"(a) goto future -> viewport={[round(x) for x in i['viewport']]} "
          f"(clamped to cursor {env4.cursor:.0f}) -- OK={i['viewport'][1] <= env4.cursor + 1e-6}")
    # (b) rewind into elapsed signal costs an extra budget unit
    for _ in range(5):
        env4.step({"op": "advance", "n": 1})  # cursor now well ahead
    before = env4.obs_used
    o, r, d, i = env4.step({"op": "goto", "t": samp["t0"] + 5, "span": 20})  # look far back
    print(f"(b) rewind -> flagged={i.get('rewind')}, budget_used +{env4.obs_used - before} "
          f"(expect 2: 1 move + 1 rewind penalty)")
    # (c) oracle latency: write each GT event exactly when the cursor first passes its onset
    env5 = EcgScrollEnv(samp, budget=100000, streaming=True, chunk_s=30.0)
    env5.reset()
    gt_sorted = sorted(gt)
    gi = 0
    while env5.cursor < samp["t1"] and gi < len(gt_sorted):
        if env5.cursor >= gt_sorted[gi][0]:  # onset has streamed in -> record immediately
            env5.step({"op": "write", "entry": {"type": "AFIB", "interval": gt_sorted[gi]}})
            gi += 1
        else:
            env5.step({"op": "advance", "n": 1})
    rep = env5.latency_report()
    print(f"(c) oracle latency: detected={rep['detected']}/{rep['n_events']}, "
          f"mean_latency={rep['mean_latency_s']}s (expect small, < chunk_s=30)")

