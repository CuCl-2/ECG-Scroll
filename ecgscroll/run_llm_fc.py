"""LLM streaming agent — HARNESS-DRIVEN chunk scan + batched LLM judgment.

The harness walks the whole recording chunk by chunk (like the Rule Agent), measuring each
freshly-arrived 30s chunk. It then shows the model the chunk measurements in time-ordered
BATCHES and asks a single question: which chunks in this batch belong to the target event? The
model never controls granularity or issues free-form tool calls -- it only *judges* chunks. This
removes the mechanical "the LLM never skims down to chunk resolution" failure and makes the
comparison a pure test of reading the signal (--dx) or using a ready label (+dx).

Two tiers:
  * signal   (--dx): each chunk is shown its signal cue (rr_cv / st_mv / wide_beat_frac); the
    model must decide from the numbers which chunks are AF / ischemic / ectopic.
  * signal+dx(+dx):  each chunk is shown the diagnostic label; the model decides from the label.

Flagged chunks are written to the env ledger with the cursor timestamp, so detection latency is
measured exactly as before. End-of-stream assembly + verifier are shared with the Rule Agent via
ecgscroll.chunking, guaranteeing an identical protocol.

Start the vLLM server with tool-call parsing:
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
from ecgscroll import chunking

# ---- per-task descriptions of what an "event" chunk is, for each tier ----------------------
_EVENT = {
    "episode_detection":     ("atrial fibrillation (AF)", "rr_cv"),
    "burden_quantification": ("atrial fibrillation (AF)", "rr_cv"),
    "change_detection":      ("atrial fibrillation (AF)", "rr_cv"),
    "ischemia_detection":    ("an ischemic ST episode", "st_mv"),
    "rare_event_search":     ("a ventricular ectopic (PVC) beat", "wide_beat_frac"),
}

_CUE_DOC = {
    "rr_cv": "rr_cv = R-R coefficient of variation. AF shows high rr_cv (>~0.15) vs a regular "
             "sinus rhythm (~0.02-0.08). hr_bpm is beats/min.",
    "st_mv": "st_mv = |ST deviation| in mV (max over leads). Ischemia shows sustained ST "
             "deviation (>~0.1 mV).",
    "wide_beat_frac": "wide_beat_frac = fraction of wide/abnormal QRS beats; ectopic_count = "
                      "number of wide beats. A PVC is a single wide beat, so a target chunk has "
                      "ectopic_count>=1.",
}

_TASK_GOAL = {
    "episode_detection":     "You are localizing every AF EPISODE. Flag every chunk that is in AF.",
    "burden_quantification": "You are quantifying AF BURDEN (fraction of the record in AF). Flag "
                             "every AF chunk; the burden is the fraction of chunks you flag.",
    "change_detection":      "You are finding the FIRST onset of AF. Flag the AF chunks; the "
                             "earliest flagged chunk is taken as the onset, so do not flag "
                             "isolated sinus noise before the real AF starts.",
    "ischemia_detection":    "You are localizing every ischemic ST EPISODE. Flag every ischemic "
                             "chunk.",
    "rare_event_search":     "You are finding the SINGLE short target beat (one PVC) in a long "
                             "otherwise-normal record. Flag only the chunk(s) that truly contain "
                             "the ectopic beat -- ideally exactly one.",
}


def system_for(task, tier="signal"):
    event, cue = _EVENT.get(task, _EVENT["episode_detection"])
    if tier == "signal+dx":
        obs = ("For each chunk you are given a ready-made diagnostic LABEL for that 30s window "
               "(e.g. rhythm class and af_frac, or an ischemia flag, or ectopic-beat times). "
               "Trust the label: flag the chunk when the label says it is " + event + ".")
    else:
        obs = ("For each chunk you are given signal MEASUREMENTS for that 30s window. "
               + _CUE_DOC.get(cue, "") + " Decide from the numbers which chunks are " + event + ".")
    return (
        "You are an expert cardiologist reading a long ambulatory ECG that streams in ONLINE, in "
        "time order, one 30-second chunk at a time. You cannot see the future. " + _TASK_GOAL.get(
            task, _TASK_GOAL["episode_detection"]) + "\n\n" + obs + "\n\n"
        "You will see chunks in time-ordered batches. For EACH batch, call flag_chunks(indices) "
        "with the list of chunk indices (from the 'i' field) that belong to the event. If none "
        "in the batch qualify, call flag_chunks with an empty list. Call flag_chunks exactly once "
        "per batch.")


FLAG_TOOL = [{"type": "function", "function": {
    "name": "flag_chunks",
    "description": "Report which chunks in the current batch belong to the target event.",
    "parameters": {"type": "object", "properties": {
        "indices": {"type": "array", "items": {"type": "integer"},
                    "description": "chunk indices (the 'i' field) that are event chunks; [] if none"}},
        "required": ["indices"]}}}]


def _chunk_line(c, tier):
    """One compact line per chunk for the batch prompt."""
    if tier == "signal+dx":
        lab = c.get("label", {})
        # keep the label compact
        return f"i={c['i']} t={c['t0']:.0f}-{c['t1']:.0f}s label={json.dumps(lab, separators=(',', ':'))}"
    parts = [f"i={c['i']}", f"t={c['t0']:.0f}-{c['t1']:.0f}s"]
    for k in ("rr_cv", "hr_bpm", "st_mv", "wide_beat_frac", "ectopic_count"):
        if k in c and c[k] is not None:
            parts.append(f"{k}={c[k]}")
    return " ".join(parts)


def run_one(sample, client, served, tier="signal", batch=60, chunk_s=30.0, temperature=0.2):
    dx = (tier == "signal+dx")
    env, chunks = chunking.scan_chunks(sample, chunk_s=chunk_s, dx=dx)
    sys_prompt = system_for(sample["task"], tier)
    etype = chunking.etype_for(sample)

    flagged_idx = set()
    n_calls = 0
    for b0 in range(0, len(chunks), batch):
        bchunks = chunks[b0:b0 + batch]
        lines = "\n".join(_chunk_line(c, tier) for c in bchunks)
        lo, hi = bchunks[0]["i"], bchunks[-1]["i"]
        user = (f"Batch chunks i={lo}..{hi} (of 0..{len(chunks)-1} total), time "
                f"{bchunks[0]['t0']:.0f}s..{bchunks[-1]['t1']:.0f}s:\n{lines}\n\n"
                f"Call flag_chunks(indices) with the event chunks in THIS batch.")
        _xb = ({"thinking": {"type": "disabled"}} if "deepseek" in served.lower()
               else {"chat_template_kwargs": {"enable_thinking": False}})
        try:
            r = client.chat.completions.create(
                model=served,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}],
                tools=FLAG_TOOL, tool_choice="required",
                temperature=temperature, max_tokens=2048, extra_body=_xb)
            n_calls += 1
            tc = r.choices[0].message.tool_calls
            if tc:
                args = json.loads(tc[0].function.arguments or "{}")
                for idx in args.get("indices", []):
                    try:
                        flagged_idx.add(int(idx))
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            # on API failure, skip this batch (flag nothing) but keep going
            _ = e

    # write flagged chunks to the ledger in time order (cursor-stamped for latency), then assemble
    by_i = {c["i"]: c for c in chunks}
    flagged, strengths = [], []
    for i in sorted(flagged_idx):
        c = by_i.get(i)
        if not c:
            continue
        flagged.append((c["t0"], c["t1"]))
        strengths.append(c.get("strength", 0.0))
        # stamp the write at the chunk end (when it would have streamed in) for a fair latency
        env.cursor = c["t1"]
        env.step({"op": "write", "entry": {"type": etype, "interval": [round(c["t0"], 2),
                                                                       round(c["t1"], 2)]}})
    env.cursor = sample["t1"]
    pred = chunking.assemble_answer(sample, flagged, strengths)
    from ecgscroll import verify
    reward = verify.score(sample["verifier"], pred, sample["answer"])

    row = {"id": sample["id"], "task": sample["task"], "pn_dir": sample["pn_dir"],
           "duration_s": round(sample["t1"] - sample["t0"], 1), "score": round(reward, 4),
           "n_chunks": len(chunks), "n_flagged": len(flagged), "n_calls": n_calls,
           # persist the full prediction + gold so metric-difficulty tweaks are OFFLINE rescoring
           # (no rerun): rescore pred vs sample["answer"] under a new verifier param.
           "pred": pred, "gold": sample["answer"], "verifier": sample["verifier"],
           "flagged_intervals": [[round(a, 2), round(b, 2)] for a, b in flagged]}
    if "intervals" in sample.get("answer", {}):
        row["latency"] = env.latency_report()
    if sample["task"] == "rare_event_search":
        row["pred_interval"] = pred.get("interval")
        row["gt_interval"] = sample["answer"].get("interval")
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
                  "mean_flagged": round(float(np.mean([r["n_flagged"] for r in rs])), 1),
                  "mean_calls": round(float(np.mean([r["n_calls"] for r in rs])), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ecgscroll_lh.jsonl")
    ap.add_argument("--task", default="episode_detection")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--served", default="qwen3-30b")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--tool_tier", choices=["signal", "signal+dx"], default="signal")
    ap.add_argument("--batch", type=int, default=60, help="chunks per LLM judgment call")
    ap.add_argument("--chunk_s", type=float, default=30.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results_fc.json")
    args = ap.parse_args()

    pool = load_pool(args.data, args.n, args.task)
    print(f"FC chunk-judge agent: {len(pool)} recs | served={args.served} | tier={args.tool_tier} | "
          f"batch={args.batch}", flush=True)
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda s: run_one(s, client, args.served, args.tool_tier,
                                             args.batch, args.chunk_s, args.temperature),
                           pool))
    summ = summarize(rows)
    print("summary:")
    for t, d in summ.items():
        print(f"  {t}: {d}", flush=True)
    os.makedirs(paths.RESULTS, exist_ok=True)
    json.dump({"meta": {"served": args.served, "tool_tier": args.tool_tier,
                        "batch": args.batch, "n": len(pool)},
               "summary": summ, "rows": rows},
              open(paths.result_path(args.out), "w"), indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
