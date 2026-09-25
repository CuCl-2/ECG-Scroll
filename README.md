# ECG-Scroll

**A long-horizon, streaming (online, causal) benchmark and agent environment for interpreting long-duration (ambulatory / Holter) ECG.**

Reading a multi-hour Holter recording is not a batch problem done after the fact — telemetry is read *as it streams in*, and the events that matter (atrial-fibrillation episodes, ischemic ST changes, isolated ectopic beats) are sparse and scattered across tens of thousands of seconds. ECG-Scroll frames this as an **online sequential decision process**: a recording is streamed to an agent, and at any instant the agent can measure only the signal that has **already elapsed** — the future has not arrived (physically unobservable), and the past streams by and, unless committed to memory, is gone.

The "cannot see everything" constraint comes from **time's arrow**, not an imposed observation budget. That makes three target competencies *necessary* rather than decorative:

- **Memory** — past signal is expensive/impossible to revisit, so evidence must be carried forward; an event not written when it passes is effectively lost.
- **Tool use** — the agent never sees pixels; it reads the waveform only through signal-grounded measurements (RR variability, ST level, QRS morphology, …).
- **Planning** — decide what to measure now, at what granularity, and when to commit.

Every answer is scored by an **objective, rule-based verifier** (no model-as-judge). The streaming setting adds a metric batch evaluation cannot express: **detection latency** — how long after an event's true onset the agent records it (each memory write is cursor-timestamped).

---

## Tasks

| Task | Source DB | Answer | Verifier / Metric | N | Mean dur. |
|---|---|---|---|---|---|
| `episode_detection` | afdb, ltafdb | interval set | temporal F1 (IoU ≥ τ) | 107 | 20.5 h |
| `burden_quantification` (AF) | afdb, ltafdb | scalar | rel. error score | 107 | 20.5 h |
| `change_detection` | afdb | change-point (s) | hit@±10 s | 23 | 10.2 h |
| `ischemia_detection` | edb | interval set | temporal F1 (IoU ≥ 0.75) | 85 | 2.0 h |
| `burden_quantification` (PVC) | mitdb, ltdb | scalar | rel. error score | 55 | 3.1 h |
| `rare_event_search` | mitdb, ltdb | interval | hit@±15 s (midpoint tol.) | 13 | 0.5 h |

All verifiers return a score in `[0, 1]` and are unit-tested, including the detection-latency metric (`python -m ecgscroll.verify`).

---

## The agent environment

A recording streams in causally through a monotonic **cursor** ("now"); any view/tool request is clamped to `[·, cursor]` (no future signal). Two agent families plug into the same environment, differing only in *who decides where to look*:

### Planning agent (`run_llm_skim.py`) — the model drives its own scan
The core action is **`skim(duration)`**: the agent chooses how big the next slice is — coarse over a long quiet stretch, fine over a suspicious one. A large skim returns a *lossy* profile (a peak feature and where it lies), so a short buried event surfaces as an elevated peak the agent must zoom into. The action set is:

```
skim(duration)          advance the cursor and return the newly-elapsed span's lossy profile
measure(tool, t0, t1)   run one CHOSEN tool over an elapsed sub-window (no time advance)
write(start, end)        commit a cursor-timestamped finding to memory
```

`measure` is a **broad action**: the model names which tool to run. **Signal tools** return numbers only and are always available; **diagnostic tools** return a ready-made label and are exposed only in the `signal+dx` tier (the with/without-label ablation). Re-measuring elapsed signal before the current skim window is a costed *rewind* (or forbidden in the strict variant), so the past is reliably available only through what was written down.

### Judge agent (`run_llm_fc.py`) — planning removed
The environment walks the record chunk by chunk; the model only judges which elapsed chunks are events. Reading the two side by side isolates planning from perception and knowledge, which are held fixed.

### Rule Agent (`run_stream.py`) — signal-grounded reference
No language model: it walks the stream in fixed chunks, measures the elapsed slice, and flags a chunk when the relevant cue crosses a threshold (`--dx`) or when the diagnostic label fires (`+dx`).

### Signal tools — a clinician's reading order
`measure` dispatches to tools organized band by band, the way a cardiologist reads a strip:

| Tool | Target | Output (cooked) |
|---|---|---|
| `get_rate_rhythm` | rate, AF, pauses | hr_bpm, rr_mean/sd_ms, rr_irregularity (CV) |
| `get_p_wave` | flutter, conduction | p_present_frac, pr_mean_ms, p_amp, pp_regularity |
| `get_qrs` | ectopy, BBB, LVH | mean_qrs_ms, wide_beat_frac, ectopic_count/times |
| `get_st` | ischemia | st_mv per lead (J+80 ms vs PR baseline), sign |
| `get_t_wave` | ischemia, K⁺ | t_amp per lead, polarity |
| `get_qt` | long/short QT | qt_mean_ms, qtc_ms (Bazett) |

Every signal tool returns numbers only (no diagnostic label), all from beats located by an XQRS (Pan-Tompkins family) detector run once per record and cached. Each reports a compact summary by default and per-beat arrays on request (`tool_mode="raw"`). **Diagnostic tools** (`classify_rhythm`, `classify_beats`, `flag_ischemia`) instead return a ready-made label read from ground-truth annotations, causally clamped to the elapsed window.

---

## Install

Python 3.10+. Construction, verification, the environment, and the Rule Agent need **no GPU**:

```bash
pip install -r requirements.txt   # wfdb, numpy, scipy, matplotlib
```

Running the LLM agents additionally needs an OpenAI-compatible endpoint (e.g. a local vLLM server with tool-call parsing, or any hosted API). The parsers, environment, verifiers, and Rule Agent are **local-first**: once records are cached under `data/raw/<db>_local/` they run fully offline (`HF_HUB_OFFLINE=1`).

---

## Reproduce

```bash
# 1. download the source databases (~10 GB; resumes on re-run)
python -m scripts.download_data                 # or: --dbs afdb edb mitdb ltafdb ltdb

# 2. build every task JSONL from the raw records
python -m scripts.build_all
python -m ecgscroll.build_tasks_lh              # long-horizon AF (offline)

# 3. sanity-check verifiers + environment
python -m ecgscroll.verify                      # verifier + detection-latency unit tests
python -m ecgscroll.stream                      # streaming self-test

# 4. Rule Agent reference (offline, CPU)
python -m ecgscroll.run_stream --data ecgscroll_lh.jsonl --n 30 --chunk_s 30 --out results_stream_tool.json

# 5. LLM agents against an OpenAI-compatible server on :8000
#    Planning agent (model drives its own scan):
python -m ecgscroll.run_llm_skim --data ecgscroll_lh.jsonl --task episode_detection \
       --n 30 --served <model> --base_url http://127.0.0.1:8000/v1 --tool_tier signal+dx --out plan.json
#    Judge agent (planning removed):
python -m ecgscroll.run_llm_fc   --data ecgscroll_lh.jsonl --task episode_detection \
       --n 30 --served <model> --base_url http://127.0.0.1:8000/v1 --tool_tier signal+dx --out judge.json

# full sweeps (6 tasks x 2 tiers) against one server:
python scripts/run_skim_full.py <served> <mtag>   # Planning agent
python scripts/run_v7.py        <served> <mtag>   # Judge agent
```

To serve a model with vLLM and tool-call parsing:

```bash
python -m vllm.entrypoints.openai.api_server --model <snapshot> --served-model-name <served> \
   --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

---

## Layout

```
ECG-Scroll/
├── ecgscroll/                    Python package (all modules address data via paths.py)
│   ├── paths.py                  single source of truth for DATA / RAW / TASKS / RESULTS
│   ├── parse_{afdb,edb,mitdb,ltafdb,ltdb}.py   PhysioNet annotations -> gold events
│   ├── build_tasks*.py           build task JSONL from raw records (windowed + long-horizon)
│   ├── render.py                 local-first partial signal reader (fs, window slices)
│   ├── tools.py                  signal + diagnostic tools; the measure(tool) registry
│   ├── env.py                    streaming environment (cursor, caliper, write ledger, latency)
│   ├── stream.py                 StreamSession: the skim/measure/write engine (Planning agent)
│   ├── chunking.py               chunk-judge harness (Judge agent)
│   ├── verify.py                 rule-based verifiers + detection_latency
│   ├── run_llm_skim.py           Planning agent driver (LLM chooses skim granularity + tool)
│   ├── run_llm_fc.py             Judge agent driver (environment walks chunks, LLM judges)
│   └── run_stream.py             Rule Agent reference (signal threshold, no LLM)
├── scripts/
│   ├── download_data.py          fetch source DBs from PhysioNet
│   ├── build_all.py              (re)build every task JSONL
│   ├── run_skim_full.py          Planning-agent full sweep (6 tasks x 2 tiers)
│   ├── run_v7.py                 Judge-agent full sweep
│   └── make_*_fig.py             benchmark-statistics + behavior figures
├── data/
│   ├── raw/<db>_local/           raw PhysioNet records (.hea/.dat/.atr; git-ignored)
│   └── tasks/*.jsonl             the benchmark itself (prompt + gold + verifier per line)
└── requirements.txt
```

---

## Task JSONL schema

One JSON object per line:

```json
{
  "id": "afdb-04015-ep-full",
  "task": "episode_detection",
  "record": "04015", "pn_dir": "afdb", "lead": 0,
  "t0": 0.0, "t1": 36823.04,
  "answer": {"intervals": [[410.34, 478.42]]},
  "verifier": {"type": "temporal_f1", "params": {"tau": 0.5}}
}
```

`lead = -1` means both leads are shown and the agent must pick (ischemia). PVC-burden instances carry `answer.kind = "pvc_burden"` to disambiguate from AF burden.

---

## Data sources

All from [PhysioNet](https://physionet.org/); cite the original databases if you use ECG-Scroll.

| DB | PhysioNet ID | Records | Used for |
|---|---|---|---|
| MIT-BIH Atrial Fibrillation | `afdb` | 23 (~10 h) | AF episode / burden / change |
| European ST-T | `edb` | 90 (2 h) | ischemia detection |
| MIT-BIH Arrhythmia | `mitdb` | 48 (30 min) | PVC rare-event / burden |
| Long-Term AF | `ltafdb` | 84 (~24 h) | long-horizon AF (episode + burden) |
| MIT-BIH Long-Term (ST) | `ltdb` | 7 (~21 h) | long-horizon ectopy |

---

## License

Released under the MIT License (see `LICENSE`). The underlying PhysioNet databases retain their
own licenses; download them from PhysioNet and cite the original sources.
