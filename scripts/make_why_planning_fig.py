#!/usr/bin/env python
"""Figure: why the Planning agent fails while the Judge agent succeeds.
Left  : per-model skims / measures / findings-written, AGGREGATED across all six tasks (+dx).
        Scans heavily, inspects a fair amount on some models, yet commits ~nothing.
Right : fraction of recordings ending with zero committed findings, per model (all tasks).
Aggregation is task-equal-weighted: we average each model's per-task means over the six tasks
so long-record tasks (episode/afburden) do not dominate the short-record ones (pvc/rare).
Data straight from data/results/ (offline)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RES = os.path.join(ROOT, "data/results")
OUT = os.path.join(ROOT, "data", "figures", "why_planning.png")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_SKIM = "#6E8CB0"   # blue-slate: advancing / scanning
C_MEAS = "#D9A441"   # amber: inspecting
C_WRITE = "#C44E52"  # red: committing
C_ZERO = "#55606E"   # gray

# model tag -> display label (only those with data; order small->large)
CAND = [("8b", "Qwen3-8B"), ("30b", "Qwen3-30B"), ("235b", "Qwen3-235B"),
        ("30bthink", "Qwen3-30B-Think"), ("glm", "GLM-4.5-Air"),
        ("llama", "Llama-3.3-70B"), ("deepseek", "DeepSeek-V4.1")]

# DeepSeek's planning run uses the earlier exp_main_* naming; the rest use exp_skim_*.
# All six tasks are aggregated (task-equal-weighted) per model.
TASKS = ["episode", "afburden", "change", "ischemia", "pvcburden", "rare"]
_DS_TASK = {"episode": "episode", "afburden": "burden", "change": "change",
            "ischemia": "ischemia", "pvcburden": "pvcburden", "rare": "rare"}


def result_path(tag, task):
    if tag == "deepseek":
        return f"{RES}/exp_main_deepseek_{_DS_TASK[task]}_dx.json"
    return f"{RES}/exp_skim_{tag}_{task}_dx.json"


labels, skims, measures, writes, zerofrac = [], [], [], [], []
for tag, disp in CAND:
    # per-task means, then average across tasks (equal weight) so long records don't dominate
    sk_t, me_t, wr_t, zf_t = [], [], [], []
    for task in TASKS:
        f = result_path(tag, task)
        if not os.path.exists(f):
            continue
        rows = json.load(open(f))["rows"]
        if not rows:
            continue
        sk_t.append(np.mean([r.get("skims", 0) for r in rows]))
        me_t.append(np.mean([r.get("measures", 0) for r in rows]))
        wr_t.append(np.mean([r.get("writes", 0) for r in rows]))
        zf_t.append(np.mean([r.get("writes", 0) == 0 for r in rows]))
    if not sk_t:
        continue
    labels.append(disp)
    skims.append(float(np.mean(sk_t)))
    measures.append(float(np.mean(me_t)))
    writes.append(float(np.mean(wr_t)))
    zerofrac.append(float(np.mean(zf_t)))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.1),
                               gridspec_kw={"width_ratios": [1.5, 1.0]})

# --- Left: skims / measures / writes, grouped horizontal bars ---
y = np.arange(len(labels))[::-1]   # largest model on top reads naturally
h = 0.26
axL.barh(y + h, skims, height=h, color=C_SKIM, label="skim (advance cursor)",
         edgecolor="white", linewidth=0.6)
axL.barh(y, measures, height=h, color=C_MEAS, label="measure (inspect signal)",
         edgecolor="white", linewidth=0.6)
axL.barh(y - h, writes, height=h, color=C_WRITE, label="write (commit finding)",
         edgecolor="white", linewidth=0.6)
axL.set_yticks(y)
axL.set_yticklabels(labels)
axL.set_xlabel("mean actions per recording")
xmax = max(skims) * 1.20
for yi, (s, m, w) in zip(y, zip(skims, measures, writes)):
    axL.text(s + xmax * 0.012, yi + h, f"{s:.0f}", va="center", fontsize=8.5, color=C_SKIM)
    axL.text(m + xmax * 0.012, yi, f"{m:.0f}", va="center", fontsize=8.5, color="#9c7620")
    wlab = f"{w:.0f}" if w >= 1 else f"{w:.2f}"
    axL.text(w + xmax * 0.012, yi - h, wlab, va="center", fontsize=8.5,
             color=C_WRITE, fontweight="bold")
axL.set_xlim(0, xmax)
axL.legend(frameon=False, loc="lower right", handlelength=1.2)
axL.set_title("(a) Actions per recording (all tasks)",
              loc="left", fontweight="bold", fontsize=11.5)
axL.grid(axis="y", visible=False)

# --- Right: zero-commit fraction ---
axR.barh(y, [z * 100 for z in zerofrac], height=0.6, color=C_ZERO,
         edgecolor="white", linewidth=0.6)
axR.set_yticks(y)
axR.set_yticklabels([])
axR.set_xlabel("% recordings with zero findings")
axR.set_xlim(0, 108)
for yi, z in zip(y, zerofrac):
    axR.text(z * 100 - 3, yi, f"{z*100:.0f}%", va="center", ha="right",
             fontsize=9, color="white", fontweight="bold")
axR.set_title("(b) Recordings with zero committed findings",
              loc="left", fontweight="bold", fontsize=11.5)
axR.grid(axis="y", visible=False)

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT)
print("saved", OUT)
for l, s, m, w, z in zip(labels, skims, measures, writes, zerofrac):
    print(f"  {l:18} skims {s:6.1f}  measures {m:5.1f}  writes {w:.2f}  zero {z*100:3.0f}%")
