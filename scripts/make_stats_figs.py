#!/usr/bin/env python
"""Generate the consolidated benchmark-statistics figures for ECG-Scroll.

The data section tells one story in TWO figures with a single shared style:

  Figure 1 (stats_core.png)   -- the AF long-horizon core:
      scale -> sparsity -> skew, as four facets of one narrative.
      (a) whole-recording durations (hours): the long-horizon premise
      (b) AF episodes per recording (log): few events, heavy tail
      (c) individual episode durations (log s): events last seconds-minutes
      (d) AF burden (U-shaped): the aggregate base rate is bimodal

  Figure 2 (stats_coverage.png) -- the full five-task suite:
      (a) instance counts per (dataset x task) across all 5 databases
      (b) target coverage fraction across localization families: the events an
          online reader must catch occupy a tiny, heavy-tailed slice of the stream

Stats are computed only from the JSONL answer/t0/t1/meta fields (offline, fast).

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/make_stats_figs.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
JSONL = os.path.join(ROOT, "data/tasks/ecgscroll_lh.jsonl")
MULTI = os.path.join(ROOT, "data/tasks/ecgscroll_lh_multi.jsonl")
OUTDIR = os.path.join(ROOT, "data", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ---- Style ------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})
# One muted academic palette shared across both figures ----------------------
C_AFDB = "#4C72B0"   # soft blue   (MIT-BIH AFib)
C_LTAF = "#8C9AB0"   # muted slate (Long-Term AF)
C_ACCENT = "#C44E52" # muted red   (median / annotation accent)
C_GRAY = "#55606E"

SRC_LABEL = {"afdb": "MIT-BIH AFib (afdb)", "ltafdb": "Long-Term AF (ltafdb)"}
SRC_COLOR = {"afdb": C_AFDB, "ltafdb": C_LTAF}

# Per-task colours, reused consistently in the coverage figure ----------------
TASK_COLOR = {
    "episode_detection":     "#4C72B0",  # blue
    "burden_quantification": "#55A868",  # green
    "ischemia_detection":    "#8172B3",  # purple
    "rare_event_search":     "#C44E52",  # red
    "change_detection":      "#CC860B",  # amber
}
TASK_LABEL = {
    "episode_detection":     "Episode detection (AF)",
    "burden_quantification": "Burden quantification",
    "ischemia_detection":    "Ischemia detection (ST)",
    "rare_event_search":     "Rare-event search",
    "change_detection":      "Change-point detection",
}


def panel_tag(ax, s, fs=11.5):
    ax.set_title(s, loc="left", fontsize=fs, fontweight="bold", pad=6)


def stat_box(ax, s, loc="upper right"):
    """A small stats caption in a corner, on a translucent white card so it never
    clashes with bars or the median line."""
    xa, ha = (0.96, "right") if "right" in loc else (0.04, "left")
    ya, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(xa, ya, s, transform=ax.transAxes, ha=ha, va=va, fontsize=9,
            color=C_GRAY, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="none", alpha=0.72))


# ---- Load -------------------------------------------------------------------
rows = [json.loads(l) for l in open(JSONL)]
mrows = [json.loads(l) for l in open(MULTI)]
ep = [r for r in rows if r["task"] == "episode_detection"]
bu = [r for r in rows if r["task"] == "burden_quantification"]


def hours(r):
    return (r["t1"] - r["t0"]) / 3600.0


dur_h = {s: np.array([hours(r) for r in ep if r["pn_dir"] == s]) for s in ("afdb", "ltafdb")}
allh = np.concatenate([dur_h["afdb"], dur_h["ltafdb"]])

n_ep = {s: np.array([len(r["answer"]["intervals"]) for r in ep if r["pn_dir"] == s])
        for s in ("afdb", "ltafdb")}
all_nep = np.concatenate([n_ep["afdb"], n_ep["ltafdb"]])

ep_durs = np.array([iv[1] - iv[0] for r in ep for iv in r["answer"]["intervals"]])
ep_durs = ep_durs[ep_durs > 0]

burden = {s: np.array([r["answer"]["value"] for r in bu if r["pn_dir"] == s])
          for s in ("afdb", "ltafdb")}
all_b = np.concatenate([burden["afdb"], burden["ltafdb"]])

# =============================================================================
# FIGURE 1 -- Long-horizon scale and paroxysmal, skewed events (AF core)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
_MED_LABEL = "median"

# (a) whole-recording durations -- the long-horizon premise -------------------
ax = axes[0, 0]
bins = np.linspace(0, 27, 28)
ax.hist([dur_h["ltafdb"], dur_h["afdb"]], bins=bins, stacked=True,
        color=[SRC_COLOR["ltafdb"], SRC_COLOR["afdb"]],
        label=[f"Long-Term AF  (n={len(dur_h['ltafdb'])})",
               f"MIT-BIH AFib  (n={len(dur_h['afdb'])})"],
        edgecolor="white", linewidth=0.5, zorder=2)
ax.axvline(np.median(allh), color=C_ACCENT, ls="--", lw=1.4, zorder=3,
           label=f"{_MED_LABEL} {np.median(allh):.0f} h")
ax.set_xlabel("Recording duration (hours)")
ax.set_ylabel("# recordings")
ax.set_xlim(0, 27)
ax.margins(y=0.18)
ax.legend(frameon=False, loc="upper left", fontsize=8.6)
stat_box(ax, f"{allh.sum():,.0f} h total", loc="upper right")
panel_tag(ax, "(a) Recordings are long-horizon")

# (b) episodes per recording -- heavy-tailed count, LOG x ---------------------
ax = axes[0, 1]
n_zero_ep = int(np.sum(all_nep == 0))
nep_plot = {s: np.clip(n_ep[s], 1, None) for s in ("afdb", "ltafdb")}
logbins = np.logspace(0, np.log10(all_nep.max()), 20)
ax.hist([nep_plot["ltafdb"], nep_plot["afdb"]], bins=logbins, stacked=True,
        color=[SRC_COLOR["ltafdb"], SRC_COLOR["afdb"]],
        edgecolor="white", linewidth=0.4, zorder=2)
ax.set_xscale("log")
ax.axvline(max(np.median(all_nep), 1), color=C_ACCENT, ls="--", lw=1.4, zorder=3)
ax.set_xlabel("# AF episodes per recording (log)")
ax.set_ylabel("# recordings")
ax.margins(y=0.18)
zero_note = f"\n{n_zero_ep} record with 0 (clipped)" if n_zero_ep else ""
stat_box(ax, f"median {np.median(all_nep):.0f}, up to {all_nep.max():,}{zero_note}",
         loc="upper right")
panel_tag(ax, "(b) Few events per recording")

# (c) individual episode durations -- seconds to minutes, LOG x ---------------
ax = axes[1, 0]
logbins = np.logspace(np.log10(ep_durs.min()), np.log10(ep_durs.max()), 30)
ax.hist(ep_durs, bins=logbins, color=C_AFDB, edgecolor="white", linewidth=0.4, zorder=2)
ax.set_xscale("log")
med = np.median(ep_durs)
frac60 = np.mean(ep_durs < 60)
ax.axvline(med, color=C_ACCENT, ls="--", lw=1.4, zorder=3)
ax.set_xlabel("Episode duration (s, log)")
ax.set_ylabel("# episodes")
ax.margins(y=0.18)
stat_box(ax, f"N = {len(ep_durs):,}\nmedian {med:.0f} s\n{frac60*100:.0f}% < 60 s",
         loc="upper right")
panel_tag(ax, "(c) Events are brief")

# (d) AF burden -- U-shaped base rate -----------------------------------------
ax = axes[1, 1]
bins = np.linspace(0, 1, 21)
ax.hist([burden["ltafdb"], burden["afdb"]], bins=bins, stacked=True,
        color=[SRC_COLOR["ltafdb"], SRC_COLOR["afdb"]],
        edgecolor="white", linewidth=0.5, zorder=2)
ax.axvline(np.median(all_b), color=C_ACCENT, ls="--", lw=1.4, zorder=3)
frac_low = np.mean(all_b < 0.1)
frac_high = np.mean(all_b > 0.9)
ax.set_xlabel("AF burden (fraction of recording in AF)")
ax.set_ylabel("# recordings")
ax.set_xlim(0, 1)
ax.margins(y=0.18)
stat_box(ax, f"{frac_low*100:.0f}% < 0.10\n{frac_high*100:.0f}% > 0.90", loc="upper center")
panel_tag(ax, "(d) Burden is U-shaped")

fig.tight_layout(w_pad=2.2, h_pad=2.4)
f1 = os.path.join(OUTDIR, "stats_core.png")
fig.savefig(f1)
plt.close(fig)

# =============================================================================
# FIGURE 2 -- Task and dataset coverage across the full suite
# =============================================================================
# Assemble instance counts per (dataset, task) over BOTH releases.
suite = []  # (dataset, task, count)
for s in ("afdb", "ltafdb"):
    suite.append((s, "episode_detection", int(np.sum([r["pn_dir"] == s for r in ep]))))
    suite.append((s, "burden_quantification", int(np.sum([r["pn_dir"] == s for r in bu]))))
mt_by = {}
for r in mrows:
    mt_by.setdefault((r["pn_dir"], r["task"]), 0)
    mt_by[(r["pn_dir"], r["task"])] += 1
for (s, t), c in mt_by.items():
    suite.append((s, t, c))

# order: group by dataset, and within a dataset keep a stable task order
DS_ORDER = ["afdb", "ltafdb", "edb", "mitdb", "ltdb"]
TASK_ORDER = list(TASK_COLOR.keys())
suite.sort(key=lambda x: (DS_ORDER.index(x[0]), TASK_ORDER.index(x[1])))
suite = suite[::-1]  # so the first dataset ends up on top after barh

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4),
                         gridspec_kw={"width_ratios": [1.15, 1.0]})

# (a) coverage bars -----------------------------------------------------------
ax = axes[0]
y = np.arange(len(suite))
vals = [c for _, _, c in suite]
cols = [TASK_COLOR[t] for _, t, _ in suite]
labs = [f"{s} · {t.replace('_', ' ')}" for s, t, _ in suite]
ax.barh(y, vals, color=cols, edgecolor="white", linewidth=0.6)
ax.set_yticks(y)
ax.set_yticklabels(labs, fontsize=8.5)
for yi, v in zip(y, vals):
    ax.text(v + 1.2, yi, str(v), va="center", fontsize=8.5, color=C_GRAY)
ax.set_xlabel("# instances")
ax.set_xlim(0, max(vals) * 1.16)
ax.grid(axis="y", visible=False)
total = len(rows) + len(mrows)
handles = [Patch(facecolor=TASK_COLOR[t], label=TASK_LABEL[t]) for t in TASK_ORDER]
ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8.2)
panel_tag(ax, f"(a) 5 tasks × 5 databases  (N={total} instances)", fs=10.5)

# (b) target coverage fraction across localization families ------------------
# For each interval-localization instance: total target duration / recording T.
# This is the slice of the stream an online reader must actually catch.
def cover_frac_intervals(rlist):
    out = []
    for r in rlist:
        T = r["meta"]["duration_s"]
        d = sum(iv[1] - iv[0] for iv in r["answer"]["intervals"])
        if T > 0 and d > 0:
            out.append(d / T)
    return np.array(out)

isch = [r for r in mrows if r["task"] == "ischemia_detection"]
rare = [r for r in mrows if r["task"] == "rare_event_search"]
cov_af = cover_frac_intervals(ep)
cov_isch = cover_frac_intervals(isch)
cov_rare = np.array([(r["answer"]["interval"][1] - r["answer"]["interval"][0]) / r["meta"]["duration_s"]
                     for r in rare if r["meta"]["duration_s"] > 0])

ax = axes[1]
series = [("AF episodes\n(afdb+ltafdb)", cov_af, TASK_COLOR["episode_detection"]),
          ("Ischemia ST\n(edb)", cov_isch, TASK_COLOR["ischemia_detection"]),
          ("Rare-event\n(mitdb)", cov_rare, TASK_COLOR["rare_event_search"])]
positions = np.arange(len(series))[::-1]
data = [np.clip(s[1], 1e-5, None) for s in series]
bp = ax.boxplot(data, positions=positions, orientation="horizontal", widths=0.55,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor=C_GRAY,
                                markeredgecolor="none", alpha=0.4),
                medianprops=dict(color=C_ACCENT, lw=1.6),
                whiskerprops=dict(color=C_GRAY, lw=1.0),
                capprops=dict(color=C_GRAY, lw=1.0),
                boxprops=dict(lw=0.6))
for patch, (_, _, c) in zip(bp["boxes"], series):
    patch.set_facecolor(c)
    patch.set_alpha(0.8)
    patch.set_edgecolor("white")
ax.set_xscale("log")
ax.set_xlim(3e-4, 2.0)
ax.set_yticks(positions)
ax.set_yticklabels([s[0] for s in series], fontsize=8.5)
ax.set_xlabel("Target coverage: event time / recording (log)")
ax.axvline(0.1, color=C_GRAY, ls=":", lw=1.0)
ax.text(0.1, positions.max() + 0.42, "10%", ha="center", va="bottom",
        fontsize=8, color=C_GRAY)
ax.grid(axis="y", visible=False)
panel_tag(ax, "(b) Targets span 4 orders of magnitude", fs=10.5)

fig.tight_layout(w_pad=2.4)
f2 = os.path.join(OUTDIR, "stats_coverage.png")
fig.savefig(f2)
plt.close(fig)

# ---- Remove stale figures from the previous piecemeal layout ----------------
STALE = ["stats_durations.png", "stats_burden.png", "stats_episodes.png",
         "stats_sparsity.png", "stats_multitask.png"]
for name in STALE:
    p = os.path.join(OUTDIR, name)
    if os.path.exists(p):
        os.remove(p)

# ---- Report numbers ---------------------------------------------------------
pvc = [r for r in mrows if r["task"] == "burden_quantification"]
pv = np.array([r["answer"]["value"] for r in pvc])
isch_durs = np.array([iv[1] - iv[0] for r in isch for iv in r["answer"]["intervals"]])
isch_durs = isch_durs[isch_durs > 0]
cp = [r for r in mrows if r["task"] == "change_detection"]
cp_frac = np.array([r["answer"]["change_point"] / r["meta"]["duration_s"] for r in cp])

print("=== KEY NUMBERS ===")
print(f"total suite instances: {total}  (lh core {len(rows)} + multi {len(mrows)})")
print(f"AF core: {len(ep)} episode + {len(bu)} burden over 107 recordings")
print(f"recorded time: {allh.sum():,.0f} h  median {np.median(allh):.1f} h "
      f"range [{allh.min():.1f}, {allh.max():.1f}] h")
print(f"episodes/rec: median {np.median(all_nep):.0f} mean {all_nep.mean():.1f} "
      f"max {all_nep.max()} total {all_nep.sum()} zero-episode records {n_zero_ep}")
print(f"episode dur (s): N {len(ep_durs):,} median {np.median(ep_durs):.0f} "
      f"frac<60s {np.mean(ep_durs<60):.2f} min {ep_durs.min():.1f} max {ep_durs.max():.0f}")
print(f"AF burden: median {np.median(all_b):.3f} <0.10 {np.mean(all_b<0.1):.2f} "
      f">0.90 {np.mean(all_b>0.9):.2f}")
print(f"ischemia ST dur: N {len(isch_durs)} median {np.median(isch_durs):.0f} s")
print(f"PVC burden: mitdb {np.sum([r['pn_dir']=='mitdb' for r in pvc])} "
      f"ltdb {np.sum([r['pn_dir']=='ltdb' for r in pvc])} median {np.median(pv):.4f} "
      f"<1% {np.mean(pv<0.01):.2f}")
print(f"change-point onset: n {len(cp)} median at {np.median(cp_frac)*100:.0f}% of record")
print(f"target coverage median: AF {np.median(cov_af)*100:.2f}%  "
      f"ischemia {np.median(cov_isch)*100:.2f}%  rare {np.median(cov_rare)*100:.3f}%")
print("=== FILES ===")
for f in (f1, f2):
    print(f)
