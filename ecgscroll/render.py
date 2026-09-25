"""ECG-Scroll reusable tile renderer.

Renders a time window of an ambulatory recording into a clinical-grid image tile.
Used both to pre-render dataset figures and (lazily) by the agent environment when
the agent navigates/zooms. The signal is read on demand via wfdb partial range reads
(no full-record download).

Conventions:
  - detail tile:  mm_s=25 (clinical standard), fine 1mm grid
  - overview tile: mm_s<=10, coarse grid + Ns time ticks (declutter)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import wfdb

from ecgscroll import paths

# Downloading records from PhysioNet may require a proxy on some networks;
# set HTTP_PROXY / HTTPS_PROXY in your environment if needed (no default here).

# ---- prefer local full records (fast); fall back to partial streaming ----
_FS = {}          # record -> fs
_MEM = {}         # record -> (signal[n,leads] float32, fs)  for locally-available records


def _local_path(record, pn_dir):
    return paths.raw_path(pn_dir, record)


def _fs_of(record, pn_dir, default=250):
    if record not in _FS:
        try:
            _FS[record] = wfdb.rdheader(record, pn_dir=pn_dir).fs
        except Exception:
            _FS[record] = default
    return _FS[record]


def _load_local(record, pn_dir):
    """Load a full local record into memory once (fast). Returns (sig, fs) or None."""
    if record in _MEM:
        return _MEM[record]
    lp = _local_path(record, pn_dir)
    if os.path.exists(lp + ".dat") and os.path.exists(lp + ".hea"):
        try:
            rec = wfdb.rdrecord(lp)
            _MEM[record] = (rec.p_signal.astype("float32"), rec.fs)
            _FS[record] = rec.fs
            return _MEM[record]
        except Exception:
            return None
    return None


def has_local(record, pn_dir):
    """True if the full record is available on disk (no PhysioNet stream needed)."""
    lp = _local_path(record, pn_dir)
    return os.path.exists(lp + ".dat") and os.path.exists(lp + ".hea")


def get_signal(record, pn_dir, fs=250):
    """API-compat helper: returns (None, fs)."""
    loc = _load_local(record, pn_dir)
    if loc is not None:
        return None, loc[1]
    return None, _fs_of(record, pn_dir, fs)


def read_window(record, pn_dir, t0, t1, fs=250):
    """Read [t0,t1) seconds. Local full record if available (fast), else partial stream."""
    loc = _load_local(record, pn_dir)
    if loc is not None:
        sig, f = loc
        i0, i1 = int(t0 * f), int(t1 * f)
        names = [f"lead{k}" for k in range(sig.shape[1])]
        return sig[i0:i1], names, f
    f = _fs_of(record, pn_dir, fs)
    rec = wfdb.rdrecord(record, pn_dir=pn_dir, sampfrom=int(t0 * f), sampto=int(t1 * f))
    return rec.p_signal, rec.sig_name, rec.fs


def render_tile(record, pn_dir, t0, t1, lead=0, out_path=None,
                mm_s=None, fs=250, title=True):
    """Render window [t0,t1)s of one lead to a clinical-grid PNG. Returns out_path."""
    dur = t1 - t0
    if mm_s is None:
        mm_s = 25 if dur <= 12 else max(2, int(1500 / dur))  # auto: fit ~ page width
    sig, names, fs = read_window(record, pn_dir, t0, t1, fs)
    s = sig[:, lead]
    t = t0 + np.arange(len(s)) / fs
    detail = dur <= 12

    width_in = min(dur * mm_s / 25.4, 26)
    fig, ax = plt.subplots(figsize=(max(width_in, 4), 2.3), dpi=110)
    ax.plot(t, s, color="black", lw=0.5 if detail else 0.35)

    if detail:  # fine clinical grid
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(which="minor", color="#f4b7b7", lw=0.3)
        ax.grid(which="major", color="#e46a6a", lw=0.6)
    else:  # decluttered overview grid
        tick = max(1, round(dur / 12))
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.grid(which="major", color="#e46a6a", lw=0.5, alpha=0.6)

    lo, hi = np.nanpercentile(s, 1), np.nanpercentile(s, 99)
    pad = (hi - lo) * 0.5 + 0.3
    ax.set_xlim(t0, t1); ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=6)
    if title:
        kind = "detail" if detail else "overview"
        ax.set_title(f"{pn_dir}/{record} {names[lead]}  {kind} "
                     f"{t0:.1f}-{t1:.1f}s @ {mm_s}mm/s", fontsize=7)
    fig.tight_layout()
    if out_path is None:
        out_path = f"{record}_{int(t0)}_{int(t1)}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grid(ax, detail, dur):
    if detail:
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(which="minor", color="#f4b7b7", lw=0.3)
        ax.grid(which="major", color="#e46a6a", lw=0.6)
    else:
        tick = max(1, round(dur / 12))
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.grid(which="major", color="#e46a6a", lw=0.5, alpha=0.6)


def render_tile_multi(record, pn_dir, t0, t1, leads=(0, 1), out_path=None,
                      mm_s=None, fs=250, title=True):
    """Render window [t0,t1)s of several leads stacked (shared x-axis) to one clinical-grid PNG.
    Used for ischemia localization, where the agent must decide WHICH lead deviates."""
    dur = t1 - t0
    if mm_s is None:
        mm_s = 25 if dur <= 12 else max(2, int(1500 / dur))
    sig, names, fs = read_window(record, pn_dir, t0, t1, fs)
    detail = dur <= 12
    leads = [l for l in leads if l < sig.shape[1]]
    width_in = min(dur * mm_s / 25.4, 26)
    fig, axes = plt.subplots(len(leads), 1, sharex=True,
                             figsize=(max(width_in, 4), 1.5 * len(leads) + 0.6), dpi=110)
    if len(leads) == 1:
        axes = [axes]
    t = t0 + np.arange(sig.shape[0]) / fs
    for ax, lead in zip(axes, leads):
        s = sig[:, lead]
        ax.plot(t, s, color="black", lw=0.5 if detail else 0.35)
        _grid(ax, detail, dur)
        lo, hi = np.nanpercentile(s, 1), np.nanpercentile(s, 99)
        pad = (hi - lo) * 0.5 + 0.3
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f"lead{lead}: {names[lead]}", fontsize=6)
        ax.tick_params(labelsize=6)
    axes[0].set_xlim(t0, t1)
    if title:
        kind = "detail" if detail else "overview"
        axes[0].set_title(f"{pn_dir}/{record}  {kind} {t0:.1f}-{t1:.1f}s @ {mm_s}mm/s "
                          f"(both leads)", fontsize=7)
    fig.tight_layout()
    if out_path is None:
        out_path = f"{record}_{int(t0)}_{int(t1)}_multi.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # smoke test: render one detail + one overview from a known AF record
    d = os.path.dirname(__file__)
    p1 = render_tile("04043", "afdb", 300, 310, out_path=os.path.join(d, "_smoke_detail.png"))
    p2 = render_tile("04043", "afdb", 0, 600, out_path=os.path.join(d, "_smoke_overview.png"))
    print("rendered:", p1, p2)
