"""Centralized data-directory layout for ECG-Scroll (open-source).

    data/raw/<db>_local/    raw PhysioNet records (.hea/.dat/.atr); large, git-ignored
    data/tasks/*.jsonl      task instances (the benchmark itself: prompt + gold + verifier)
    data/results/*.json     evaluation outputs
    data/figures/           rendered curves

All modules import from here so the tree can be relocated by editing one file.
"""
import os

_PKG = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(_PKG, "..", "data"))
RAW = os.path.join(DATA, "raw")
TASKS = os.path.join(DATA, "tasks")
RESULTS = os.path.join(DATA, "results")
FIGURES = os.path.join(DATA, "figures")

# The paper lives outside the open-source package (one level up from benchmark/).
# Plot/fill utilities that feed the manuscript reference it; the core benchmark
# (construction + eval + baseline) never depends on this.
REPO = os.path.abspath(os.path.join(_PKG, "..", ".."))
PAPER = os.path.join(REPO, "paper")


def raw_path(pn_dir, record=""):
    """Path to a raw record (or the db dir if record='')."""
    return os.path.join(RAW, f"{pn_dir}_local", record)


def task_path(name):
    return os.path.join(TASKS, name)


def result_path(name):
    return os.path.join(RESULTS, name)


def data_path(name):
    """Data-root file (SFT split / caches / other scratch artifacts)."""
    return os.path.join(DATA, name)
