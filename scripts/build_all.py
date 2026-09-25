"""Build all ECG-Scroll task JSONL files from downloaded raw records.

Runs the three task-construction modules (afdb, edb, mitdb) end to end and reports
how many samples per task landed in benchmark/data/tasks/.

Usage:
    python -m scripts.build_all              # (re)build everything
    python -m scripts.build_all --dry-run    # report which raw DBs are present
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ecgscroll import paths  # noqa: E402


def _summarise(jsonl_path):
    if not os.path.exists(jsonl_path):
        return None
    tasks = Counter()
    records = set()
    with open(jsonl_path) as f:
        for line in f:
            o = json.loads(line)
            tasks[o["task"]] += 1
            records.add(o["record"])
    return {"total": sum(tasks.values()), "by_task": dict(tasks),
            "n_records": len(records)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report on downloaded raw databases without rebuilding")
    args = ap.parse_args()

    print("Raw data directory:", paths.RAW)
    for db in ("afdb", "edb", "mitdb", "ltafdb", "ltdb"):
        d = paths.raw_path(db)
        n_hea = len([f for f in os.listdir(d) if f.endswith(".hea")]) if os.path.isdir(d) else 0
        print(f"  {db:8s} {'ok' if n_hea else 'MISSING':>8s}  ({n_hea} .hea files)")

    if args.dry_run:
        return

    from ecgscroll import build_tasks, build_tasks_edb, build_tasks_mitdb, build_tasks_lh  # noqa: E402
    print("\nBuilding tasks ...")
    os.makedirs(paths.TASKS, exist_ok=True)
    # each module's __main__ block dumps to paths.TASKS; import & invoke build() directly
    # instead of shelling out so any exception surfaces here.
    for mod, name in [(build_tasks, "afdb"),
                      (build_tasks_edb, "edb"),
                      (build_tasks_mitdb, "mitdb"),
                      (build_tasks_lh, "lh")]:
        print(f"\n=== {name} ===")
        samples, stats = mod.build()
        out = paths.task_path(f"ecgscroll_{name}.jsonl")
        with open(out, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"wrote {len(samples)} samples -> {out}")
        print("stats:", stats)

    print("\nSummary:")
    for name in ("afdb", "edb", "mitdb", "lh"):
        s = _summarise(paths.task_path(f"ecgscroll_{name}.jsonl"))
        print(f"  {name:8s} {s}")


if __name__ == "__main__":
    main()
