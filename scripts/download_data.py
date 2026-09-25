"""Download all source databases used by ECG-Scroll from PhysioNet.

Each database is downloaded into benchmark/data/raw/<db>_local/ (roughly ~10 GB total
across all five). Uses wfdb.dl_database which resumes cleanly on re-run.

Usage:
    python -m scripts.download_data                # all five databases
    python -m scripts.download_data --dbs afdb edb # a subset

Behind a corporate proxy set http_proxy/https_proxy before invoking.
"""
import argparse
import os
import sys

import wfdb

# make `from ecgscroll import paths` work when this script is run as a module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ecgscroll import paths  # noqa: E402

# database: (physionet id, one-line description)
DATABASES = {
    "afdb": "MIT-BIH Atrial Fibrillation Database (23 usable records, ~10h each, 234h)",
    "edb": "European ST-T Database (90 records, 2h each, 180h)",
    "mitdb": "MIT-BIH Arrhythmia Database (48 records, 30min each, 24h)",
    "ltafdb": "Long-Term AF Database (84 records, ~24h each, 1961h; ~3.4 GB)",
    "ltdb": "MIT-BIH Long-Term Database (7 records, ~21h each, 147h)",
}


def download(db):
    dst = paths.raw_path(db)
    os.makedirs(dst, exist_ok=True)
    print(f"[{db}] -> {dst}", flush=True)
    wfdb.dl_database(db, dl_dir=dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="+", default=list(DATABASES),
                    choices=list(DATABASES),
                    help="which databases to download (default: all five)")
    args = ap.parse_args()

    print("Downloading to:", paths.RAW)
    print("Selected:", ", ".join(args.dbs))
    for db in args.dbs:
        print(f"\n{db}: {DATABASES[db]}")
        download(db)
    print("\nDone. Verify with:")
    print("  python -m scripts.build_all --dry-run")


if __name__ == "__main__":
    main()
