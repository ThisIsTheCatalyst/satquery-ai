#!/usr/bin/env python3
"""
One-command data preparation for the demo.

Downloads EuroSAT, caches a 200-image subset, and builds the OSCD demo pairs
if OSCD has been extracted. Everything the demo and the two evaluation
scripts need comes from here.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --n-per-class 20 --n-pairs 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--skip-eurosat", action="store_true")
    ap.add_argument("--skip-oscd", action="store_true")
    args = ap.parse_args()

    ok = True

    if not args.skip_eurosat:
        print("=" * 55)
        print("EuroSAT")
        print("=" * 55)
        from src.data.eurosat import download_eurosat, build_demo_subset
        src = download_eurosat()
        if src is None:
            print("  FAILED — see message above.")
            ok = False
        else:
            build_demo_subset(src, n_per_class=args.n_per_class)
            print("  OK")

    if not args.skip_oscd:
        print("\n" + "=" * 55)
        print("OSCD")
        print("=" * 55)
        from src.data.oscd import download_oscd, build_demo_pairs
        src = download_oscd()
        if src is None:
            print("  SKIPPED — OSCD needs a manual download. "
                  "Change detection will fall back to uploaded image pairs.")
        else:
            build_demo_pairs(src, n_pairs=args.n_pairs)
            print("  OK")

    print("\n" + "=" * 55)
    print("Next: python scripts/eval_clip_eurosat.py")
    print("=" * 55)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
