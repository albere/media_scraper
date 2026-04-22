"""
aggregate_corpus.py — Combine monthly CSVs into a single corpus file per outlet.

Usage:
    python aggregate_corpus.py --input-dir express_monthly/ --output express_corpus.csv
    python aggregate_corpus.py --input-dir star_monthly/ --output star_corpus.csv
"""

import argparse
import glob
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate monthly corpus CSVs into a single file"
    )
    parser.add_argument("--input-dir", required=True, help="Directory of monthly CSVs")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(files)} CSV files in {args.input_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)

    print(f"Combined: {len(combined):,} articles → {args.output}")


if __name__ == "__main__":
    main()
