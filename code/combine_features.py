"""
combine_features.py — Merge per-outlet NLP feature CSVs into one all_features.csv.

Combines the four *_features_v2.csv files produced by nlp_pipeline.py into a
single file ready for statistical_analysis.py. Normalises outlet labels to a
consistent lowercase scheme and repairs any missing outlet values.

Usage:
    python combine_features.py \
        --inputs guardian_features_v2.csv express_features_v2.csv \
                 independent_features_v2.csv mirror_features_v2.csv \
        --output all_features.csv
"""

import argparse
from pathlib import Path

import pandas as pd


# Map every observed outlet label variant to a canonical lowercase code.
# Add new variants here if outlet strings change in future runs.
OUTLET_MAP = {
    "guardian":        "guardian",
    "the guardian":    "guardian",
    "daily express":   "express",
    "express":         "express",
    "the independent": "independent",
    "independent":     "independent",
    "daily mirror":    "mirror",
    "mirror":          "mirror",
}

# If a file has no outlet value at all (NaN), fall back to inferring from
# the filename stem. This rescued the 1 missing Independent row.
FILENAME_HINTS = {
    "guardian":    "guardian",
    "express":     "express",
    "independent": "independent",
    "mirror":      "mirror",
}


def infer_from_filename(path):
    stem = Path(path).stem.lower()
    for hint, canonical in FILENAME_HINTS.items():
        if hint in stem:
            return canonical
    return None


def normalise_outlet(df, source_path):
    """Map outlet labels to canonical codes; fill NaN from filename."""
    file_outlet = infer_from_filename(source_path)

    if "outlet" not in df.columns:
        if file_outlet is None:
            raise ValueError(f"No outlet column and cannot infer from {source_path}")
        df["outlet"] = file_outlet
        return df

    # Fill missing outlet values from the filename hint
    n_missing = df["outlet"].isna().sum()
    if n_missing > 0:
        if file_outlet is None:
            raise ValueError(
                f"{n_missing} missing outlet values in {source_path} "
                f"and cannot infer from filename"
            )
        df["outlet"] = df["outlet"].fillna(file_outlet)
        print(f"    filled {n_missing} missing outlet value(s) -> {file_outlet}")

    # Map every label to canonical form (case-insensitive)
    mapped = df["outlet"].astype(str).str.strip().str.lower().map(OUTLET_MAP)

    unmapped = df["outlet"][mapped.isna()].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"Unrecognised outlet label(s) in {source_path}: {list(unmapped)}. "
            f"Add them to OUTLET_MAP."
        )

    df["outlet"] = mapped
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-outlet feature CSVs into one all_features.csv"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="The per-outlet *_features_v2.csv files")
    parser.add_argument("--output", required=True,
                        help="Output combined CSV path")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        print(f"Loading {path} ...")
        df = pd.read_csv(path, low_memory=False)
        df = normalise_outlet(df, path)
        print(f"    {len(df):,} rows, outlet -> {sorted(df['outlet'].unique())}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nCombined: {len(combined):,} total articles")
    print("Per-outlet counts:")
    for outlet, n in combined["outlet"].value_counts().sort_index().items():
        print(f"    {outlet:12s} {n:>7,}")

    # Sanity: confirm the columns the stats script depends on are present
    required = {"outlet", "date", "dehum_density"}
    missing = required - set(combined.columns)
    if missing:
        print(f"\nWARNING: missing expected columns: {missing}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(combined):,} rows, {len(combined.columns)} columns)")


if __name__ == "__main__":
    main()
