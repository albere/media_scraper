"""
sample_for_validation.py — Draw stratified random samples for human validation.

Produces two CSVs:
  1. topic_validation_sample.csv  — 300 articles (150 above + 150 below threshold)
                                     for validating the immigration topic filter
  2. stance_validation_sample.csv — 200 articles with body text
                                     for validating the stance classifier

Usage:
    python sample_for_validation.py
"""

import pandas as pd
import random

# ============================================================
# CONFIGURATION — fill in your paths here
# ============================================================

SCORED_FILES = [
    "data/scored/guardian_scored.csv",
    "data/scored/express_scored.csv",
    "data/scored/independent_scored.csv",
    "data/scored/mirror_scored.csv",
]

STANCE_FILES = [
    "data/stance/guardian_stance.csv",
    "data/stance/express_stance.csv",
    "data/stance/independent_stance.csv",
    "data/stance/mirror_stance.csv",
]

THRESHOLD = 0.5          # immigration score threshold used in filtering
TOPIC_SAMPLE_N = 300     # total topic sample (half above, half below threshold)
STANCE_SAMPLE_N = 200    # total stance sample
SEED = 42

# ============================================================

def load_scored():
    """Load all scored files into one dataframe."""
    dfs = []
    for path in SCORED_FILES:
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        # Guardian doesn't have an outlet column
        if "outlet" not in df.columns:
            df["outlet"] = "guardian"
        df = df[["outlet", "url", "body", "immigration_score"]]
        dfs.append(df)
        print(f"  {len(df)} rows")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total scored articles: {len(combined)}")
    return combined

def load_stance():
    """Load all stance files into one dataframe."""
    dfs = []
    for path in STANCE_FILES:
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  {len(df)} rows")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total stance articles: {len(combined)}")
    return combined


def sample_topic(scored_df):
    """
    Stratified sample: half from above threshold, half from below.
    This lets us measure both precision (are the kept articles correct?)
    and recall (did we miss any immigration articles below threshold?).
    """
    half = TOPIC_SAMPLE_N // 2

    above = scored_df[scored_df["immigration_score"] >= THRESHOLD]
    below = scored_df[scored_df["immigration_score"] < THRESHOLD]

    print(f"\nAbove threshold: {len(above)}")
    print(f"Below threshold: {len(below)}")

    above_sample = above.sample(n=min(half, len(above)), random_state=SEED)
    below_sample = below.sample(n=min(half, len(below)), random_state=SEED)

    sample = pd.concat([above_sample, below_sample], ignore_index=True)
    # Shuffle so the annotator doesn't see all above-threshold first
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Add empty column for human label
    sample["human_label"] = ""

    # Columns for output
    out = sample[["outlet", "url", "body", "immigration_score", "human_label"]]
    out.to_csv("topic_validation_sample.csv", index=False)
    print(f"\nSaved topic_validation_sample.csv ({len(out)} articles)")

    # Per-outlet breakdown
    for outlet, group in sample.groupby("outlet"):
        n_above = (group["immigration_score"] >= THRESHOLD).sum()
        n_below = (group["immigration_score"] < THRESHOLD).sum()
        print(f"  {outlet}: {len(group)} ({n_above} above, {n_below} below)")


def sample_stance(scored_df, stance_df):
    """
    Sample from stance-classified articles, joining body text from scored files.
    Stratified by stance label so we get adequate representation of each class.
    """
    # Join body text onto stance data via URL
    merged = stance_df.merge(
        scored_df[["url", "outlet", "body"]].drop_duplicates(subset="url"),
        on="url",
        how="inner"
    )
    print(f"\nStance articles with body text: {len(merged)}")

    # Stratified by stance label — proportional allocation
    sampled_indices = merged.groupby("stance_label", group_keys=False).apply(
        lambda g: g.sample(
            n=min(len(g), max(10, int(STANCE_SAMPLE_N * len(g) / len(merged)))),
            random_state=SEED
        )
    ).index
    sampled = merged.loc[sampled_indices]
    # Trim or pad to target
    if len(sampled) > STANCE_SAMPLE_N:
        sampled = sampled.sample(n=STANCE_SAMPLE_N, random_state=SEED)

    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Add empty column for human label
    sampled["human_label"] = ""

    out = sampled[["outlet", "url", "body", "pro_score", "anti_score", "stance_label", "human_label"]]
    out.to_csv("stance_validation_sample.csv", index=False)
    print(f"\nSaved stance_validation_sample.csv ({len(out)} articles)")

    # Breakdown
    for label, group in sampled.groupby("stance_label"):
        print(f"  {label}: {len(group)}")
    for outlet, group in sampled.groupby("outlet"):
        print(f"  {outlet}: {len(group)}")


def main():
    random.seed(SEED)

    print("=" * 60)
    print("SAMPLING FOR HUMAN VALIDATION")
    print("=" * 60)

    scored_df = load_scored()

    print("\n--- TOPIC VALIDATION ---")
    sample_topic(scored_df)

    print("\n--- STANCE VALIDATION ---")
    stance_df = load_stance()
    sample_stance(scored_df, stance_df)

    print("\n" + "=" * 60)
    print("Done. Now run:")
    print("  python annotate.py topic")
    print("  python annotate.py stance")
    print("=" * 60)


if __name__ == "__main__":
    main()
