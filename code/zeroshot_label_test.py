"""
zeroshot_label_test.py — Zero-shot classification label comparison

Tests 4 candidate labels on a random sample of the Guardian corpus
to find the best discriminator for immigration-focused articles.

Model: facebook/bart-large-mnli (zero-shot NLI)

Usage:
    python zeroshot_label_test.py --input guardian_corpus.csv --sample 500

Output:
    zeroshot_label_scores.csv — per-article scores for all 4 labels
    Console summary of score distributions

Requirements:
    pip install transformers torch pandas
"""

import argparse
import pandas as pd
import time
from transformers import pipeline

# ── Candidate labels to compare ─────────────────────────────────────────
CANDIDATE_LABELS = [
    "immigration and asylum policy",
    "immigration, asylum seekers, and refugees",
    "UK immigration and border control",
    "immigration",
]

# How much article text to feed the classifier.
# BART's context is 1024 tokens (~750 words). Title + 512 chars of body
# is well within that and keeps inference fast.
BODY_TRUNCATE = 512


def main():
    parser = argparse.ArgumentParser(
        description="Test zero-shot classification labels on a Guardian sample"
    )
    parser.add_argument(
        "--input", required=True, help="Path to Guardian corpus CSV"
    )
    parser.add_argument(
        "--sample", type=int, default=500,
        help="Number of articles to sample (default: 500)"
    )
    parser.add_argument(
        "--output", default="zeroshot_label_scores.csv",
        help="Output CSV path (default: zeroshot_label_scores.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    args = parser.parse_args()

    # ── Load and sample ─────────────────────────────────────────────────
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} total articles")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)
        print(f"  Sampled {len(df)} articles (seed={args.seed})")
    else:
        print(f"  Using all {len(df)} articles (corpus smaller than sample size)")

    # ── Load model ──────────────────────────────────────────────────────
    print("Loading facebook/bart-large-mnli...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU — change to 0 if you have a GPU
    )
    print("  Model loaded.")

    # ── Score each article ──────────────────────────────────────────────
    results = []
    total = len(df)
    start_time = time.time()

    for count, (i, row) in enumerate(df.iterrows(), 1):
        # Combine title + truncated body
        body = str(row.get("body", ""))[:BODY_TRUNCATE]
        text = f"{row['title']}. {body}"

        # Score against ALL labels simultaneously (multi-label mode)
        out = classifier(text, CANDIDATE_LABELS, multi_label=True)

        # Build result row
        score_dict = {
            "idx": i,
            "year": row.get("year", ""),
            "title": str(row.get("title", ""))[:100],
            "section": row.get("section", ""),
            "wordcount": row.get("wordcount", ""),
        }
        for label, score in zip(out["labels"], out["scores"]):
            col = "label_" + label.replace(",", "").replace(" ", "_")
            score_dict[col] = round(score, 4)

        results.append(score_dict)

        # Progress every 25 articles
        if count % 25 == 0:
            elapsed = time.time() - start_time
            rate = count / (elapsed / 60)
            remaining = (total - count) / (rate / 60)
            print(
                f"  {count}/{total} — "
                f"{rate:.1f} articles/min — "
                f"~{remaining:.0f}s remaining"
            )

    # ── Save ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    elapsed_total = time.time() - start_time
    print(
        f"\nDone. {total} articles scored in "
        f"{elapsed_total:.0f}s ({elapsed_total/60:.1f} min)"
    )
    print(f"Results saved to {args.output}")

    # ── Summary stats ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTIONS PER LABEL")
    print("=" * 60)

    score_cols = [c for c in results_df.columns if c.startswith("label_")]
    for col in score_cols:
        scores = results_df[col]
        label_name = col.replace("label_", "").replace("_", " ")
        print(f"\n  {label_name}:")
        print(f"    mean  = {scores.mean():.3f}")
        print(f"    median= {scores.median():.3f}")
        print(f"    std   = {scores.std():.3f}")
        print(f"    >0.5  : {(scores > 0.5).sum():>4d} / {total}  "
              f"({(scores > 0.5).sum()/total*100:.1f}%)")
        print(f"    >0.7  : {(scores > 0.7).sum():>4d} / {total}  "
              f"({(scores > 0.7).sum()/total*100:.1f}%)")
        print(f"    >0.9  : {(scores > 0.9).sum():>4d} / {total}  "
              f"({(scores > 0.9).sum()/total*100:.1f}%)")

    # ── Quick comparison: which label is most discriminating? ───────────
    print("\n" + "=" * 60)
    print("DISCRIMINATION POWER (higher std = better separation)")
    print("=" * 60)
    for col in sorted(score_cols, key=lambda c: results_df[c].std(), reverse=True):
        label_name = col.replace("label_", "").replace("_", " ")
        print(f"  {results_df[col].std():.3f}  {label_name}")

    print("\nNext step: upload the output CSV here so we can inspect")
    print("edge cases and pick the best label + threshold.")


if __name__ == "__main__":
    main()
