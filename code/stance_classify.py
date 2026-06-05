"""
stance_classify.py
Immigration News Corpus — Stance Classification

Classifies each immigration article as pro- or anti-immigration using
facebook/bart-large-mnli zero-shot inference.

Input:  CSV files with 'body' and 'url' columns (the scored corpus files
        before body was dropped, or the original corpus files)
Output: CSV with url, stance_label, pro_score, anti_score

The output is then merged with all_features.csv on url.

Usage:
    # Single file
    python stance_classify.py --input data/guardian_scored.csv --output results/guardian_stance.csv

    # Directory of scored files
    python stance_classify.py --input-dir data/scored/ --output results/all_stance.csv

    # With custom batch size (adjust for GPU memory)
    python stance_classify.py --input-dir data/scored/ --output results/all_stance.csv --batch-size 32

    # Resume after interruption (skips already-processed URLs)
    python stance_classify.py --input-dir data/scored/ --output results/all_stance.csv --resume

Requirements:
    pip install transformers torch pandas
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = "facebook/bart-large-mnli"

# Candidate labels for zero-shot stance classification
LABEL_PRO  = "supports immigration"
LABEL_ANTI = "opposes immigration"
CANDIDATE_LABELS = [LABEL_PRO, LABEL_ANTI]

NEUTRAL_THRESHOLD = 0.1

# Minimum word count to classify (very short articles produce noisy scores)
MIN_WORDS = 50

# Truncate body text to this many characters before classification
# BART has a 1024-token limit; ~3000 chars is roughly 500-600 tokens,
# leaving headroom for the hypothesis template
MAX_CHARS = 3000

# Progress reporting
PROGRESS_EVERY = 500

# Columns required in input
REQUIRED_COLS = {"body", "url"}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_classifier(device=None):
    """Load the zero-shot classification pipeline."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    print(f"Loading {MODEL}...")
    print(f"  Device: {'GPU ' + torch.cuda.get_device_name(0) if device >= 0 else 'CPU'}")

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL,
        device=device,
    )
    print("  Model loaded.\n")
    return classifier


def classify_batch(classifier, texts, batch_size=16):
    """
    Classify a batch of texts. Returns list of dicts with
    pro_score and anti_score.
    """
    # Truncate texts
    truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]

    results = classifier(
        truncated,
        candidate_labels=CANDIDATE_LABELS,
        batch_size=batch_size,
    )

    # Handle single result (not wrapped in list)
    if isinstance(results, dict):
        results = [results]

    parsed = []
    for r in results:
        scores = dict(zip(r["labels"], r["scores"]))
        pro = scores.get(LABEL_PRO, 0)
        anti = scores.get(LABEL_ANTI, 0)
        margin = abs(pro - anti)
        if margin < NEUTRAL_THRESHOLD:
            label = "neutral"
        elif pro > anti:
            label = "pro"
        else:
            label = "anti"
        parsed.append({
            "pro_score": round(pro, 4),
            "anti_score": round(anti, 4),
            "stance_label": label,
        })

    return parsed


def run_stance(input_paths, output_path, batch_size=16, resume=False):
    """
    Main pipeline: load articles, classify stance, save results.
    """
    # ── Load already-processed URLs if resuming ──
    done_urls = set()
    existing_rows = []
    if resume and Path(output_path).exists():
        existing = pd.read_csv(output_path)
        done_urls = set(existing["url"].tolist())
        existing_rows = existing.to_dict("records")
        print(f"Resuming: {len(done_urls)} URLs already processed\n")

    # ── Read input files ──
    frames = []
    for p in input_paths:
        print(f"Reading {p} ... ", end="", flush=True)
        df = pd.read_csv(p, low_memory=False)

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"SKIP — missing columns: {missing}")
            continue

        print(f"{len(df)} articles")
        frames.append(df)

    if not frames:
        print("ERROR: No valid input files.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal articles loaded: {len(df)}")

    # ── Filter ──
    df = df[df["body"].notna()].copy()
    df["_wc"] = df["body"].str.split().str.len()
    df = df[df["_wc"] >= MIN_WORDS].copy()
    df.drop(columns=["_wc"], inplace=True)

    # Skip already-processed if resuming
    if done_urls:
        before = len(df)
        df = df[~df["url"].isin(done_urls)]
        print(f"After resume filter: {len(df)} remaining (skipped {before - len(df)})")

    print(f"Articles to classify: {len(df)}\n")

    if len(df) == 0:
        print("Nothing to do.")
        if existing_rows:
            pd.DataFrame(existing_rows).to_csv(output_path, index=False)
        return

    # ── Load model ──
    classifier = load_classifier()

    # ── Classify ──
    results = list(existing_rows)  # start with any existing results
    t0 = time.time()

    urls = df["url"].tolist()
    bodies = df["body"].tolist()

    # Process in batches
    for i in range(0, len(bodies), batch_size):
        batch_texts = bodies[i:i + batch_size]
        batch_urls = urls[i:i + batch_size]

        batch_results = classify_batch(classifier, batch_texts, batch_size)

        for url, stance in zip(batch_urls, batch_results):
            stance["url"] = url
            results.append(stance)

        # Progress
        done = i + len(batch_texts)
        if done % PROGRESS_EVERY < batch_size or done == len(bodies):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(bodies) - done) / rate if rate > 0 else 0
            print(f"  [{done:,}/{len(bodies):,}] {rate:.1f} articles/sec | "
                  f"~{remaining / 60:.1f} min remaining")

        # Periodic save (every 5000 articles) for crash resilience
        if done % 5000 < batch_size:
            _save(results, output_path)

    elapsed = time.time() - t0
    print(f"\nClassification complete: {len(results):,} articles in {elapsed:.1f}s")

    # ── Save ──
    _save(results, output_path)

    # ── Summary ──
    out_df = pd.DataFrame(results)
    print(f"\n── STANCE SUMMARY ──")
    print(f"  Total classified: {len(out_df)}")
    print(f"  Pro-immigration:  {(out_df['stance_label'] == 'pro').sum()} "
          f"({(out_df['stance_label'] == 'pro').mean() * 100:.1f}%)")
    print(f"  Anti-immigration: {(out_df['stance_label'] == 'anti').sum()} "
          f"({(out_df['stance_label'] == 'anti').mean() * 100:.1f}%)")
    print(f"  Neutral:          {(out_df['stance_label'] == 'neutral').sum()} "
          f"({(out_df['stance_label'] == 'neutral').mean() * 100:.1f}%)")
    print(f"  Mean pro_score:   {out_df['pro_score'].mean():.4f}")
    print(f"  Mean anti_score:  {out_df['anti_score'].mean():.4f}")


def _save(results, output_path):
    out_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path} ({len(out_df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def merge_stance(features_path, stance_path, output_path):
    """
    Merge stance scores into the features file on URL.
    Run this after stance classification is complete.

    Usage:
        python stance_classify.py --merge \
            --features all_features_4outlet.csv \
            --stance all_stance.csv \
            --output all_features_with_stance.csv
    """
    print("Merging stance scores into features file...")
    features = pd.read_csv(features_path, low_memory=False)
    stance = pd.read_csv(stance_path)

    # Keep only stance columns
    stance_cols = stance[["url", "pro_score", "anti_score", "stance_label"]]

    merged = features.merge(stance_cols, on="url", how="left")

    matched = merged["stance_label"].notna().sum()
    print(f"  Features: {len(features)} rows")
    print(f"  Stance:   {len(stance)} rows")
    print(f"  Matched:  {matched} ({matched / len(features) * 100:.1f}%)")

    merged.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stance Classification — Immigration News Corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── Classify command ──
    classify_parser = subparsers.add_parser("classify", help="Run stance classification")
    input_group = classify_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", nargs="+",
                             help="One or more CSV files with body and url columns")
    input_group.add_argument("--input-dir",
                             help="Directory of CSV files to process")
    classify_parser.add_argument("--output", required=True,
                                 help="Output CSV path")
    classify_parser.add_argument("--batch-size", type=int, default=16,
                                 help="Batch size for inference (default: 16)")
    classify_parser.add_argument("--resume", action="store_true",
                                 help="Resume from existing output file")

    # ── Merge command ──
    merge_parser = subparsers.add_parser("merge", help="Merge stance into features")
    merge_parser.add_argument("--features", required=True)
    merge_parser.add_argument("--stance", required=True)
    merge_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "classify":
        if args.input:
            input_paths = [Path(p) for p in args.input]
        else:
            input_dir = Path(args.input_dir)
            input_paths = sorted(input_dir.glob("*.csv"))
            if not input_paths:
                print(f"ERROR: No CSV files in {input_dir}")
                sys.exit(1)

        for p in input_paths:
            if not p.exists():
                print(f"ERROR: File not found: {p}")
                sys.exit(1)

        print("=" * 65)
        print("Stance Classification Pipeline")
        print(f"  Input:      {len(input_paths)} file(s)")
        print(f"  Output:     {args.output}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Resume:     {args.resume}")
        print("=" * 65 + "\n")

        run_stance(input_paths, args.output,
                   batch_size=args.batch_size, resume=args.resume)

    elif args.command == "merge":
        merge_stance(args.features, args.stance, args.output)

    else:
        parser.print_help()
        print("\nUse 'classify' to run stance inference or 'merge' to add scores to features.")


if __name__ == "__main__":
    main()
