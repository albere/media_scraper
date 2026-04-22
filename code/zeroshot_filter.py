"""
zeroshot_filter.py — Score all articles with BART zero-shot classification.

Runs facebook/bart-large-mnli on each article to score how strongly it
relates to immigration. Designed to run on a rented GPU (Vast.ai/RunPod).

Usage:
    # Score a single corpus file
    python zeroshot_filter.py --input express_corpus.csv --output express_scored.csv

    # Score all corpus files in a directory
    python zeroshot_filter.py --input-dir data/ --output-dir data/scored/

    # Resume a partially completed run
    python zeroshot_filter.py --input express_corpus.csv --output express_scored.csv --resume

Requirements:
    pip install transformers torch pandas

GPU is expected. On a T4 expect ~100 articles/min, on an A100 ~300-500/min.
Full corpus (~268k articles) should take 1-3 hours depending on GPU.
"""

import argparse
import glob
import os
import time
import pandas as pd
import torch
from transformers import pipeline

# ── Config ──────────────────────────────────────────────────────────────
LABEL = "immigration"
BODY_TRUNCATE = 512  # chars of body to feed classifier (BART limit ~1024 tokens)
BATCH_SIZE = 16      # articles per batch — tune based on GPU memory
SAVE_EVERY = 500     # save progress every N articles


def score_corpus(input_path, output_path, classifier, resume=False, batch_size=16):
    """Score a single corpus CSV."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"Output:     {output_path}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path)
    print(f"  {len(df):,} articles loaded")

    # ── Resume support ──────────────────────────────────────────────
    start_idx = 0
    if resume and os.path.exists(output_path):
        done = pd.read_csv(output_path)
        start_idx = len(done)
        print(f"  Resuming from article {start_idx:,} ({start_idx}/{len(df)})")
        if start_idx >= len(df):
            print(f"  Already complete — skipping")
            return

    # ── Prepare texts ───────────────────────────────────────────────
    texts = []
    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        body = str(row.get('body', ''))[:BODY_TRUNCATE]
        texts.append(f"{title}. {body}")

    # ── Score in batches ────────────────────────────────────────────
    scores = []
    if resume and start_idx > 0:
        # Load existing scores
        done = pd.read_csv(output_path)
        scores = done['immigration_score'].tolist()

    total = len(df)
    start_time = time.time()

    for i in range(start_idx, total, batch_size):
        batch = texts[i:i + batch_size]

        results = classifier(
            batch,
            candidate_labels=[LABEL],
            batch_size=len(batch),
        )

        # classifier returns a list of dicts when given a list of texts
        if isinstance(results, dict):
            results = [results]

        for r in results:
            scores.append(round(r['scores'][0], 4))

        # Progress
        done_count = len(scores)
        if done_count % (batch_size * 5) == 0 or done_count == total:
            elapsed = time.time() - start_time
            rate = (done_count - start_idx) / (elapsed / 60) if elapsed > 0 else 0
            remaining = (total - done_count) / rate if rate > 0 else 0
            print(
                f"  {done_count:>7,}/{total:,}  "
                f"({done_count/total*100:.1f}%)  "
                f"{rate:.0f} articles/min  "
                f"~{remaining:.0f} min remaining"
            )

        # Periodic save
        if done_count % SAVE_EVERY == 0:
            _save_progress(df, scores, output_path)

    # ── Final save ──────────────────────────────────────────────────
    _save_progress(df, scores, output_path)

    elapsed_total = time.time() - start_time
    print(f"  Done in {elapsed_total/60:.1f} min")

    # ── Quick summary ───────────────────────────────────────────────
    s = pd.Series(scores)
    print(f"  Score distribution:")
    print(f"    mean={s.mean():.3f}  median={s.median():.3f}  std={s.std():.3f}")
    print(f"    >0.5: {(s > 0.5).sum():,} ({(s > 0.5).mean()*100:.1f}%)")
    print(f"    >0.7: {(s > 0.7).sum():,} ({(s > 0.7).mean()*100:.1f}%)")
    print(f"    >0.9: {(s > 0.9).sum():,} ({(s > 0.9).mean()*100:.1f}%)")


def _save_progress(df, scores, output_path):
    """Save current progress to CSV."""
    out = df.iloc[:len(scores)].copy()
    out['immigration_score'] = scores
    out.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Score articles with BART zero-shot classification"
    )
    parser.add_argument("--input", help="Single corpus CSV to score")
    parser.add_argument("--input-dir", help="Directory of corpus CSVs to score")
    parser.add_argument("--output", help="Output CSV path (for single file)")
    parser.add_argument("--output-dir", help="Output directory (for batch mode)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for GPU inference (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide either --input or --input-dir")

    batch_size = args.batch_size

    # ── Load model ──────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")

    if device == -1:
        print("WARNING: No GPU detected — this will be very slow.")
        print("Consider using a GPU rental service (Vast.ai, RunPod, Colab).")

    print("Loading facebook/bart-large-mnli...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )
    print("Model loaded.\n")

    # ── Process files ───────────────────────────────────────────────
    if args.input:
        output = args.output or args.input.replace('.csv', '_scored.csv')
        score_corpus(args.input, output, classifier, args.resume, batch_size)

    if args.input_dir:
        output_dir = args.output_dir or os.path.join(args.input_dir, 'scored')
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(args.input_dir, '*_corpus.csv')))
        print(f"Found {len(files)} corpus files in {args.input_dir}")

        for f in files:
            out_name = os.path.basename(f).replace('_corpus.csv', '_scored.csv')
            out_path = os.path.join(output_dir, out_name)
            score_corpus(f, out_path, classifier, args.resume, batch_size)

    print("\nAll done.")


if __name__ == "__main__":
    main()
