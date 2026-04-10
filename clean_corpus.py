"""
clean_corpus.py — Strip boilerplate from scraped news corpora

Handles:
  - Express: removes "READ MORE: ..." lines (cross-promotion links)
  - Independent: removes "- Bookmark" lines, "- CommentsGo to comments",
    and the "Join our commenting forum..." footer
  - Recalculates wordcount after cleaning
  - Leaves Guardian untouched (API data is clean)

Usage:
    # Clean a single file
    python clean_corpus.py --input express_monthly/daily_express_2025_12_corpus.csv

    # Clean all files in a directory
    python clean_corpus.py --input-dir express_monthly/

    # Dry run — show what would change without modifying files
    python clean_corpus.py --input-dir express_monthly/ --dry-run

Output:
    Overwrites input files in-place (back up first if nervous).
    Use --output-dir to write cleaned files to a separate folder instead.
"""

import argparse
import re
import glob
import os
import pandas as pd


# ── Cleaning functions ──────────────────────────────────────────────────

def clean_express(body: str) -> str:
    """Remove READ MORE cross-promotion lines from Express articles."""
    # READ MORE lines always start on their own line
    body = re.sub(r'\nREAD MORE:.*', '', body)
    # Handle case where READ MORE is at the very start (unlikely but safe)
    body = re.sub(r'^READ MORE:.*', '', body)
    return body.strip()


def clean_independent(body: str) -> str:
    """Remove Bookmark markers, comment prompts, and footer from Independent articles."""
    # Remove "- Bookmark" lines (appear near article start as UI artifact)
    body = re.sub(r'\n- Bookmark', '', body)
    body = re.sub(r'^- Bookmark', '', body)

    # Remove "- CommentsGo to comments" (no space — that's how it appears)
    body = re.sub(r'\n- CommentsGo to comments', '', body)
    body = re.sub(r'^- CommentsGo to comments', '', body)

    # Remove the footer block
    # "Join our commenting forum\nJoin thought-provoking conversations,
    #  follow other Independent readers and see their replies\nComments"
    body = re.sub(
        r'\nJoin our commenting forum\n.*?see their replies\nComments\s*$',
        '',
        body,
        flags=re.DOTALL,
    )

    # Catch partial variants of the footer
    body = re.sub(
        r'\nJoin our commenting forum.*$',
        '',
        body,
        flags=re.DOTALL,
    )

    return body.strip()


def clean_body(body: str, outlet: str) -> str:
    """Route to the appropriate cleaner based on outlet."""
    if pd.isna(body):
        return body

    body = str(body)

    if 'express' in outlet.lower():
        body = clean_express(body)
    elif 'independent' in outlet.lower():
        body = clean_independent(body)

    return body


def recount_words(body: str) -> int:
    """Recount words after cleaning."""
    if pd.isna(body):
        return 0
    return len(str(body).split())


# ── Main ────────────────────────────────────────────────────────────────

def process_file(filepath: str, output_path: str = None, dry_run: bool = False):
    """Clean a single CSV file."""
    df = pd.read_csv(filepath)

    if 'body' not in df.columns:
        print(f"  SKIP {filepath} — no 'body' column")
        return

    # Detect outlet from filename or column
    if 'outlet' in df.columns:
        outlet = df['outlet'].iloc[0]
    elif 'express' in filepath.lower():
        outlet = 'Daily Express'
    elif 'independent' in filepath.lower():
        outlet = 'The Independent'
    elif 'star' in filepath.lower():
        outlet = 'Daily Star'
    elif 'guardian' in filepath.lower():
        outlet = 'The Guardian'
    else:
        print(f"  SKIP {filepath} — cannot detect outlet")
        return

    if 'guardian' in outlet.lower():
        print(f"  SKIP {filepath} — Guardian data is clean")
        return

    # Clean
    df['wordcount'] = pd.to_numeric(df['wordcount'], errors='coerce').fillna(0).astype(int)
    original_wc = df['wordcount'].sum()
    df['body'] = df['body'].apply(lambda b: clean_body(b, outlet))
    df['wordcount'] = df['body'].apply(recount_words)
    new_wc = df['wordcount'].sum()

    words_removed = original_wc - new_wc
    pct = (words_removed / original_wc * 100) if original_wc > 0 else 0

    print(
        f"  {os.path.basename(filepath)}: "
        f"{len(df)} articles, "
        f"{words_removed:,} words removed ({pct:.1f}%)"
    )

    if not dry_run:
        out = output_path or filepath
        df.to_csv(out, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Clean boilerplate from news corpus CSVs"
    )
    parser.add_argument("--input", help="Path to a single CSV file")
    parser.add_argument("--input-dir", help="Path to directory of CSV files")
    parser.add_argument(
        "--output-dir",
        help="Write cleaned files here instead of overwriting originals",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide either --input or --input-dir")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        output_path = None
        if args.output_dir:
            output_path = os.path.join(
                args.output_dir, os.path.basename(args.input)
            )
        process_file(args.input, output_path, args.dry_run)

    if args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
        print(f"Found {len(files)} CSV files in {args.input_dir}")
        if args.dry_run:
            print("DRY RUN — no files will be modified\n")
        for f in files:
            output_path = None
            if args.output_dir:
                output_path = os.path.join(
                    args.output_dir, os.path.basename(f)
                )
            process_file(f, output_path, args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No files were modified.")


if __name__ == "__main__":
    main()
