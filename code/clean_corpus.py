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

def _strip_dont_miss_block(body: str) -> str:
    """
    Remove 'DON'T MISS' followed by consecutive headline lines.

    After a 'DON'T MISS' (with optional colon) header, we consume consecutive
    lines that look like headlines and stop at the first line that looks like
    article prose. The distinction is based on two combined signals
    observed across the 2022 Express corpus:

      - Headlines are short (typically <= 85 chars) and often end in a
        tag marker like [INSIGHT] or [ANALYSIS]
      - Prose sentences are long (typically >= 100 chars) AND end with
        sentence-ending punctuation (. " \u201d)

    A line is treated as prose (and stops consumption) when it's both
    long AND ends in prose punctuation. Otherwise it's consumed as a
    headline.
    """
    # Match "DON'T MISS" followed by optional colon, ellipsis (... or \u2026),
    # then a newline. The ellipsis variant appears occasionally in 2021 and 2025.
    header_re = re.compile(
        r"\n(?:DON'T MISS(?::|\.{3}|\u2026)?)[ \t]*\n",
        re.IGNORECASE,
    )

    def is_prose(line: str) -> bool:
        """Return True if this line looks like article prose, not a headline."""
        stripped = line.rstrip()
        if not stripped:
            # Blank line ends the block
            return True
        # Combined signal: long enough AND ends in sentence punctuation
        long_enough = len(stripped) >= 100
        ends_like_sentence = stripped.endswith(('.', '"', '\u201d'))
        return long_enough and ends_like_sentence

    # Safety cap — never consume more than 10 lines, even if heuristic is
    # ambiguous. Real blocks are 2-5 lines.
    MAX_LINES = 10

    while True:
        m = header_re.search(body)
        if not m:
            break

        pos = m.end()
        end = pos
        consumed = 0
        while end < len(body) and consumed < MAX_LINES:
            nl = body.find('\n', end)
            line = body[end:nl] if nl >= 0 else body[end:]
            line_end = nl + 1 if nl >= 0 else len(body)

            if is_prose(line):
                break

            end = line_end
            consumed += 1

        # Strip header + headline lines, keep a single newline separator
        body = body[:m.start()] + '\n' + body[end:]

    return body


def clean_express(body: str) -> str:
    """
    Remove cross-promotional content from Express articles.

    Handles patterns observed in the corpus:
      1. 'Read more:' / 'READ MORE:' single-line cross-promotion links
         (2019-2025)
      2. 'DON'T MISS' header followed by headline block
         (tagged with [INSIGHT] etc. or untagged) — common in 2022
      3. 'DON'T MISS:' single-line format (2024+)
      4. 'JUST IN:' single-line cross-promotion (2022)

    Both straight (') and fancy (\u2019) apostrophes are handled by
    normalising to straight apostrophes first.
    """
    # Normalise fancy apostrophe (typographic variant of the same character)
    body = body.replace("\u2019", "'")

    # ── DON'T MISS blocks (tagged or untagged headline list) ────────────
    body = _strip_dont_miss_block(body)

    # ── DON'T MISS single-line format (2024+) ───────────────────────────
    # Case-insensitive to catch "Don't miss:" (title case) variant too
    body = re.sub(r"\nDON'T MISS:.*", '', body, flags=re.IGNORECASE)
    body = re.sub(r"^DON'T MISS:.*", '', body, flags=re.IGNORECASE)

    # ── JUST IN: single-line format ─────────────────────────────────────
    body = re.sub(r'\nJUST IN:.*', '', body, flags=re.IGNORECASE)
    body = re.sub(r'^JUST IN:.*', '', body, flags=re.IGNORECASE)

    # ── Read more: single-line format ───────────────────────────────────
    body = re.sub(r'\nRead more:.*', '', body, flags=re.IGNORECASE)
    body = re.sub(r'^Read more:.*', '', body, flags=re.IGNORECASE)

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


def clean_guardian(body: str) -> str:
    """
    Remove promotional boilerplate from Guardian articles.

    Guardian API data is mostly clean but contains two patterns worth
    stripping:

      1. 'Do you have an opinion...' letter submission blocks — appear at
         article end, promoting the letters section. ~2% of articles.

      2. Newsletter sign-up prompts — variations of "Sign up...newsletter",
         "Sign up here for a weekly...", etc. ~6% of articles. Careful
         to distinguish from legitimate uses of "sign up" in article text
         (e.g. "agreement to sign up to net zero").
    """
    # Normalise fancy apostrophes for consistency
    body = body.replace("\u2019", "'")

    # ── Letter submission blocks ─────────────────────────────────────────
    # Common variants, always at article end after prose. We strip from
    # the start of the promo block through to end of article.
    #
    # Variant 1: "Do you have an opinion on the issues raised..."
    body = re.sub(
        r'(?:\n|\s+)Do you have an opinion on the issues raised.*$',
        '',
        body,
        flags=re.DOTALL,
    )

    # Variant 2: "Have an opinion on anything you've read..."
    body = re.sub(
        r"(?:\n|\s+)Have an opinion on anything you(?:'ve| have) read.*$",
        '',
        body,
        flags=re.DOTALL,
    )

    # Variant 3: "Join the debate – email guardian.letters@..."
    # Uses an en-dash (–) which appears in Guardian's letters section
    body = re.sub(
        r'(?:\n|\s+)(?:\u2022\s*)?Join the debate\s*[\u2013\-].*$',
        '',
        body,
        flags=re.DOTALL,
    )

    # ── Newsletter sign-up blocks ───────────────────────────────────────
    # Common patterns: "Sign up here for...", "Sign up to receive...",
    # "Sign up for our...", "Sign up If you would like to receive..."
    # The signal is 'Sign up' followed within ~150 chars by newsletter
    # promotional language (email, newsletter, inbox, weekly, morning mail,
    # briefing). Strip from 'Sign up' through the end of that sentence.
    #
    # We use a multi-step approach:
    # - Match 'Sign up' followed within 120 chars by a newsletter keyword
    # - Strip from 'Sign up' to the next sentence-ending punctuation
    newsletter_kw = r'(?:newsletter|inbox|email briefing|email with|weekly email|weekly roundup|morning mail|morning briefing|long read weekly|daily email|delivered to|subscribe now)'

    # Match "Sign up..." through the first sentence ending (. ! ? ")
    # where the sentence contains a newsletter keyword
    body = re.sub(
        r'(?i)(?:\n|\s+)Sign up[^.!?]{0,200}' + newsletter_kw + r'[^.!?]*[.!?]',
        '',
        body,
    )

    # Also catch "Sign up" sentences where keyword appears shortly after
    # in the next clause (e.g. "Sign up here. The newsletter will arrive...")
    # Keep this narrower — only if the ENTIRE next short chunk is promo
    body = re.sub(
        r'(?i)(?:\n|\s+)Sign up[^.!?\n]{0,80}(?:here|now|today)[.!]\s*(?:[A-Z][^.!?]{0,150}' + newsletter_kw + r'[^.!?]*[.!?])?',
        '',
        body,
    )

    return body.strip()


def clean_star(body: str) -> str:
    """
    Remove cross-promotional content from Daily Star articles.

    The Star is published by Reach plc (same as Express) and shares
    similar boilerplate patterns with some Star-specific variants:

      1. 'READ MORE:' single-line cross-promotion
      2. 'READ NEXT:' followed by bullet-pointed headline list
      3. Newsletter sign-up variants
      4. 'Don't miss:' title-case variants
      5. Trailing bullet-point cross-promo lists
    """
    body = body.replace("\u2019", "'")

    # ── Newsletter sign-up blocks ───────────────────────────────────────
    # Multiple variants observed across 2019-2025:
    # "To get more stories from Daily Star...sign up..."
    # "To stay up to date with all the latest news...sign up..."
    # "For the latest breaking news...sign up for our newsletters"
    # "Want all the biggest Showbiz...Sign up for our free Daily Star..."
    # "sign up for our newsletter by clicking here"
    # "sign up to one of our newsletters here"
    # These appear mid-article, so strip just the sign-up sentence/line.
    body = re.sub(
        r'\nTo (?:get more|stay up to date)[^\n]*sign up[^\n]*\.?',
        '',
        body,
        flags=re.IGNORECASE,
    )

    body = re.sub(
        r'\nFor the latest breaking news[^\n]*sign up[^\n]*\.?',
        '',
        body,
        flags=re.IGNORECASE,
    )

    body = re.sub(
        r'\nWant all the biggest[^\n]*[Ss]ign up[^\n]*\.?',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # Generic "sign up for our newsletter(s)" anywhere on a line
    body = re.sub(
        r',?\s*sign up for our (?:free )?(?:Daily Star \w+ )?newsletters?\b[^\n]*\.?',
        '',
        body,
        flags=re.IGNORECASE,
    )

    body = re.sub(
        r',?\s*sign up to one of our (?:free )?newsletters here\s*\.?',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # ── READ NEXT block ─────────────────────────────────────────────────
    # Header (with colon or ellipsis) followed by headline lines
    body = re.sub(
        r'\nREAD NEXT(?::|\.{3}|\u2026)?\s*\n(?:[^\n]+\n?)+?(?=\n[A-Z][a-z]|\Z)',
        '\n',
        body,
        flags=re.IGNORECASE,
    )

    # Standalone READ NEXT header (orphan, no following lines matched)
    body = re.sub(
        r'\nREAD NEXT(?::|\.{3}|\u2026)?\s*$',
        '',
        body,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # ── READ MORE: lines ────────────────────────────────────────────────
    # Star format concatenates the cross-promo headline with the next
    # sentence of article text on the SAME line, e.g.:
    #   "READ MORE: Headline textArticle continues..."
    # We cannot reliably separate headline from article text, so we
    # just strip the "READ MORE: " prefix tag. The headline text remains
    # as noise but this is far less damaging than removing article content.
    body = re.sub(r'\nREAD MORE:\s*', '\n', body, flags=re.IGNORECASE)
    body = re.sub(r'^READ MORE:\s*', '', body, flags=re.IGNORECASE)

    # ── DON'T MISS: single-line ─────────────────────────────────────────
    body = re.sub(r"\nDON'T MISS:.*", '', body, flags=re.IGNORECASE)
    body = re.sub(r"^DON'T MISS:.*", '', body, flags=re.IGNORECASE)

    return body.strip()


def clean_mirror(body: str) -> str:
    """
    Remove cross-promotional content from Daily Mirror articles.

    The Mirror is published by Reach plc (same as Express/Star) and
    shares similar boilerplate patterns:

      1. 'READ MORE:' cross-promo headlines (concatenated with article
         text on same line, same issue as Star — strip prefix only)
      2. 'Read more:' title-case variant (2016 era, on own lines)
      3. Newsletter/WhatsApp sign-up: "Sign up for all the best...",
         "Join our Mirror politics WhatsApp group..."
      4. Social media footer: "Join The Mirror's WhatsApp Community
         or follow us on Google News..."
    """
    body = body.replace("\u2019", "'")

    # ── Newsletter / WhatsApp sign-up blocks ────────────────────────────
    # "Sign up for all the best travel/celeb updates from the Mirror here"
    body = re.sub(
        r',?\s*[Ss]ign up for all the best[^\n]*from the Mirror[^\n]*\.',
        '',
        body,
    )

    # "NEWSLETTER: Or sign up here to the Mirror's..."
    body = re.sub(
        r'\nNEWSLETTER:[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # "Join our Mirror politics/free ITV/etc WhatsApp group..."
    # "Join our free [topic] WhatsApp community..."
    # "Join our new WhatsApp community..."
    # "Join our new MAN UTD WhatsApp community..."
    body = re.sub(
        r'\nREAD MORE: Join our[^\n]*WhatsApp[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    body = re.sub(
        r'\nJoin our[^\n]*WhatsApp[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # "Get Donald Trump updates straight to your WhatsApp!"
    # and following promo paragraph
    body = re.sub(
        r'\nGet [^\n]*straight to your WhatsApp[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # Promo paragraph: "As the world attempts to keep up with Trump's antics,
    # the Mirror has launched its very own..."
    body = re.sub(
        r'\n[^\n]*the Mirror has launched its very own[^\n]*WhatsApp[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # Follow-on promo paragraph: "We'll send you the latest breaking
    # updates and exclusives all directly to your phone..."
    body = re.sub(
        r"\nWe'll send you the latest[^\n]*WhatsApp[^\n]*",
        '',
        body,
        flags=re.IGNORECASE,
    )

    # "All you have to do to join is click on..."
    body = re.sub(
        r'\nAll you have to do to join[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # ── Social media footer ─────────────────────────────────────────────
    # "Join The Mirror's WhatsApp Community or follow us on Google News..."
    # Also: "* Join The Mirror's WhatsApp Community..."
    body = re.sub(
        r'(?:\n\*?\s*)?Join The Mirror.s WhatsApp Community[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # "PARTY GAMES: Watch our new YouTube series..."
    body = re.sub(
        r'\nPARTY GAMES:[^\n]*',
        '',
        body,
        flags=re.IGNORECASE,
    )

    # ── Read more: on own line (2016 era) ───────────────────────────────
    body = re.sub(r'\nRead more:.*\n', '\n', body, flags=re.IGNORECASE)

    # ── READ MORE: concatenated format (2023+) ──────────────────────────
    # Same as Star — headline jammed onto same line as article text.
    # Strip just the "READ MORE: " prefix tag.
    body = re.sub(r'\nREAD MORE:\s*', '\n', body, flags=re.IGNORECASE)
    body = re.sub(r'^READ MORE:\s*', '', body, flags=re.IGNORECASE)

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
    elif 'guardian' in outlet.lower():
        body = clean_guardian(body)
    elif 'star' in outlet.lower():
        body = clean_star(body)
    elif 'mirror' in outlet.lower():
        body = clean_mirror(body)

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

    if len(df) == 0:
        print(f"  SKIP {filepath} — empty file")
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
