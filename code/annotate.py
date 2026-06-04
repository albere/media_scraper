"""
annotate.py — Interactive annotation tool for classifier validation.

Shows one article at a time in the terminal, records your keypress.
Progress is saved after every article so you can quit and resume.

Usage:
    python annotate.py topic     # annotate topic relevance (y/n)
    python annotate.py stance    # annotate stance (p/a/n)

Controls:
    Topic mode:  y = immigration,  n = not immigration,  s = skip,  q = quit
    Stance mode: p = pro,  a = anti,  n = neutral,  s = skip,  q = quit

Progress saves automatically. Re-run the same command to resume where you left off.
"""

import pandas as pd
import sys
import os

# ============================================================
# CONFIGURATION
# ============================================================

TOPIC_FILE = "topic_validation_sample.csv"
STANCE_FILE = "stance_validation_sample.csv"

# How many characters of the article to show. Set to 0 for full text.
TRUNCATE_BODY = 0

# ============================================================


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def show_article(row, index, total, mode):
    """Display one article for annotation."""
    clear_screen()

    print(f"{'=' * 70}")
    print(f"  Article {index + 1} / {total}    |    Outlet: {row['outlet']}")
    print(f"  URL: {row['url']}")

    if mode == "topic":
        print(f"  BART score: {row['immigration_score']:.4f}")
        print(f"  BART decision: {'KEEP' if row['immigration_score'] >= 0.5 else 'REJECT'}")
    elif mode == "stance":
        print(f"  BART stance: {row['stance_label']}")
        print(f"  Pro: {row['pro_score']:.4f}  Anti: {row['anti_score']:.4f}")

    print(f"{'=' * 70}\n")

    body = str(row["body"])
    if TRUNCATE_BODY > 0 and len(body) > TRUNCATE_BODY:
        print(body[:TRUNCATE_BODY])
        print(f"\n... [truncated, {len(body)} chars total] ...")
    else:
        print(body)

    print(f"\n{'─' * 70}")


def get_input_topic():
    """Get topic annotation from user."""
    while True:
        print("  [y] immigration   [n] not immigration   [s] skip   [q] quit")
        key = input("  > ").strip().lower()
        if key in ("y", "n", "s", "q"):
            return key
        print("  Invalid input, try again.")


def get_input_stance():
    """Get stance annotation from user."""
    while True:
        print("  [p] pro-immigration   [a] anti-immigration   [n] neutral   [s] skip   [q] quit")
        key = input("  > ").strip().lower()
        if key in ("p", "a", "n", "s", "q"):
            return key
        print("  Invalid input, try again.")


def label_to_text(key, mode):
    """Convert keypress to label string."""
    if mode == "topic":
        return {"y": "immigration", "n": "not_immigration", "s": "skip"}[key]
    elif mode == "stance":
        return {"p": "pro", "a": "anti", "n": "neutral", "s": "skip"}[key]


def run_annotation(mode):
    """Main annotation loop."""
    if mode == "topic":
        filepath = TOPIC_FILE
        get_input = get_input_topic
    elif mode == "stance":
        filepath = STANCE_FILE
        get_input = get_input_stance
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Run sample_for_validation.py first.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    df["human_label"] = df["human_label"].astype(str).replace("nan", "")

    # Find where we left off
    already_done = df["human_label"].notna() & (df["human_label"] != "")
    start_index = already_done.sum()
    total = len(df)

    if start_index >= total:
        print(f"All {total} articles already annotated!")
        print_summary(df, mode)
        return

    print(f"\nResuming from article {start_index + 1} / {total}")
    print(f"({start_index} already annotated)")
    input("Press Enter to start...")

    for i in range(start_index, total):
        row = df.iloc[i]
        show_article(row, i, total, mode)

        key = get_input()

        if key == "q":
            df.to_csv(filepath, index=False)
            print(f"\nProgress saved. {i - start_index} new annotations this session.")
            print(f"Total annotated: {i} / {total}")
            return

        df.at[i, "human_label"] = label_to_text(key, mode)

        # Save after every article
        df.to_csv(filepath, index=False)

    print(f"\nDone! All {total} articles annotated.")
    print_summary(df, mode)


def print_summary(df, mode):
    """Print agreement summary after annotation is complete."""
    labelled = df[df["human_label"].notna() & (df["human_label"] != "") & (df["human_label"] != "skip")]

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION SUMMARY ({len(labelled)} articles, excluding skips)")
    print(f"{'=' * 70}")

    if mode == "topic":
        # BART decision vs human label
        labelled = labelled.copy()
        labelled["bart_decision"] = (labelled["immigration_score"] >= 0.5).map(
            {True: "immigration", False: "not_immigration"}
        )

        agree = (labelled["human_label"] == labelled["bart_decision"]).sum()
        print(f"\n  Agreement: {agree}/{len(labelled)} ({100*agree/len(labelled):.1f}%)")

        # Confusion matrix
        tp = ((labelled["bart_decision"] == "immigration") & (labelled["human_label"] == "immigration")).sum()
        fp = ((labelled["bart_decision"] == "immigration") & (labelled["human_label"] == "not_immigration")).sum()
        fn = ((labelled["bart_decision"] == "not_immigration") & (labelled["human_label"] == "immigration")).sum()
        tn = ((labelled["bart_decision"] == "not_immigration") & (labelled["human_label"] == "not_immigration")).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  Confusion matrix:")
        print(f"                    Human: immigration   Human: not_immigration")
        print(f"  BART: immigration      {tp:>6}              {fp:>6}")
        print(f"  BART: not_immigration  {fn:>6}              {tn:>6}")
        print(f"\n  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1:        {f1:.3f}")

    elif mode == "stance":
        labelled = labelled.copy()

        agree = (labelled["human_label"] == labelled["stance_label"]).sum()
        print(f"\n  Agreement: {agree}/{len(labelled)} ({100*agree/len(labelled):.1f}%)")

        # Per-class breakdown
        labels = ["pro", "anti", "neutral"]
        print(f"\n  {'':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        for label in labels:
            tp = ((labelled["stance_label"] == label) & (labelled["human_label"] == label)).sum()
            fp = ((labelled["stance_label"] == label) & (labelled["human_label"] != label)).sum()
            fn = ((labelled["stance_label"] != label) & (labelled["human_label"] == label)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            support = (labelled["human_label"] == label).sum()

            print(f"  {label:>12} {p:>10.3f} {r:>10.3f} {f:>10.3f} {support:>10}")

    print(f"\n{'=' * 70}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("topic", "stance"):
        print("Usage: python annotate.py topic|stance")
        sys.exit(1)

    run_annotation(sys.argv[1])


if __name__ == "__main__":
    main()
