"""
lemma_test.py — Guardian API keyword expansion test

Compares article counts for one month using original keywords vs
expanded keywords (adding singular forms and verb forms).

Usage:
    python lemma_test.py --api-key YOUR_KEY
    python lemma_test.py --api-key YOUR_KEY --year 2016 --month 6
"""

import requests
import argparse
import time
from calendar import monthrange

BASE_URL = "https://content.guardianapis.com/search"

ORIGINAL_QUERY = "immigration OR asylum OR migrants OR refugees OR borders OR migration"

# Expanded: add singulars (migrant, refugee, border, immigrant)
# and verb forms (immigrate, immigrating, immigrated, migrate, migrating, migrated)
EXPANDED_QUERY = (
    "immigration OR asylum OR migrants OR refugees OR borders OR migration "
    "OR migrant OR refugee OR border OR immigrant OR immigrants "
)


def count_articles(api_key, query, year, month):
    """Return total article count for a query in a given month."""
    last_day = monthrange(year, month)[1]
    params = {
        "q": query,
        "from-date": f"{year}-{month:02d}-01",
        "to-date": f"{year}-{month:02d}-{last_day}",
        "show-fields": "wordcount",
        "page-size": 1,  # we only need the total count
        "api-key": api_key,
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"  API error: {response.status_code}")
        print(f"  {response.text[:300]}")
        return None
    data = response.json()
    return data["response"]["total"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--month", type=int, default=1)
    args = parser.parse_args()

    print(f"Testing {args.year}-{args.month:02d}\n")

    print("Original keywords:")
    print(f"  {ORIGINAL_QUERY}\n")
    original = count_articles(args.api_key, ORIGINAL_QUERY, args.year, args.month)
    print(f"  Total articles: {original}\n")

    time.sleep(0.5)

    print("Expanded keywords (+ singulars + verb forms):")
    print(f"  {EXPANDED_QUERY}\n")
    expanded = count_articles(args.api_key, EXPANDED_QUERY, args.year, args.month)
    print(f"  Total articles: {expanded}\n")

    if original and expanded:
        diff = expanded - original
        pct = (diff / original) * 100 if original > 0 else 0
        print(f"Difference: {diff} articles ({pct:.1f}% increase)")
        if pct < 5:
            print("-> Marginal difference. Lemmatisation unlikely to change results.")
        elif pct < 15:
            print("-> Moderate difference. Worth investigating further.")
        else:
            print("-> Substantial difference. Re-collection with expanded keywords recommended.")


if __name__ == "__main__":
    main()
