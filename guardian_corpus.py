import os
import requests
import csv
import time
import argparse
from calendar import monthrange
from pathlib import Path

BASE_URL = "https://content.guardianapis.com/search"

def get_articles_for_month(year, month, query, min_words, api_key):
    start_date = f"{year}-{month:02d}-01"
    last_day = monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day}"

    articles = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        params = {
            "q": query,
            "from-date": start_date,
            "to-date": end_date,
            "show-fields": "bodyText,wordcount",
            "page-size": 200,
            "page": page,
            "api-key": api_key
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print(f"Error {response.status_code} for {year}-{month:02d}, page {page}")
            break

        data = response.json()["response"]
        total_pages = data["pages"]

        for article in data["results"]:
            fields = article.get("fields", {})
            wordcount = int(fields.get("wordcount", 0) or 0)
            body = fields.get("bodyText", "") or ""

            if wordcount >= min_words and body:
                articles.append({
                    "year": year,
                    "month": month,
                    "date": article["webPublicationDate"],
                    "title": article["webTitle"],
                    "url": article["webUrl"],
                    "section": article["sectionId"],
                    "wordcount": wordcount,
                    "body": body
                })

        page += 1
        time.sleep(0.1)  # be polite to the API

    return articles

def main():
    parser = argparse.ArgumentParser(description="Fetch Guardian articles into a CSV corpus")
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--year-end", type=int, default=2026,
                        help="Exclusive end year (e.g. 2026 processes up to and including 2025)")
    parser.add_argument("--month-start", type=int, default=1)
    parser.add_argument("--month-end", type=int, default=13,
                        help="Exclusive end month (e.g. 13 processes all months 1-12)")
    parser.add_argument(
        "--keywords",
        default="immigration,asylum,migrants,refugees,borders",
        help="Comma-separated keywords used to build the API query (joined with OR)",
    )
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument("--output-file", default="guardian_corpus.csv")
    args = parser.parse_args()

    api_key = os.environ.get("GUARDIAN_API_KEY")
    if not api_key:
        raise SystemExit("Error: GUARDIAN_API_KEY environment variable is not set")
    query = " OR ".join(kw.strip() for kw in args.keywords.split(","))

    with Path(args.output_file).open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["year", "month", "date", "title", "url", "section", "wordcount", "body"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for year in range(args.year_start, args.year_end):
            for month in range(args.month_start, args.month_end):
                print(f"Fetching {year}-{month:02d}...")
                articles = get_articles_for_month(year, month, query, args.min_words, api_key)
                writer.writerows(articles)
                f.flush()  # save progress as we go
                print(f"  -> {len(articles)} articles saved")
                time.sleep(0.5)

if __name__ == "__main__":
    main()
