import os
import requests
import csv
import time
from datetime import datetime, date
from calendar import monthrange

API_KEY = os.environ.get("GUARDIAN_API_KEY")
BASE_URL = "https://content.guardianapis.com/search"
QUERY = "immigration OR asylum OR migrants OR refugees OR borders"
OUTPUT_FILE = "guardian_corpus.csv"
MIN_WORDS = 200

def get_articles_for_month(year, month):
    start_date = f"{year}-{month:02d}-01"
    last_day = monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day}"

    articles = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        params = {
            "q": QUERY,
            "from-date": start_date,
            "to-date": end_date,
            "show-fields": "bodyText,wordcount",
            "page-size": 200,
            "page": page,
            "api-key": API_KEY
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

            if wordcount >= MIN_WORDS and body:
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
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["year", "month", "date", "title", "url", "section", "wordcount", "body"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for year in range(2010, 2026):
            for month in range(1, 13):
                print(f"Fetching {year}-{month:02d}...")
                articles = get_articles_for_month(year, month)
                writer.writerows(articles)
                f.flush()  # save progress as we go
                print(f"  -> {len(articles)} articles saved")
                time.sleep(0.5)

if __name__ == "__main__":
    main()
