import asyncio
import aiohttp
import csv
import os
import time
import trafilatura
import xml.etree.ElementTree as ET
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ── Choose your outlet here ──────────────────────────────────────────────
OUTLET = "The Independent"  # change to run a different outlet

OUTLET_CONFIGS = {
    "Daily Mirror": {
        "domain": "mirror.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.mirror.co.uk/sitemaps/map_art_{year}-{month:02d}-01.xml"
        ],
        "date_source": "trafilatura",
    },
    "Daily Express": {
        "domain": "express.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.express.co.uk/news/{year}{month:02d}.xml"
        ],
        "date_source": "trafilatura",
    },
    "Daily Star": {
        "domain": "dailystar.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.dailystar.co.uk/sitemaps/map_art_{year}-{month:02d}-01.xml"
        ],
        "date_source": "trafilatura",
    },
    "The Independent": {
        "domain": "independent.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.independent.co.uk/sitemaps/sitemap-articles-{year}-{month:02d}-{day:02d}.xml"
            for day in range(1, monthrange(year, month)[1] + 1)
        ],
        "date_source": "trafilatura",
    },
}

# ── Config ───────────────────────────────────────────────────────────────
KEYWORDS = ["immigration", "asylum", "migrants", "refugees", "borders", "migration"]
MIN_WORDS = 200
YEAR_START, YEAR_END = 2010, 2011
MONTH_START, MONTH_END = 1, 2

DOMAIN_CONCURRENCY = 15
GLOBAL_CONCURRENCY = 30
PARSE_WORKERS = os.cpu_count()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ── Derived from config ──────────────────────────────────────────────────
config = OUTLET_CONFIGS[OUTLET]
OUTPUT_FILE = f"{OUTLET.lower().replace(' ', '_')}_corpus.csv"
URL_QUEUE_FILE = f"{OUTLET.lower().replace(' ', '_')}_url_queue.csv"

# ── Helpers ──────────────────────────────────────────────────────────────

def contains_keyword(text: str) -> bool:
    return any(kw in text.lower() for kw in KEYWORDS)


def _extract_article(html: str):
    """CPU-bound — runs in process pool."""
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        metadata = trafilatura.extract_metadata(html)
        date = metadata.date if metadata else None
        return text, date
    except Exception:
        return None, None


# ── Phase 1: URL Discovery ───────────────────────────────────────────────

async def discover_urls_for_month(session, sem, year, month):
    sitemap_urls = config["sitemap_urls"](year, month)
    results = []

    for sitemap_url in sitemap_urls:
        async with sem:
            try:
                async with session.get(
                    sitemap_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        continue
                    body = await r.read()
            except Exception as e:
                print(f"  [{OUTLET}] Sitemap error {sitemap_url}: {e}")
                continue

        try:
            root = ET.fromstring(body)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for url_elem in root.findall("sm:url", ns):
                loc = url_elem.find("sm:loc", ns)
                lastmod = url_elem.find("sm:lastmod", ns)
                if loc is not None:
                    date_str = (
                        lastmod.text.strip()[:10] if lastmod is not None else None
                    )
                    results.append({
                        "outlet": OUTLET,
                        "url": loc.text.strip(),
                        "sitemap_date": date_str,
                        "year": year,
                        "month": month,
                    })
        except Exception as e:
            print(f"  [{OUTLET}] XML parse error {sitemap_url}: {e}")
            continue

    return results


async def run_discovery():
    if Path(URL_QUEUE_FILE).exists():
        print(f"✓ {URL_QUEUE_FILE} already exists — skipping discovery")
        return

    sem = asyncio.Semaphore(GLOBAL_CONCURRENCY)
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = [
            discover_urls_for_month(session, sem, year, month)
            for year in range(YEAR_START, YEAR_END)
            for month in range(MONTH_START, MONTH_END)
        ]
        print(f"Discovering URLs across {len(tasks)} months for {OUTLET}...")
        results = await asyncio.gather(*tasks)

    all_urls = [item for sublist in results for item in sublist]
    print(f"  Total URLs found: {len(all_urls)}")

    # Deduplicate
    seen = set()
    deduped = []
    for item in all_urls:
        if item["url"] not in seen:
            seen.add(item["url"])
            deduped.append(item)
    print(f"  After dedup: {len(deduped)}")

    with open(URL_QUEUE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["outlet", "url", "sitemap_date", "year", "month"]
        )
        writer.writeheader()
        writer.writerows(deduped)

    print(f"✓ Wrote {len(deduped)} URLs to {URL_QUEUE_FILE}")


# ── Phase 2: Fetch + Extract ─────────────────────────────────────────────

def load_queue():
    with open(URL_QUEUE_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done():
    if not Path(OUTPUT_FILE).exists():
        return set()
    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        return {row["url"] for row in csv.DictReader(f)}


async def fetch_and_extract(session, domain_sem, global_sem, pool, item):
    url = item["url"]

    async with global_sem, domain_sem:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status != 200:
                    return None
                html = await r.text()
        except Exception:
            return None

    loop = asyncio.get_running_loop()
    text, extracted_date = await loop.run_in_executor(pool, _extract_article, html)

    if not text:
        return None

    # Validate date matches target year/month
    date = extracted_date
    if config["date_source"] == "trafilatura":
        if not date:
            return None
        try:
            art_year = int(date[:4])
            art_month = int(date[5:7])
            if art_year != int(item["year"]) or art_month != int(item["month"]):
                return None
        except Exception:
            return None
    else:
        date = item["sitemap_date"] or extracted_date

    wordcount = len(text.split())
    if wordcount < MIN_WORDS:
        return None
    if not contains_keyword(text):
        return None

    return {
        "outlet": OUTLET,
        "year": item["year"],
        "month": item["month"],
        "date": date,
        "url": url,
        "wordcount": wordcount,
        "body": text,
    }


async def run_extraction():
    queue = load_queue()
    done = already_done()
    remaining = [item for item in queue if item["url"] not in done]
    print(f"Extraction: {len(remaining)} to fetch ({len(done)} already done)")

    if not remaining:
        print("Nothing to do.")
        return

    fieldnames = ["outlet", "year", "month", "date", "url", "wordcount", "body"]
    write_header = not Path(OUTPUT_FILE).exists() or len(done) == 0
    f = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    domain_sem = asyncio.Semaphore(DOMAIN_CONCURRENCY)
    global_sem = asyncio.Semaphore(GLOBAL_CONCURRENCY)

    saved = 0
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=PARSE_WORKERS) as pool:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            BATCH = 200
            for i in range(0, len(remaining), BATCH):
                batch = remaining[i: i + BATCH]
                tasks = [
                    fetch_and_extract(session, domain_sem, global_sem, pool, item)
                    for item in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict):
                        writer.writerow(result)
                        saved += 1

                f.flush()
                elapsed = time.monotonic() - t0
                total_processed = i + len(batch)
                rate = total_processed / elapsed if elapsed else 0
                print(
                    f"  [{total_processed}/{len(remaining)}] "
                    f"{saved} saved · {rate:.1f} articles/sec"
                )

    f.close()
    elapsed = time.monotonic() - t0
    print(f"✓ Done. {saved} articles saved in {elapsed / 60:.1f} minutes")


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    print(f"=== Scraping {OUTLET} ===")
    await run_discovery()
    await run_extraction()


if __name__ == "__main__":
    asyncio.run(main())
