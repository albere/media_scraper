import argparse
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

OUTLET_CONFIGS = {
    "Daily Mirror": {
        "domain": "mirror.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.mirror.co.uk/sitemaps/map_art_{year}-{month:02d}-01.xml"
        ],
        "date_source": "trafilatura",
        "crawl_delay": 2,
    },
    "Daily Express": {
        "domain": "express.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.express.co.uk/news/{year}{month:02d}.xml"
        ],
        "date_source": "trafilatura",
        "crawl_delay": 2,
    },
    "Daily Star": {
        "domain": "dailystar.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.dailystar.co.uk/sitemaps/map_art_{year}-{month:02d}-01.xml"
        ],
        "date_source": "trafilatura",
        "crawl_delay": 10,
    },
    "The Independent": {
        "domain": "independent.co.uk",
        "sitemap_urls": lambda year, month: [
            f"https://www.independent.co.uk/sitemaps/sitemap-articles-{year}-{month:02d}-{day:02d}.xml"
            for day in range(1, monthrange(year, month)[1] + 1)
        ],
        "date_source": "trafilatura",
        "crawl_delay": 2,
    },
}

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── Rate limiter ─────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self.lock = asyncio.Lock()
        self.last_request = 0.0

    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            wait = self.delay - (now - self.last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_request = asyncio.get_event_loop().time()


# ── Helpers ──────────────────────────────────────────────────────────────

def contains_keyword(text: str, keywords: list) -> bool:
    return any(kw in text.lower() for kw in keywords)


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

async def discover_urls_for_month(session, sem, rate_limiter, year, month, args):
    sitemap_urls = args.config["sitemap_urls"](year, month)

    async def fetch_one(sitemap_url):
        await rate_limiter.acquire()
        async with sem:
            try:
                async with session.get(
                    sitemap_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        return []
                    body = await r.read()
            except Exception as e:
                print(f"  [{args.outlet}] Sitemap error {sitemap_url}: {e}")
                return []

        try:
            root = ET.fromstring(body)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            found = []
            for url_elem in root.findall("sm:url", ns):
                loc = url_elem.find("sm:loc", ns)
                lastmod = url_elem.find("sm:lastmod", ns)
                if loc is not None:
                    date_str = (
                        lastmod.text.strip()[:10] if lastmod is not None else None
                    )
                    found.append({
                        "outlet": args.outlet,
                        "url": loc.text.strip(),
                        "sitemap_date": date_str,
                        "year": year,
                        "month": month,
                    })
            return found
        except Exception as e:
            print(f"  [{args.outlet}] XML parse error {sitemap_url}: {e}")
            return []

    all_results = await asyncio.gather(*[fetch_one(u) for u in sitemap_urls])
    return [item for sublist in all_results for item in sublist]


async def run_discovery(args):
    if Path(args.url_queue_file).exists():
        print(f"✓ {args.url_queue_file} already exists — skipping discovery")
        return

    sem = asyncio.Semaphore(args.global_concurrency)
    rate_limiter = RateLimiter(args.crawl_delay)
    headers = {"User-Agent": args.user_agent}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            discover_urls_for_month(session, sem, rate_limiter, year, month, args)
            for year in range(args.year_start, args.year_end)
            for month in range(args.month_start, args.month_end)
        ]
        print(f"Discovering URLs across {len(tasks)} months for {args.outlet}...")
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

    with open(args.url_queue_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["outlet", "url", "sitemap_date", "year", "month"]
        )
        writer.writeheader()
        writer.writerows(deduped)

    print(f"✓ Wrote {len(deduped)} URLs to {args.url_queue_file}")


# ── Phase 2: Fetch + Extract ─────────────────────────────────────────────

def load_queue(args):
    with open(args.url_queue_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(args):
    if not Path(args.output_file).exists():
        return set()
    with open(args.output_file, newline="", encoding="utf-8") as f:
        return {row["url"] for row in csv.DictReader(f)}


async def fetch_and_extract(session, domain_sem, global_sem, rate_limiter, pool, item, args):
    url = item["url"]

    await rate_limiter.acquire()
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
    if args.config["date_source"] == "trafilatura":
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
    if wordcount < args.min_words:
        return None
    if not contains_keyword(text, args.keywords):
        return None

    return {
        "outlet": args.outlet,
        "year": item["year"],
        "month": item["month"],
        "date": date,
        "url": url,
        "wordcount": wordcount,
        "body": text,
    }


async def run_extraction(args):
    queue = load_queue(args)
    done = already_done(args)
    remaining = [item for item in queue if item["url"] not in done]
    print(f"Extraction: {len(remaining)} to fetch ({len(done)} already done)")

    if not remaining:
        print("Nothing to do.")
        return

    fieldnames = ["outlet", "year", "month", "date", "url", "wordcount", "body"]
    write_header = not Path(args.output_file).exists() or len(done) == 0
    f = open(args.output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    domain_sem = asyncio.Semaphore(args.domain_concurrency)
    global_sem = asyncio.Semaphore(args.global_concurrency)
    rate_limiter = RateLimiter(args.crawl_delay)
    headers = {"User-Agent": args.user_agent}

    saved = 0
    errors = 0
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.parse_workers) as pool:
        async with aiohttp.ClientSession(headers=headers) as session:
            for i in range(0, len(remaining), args.batch_size):
                batch = remaining[i: i + args.batch_size]
                tasks = [
                    fetch_and_extract(session, domain_sem, global_sem, rate_limiter, pool, item, args)
                    for item in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        errors += 1
                        print(f"  [error] {type(result).__name__}: {result}")
                    elif isinstance(result, dict):
                        writer.writerow(result)
                        saved += 1

                f.flush()
                elapsed = time.monotonic() - t0
                total_processed = i + len(batch)
                rate = total_processed / elapsed if elapsed else 0
                print(
                    f"  [{total_processed}/{len(remaining)}] "
                    f"{saved} saved · {errors} errors · {rate:.1f} articles/sec"
                )

    f.close()
    elapsed = time.monotonic() - t0
    print(f"✓ Done. {saved} articles saved in {elapsed / 60:.1f} minutes")


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Media scraper")
    parser.add_argument(
        "outlet",
        choices=list(OUTLET_CONFIGS.keys()),
        help="News outlet to scrape",
    )
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--year-end", type=int, default=2022,
                        help="Exclusive end year (e.g. 2022 processes up to and including 2021)")
    parser.add_argument("--month-start", type=int, default=1)
    parser.add_argument("--month-end", type=int, default=13,
                        help="Exclusive end month (e.g. 13 processes all months 1-12)")
    parser.add_argument("--crawl-delay", type=float, default=None,
                        help="Seconds between requests (default: per-outlet)")
    parser.add_argument("--domain-concurrency", type=int, default=5)
    parser.add_argument("--global-concurrency", type=int, default=10)
    parser.add_argument("--parse-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument(
        "--keywords",
        default="immigration,asylum,migrants,refugees,borders,migration",
        help="Comma-separated keywords to filter articles",
    )
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--url-queue-file", default=None)
    parser.add_argument("--user-agent", default=_DEFAULT_USER_AGENT)

    args = parser.parse_args()

    # Resolve config and derived values
    args.config = OUTLET_CONFIGS[args.outlet]
    if args.crawl_delay is None:
        args.crawl_delay = args.config["crawl_delay"]
    outlet_slug = args.outlet.lower().replace(" ", "_")
    if args.output_file is None:
        args.output_file = f"{outlet_slug}_corpus.csv"
    if args.url_queue_file is None:
        args.url_queue_file = f"{outlet_slug}_url_queue.csv"
    if args.parse_workers is None:
        args.parse_workers = os.cpu_count()
    args.keywords = [kw.strip() for kw in args.keywords.split(",")]

    return args


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    print(f"=== Scraping {args.outlet} ===")
    print(f"    Years: {args.year_start}-{args.year_end}")
    print(f"    Months: {args.month_start}-{args.month_end}")
    print(f"    Crawl delay: {args.crawl_delay}s")
    print(f"    Domain concurrency: {args.domain_concurrency}")
    print(f"    Parse workers: {args.parse_workers}")
    print(f"    Output: {args.output_file}")

    await run_discovery(args)
    await run_extraction(args)


if __name__ == "__main__":
    asyncio.run(main())
