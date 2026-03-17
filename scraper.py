import asyncio
import aiohttp
import csv
import os
import time
import trafilatura
import xml.etree.ElementTree as ET
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from pathlib import Path

import aio.run.runner as runner

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


# ── Runner ───────────────────────────────────────────────────────────────

class CorpusScraper(runner.Runner):

    def add_arguments(self, parser):
        super().add_arguments(parser)
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

    @cached_property
    def config(self):
        return OUTLET_CONFIGS[self.args.outlet]

    @cached_property
    def crawl_delay(self):
        return self.args.crawl_delay or self.config["crawl_delay"]

    @cached_property
    def keywords(self):
        return [kw.strip() for kw in self.args.keywords.split(",")]

    @cached_property
    def output_file(self):
        return self.args.output_file or (
            f"{self.args.outlet.lower().replace(' ', '_')}_corpus.csv"
        )

    @cached_property
    def url_queue_file(self):
        return self.args.url_queue_file or (
            f"{self.args.outlet.lower().replace(' ', '_')}_url_queue.csv"
        )

    @cached_property
    def rate_limiter(self):
        return RateLimiter(self.crawl_delay)

    @cached_property
    def session(self):
        return aiohttp.ClientSession(headers={"User-Agent": self.args.user_agent})

    @cached_property
    def pool(self):
        return ProcessPoolExecutor(max_workers=self.args.parse_workers or os.cpu_count())

    async def cleanup(self):
        await super().cleanup()
        if "session" in self.__dict__:
            await self.session.close()
        if "pool" in self.__dict__:
            self.pool.shutdown(wait=False)

    async def run_discovery(self):
        if Path(self.url_queue_file).exists():
            self.log.info(f"✓ {self.url_queue_file} already exists — skipping discovery")
            return

        sem = asyncio.Semaphore(self.args.global_concurrency)
        tasks = [
            self._discover_urls_for_month(sem, year, month)
            for year in range(self.args.year_start, self.args.year_end)
            for month in range(self.args.month_start, self.args.month_end)
        ]
        self.log.info(f"Discovering URLs across {len(tasks)} months for {self.args.outlet}...")
        results = await asyncio.gather(*tasks)

        all_urls = [item for sublist in results for item in sublist]
        self.log.info(f"  Total URLs found: {len(all_urls)}")

        # Deduplicate
        seen = set()
        deduped = []
        for item in all_urls:
            if item["url"] not in seen:
                seen.add(item["url"])
                deduped.append(item)
        self.log.info(f"  After dedup: {len(deduped)}")

        with open(self.url_queue_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["outlet", "url", "sitemap_date", "year", "month"]
            )
            writer.writeheader()
            writer.writerows(deduped)

        self.log.success(f"✓ Wrote {len(deduped)} URLs to {self.url_queue_file}")

    async def run_extraction(self):
        queue = self._load_queue()
        done = self._already_done()
        remaining = [item for item in queue if item["url"] not in done]
        self.log.info(f"Extraction: {len(remaining)} to fetch ({len(done)} already done)")

        if not remaining:
            self.log.info("Nothing to do.")
            return

        fieldnames = ["outlet", "year", "month", "date", "url", "wordcount", "body"]
        write_header = not Path(self.output_file).exists() or len(done) == 0
        f = open(self.output_file, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        domain_sem = asyncio.Semaphore(self.args.domain_concurrency)
        global_sem = asyncio.Semaphore(self.args.global_concurrency)

        saved = 0
        errors = 0
        t0 = time.monotonic()

        try:
            for i in range(0, len(remaining), self.args.batch_size):
                batch = remaining[i: i + self.args.batch_size]
                tasks = [
                    self._fetch_and_extract(domain_sem, global_sem, item)
                    for item in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        errors += 1
                        self.log.error(f"  [error] {type(result).__name__}: {result}")
                    elif isinstance(result, dict):
                        writer.writerow(result)
                        saved += 1

                f.flush()
                elapsed = time.monotonic() - t0
                total_processed = i + len(batch)
                rate = total_processed / elapsed if elapsed else 0
                self.log.info(
                    f"  [{total_processed}/{len(remaining)}] "
                    f"{saved} saved · {errors} errors · {rate:.1f} articles/sec"
                )
        finally:
            f.close()

        elapsed = time.monotonic() - t0
        self.log.success(f"✓ Done. {saved} articles saved in {elapsed / 60:.1f} minutes")

    # ── Phase 1: URL Discovery ───────────────────────────────────────────

    async def _fetch_sitemap(self, sem, sitemap_url, year, month):
        await self.rate_limiter.acquire()
        async with sem:
            try:
                async with self.session.get(
                    sitemap_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        return []
                    body = await r.read()
            except Exception as e:
                self.log.warning(f"  [{self.args.outlet}] Sitemap error {sitemap_url}: {e}")
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
                        "outlet": self.args.outlet,
                        "url": loc.text.strip(),
                        "sitemap_date": date_str,
                        "year": year,
                        "month": month,
                    })
            return found
        except Exception as e:
            self.log.warning(f"  [{self.args.outlet}] XML parse error {sitemap_url}: {e}")
            return []

    async def _discover_urls_for_month(self, sem, year, month):
        sitemap_urls = self.config["sitemap_urls"](year, month)
        all_results = await asyncio.gather(
            *[self._fetch_sitemap(sem, u, year, month) for u in sitemap_urls]
        )
        return [item for sublist in all_results for item in sublist]

    # ── Phase 2: Fetch + Extract ─────────────────────────────────────────

    async def _fetch_and_extract(self, domain_sem, global_sem, item):
        url = item["url"]

        await self.rate_limiter.acquire()
        async with global_sem, domain_sem:
            try:
                async with self.session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        return None
                    html = await r.text()
            except Exception:
                return None

        loop = asyncio.get_running_loop()
        text, extracted_date = await loop.run_in_executor(self.pool, _extract_article, html)

        if not text:
            return None

        # Validate date matches target year/month
        date = extracted_date
        if self.config["date_source"] == "trafilatura":
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
        if wordcount < self.args.min_words:
            return None
        if not contains_keyword(text, self.keywords):
            return None

        return {
            "outlet": self.args.outlet,
            "year": item["year"],
            "month": item["month"],
            "date": date,
            "url": url,
            "wordcount": wordcount,
            "body": text,
        }

    def _load_queue(self):
        with open(self.url_queue_file, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _already_done(self):
        if not Path(self.output_file).exists():
            return set()
        with open(self.output_file, newline="", encoding="utf-8") as f:
            return {row["url"] for row in csv.DictReader(f)}

    @runner.cleansup
    @runner.catches((aiohttp.ClientError, KeyboardInterrupt))
    async def run(self) -> int | None:
        self.log.info(f"=== Scraping {self.args.outlet} ===")
        self.log.info(f"    Years: {self.args.year_start}-{self.args.year_end}")
        self.log.info(f"    Months: {self.args.month_start}-{self.args.month_end}")
        self.log.info(f"    Crawl delay: {self.crawl_delay}s")
        self.log.info(f"    Domain concurrency: {self.args.domain_concurrency}")
        self.log.info(f"    Parse workers: {self.args.parse_workers or os.cpu_count()}")
        self.log.info(f"    Output: {self.output_file}")

        await self.run_discovery()
        await self.run_extraction()


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    import sys
    return CorpusScraper(*sys.argv[1:])()


if __name__ == "__main__":
    raise SystemExit(main())
