import aiohttp
import asyncio
import csv
import logging
import os
import sys
import time
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from pathlib import Path
import re
from typing import Any, Sequence
import argparse
import xml.etree.ElementTree as ET

import yaml

import trafilatura

import aio.run.runner as runner

_log = logging.getLogger(__name__)

_DAY_PLACEHOLDER_PATTERN = re.compile(r"{day(?::[^}]*)?}")


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ── Rate limiter ─────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.lock = asyncio.Lock()
        self.last_request = 0.0

    async def acquire(self) -> None:
        async with self.lock:
            now = asyncio.get_event_loop().time()
            wait = self.delay - (now - self.last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_request = asyncio.get_event_loop().time()


# ── Helpers ──────────────────────────────────────────────────────────────

def contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    return any(kw in text.lower() for kw in keywords)


def _extract_article(html: str) -> tuple[str | None, str | None]:
    """CPU-bound — runs in process pool."""
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        metadata = trafilatura.extract_metadata(html)
        date = metadata.date if metadata else None
        return text, date
    except Exception as e:
        _log.warning(f"trafilatura extraction failed: {type(e).__name__}: {e}")
        return None, None


class CorpusScraper(runner.Runner):

    @staticmethod
    def _load_outlet_configs(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Outlet config file not found at {path} "
                "(set --outlet-config-path to override)"
            )

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Outlet config file {path} is empty or invalid")
        if not isinstance(data, dict):
            raise ValueError(
                f"Outlet config must be a mapping, got {type(data).__name__}"
            )

        return data

    @cached_property
    def outlet_config_path(self) -> Path:
        if self.args.outlet_config_path:
            return Path(self.args.outlet_config_path)
        return Path(__file__).with_name("outlet_configs.yaml")

    @cached_property
    def outlet_configs(self) -> dict[str, dict[str, Any]]:
        return self._load_outlet_configs(self.outlet_config_path)

    @cached_property
    def config(self) -> dict[str, Any]:
        configs = self.outlet_configs
        if self.args.outlet not in configs:
            raise ValueError(
                f"Outlet '{self.args.outlet}' not found in {self.outlet_config_path}. "
                "Check the YAML configuration."
            )
        return configs[self.args.outlet]

    @cached_property
    def crawl_delay(self) -> float:
        return self.args.crawl_delay or self.config["crawl_delay"]

    @cached_property
    def keywords(self) -> list[str]:
        return [kw.strip() for kw in self.args.keywords.split(",")]

    @cached_property
    def output_path(self) -> Path:
        return Path(self.args.output_file or f"{self._slug}_corpus.csv")

    @cached_property
    def url_queue_path(self) -> Path:
        return Path(self.args.url_queue_file or f"{self._slug}_url_queue.csv")

    @cached_property
    def rate_limiter(self) -> RateLimiter:
        return RateLimiter(self.crawl_delay)

    @cached_property
    def session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(headers={"User-Agent": self.args.user_agent})

    @cached_property
    def pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=self.args.parse_workers or os.cpu_count())

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "outlet",
            help="News outlet to scrape (must exist in outlet config YAML)",
        )
        parser.add_argument(
            "--outlet-config-path",
            default=os.environ.get("OUTLET_CONFIG_PATH"),
            help="Path to outlet config YAML (default: outlet_configs.yaml)",
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
        parser.add_argument(
            "--s3-bucket",
            default=None,
            help="S3 bucket for caching completed months (overrides S3_BUCKET env var)",
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        if "session" in self.__dict__:
            await self.session.close()
        if "pool" in self.__dict__:
            self.pool.shutdown(wait=False)

    async def run_discovery(self) -> None:
        if self.url_queue_path.exists():
            self.log.info(f"✓ {self.url_queue_path} already exists — skipping discovery")
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

        with self.url_queue_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["outlet", "url", "sitemap_date", "year", "month"]
            )
            writer.writeheader()
            writer.writerows(deduped)

        self.log.success(f"✓ Wrote {len(deduped)} URLs to {self.url_queue_path}")

    async def run_extraction(self) -> None:
        queue = self._load_queue()
        done = self._already_done()
        remaining = [item for item in queue if item["url"] not in done]
        self.log.info(f"Extraction: {len(remaining)} to fetch ({len(done)} already done)")

        if not remaining:
            self.log.info("Nothing to do.")
            return

        fieldnames = ["outlet", "year", "month", "date", "url", "wordcount", "body"]
        write_header = not self.output_path.exists() or len(done) == 0
        f = self.output_path.open("a", newline="", encoding="utf-8")
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

    @runner.cleansup
    @runner.catches((aiohttp.ClientError, KeyboardInterrupt))
    async def run(self) -> int | None:
        from s3_cache import S3Cache

        self.log.info(f"=== Scraping {self.args.outlet} ===")
        self.log.info(f"    Years:  {self.args.year_start}–{self.args.year_end - 1}")
        self.log.info(f"    Months: {self.args.month_start}–{self.args.month_end - 1} (inclusive)")
        self.log.info(f"    Crawl delay: {self.crawl_delay}s")
        self.log.info(f"    Domain concurrency: {self.args.domain_concurrency}")
        self.log.info(f"    Parse workers: {self.args.parse_workers or os.cpu_count()}")

        async with S3Cache(self.args.s3_bucket) as cache:
            for year in range(self.args.year_start, self.args.year_end):
                for month in range(self.args.month_start, self.args.month_end):
                    if await cache.exists(self.args.outlet, year, month):
                        self.log.info(
                            f"  Skipping {year}-{month:02d} (S3 cache hit)"
                        )
                        continue

                    queue_file = self._month_queue_path(year, month)
                    output_file = self._month_output_path(year, month)

                    try:
                        await self._discover_month(year, month, queue_file)
                        await self._extract_month(year, month, queue_file, output_file)

                        if output_file.exists() and output_file.stat().st_size > 0:
                            await cache.upload(
                                output_file, self.args.outlet, year, month
                            )
                            self.log.success(
                                f"  ✓ {year}-{month:02d} complete and uploaded to S3"
                            )
                        else:
                            self.log.warning(
                                f"  ⚠ {year}-{month:02d}: no output produced — "
                                "not uploading to S3"
                            )
                    except Exception as exc:
                        self.log.error(
                            f"  ✗ {year}-{month:02d} failed: "
                            f"{type(exc).__name__}: {exc} — "
                            "will retry on next run"
                        )

    @cached_property
    def _slug(self) -> str:
        return self.args.outlet.lower().replace(" ", "_")

    def _already_done(self) -> set[str]:
        if not self.output_path.exists():
            return set()
        with self.output_path.open(newline="", encoding="utf-8") as f:
            return {row["url"] for row in csv.DictReader(f)}

    def _sitemap_urls(self, year: int, month: int) -> list[str]:
        if "sitemap_templates" not in self.config:
            raise KeyError(f"No sitemap_templates configured for {self.args.outlet}")

        templates = self.config["sitemap_templates"]
        if not templates:
            raise ValueError(f"sitemap_templates for {self.args.outlet} is empty")

        urls: list[str] = []
        last_day = monthrange(year, month)[1]

        for template in templates:
            if not isinstance(template, str):
                raise TypeError(
                    f"Sitemap template for {self.args.outlet} must be a string, "
                    f"got {type(template).__name__}"
                )

            try:
                if _DAY_PLACEHOLDER_PATTERN.search(template):
                    urls.extend(
                        template.format(year=year, month=month, day=day)
                        for day in range(1, last_day + 1)
                    )
                else:
                    urls.append(template.format(year=year, month=month))
            except KeyError as exc:
                raise KeyError(
                    f"Unknown placeholder {exc} in sitemap template for {self.args.outlet}"
                ) from exc

        return urls

    async def _discover_month(self, year: int, month: int, queue_file: Path) -> int:
        """Discover URLs for a single (year, month) and write them to *queue_file*.

        Returns the number of (deduplicated) URLs found.  If *queue_file*
        already exists the step is skipped and the existing row count returned.
        """
        if queue_file.exists():
            self.log.info(f"  ✓ {queue_file} already exists — skipping discovery")
            with queue_file.open(newline="", encoding="utf-8") as f:
                return sum(1 for _ in csv.DictReader(f))

        sem = asyncio.Semaphore(self.args.global_concurrency)
        urls = await self._discover_urls_for_month(sem, year, month)

        seen: set[str] = set()
        deduped = []
        for item in urls:
            if item["url"] not in seen:
                seen.add(item["url"])
                deduped.append(item)

        with queue_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["outlet", "url", "sitemap_date", "year", "month"]
            )
            writer.writeheader()
            writer.writerows(deduped)

        self.log.info(f"  Discovered {len(deduped)} URLs for {year}-{month:02d}")
        return len(deduped)

    async def _discover_urls_for_month(
        self, sem: asyncio.Semaphore, year: int, month: int
    ) -> list[dict[str, Any]]:
        sitemap_urls = self._sitemap_urls(year, month)
        all_results = await asyncio.gather(
            *[self._fetch_sitemap(sem, u, year, month) for u in sitemap_urls]
        )
        return [item for sublist in all_results for item in sublist]

    async def _fetch_and_extract(
        self,
        domain_sem: asyncio.Semaphore,
        global_sem: asyncio.Semaphore,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        url = item["url"]

        async with global_sem, domain_sem:
            await self.rate_limiter.acquire()
            try:
                async with self.session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        self.log.debug(f"  [{url}] Skipping: HTTP {r.status}")
                        return None
                    html = await r.text()
            except aiohttp.ClientError as e:
                self.log.warning(f"  [{url}] Fetch error: {type(e).__name__}: {e}")
                return None

        loop = asyncio.get_running_loop()
        text, extracted_date = await loop.run_in_executor(self.pool, _extract_article, html)

        if not text:
            self.log.debug(f"  [{url}] Skipping: no text extracted")
            return None

        # Validate date matches target year/month
        date = extracted_date
        if self.config["date_source"] == "trafilatura":
            if not date:
                self.log.debug(f"  [{url}] Skipping: no date extracted")
                return None
            try:
                art_year = int(date[:4])
                art_month = int(date[5:7])
                if art_year != int(item["year"]) or art_month != int(item["month"]):
                    self.log.debug(
                        f"  [{url}] Skipping: date {date} does not match "
                        f"target {item['year']}-{int(item['month']):02d}"
                    )
                    return None
            except (ValueError, IndexError) as e:
                self.log.warning(f"  [{url}] Skipping: could not parse date {date!r}: {e}")
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

    async def _extract_month(
        self, year: int, month: int, queue_file: Path, output_file: Path
    ) -> int:
        """Extract articles for a single (year, month) and append to *output_file*.

        Returns the number of articles saved in this run.
        """
        with queue_file.open(newline="", encoding="utf-8") as f:
            queue = list(csv.DictReader(f))

        done: set[str] = set()
        if output_file.exists():
            with output_file.open(newline="", encoding="utf-8") as f:
                done = {row["url"] for row in csv.DictReader(f)}

        remaining = [item for item in queue if item["url"] not in done]
        self.log.info(
            f"  Extraction {year}-{month:02d}: {len(remaining)} to fetch "
            f"({len(done)} already done)"
        )

        if not remaining:
            return 0

        fieldnames = ["outlet", "year", "month", "date", "url", "wordcount", "body"]
        write_header = not output_file.exists() or len(done) == 0

        domain_sem = asyncio.Semaphore(self.args.domain_concurrency)
        global_sem = asyncio.Semaphore(self.args.global_concurrency)

        saved = 0
        errors = 0
        t0 = time.monotonic()

        with output_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

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

        elapsed = time.monotonic() - t0
        self.log.info(
            f"  ✓ {year}-{month:02d}: {saved} articles in {elapsed / 60:.1f} min"
        )
        return saved

    async def _fetch_sitemap(
        self, sem: asyncio.Semaphore, sitemap_url: str, year: int, month: int
    ) -> list[dict[str, Any]]:
        async with sem:
            await self.rate_limiter.acquire()
            try:
                async with self.session.get(
                    sitemap_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status != 200:
                        return []
                    body = await r.read()
            except aiohttp.ClientError as e:
                self.log.warning(f"  [{self.args.outlet}] Sitemap error {sitemap_url}: {type(e).__name__}: {e}")
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
        except ET.ParseError as e:
            self.log.warning(f"  [{self.args.outlet}] XML parse error {sitemap_url}: {type(e).__name__}: {e}")
            return []

    def _load_queue(self) -> list[dict[str, str]]:
        with self.url_queue_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _month_output_path(self, year: int, month: int) -> Path:
        return Path(f"{self._slug}_{year}_{month:02d}_corpus.csv")

    def _month_queue_path(self, year: int, month: int) -> Path:
        return Path(f"{self._slug}_{year}_{month:02d}_url_queue.csv")


def main() -> int | None:
    return CorpusScraper(*sys.argv[1:])()


if __name__ == "__main__":
    raise SystemExit(main())
