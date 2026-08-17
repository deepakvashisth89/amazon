#!/usr/bin/env python
# coding: utf-8

# # Amazon ISBN-13 Web Scraping Pipeline (CREP Dataset Collection)
# 
# Converted from `amazon_scraper.py` into a notebook so each stage of the pipeline
# can be run and inspected interactively.
# 
# For each ISBN-13, this collects:
# 
# 1. **Book metadata** -> `data/book_metadata/amazon_metadata.json`
# 2. **Book cover image** -> `data/book_coverpage/{isbn}_cp_amazon_1.jpg`
# 3. **Book blurb** -> `data/book_blurb/{isbn}_b_amazon_1.txt`
# 4. **Book reviews** -> `data/book_reviews/{isbn}_r_amazon_{n}.txt` (>= 25 per source)
# 5. **Genres** -> parsed from the breadcrumb / used inline in metadata
# 
# > Amazon pages are more likely to block automated requests than most sites.
# > This script uses a persistent `requests` session first and falls back to
# > Selenium (headless Chrome) when the plain HTML response looks blocked or
# > doesn't contain enough review text.
# 
# **Optimizations applied while converting** (see the final markdown cell for the full list):
# - Removed a duplicate `_text()` definition that silently overrode the safer version.
# - Removed ~150 lines of dead/commented-out code.
# - `extract_ld_json()` was being called twice per ISBN — now called once and reused.
# - Metadata storage switched from *read-modify-write the whole JSON file per ISBN* (O(n²) over a large ISBN list) to an in-memory dict keyed by ISBN, flushed to disk once per ISBN (O(n) reads instead of O(n²)).
# - `requests.Session` now uses an `HTTPAdapter` with retry/backoff instead of failing immediately on transient 429/5xx errors.
# - BeautifulSoup now uses `lxml` when available (noticeably faster than the stdlib `html.parser`), falling back gracefully if it isn't installed.
# - Review-noise regex patterns are now precompiled once instead of being recompiled on every call to `clean_review_text`.
# - **New: a dedicated log file** (`data/logs/pipeline.log`) records every blocked/captcha page, request failure, Selenium error, and unexpected exception with a timestamp — a persistent audit trail separate from the interactive cell output.
# 

# ## 1. Install requirements
# 
# Run once per environment.

# Run this once in your terminal before executing the script (not valid
# inside a plain .py file, so it's commented out here — this was an
# IPython/Jupyter magic command):
#   pip install -q requests beautifulsoup4 selenium webdriver-manager isbnlib lxml


# ## 2. Imports

import os
import re
import sys
import json
import time
import random
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

try:
    import isbnlib  # optional; used only for origin_country inference
except ImportError:
    isbnlib = None

# Prefer lxml (faster C parser); fall back to the stdlib parser if unavailable.
try:
    import lxml  # noqa: F401
    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"


# ## 3. Config

BASE_DIR = "data"
FOLDERS = {
    "metadata": os.path.join(BASE_DIR, "book_metadata"),
    "coverpage": os.path.join(BASE_DIR, "book_coverpage"),
    "blurb": os.path.join(BASE_DIR, "book_blurb"),
    "reviews": os.path.join(BASE_DIR, "book_reviews"),
}
AMAZON_METADATA_FILE = os.path.join(FOLDERS["metadata"], "amazon_metadata.json")

# Dedicated error/mis-occurrence log file (new)
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

MIN_REVIEWS = 25
USE_SELENIUM_FOR_REVIEWS = True
REQUEST_DELAY_RANGE = (1.5, 3.5)

# --- Chrome driver health/balance settings (overnight-run reliability) ---
# A single long-lived Chrome process slowly builds up memory/tabs/zombie
# processes over an 8+ hour unattended run and eventually crashes or hangs.
# To "balance" it: (1) recycle the driver every N Selenium uses even if it
# looks healthy, and (2) if a call raises a WebDriver-level error (crashed
# session, disconnected, etc.), throw the broken driver away immediately so
# the *next* call gets a fresh one instead of retrying against a dead browser.
DRIVER_RESTART_EVERY = 40  # recycle Chrome after this many Selenium page loads
_SELENIUM_CALL_COUNT = 0


# ## 4. Logging Setup (new)
# 
# Every blocked/captcha page, failed request, Selenium error, and unexpected
# exception is written to `data/logs/pipeline.log` with a timestamp — a
# persistent record you can review after a long run, independent of the
# interactive cell output. The console only surfaces genuine crashes through
# the logger (level `ERROR`); the existing `print()` statements throughout the
# pipeline keep providing normal progress feedback as before, so this doesn't
# add console noise.
# 
# Re-running this cell clears old handlers first, so re-executing the notebook
# doesn't duplicate log lines or open the file multiple times.

def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("amazon_scraper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if this cell is re-run

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)  # only genuine crashes surface on the console via the logger
    console_handler.setFormatter(logging.Formatter("[log] %(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


logger = setup_logger()
logger.info(f"Pipeline logger initialized. Writing to: {LOG_FILE}")
print(f"[info] Logging ready — blocked pages, failures, and errors will be recorded in: {LOG_FILE}")


# ## 5. Session & Selenium driver management
# 
# **Optimization:** the session now mounts an `HTTPAdapter` configured with a
# `Retry` policy (3 retries, exponential backoff, retrying on 429/500/502/503/504).
# The original script had no retry logic at all — a single transient error on a
# long ISBN list meant an immediate fallback to the much slower Selenium path.

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = _make_session()

_DRIVER = None


def get_driver():
    """Return a live Chrome driver, recycling it when it's either stale
    (used more than DRIVER_RESTART_EVERY times) or actually crashed."""
    global _DRIVER, _SELENIUM_CALL_COUNT

    if _DRIVER is not None and _SELENIUM_CALL_COUNT >= DRIVER_RESTART_EVERY:
        print(f"[driver] recycling Chrome after {_SELENIUM_CALL_COUNT} uses (routine restart)")
        logger.info(f"Recycling Chrome driver after {_SELENIUM_CALL_COUNT} uses (routine restart)")
        close_driver()

    if _DRIVER is not None:
        # Cheap liveness check — catches a crashed/disconnected session
        # before we hand it back to the caller.
        try:
            _ = _DRIVER.window_handles
        except Exception:
            print("[driver] existing Chrome session looks dead — restarting it")
            logger.warning("Existing Chrome session looks dead — restarting it")
            close_driver()

    if _DRIVER is None:
        _DRIVER = _make_chrome_driver()
        _SELENIUM_CALL_COUNT = 0

    _SELENIUM_CALL_COUNT += 1
    return _DRIVER


def close_driver():
    global _DRIVER, _SELENIUM_CALL_COUNT
    if _DRIVER is not None:
        try:
            _DRIVER.quit()
        except Exception:
            pass
        _DRIVER = None
    _SELENIUM_CALL_COUNT = 0


def _make_chrome_driver(retries: int = 2):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    def _build():
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--log-level=3")
        # Caps per-process memory growth over a long run and avoids
        # crash-recovery popups/infobars that can block automation.
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-crash-reporter")
        options.add_argument("--disable-infobars")
        options.add_argument(f"user-agent={HEADERS['User-Agent']}")

        try:
            return webdriver.Chrome(options=options)
        except Exception:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            return _build()
        except Exception as e:
            last_err = e
            print(f"[driver] failed to start Chrome (attempt {attempt}): {e}")
            logger.error(f"Failed to start Chrome (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
    raise last_err


# ## 6. Small utilities
# 
# **Bug fix:** the original file defined `_text()` twice. The second definition
# (plain `tag.get_text(strip=True)`) silently overrode the first, more careful
# version that stripped invisible Unicode marks (`\u200e`, `\u200b`, `\ufeff`,
# non-breaking spaces) and normalized empty strings to `None`. Amazon pages are
# full of these invisible marks around prices/titles, so this was a real
# correctness regression, not just style. Kept only the safer version here.
# 
# **Optimization:** `REVIEW_NOISE_PATTERNS` are now precompiled once at import
# time instead of being re-compiled by `re.sub` on every single review.

def setup_folders():
    for path in FOLDERS.values():
        os.makedirs(path, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def polite_sleep():
    time.sleep(random.uniform(*REQUEST_DELAY_RANGE))


INVISIBLE_CHARS_RE = re.compile("[\u200e\u200f\u200b\ufeff]")


def _strip_invisible_chars(text: str) -> str:
    text = INVISIBLE_CHARS_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    return text


def _text(tag):
    if not tag:
        return None
    text = tag.get_text(strip=True)
    if not text:
        return None
    text = _strip_invisible_chars(text)
    return text.strip() or None


def _looks_blocked(html: str) -> bool:
    if not html:
        return True
    lowered = html.lower()
    block_markers = [
        "sorry! we couldn't find that page",
        "enter the characters you see below",
        "to discuss automated access",
        "are you a robot",
        "captcha",
        "unusual traffic",
    ]
    return any(marker in lowered for marker in block_markers) or len(html) < 600


REVIEW_NOISE_PATTERNS = [
    re.compile(r"Brief content visible, double tap to read full content\.", re.IGNORECASE),
    re.compile(r"Full content visible, double tap to read brief content\.", re.IGNORECASE),
]


def clean_review_text(text: str) -> str:
    if not text:
        return text
    for pattern in REVIEW_NOISE_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ## 7. Metadata storage
# 
# **Optimization:** the original `append_metadata_record()` re-read and
# re-parsed the *entire* JSON file, filtered it in Python, and rewrote the
# whole file — on every single ISBN. For a run of *n* ISBNs that's an O(n²)
# amount of I/O and JSON (de)serialization just for bookkeeping. `MetadataStore`
# loads the file once, keeps an in-memory dict keyed by `isbn13` (O(1) upsert
# instead of a linear scan-and-filter), and only serializes the full list back
# to disk on `save()`. It's still called after every ISBN by default for crash
# safety, but the expensive *read* + *scan* only happens once per run instead
# of once per ISBN.

def load_existing_metadata_records(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load existing metadata file {path}: {e}")
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


class MetadataStore:
    """In-memory ISBN -> record map, backed by a single JSON file.

    Replaces the old read-whole-file / filter / rewrite-whole-file pattern
    that ran on every ISBN with an O(1) in-memory upsert plus a single write.
    """

    def __init__(self, path: str):
        self.path = path
        self.records: dict[str, dict] = {
            r.get("isbn13"): r for r in load_existing_metadata_records(path)
        }

    def upsert(self, metadata: dict) -> None:
        self.records[metadata.get("isbn13")] = metadata

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(list(self.records.values()), f, ensure_ascii=False, indent=2)


METADATA_STORE = MetadataStore(AMAZON_METADATA_FILE)


# ## 8. Finding the Amazon product page for an ISBN

def _is_non_print_url(url: str) -> bool:
    lowered = url.lower()
    return any(marker in lowered for marker in ("audiobook", "/audible/", "audible.com"))


def _amazon_search_with_selenium(isbn13: str) -> str | None:
    search_url = f"https://www.amazon.com/s?k={isbn13}&i=stripbooks"
    html = fetch_html_via_selenium(search_url, wait_css="div[data-component-type='s-search-result']")
    if not html:
        logger.warning(f"Selenium search fallback returned no HTML for isbn={isbn13}")
        return None
    soup = BeautifulSoup(html, PARSER)
    selectors = [
        "div[data-asin][data-component-type='s-search-result'] a.a-link-normal[href]",
        "div[data-asin][data-component-type='s-search-result'] h2 a[href]",
        "a.a-link-normal.s-no-outline[href]",
        "span.a-size-medium.a-color-base.a-text-normal > a[href]",
    ]
    for sel in selectors:
        for result in soup.select(sel):
            href = result.get("href")
            if not href:
                continue
            href = href.split("?")[0]
            if href.startswith("/"):
                href = "https://www.amazon.com" + href
            if href.startswith("http") and not _is_non_print_url(href):
                return href
    logger.warning(f"Selenium search fallback found no product link for isbn={isbn13}")
    return None


def get_book_url_from_isbn(isbn13: str) -> str | None:
    search_url = "https://www.amazon.com/s"
    params = {"k": isbn13, "i": "stripbooks"}

    try:
        r = SESSION.get(search_url, params=params, timeout=15)
        r.raise_for_status()
        if _looks_blocked(r.text):
            print(f"[warn] Amazon search page for {isbn13} looks blocked or invalid")
            logger.warning(f"Amazon search page for isbn={isbn13} looks blocked/captcha'd (len={len(r.text)})")
            raise requests.RequestException("blocked or invalid response")

        soup = BeautifulSoup(r.text, PARSER)
        results = soup.select("div[data-asin][data-component-type='s-search-result'] a.a-link-normal[href]")
        for result in results:
            href = result.get("href", "").split("?")[0]
            if not href:
                continue
            if href.startswith("/"):
                href = "https://www.amazon.com" + href
            if href.startswith("http") and not _is_non_print_url(href):
                return href

    except requests.RequestException as e:
        print(f"[warn] Amazon search request failed for {isbn13}: {e}")
        print(f"[info] trying Selenium search fallback for {isbn13}")
        logger.warning(f"Amazon search request failed for isbn={isbn13}: {e} — trying Selenium fallback")
        return _amazon_search_with_selenium(isbn13)

    print(f"[warn] Amazon search found no result for {isbn13} via requests; trying Selenium fallback")
    logger.warning(f"Amazon search found no result for isbn={isbn13} via requests — trying Selenium fallback")
    return _amazon_search_with_selenium(isbn13)


# ## 9. Fetching pages (requests + Selenium fallback) and LD+JSON extraction

def fetch_soup(url: str, params: dict | None = None) -> BeautifulSoup:
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    if _looks_blocked(r.text):
        print(f"[warn] Amazon page looks blocked/captcha'd (status={r.status_code}, len={len(r.text)})")
        logger.warning(f"Page looks blocked/captcha'd: url={url} status={r.status_code} len={len(r.text)}")
    return BeautifulSoup(r.text, PARSER)


def fetch_html_via_selenium(url: str, wait_css: str = "#productTitle", timeout: int = 15) -> str | None:
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, WebDriverException
    except ImportError:
        print("[warn] selenium not installed; cannot use browser fallback. run: pip install selenium")
        logger.error("Selenium not installed — cannot use browser fallback.")
        return None

    for attempt in (1, 2):  # 1 retry with a fresh driver if Chrome itself crashed
        try:
            driver = get_driver()
        except Exception as e:
            print(f"[warn] could not start Chrome for browser fallback: {e}")
            logger.error(f"Could not start Chrome for browser fallback (url={url}): {e}")
            return None

        try:
            driver.get(url)
            try:
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_css))
                )
            except TimeoutException:
                print(f"[warn] page did not fully load within {timeout}s via browser fallback.")
                logger.warning(f"Selenium page did not fully load within {timeout}s: {url}")
            return driver.page_source
        except WebDriverException as e:
            print(f"[warn] Chrome session error on attempt {attempt}, discarding driver: {e}")
            logger.warning(f"Chrome session error (attempt {attempt}) for {url}: {e}")
            close_driver()  # force a fresh driver on the next attempt/call
            if attempt == 2:
                return None
        except Exception as e:
            print(f"[warn] browser fallback failed to load page: {e}")
            logger.error(f"Selenium browser fallback failed to load {url}: {e}")
            return None


def extract_ld_json(soup: BeautifulSoup) -> dict:
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("@type") in ("Book", "Product"):
            return data
    return {}


# ## 10. Parsing publisher / language / country

def _parse_amazon_publication_info(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    publisher = None
    pub_date = None

    # Old-style detail bullets
    for item in soup.select("#detailBullets_feature_div li"):
        text = _text(item)
        if not text:
            continue
        if text.lower().startswith("publisher"):
            publisher = text.split(":", 1)[-1].strip()
        if "publication date" in text.lower() or "date first available" in text.lower():
            pub_date = text.split(":", 1)[-1].strip()

    # Newer "Product details" overview grid
    if not publisher or not pub_date:
        for row in soup.select("#productOverview_feature_div tr"):
            cells_ = row.select("td")
            if len(cells_) < 2:
                continue
            label = _text(cells_[0])
            value = _text(cells_[1])
            if not label or not value:
                continue
            lower = label.lower()
            if "publisher" in lower and not publisher:
                publisher = value
            elif ("publication date" in lower or "date first available" in lower) and not pub_date:
                pub_date = value

    # "product-facts" grid variant
    if not publisher or not pub_date:
        for block in soup.select("div.product-facts-detail"):
            label = _text(block.select_one(".a-col-left"))
            value = _text(block.select_one(".a-col-right"))
            if not label or not value:
                continue
            lower = label.lower()
            if "publisher" in lower and not publisher:
                publisher = value
            elif ("publication date" in lower or "date first available" in lower) and not pub_date:
                pub_date = value

    # Old fallback: tech spec table
    if not publisher:
        table = soup.select_one("#productDetails_techSpec_section_1, #productDetails_detailBullets_sections1")
        if table:
            for row in table.select("tr"):
                label = _text(row.select_one("th"))
                value = _text(row.select_one("td"))
                if label and value:
                    lower = label.lower()
                    if "publisher" in lower and not publisher:
                        publisher = value
                    elif ("publication date" in lower or "date first available" in lower) and not pub_date:
                        pub_date = value

    if publisher and "(" in publisher and publisher.endswith(")"):
        publisher = publisher[:publisher.rfind("(")].strip()

    return publisher, pub_date


def _parse_amazon_language(soup: BeautifulSoup) -> str | None:
    for item in soup.select("#detailBullets_feature_div li"):
        text = _text(item)
        if text and text.lower().startswith("language"):
            return text.split(":", 1)[-1].strip()
    for row in soup.select("#productOverview_feature_div tr"):
        cells_ = row.select("td")
        if len(cells_) >= 2 and _text(cells_[0]) and "language" in _text(cells_[0]).lower():
            return _text(cells_[1])
    for block in soup.select("div.product-facts-detail"):
        label = _text(block.select_one(".a-col-left"))
        if label and "language" in label.lower():
            return _text(block.select_one(".a-col-right"))
    return None


def infer_country_from_isbn(isbn13: str) -> str:
    isbn13 = isbn13.strip().replace("-", "").replace(" ", "")
    if isbnlib is not None:
        try:
            info = isbnlib.info(isbn13)
            if info:
                return info
        except Exception as e:
            logger.warning(f"isbnlib.info() failed for isbn={isbn13}: {e}")

    # Basic fallback by registration group.
    digits = isbn13[3:] if isbn13[:3] in ("978", "979") else isbn13
    prefixes = {
        "0": "English-language region (US/UK/etc.)",
        "1": "English-language region (US/UK/etc.)",
        "2": "French-language region",
        "3": "German-language region",
        "4": "Japan",
        "5": "Russian-language region",
        "7": "China",
        "80": "Czech Republic / Slovakia",
        "81": "India",
        "82": "Norway",
        "83": "Poland",
        "84": "Spain",
        "85": "Brazil",
        "86": "Former Yugoslav region",
        "87": "Denmark",
        "88": "Italy",
        "89": "South Korea",
        "90": "Netherlands / Belgium (Dutch)",
        "91": "Sweden",
        "93": "India",
        "94": "Netherlands",
    }
    for prefix_len in (2, 1):
        prefix = digits[:prefix_len]
        if prefix in prefixes:
            return prefixes[prefix]
    return ""


# ## 11. Metadata scraping
# 
# **Optimization:** now upserts into `METADATA_STORE` (O(1)) and saves once,
# instead of the old read-full-file / filter / rewrite-full-file per ISBN.

def scrape_metadata(soup: BeautifulSoup, ld: dict, isbn13: str, book_url: str, genres: list[str] | None = None) -> dict:
    title = ld.get("name") or _text(soup.select_one("#productTitle"))

    authors = []
    if isinstance(ld.get("author"), list):
        authors = [a.get("name") for a in ld["author"] if isinstance(a, dict) and a.get("name")]
    elif isinstance(ld.get("author"), dict):
        authors = [ld["author"].get("name")]
    if not authors:
        for author_tag in soup.select(".author a, .contributorNameID"):
            name = _text(author_tag)
            if name and name not in authors:
                authors.append(name)

    language = ld.get("inLanguage") or _parse_amazon_language(soup)
    publisher, pub_date = _parse_amazon_publication_info(soup)
    if not publisher and isinstance(ld.get("publisher"), str):
        publisher = ld.get("publisher").strip()

    origin_country = infer_country_from_isbn(isbn13)

    if not title:
        logger.warning(f"No title resolved for isbn={isbn13} (url={book_url})")

    metadata = {
        "isbn13": isbn13,
        "title": title,
        "authors": authors,
        "publisher": publisher,
        "origin_country": origin_country,
        "date_of_publication": pub_date,
        "language": language,
        "source_url": book_url,
        "sources": ["Amazon"],
        "genres": genres or [],
    }

    METADATA_STORE.upsert(metadata)
    METADATA_STORE.save()
    print(f"[metadata] upserted -> {AMAZON_METADATA_FILE}")
    return metadata


# ## 12. Cover image & blurb scraping

def scrape_coverpage(soup: BeautifulSoup, ld: dict, isbn13: str):
    image_url = ld.get("image") if isinstance(ld.get("image"), str) else None
    if not image_url:
        img_tag = soup.select_one("img#imgBlkFront, img#landingImage, img#main-image")
        image_url = img_tag["src"] if img_tag and img_tag.get("src") else None

    if not image_url:
        print(f"[coverpage] no image found for {isbn13}")
        logger.warning(f"No cover image found for isbn={isbn13}")
        return

    resp = SESSION.get(image_url, timeout=15)
    resp.raise_for_status()
    out_path = os.path.join(FOLDERS["coverpage"], f"{isbn13}_cp_amazon_1.jpg")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[coverpage] saved -> {out_path}")


def scrape_blurb(soup: BeautifulSoup, ld: dict, isbn13: str):
    blurb = None
    if isinstance(ld.get("description"), str):
        blurb = ld.get("description")
    if not blurb:
        desc_tag = soup.select_one("#bookDescription_feature_div, #productDescription, #feature-bullets")
        blurb = _text(desc_tag)

    if not blurb:
        print(f"[blurb] no description found for {isbn13}")
        logger.warning(f"No description/blurb found for isbn={isbn13}")
        return

    blurb = BeautifulSoup(blurb, PARSER).get_text("\n").strip()
    out_path = os.path.join(FOLDERS["blurb"], f"{isbn13}_b_amazon_1.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(blurb)
    print(f"[blurb] saved -> {out_path}")


# ## 13. Review scraping (requests first, Selenium fallback)

def _extract_review_text(review: BeautifulSoup) -> str | None:
    candidates = [
        review.select_one("div[data-hook='reviewText']"),
        review.select_one("div[data-hook='reviewTextContainer']"),
        review.select_one("div.reviewText"),
        review.select_one("div[data-hook='reviewRichContentContainer']"),
        review.select_one("div.a-cardui-content"),
    ]
    for node in candidates:
        if node:
            text = _text(node)
            if text:
                return clean_review_text(text)

    # Fallback: return the full review block text if no specific review node is found.
    text = _text(review)
    return clean_review_text(text) if text else None


def scrape_reviews_requests(soup: BeautifulSoup) -> list[str]:
    reviews = []
    for review in soup.select("div[data-hook='review']"):
        text = _extract_review_text(review)
        if text:
            reviews.append(text)
    return reviews


def _extract_asin(url: str, soup: BeautifulSoup) -> str | None:
    match = re.search(r"/(dp|gp/product)/([A-Z0-9]{10})", url)
    if match:
        return match.group(2)
    asin_tag = soup.find(attrs={"data-asin": True})
    if asin_tag:
        return asin_tag["data-asin"]
    return None


REVIEW_CARD_SELECTORS = [
    "div[data-hook='reviewText']",
    "div[data-hook='reviewTextContainer']",
    "div.reviewText",
    "div[data-hook='reviewRichContentContainer']",
    "div.a-cardui-content",
]


def scrape_reviews_selenium(book_url: str, min_reviews: int = MIN_REVIEWS) -> list[str]:
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.common.exceptions import TimeoutException, WebDriverException
    except ImportError:
        print("[reviews] selenium not installed; run: pip install selenium")
        logger.error("Selenium not installed — cannot scrape reviews via browser fallback.")
        return []

    try:
        driver = get_driver()
    except Exception as e:
        print(f"[reviews] could not start Chrome for Selenium fallback: {e}")
        logger.error(f"Could not start Chrome for review Selenium fallback (url={book_url}): {e}")
        return []

    reviews = []
    try:
        driver.get(book_url)
        time.sleep(2)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, PARSER)
        asin = _extract_asin(book_url, soup)
        if not asin:
            see_all = driver.find_elements("css selector", "a[href*='/product-reviews/']")
            href = see_all[0].get_attribute("href") if see_all else None
            if href:
                match = re.search(r"/product-reviews/([A-Z0-9]{10})", href)
                asin = match.group(1) if match else None
        if not asin:
            print("[reviews] cannot determine ASIN for review page")
            logger.warning(f"Cannot determine ASIN for review page: {book_url}")
            return []
        print(f"[reviews] using ASIN: {asin}")

        seen_texts = set()
        page_num = 1
        max_pages = 10

        while len(reviews) < min_reviews and page_num <= max_pages:
            review_url = (
                f"https://www.amazon.com/product-reviews/{asin}/"
                f"?sortBy=recent&pageNumber={page_num}"
            )
            driver.get(review_url)
            time.sleep(1.5)  # let JS settle before checking

            current_html = driver.page_source
            if _looks_blocked(current_html):
                print(f"[reviews] page {page_num} looks blocked/captcha'd "
                      f"(len={len(current_html)}). Saving snapshot to debug_review_page.html")
                logger.warning(f"Review page {page_num} looks blocked/captcha'd for ASIN={asin} (len={len(current_html)})")
                with open("debug_review_page.html", "w", encoding="utf-8") as f:
                    f.write(current_html)
                break

            try:
                WebDriverWait(driver, 15).until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "div[data-hook='review']")
                )
            except TimeoutException:
                print(f"[reviews] no reviews loaded on page {page_num}, stopping. "
                      f"Page title was: {driver.title!r}")
                logger.warning(f"No reviews loaded on page {page_num} for ASIN={asin}; page title={driver.title!r}")
                with open("debug_review_page.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                break

            cards = driver.find_elements(By.CSS_SELECTOR, "div[data-hook='review']")
            if not cards:
                break

            new_this_page = 0
            for card in cards:
                body = None
                for selector in REVIEW_CARD_SELECTORS:
                    try:
                        body = card.find_element(By.CSS_SELECTOR, selector)
                        if body:
                            break
                    except Exception:
                        body = None
                text = (body.text.strip() if body else card.text.strip())
                text = clean_review_text(text)
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    reviews.append(text)
                    new_this_page += 1

            print(f"[reviews] page {page_num}: +{new_this_page} (total {len(reviews)})")

            if new_this_page == 0:
                break

            page_num += 1
            time.sleep(random.uniform(1.5, 3.0))

    except WebDriverException as e:
        print(f"[reviews] Chrome session error, discarding driver: {type(e).__name__}: {e}")
        logger.warning(f"Chrome session error during review scraping for {book_url}: {e}")
        close_driver()  # next ISBN gets a fresh driver instead of the crashed one
    except Exception as e:
        print(f"[reviews] Selenium error: {type(e).__name__}: {e}")
        logger.exception(f"Selenium review scraping error for {book_url}")

    return reviews[:max(min_reviews, len(reviews))]


def scrape_reviews(soup: BeautifulSoup, book_url: str, isbn13: str, source: str = "amazon"):
    reviews = scrape_reviews_requests(soup)
    if len(reviews) < MIN_REVIEWS and USE_SELENIUM_FOR_REVIEWS:
        print(f"[reviews] only {len(reviews)} found via requests; trying Selenium fallback...")
        selenium_reviews = scrape_reviews_selenium(book_url, MIN_REVIEWS)
        if len(selenium_reviews) > len(reviews):
            reviews = selenium_reviews

    if not reviews:
        print(f"[reviews] no reviews found for {isbn13}")
        logger.warning(f"No reviews found at all for isbn={isbn13}")
        return

    if len(reviews) < MIN_REVIEWS:
        print(f"[reviews] WARNING: only found {len(reviews)}/{MIN_REVIEWS} reviews for {isbn13}")
        logger.warning(f"Only found {len(reviews)}/{MIN_REVIEWS} reviews for isbn={isbn13}")

    for i, review_text in enumerate(reviews[:MIN_REVIEWS], start=1):
        out_path = os.path.join(FOLDERS["reviews"], f"{isbn13}_r_{source}_{i}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(review_text)
    print(f"[reviews] saved {min(len(reviews), MIN_REVIEWS)} reviews -> {FOLDERS['reviews']}/")


# ## 14. Genres, ISBN verification, and the per-ISBN driver
# 
# **Optimization:** `run_for_isbn` previously called `extract_ld_json(soup)`
# twice — once before the "looks blocked -> retry with Selenium" check and
# again right after, even on the normal (non-blocked) path where `soup` hadn't
# changed. It's now computed once, and only recomputed if the Selenium retry
# actually swapped in a new `soup`.

def scrape_genres(soup: BeautifulSoup, isbn13: str) -> list[str]:
    genres = []
    for tag in soup.select("#wayfinding-breadcrumbs_feature_div li a, #wayfinding-breadcrumbs_feature_div span.a-list-item"):
        text = _text(tag)
        if text and text not in genres:
            genres.append(text)
    return genres


def _page_isbn_matches(soup: BeautifulSoup, isbn13: str) -> bool:
    for item in soup.select("#detailBullets_feature_div li, #productOverview_feature_div tr"):
        text = _text(item)
        if text and isbn13 in text.replace("-", ""):
            return True
    return False


def run_for_isbn(isbn13: str):
    isbn13 = isbn13.strip().replace("-", "")
    print(f"\n=== Processing ISBN-13: {isbn13} ===")
    book_url = get_book_url_from_isbn(isbn13)
    if not book_url:
        print(f"[!] Could not find an Amazon page for ISBN {isbn13}. Skipping.")
        logger.error(f"Could not find an Amazon page for isbn={isbn13}")
        return

    polite_sleep()
    soup = fetch_soup(book_url)
    ld = extract_ld_json(soup)  # computed once; reused below unless soup is replaced

    if not _page_isbn_matches(soup, isbn13):
        print(f"[warn] landed page doesn't confirm ISBN {isbn13} — may be wrong edition (audiobook/kindle)")
        logger.warning(f"Landed page doesn't confirm isbn={isbn13} (url={book_url}) — may be wrong edition")

    if _looks_blocked(str(soup)):
        print(f"[warn] requests-based fetch for {isbn13} looks blocked — retrying with browser fallback...")
        logger.warning(f"requests-based fetch for isbn={isbn13} looks blocked — retrying with Selenium")
        html = fetch_html_via_selenium(book_url)
        if html:
            soup = BeautifulSoup(html, PARSER)
            ld = extract_ld_json(soup)  # only recompute because soup actually changed

    genres = scrape_genres(soup, isbn13)
    scrape_metadata(soup, ld, isbn13, book_url, genres=genres)
    scrape_coverpage(soup, ld, isbn13)
    scrape_blurb(soup, ld, isbn13)
    scrape_reviews(soup, book_url, isbn13, source="amazon")


# ## 15. Parsing an ISBN list file

def _parse_isbn_file(path: str) -> list[str]:
    isbns = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            candidate = line.split(",")[0].strip().strip('"').strip("'")
            cleaned = candidate.replace("-", "").replace(" ", "")
            if cleaned.isdigit() and len(cleaned) in (10, 13):
                isbns.append(cleaned)
            else:
                print(f"[skip] not a valid ISBN-10/13, skipping line: {line!r}")
                logger.warning(f"Skipping invalid ISBN-10/13 line from {path}: {line!r}")
    return isbns


# ## 16. Run configuration
# 
# This replaces the original `argparse` CLI (which doesn't apply in a notebook).
# Set the variables below, then run the execution cell.
# 
# - `ISBN`: a single ISBN-13 to scrape (leave as `None` to skip)
# - `ISBN_FILE`: path to a `.txt`/`.csv` file with one ISBN per line (leave as `None` to skip)
# - `LIMIT`: only process the first N ISBNs from `ISBN_FILE` (`None` = no limit)
# - `START_FROM`: skip the first N ISBNs in `ISBN_FILE` before starting

ISBN: str | None = None              # e.g. a single ISBN-13, or None
ISBN_FILE: str | None = "2602193001.csv"          # e.g. "isbn.csv", or None
LIMIT: int | None = None
START_FROM: int = 0


# ## 17. Run the pipeline
# 
# **Reliability note:** each ISBN is still processed inside its own
# `try/except` so one failure doesn't stop the batch — but now the exception is
# also logged with a full traceback via `logger.exception(...)`, and the run's
# start/finish are recorded to the log file too.

setup_folders()

isbn_list = []
if ISBN:
    isbn_list.append(ISBN)
if ISBN_FILE:
    try:
        isbn_list.extend(_parse_isbn_file(ISBN_FILE))
    except FileNotFoundError:
        print(f"[error] ISBN_FILE not found: {ISBN_FILE}")
        logger.error(f"ISBN_FILE not found: {ISBN_FILE}")
    except Exception as e:
        print(f"[error] Failed to read ISBN_FILE {ISBN_FILE}: {e}")
        logger.exception(f"Failed to read ISBN_FILE {ISBN_FILE}")

if not isbn_list:
    # Unattended/overnight runs must never block on input() — if there's
    # nothing to process, log it and stop instead of hanging forever.
    print("[error] No ISBNs to process (ISBN and ISBN_FILE are both empty/unreadable). Exiting.")
    logger.error("No ISBNs to process — ISBN and ISBN_FILE both empty/unreadable. Exiting.")
    sys.exit(1)

if START_FROM:
    isbn_list = isbn_list[START_FROM:]
if LIMIT is not None:
    isbn_list = isbn_list[:LIMIT]

print(f"[info] {len(isbn_list)} ISBN(s) queued for processing.")
logger.info(f"Run started — {len(isbn_list)} ISBN(s) queued.")

error_count = 0
consecutive_errors = 0
try:
    for idx, isbn in enumerate(isbn_list, start=1):
        print(f"\n[progress] {idx}/{len(isbn_list)}")
        try:
            run_for_isbn(isbn)
            consecutive_errors = 0
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Catches EVERYTHING for this one ISBN — network errors, parsing
            # bugs, a crashed Chrome session that couldn't recover, disk
            # write errors, etc. — so a single bad ISBN never takes down an
            # 8+ hour overnight run.
            print(f"[!] Error processing {isbn}: {e}")
            logger.exception(f"Unexpected error processing isbn={isbn} ({idx}/{len(isbn_list)})")
            error_count += 1
            consecutive_errors += 1
            # If Chrome itself is wedged, get_driver()'s liveness check will
            # normally catch and recycle it on the next call — but as a
            # backstop, force a full driver restart after a run of failures
            # so we don't spend the rest of the night failing every ISBN.
            if consecutive_errors >= 5:
                print("[driver] 5 consecutive failures — forcing a full Chrome restart as a precaution")
                logger.warning("5 consecutive ISBN failures — forcing a full Chrome driver restart")
                close_driver()
                consecutive_errors = 0
        polite_sleep()
except KeyboardInterrupt:
    print("\n[info] Interrupted by user — shutting down cleanly.")
    logger.info("Run interrupted by user (KeyboardInterrupt).")
finally:
    close_driver()

logger.info(f"Run finished. {len(isbn_list) - error_count}/{len(isbn_list)} ISBN(s) completed without an unhandled error.")


# ## 18. Reading the error/mis-occurrence log (new)
# 
# Everything at `WARNING` level or above in `data/logs/pipeline.log` is a
# blocked page, missing component, request/Selenium failure, or an unexpected
# exception worth reviewing — this lets you scan it from inside the notebook
# without opening the file separately.

def show_recent_log_lines(n: int = 40):
    """Print the last n lines of the pipeline log file, if it exists."""
    if not os.path.exists(LOG_FILE):
        print(f"[info] No log file yet at {LOG_FILE} — run the pipeline first.")
        return
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"[info] Showing last {min(n, len(lines))} of {len(lines):,} line(s) in {LOG_FILE}\n")
    for line in lines[-n:]:
        print(line.rstrip())

show_recent_log_lines()


# ## Optimization notes & further suggestions
# 
# **Applied in this conversion:**
# 
# 1. **Duplicate `_text()` bug** — the second definition silently shadowed the
#    first and dropped invisible-character stripping / empty-string handling.
#    Fixed by keeping a single definition.
# 2. **Redundant `extract_ld_json()` call** in `run_for_isbn` — was called
#    twice per ISBN on the common path; now called once and only recomputed
#    when the Selenium fallback actually replaces `soup`.
# 3. **O(n²) metadata I/O** — `append_metadata_record()` re-read, re-parsed,
#    filtered, and rewrote the entire JSON file for every ISBN. Replaced with
#    `MetadataStore`, an in-memory dict keyed by `isbn13` that's loaded once
#    and upserted in O(1); disk writes are still per-ISBN for crash safety but
#    the expensive full-file *read+scan* no longer repeats per ISBN.
# 4. **No HTTP retry policy** — a single transient 429/500/502/503/504 used to
#    fall straight through to the (much slower) Selenium path. The session now
#    mounts an `HTTPAdapter` with a 3-attempt exponential-backoff `Retry`.
# 5. **Parser choice** — `BeautifulSoup(..., "html.parser")` was hardcoded
#    everywhere. Switched to `lxml` when installed (it's meaningfully faster
#    on large Amazon pages), with an automatic fallback.
# 6. **Regex recompilation** — `REVIEW_NOISE_PATTERNS` are now compiled once at
#    import time instead of being recompiled inside `clean_review_text` on
#    every review.
# 7. **Dead code removed** — ~150 lines of commented-out earlier versions of
#    `_amazon_search_with_selenium`, `get_book_url_from_isbn`,
#    `_parse_amazon_publication_info`, and `scrape_reviews_selenium` were
#    deleted; the notebook keeps only the live implementations plus one markdown
#    note per section explaining what changed.
# 8. **New: dedicated error/mis-occurrence log file** (`data/logs/pipeline.log`)
#    — every blocked/captcha page, request failure, Selenium error, retry, and
#    unexpected exception is now recorded there with a timestamp, separate from
#    the interactive cell output. Unexpected exceptions in the main run loop are
#    logged with a full traceback via `logger.exception(...)`, and the run's
#    start/finish are recorded too. Use `show_recent_log_lines()` (section 18)
#    to review it without leaving the notebook.
# 
# **Not applied, but worth considering:**
# 
# - **Concurrency**: the pipeline is intentionally sequential with randomized
#   delays between requests. Adding `ThreadPoolExecutor` for the plain-`requests`
#   calls would speed things up, but for a site that already flags automated
#   traffic this trades speed for a much higher chance of getting blocked —
#   worth doing only with careful rate limiting and consideration of Amazon's
#   Terms of Service / robots.txt.
# - **Selective Selenium reads**: `scrape_reviews_selenium` calls
#   `driver.page_source` and re-parses with BeautifulSoup on the *first* page
#   load just to extract the ASIN — for ISBNs where `_extract_asin(book_url, ...)`
#   already succeeds from the URL alone, this parse is unnecessary and could be
#   skipped.
# - **Batch metadata saves**: `MetadataStore.save()` is currently called after
#   every ISBN. If you're processing a very large list and can tolerate losing
#   the last few records on a crash, calling `save()` every N ISBNs (or only in
#   the `finally` block) would cut disk writes further.
# - **Log rotation**: `pipeline.log` grows unbounded across a long run.
#   `logging.handlers.RotatingFileHandler` would cap it at a fixed size with
#   backups if that becomes a concern.
# 
