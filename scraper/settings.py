# scraper/settings.py
# Scrapy configuration for LaborLens

BOT_NAME = "laborlens"

# ── Spider Settings ────────────────────────────────────
SPIDER_MODULES = ["scraper.spiders"]
NEWSPIDER_MODULE = "scraper.spiders"

# ── Crawling Behaviour ─────────────────────────────────
# Be polite to government websites
DOWNLOAD_DELAY = 2               # wait 2 seconds between requests
RANDOMIZE_DOWNLOAD_DELAY = True  # adds randomness (1-3 seconds)
CONCURRENT_REQUESTS = 1          # one request at a time
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# ── Respect robots.txt ─────────────────────────────────
ROBOTSTXT_OBEY = True

# ── Timeouts ───────────────────────────────────────────
DOWNLOAD_TIMEOUT = 30

# ── Retry Settings ─────────────────────────────────────
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# ── User Agent ─────────────────────────────────────────
USER_AGENT = "LaborLens Research Bot 1.0 (Educational Project)"

# ── Item Pipelines ─────────────────────────────────────
# Processes scraped items in order (lower number = runs first)
ITEM_PIPELINES = {
    "scraper.pipelines.ValidationPipeline": 100,
    "scraper.pipelines.DatabasePipeline": 200,
}

# ── HTTP Cache (speeds up development) ─────────────────
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # cache for 24 hours
HTTPCACHE_DIR = ".scrapy/httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [404, 500, 502, 503]

# ── Feed Export ────────────────────────────────────────
FEEDS = {
    "logs/scraped_items.jsonl": {
        "format": "jsonlines",
        "overwrite": False,
    }
}

# ── Logging ────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = "logs/scrapy.log"

# ── PDF Download Settings ──────────────────────────────
# Allow downloading PDFs (binary files)
MEDIA_ALLOW_REDIRECTS = True

# ── Headers ────────────────────────────────────────────
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
}