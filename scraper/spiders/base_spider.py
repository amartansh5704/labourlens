# scraper/spiders/base_spider.py
# Base class that all spiders inherit from
# Contains shared logic for all spiders

import scrapy
from database.connection import get_db_session
from database.models import Document
from scraper.processors.html_parser import (
    extract_text_from_html,
    extract_title_from_html
)
from scraper.processors.pdf_parser import extract_text_from_pdf
from scraper.processors.cleaner import (
    clean_document_text,
    detect_topic_from_url,
    detect_topic_from_text
)
from shared.constants import SCRAPER_SETTINGS
from loguru import logger
import hashlib


class BaseLaborSpider(scrapy.Spider):
    """
    Base spider with shared logic for all LaborLens spiders.
    All jurisdiction spiders inherit from this class.

    Child classes must define:
    - name: spider name string
    - jurisdiction: jurisdiction string
    - start_urls: list of URLs to start crawling
    - allowed_domains: list of domains to stay within
    """

    # override these in child spiders
    name = "base_spider"
    jurisdiction = "Unknown"
    agency = "Unknown"

    custom_settings = {
        "DOWNLOAD_DELAY": SCRAPER_SETTINGS["DOWNLOAD_DELAY"],
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS": 1,
        "ROBOTSTXT_OBEY": True,
    }

    # ── URL Helpers ───────────────────────────────────────

    def make_doc_id(self, url: str) -> str:
        """Create consistent ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF"""
        return (
            url.lower().endswith(".pdf") or
            "pdf" in url.lower().split("?")[0]
        )

    def is_already_scraped(self, url: str) -> bool:
        """Check if URL is already in our database"""
        db = get_db_session()
        try:
            exists = db.query(Document).filter(
                Document.url == url
            ).first()
            return exists is not None
        finally:
            db.close()

    def should_follow_url(self, url: str) -> bool:
        """
        Decide whether to follow a link.
        Override in child class for custom logic.
        """
        # skip common non-content URLs
        skip_patterns = [
            "login", "logout", "signin",
            "register", "contact", "sitemap",
            "feed", "rss", "xml",
            "javascript:", "mailto:", "tel:",
            "#",
        ]
        url_lower = url.lower()
        return not any(p in url_lower for p in skip_patterns)

    # ── Response Handlers ─────────────────────────────────

    def parse(self, response):
        """
        Default parse method.
        Handles both HTML pages and PDFs.
        Follows links to find more pages.
        """

        # skip already scraped URLs
        if self.is_already_scraped(response.url):
            logger.debug(f"Already scraped, skipping: {response.url[:60]}")
            return

        # check content type
        content_type = response.headers.get(
            "Content-Type", b""
        ).decode("utf-8", errors="ignore").lower()

        # handle PDFs
        if "pdf" in content_type or self.is_pdf_url(response.url):
            yield from self._parse_pdf(response)

        # handle HTML
        elif "html" in content_type:
            yield from self._parse_html(response)

            # follow links to more pages
            yield from self._follow_links(response)

    def _parse_html(self, response):
        """Extract item from HTML page"""

        # extract text
        raw_text = extract_text_from_html(response.text)

        if not raw_text or len(raw_text) < 100:
            logger.debug(f"Not enough text in: {response.url[:60]}")
            return

        # clean text
        clean_text = clean_document_text(raw_text, response.url)

        # get title
        title = extract_title_from_html(response.text)

        # detect topic
        topic = self.detect_topic(response.url, clean_text)

        if topic == "general":
            logger.debug(
                f"No topic match, skipping: {response.url[:60]}"
            )
            return

        yield {
            "url": response.url,
            "title": title,
            "raw_text": clean_text,
            "jurisdiction": self.jurisdiction,
            "topic": topic,
            "document_type": self.detect_document_type(response.url),
            "law_name": self.detect_law_name(response.url, clean_text),
            "agency": self.agency,
            "effective_date": "",
            "file_type": "html",
            "http_status": response.status,
        }

    def _parse_pdf(self, response):
        """Extract item from PDF response"""

        raw_text = extract_text_from_pdf(response.body)

        if not raw_text or len(raw_text) < 100:
            logger.debug(
                f"Not enough text in PDF: {response.url[:60]}"
            )
            return

        clean_text = clean_document_text(raw_text, response.url)
        topic = self.detect_topic(response.url, clean_text)

        if topic == "general":
            logger.debug(
                f"No topic match in PDF, skipping: {response.url[:60]}"
            )
            return

        # use filename as title for PDFs
        filename = response.url.split("/")[-1].replace("-", " ").replace("_", " ")
        title = filename.replace(".pdf", "").replace(".PDF", "")

        yield {
            "url": response.url,
            "title": title,
            "raw_text": clean_text,
            "jurisdiction": self.jurisdiction,
            "topic": topic,
            "document_type": "notification",
            "law_name": self.detect_law_name(response.url, clean_text),
            "agency": self.agency,
            "effective_date": "",
            "file_type": "pdf",
            "http_status": response.status,
        }

    def _follow_links(self, response):
        """Find and follow links on the page"""
        for href in response.css("a::attr(href)").getall():
            full_url = response.urljoin(href)

            # only follow if in allowed domains
            if not self.should_follow_url(full_url):
                continue

            # check domain restriction
            if not any(
                domain in full_url
                for domain in self.allowed_domains
            ):
                continue

            yield scrapy.Request(
                url=full_url,
                callback=self.parse,
                errback=self.handle_error,
            )

    def handle_error(self, failure):
        """Handle request errors gracefully"""
        logger.warning(
            f"Request failed: {failure.request.url} "
            f"- {failure.value}"
        )

    # ── Topic and Law Detection ───────────────────────────

    def detect_topic(self, url: str, text: str = "") -> str:
        """
        Detect topic from URL first, then from text content.
        Override in child class for custom logic.
        """
        # try URL first (faster)
        topic = detect_topic_from_url(url)

        # if no match, try text content
        if topic == "general" and text:
            topic = detect_topic_from_text(text, url)

        return topic

    def detect_document_type(self, url: str) -> str:
        """Detect document type from URL"""
        url_lower = url.lower()

        if "notification" in url_lower:
            return "notification"
        elif "amendment" in url_lower:
            return "amendment"
        elif "circular" in url_lower:
            return "circular"
        elif "order" in url_lower:
            return "order"
        elif "faq" in url_lower:
            return "FAQ"
        elif "act" in url_lower or "rules" in url_lower:
            return "statute"
        else:
            return "guidance"

    def detect_law_name(self, url: str, text: str = "") -> str:
        """
        Try to detect law name from URL or text.
        Returns empty string if cannot detect.
        """
        # common Indian labor law names to look for
        law_patterns = [
            "Minimum Wages Act",
            "Payment of Wages Act",
            "Factories Act",
            "Industrial Disputes Act",
            "Employees Provident Fund",
            "Employees State Insurance",
            "Contract Labour Act",
            "Maternity Benefit Act",
            "Payment of Gratuity Act",
            "Shops and Establishments Act",
            "Code on Wages",
            "Labour Codes",
        ]

        text_to_search = (url + " " + text[:500]).lower()

        for law in law_patterns:
            if law.lower() in text_to_search:
                return law

        return ""