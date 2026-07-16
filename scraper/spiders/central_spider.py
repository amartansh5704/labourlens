# scraper/spiders/central_spider.py
# Scrapes central government labor law websites
# Primary source: labour.gov.in

import scrapy
from scraper.spiders.base_spider import BaseLaborSpider
from loguru import logger


class CentralLaborSpider(BaseLaborSpider):
    """
    Scrapes Ministry of Labour and Employment website
    and other central government labor law sources
    """

    name = "central_labor"
    jurisdiction = "Central"
    agency = "Ministry of Labour and Employment, India"

    allowed_domains = [
        "labour.gov.in",
        "epfindia.gov.in",
        "esic.gov.in",
    ]

    # Pages to start scraping from
    start_urls = [
        # Ministry of Labour main sections
        "https://labour.gov.in/acts-rules",
        "https://labour.gov.in/labour-laws",
        "https://labour.gov.in/minimum-wages",
        "https://labour.gov.in/overtime",

        # EPF India
        "https://www.epfindia.gov.in/site_en/FAQ.php",
        "https://www.epfindia.gov.in/site_en/EPF_Schemes.php",

        # ESI Corporation
        "https://www.esic.gov.in/employees",
        "https://www.esic.gov.in/employers",
    ]

    def parse(self, response):
        """Override parse to add central-specific logic"""

        logger.info(
            f"Scraping: [{self.jurisdiction}] {response.url[:70]}"
        )

        # use parent class parse logic
        yield from super().parse(response)

    def should_follow_url(self, url: str) -> bool:
        """
        Custom URL filtering for central government sites.
        Only follow URLs likely to have labor law content.
        """

        # first apply base class filtering
        if not super().should_follow_url(url):
            return False

        url_lower = url.lower()

        # keywords that suggest labor law content
        relevant_keywords = [
            "wage", "salary", "overtime", "leave",
            "provident", "epf", "esic", "esi",
            "labour", "labor", "worker", "employee",
            "act", "rule", "notification", "circular",
            "minimum", "payment", "gratuity", "contract",
            "maternity", "factory", "industrial",
            "faq", "scheme", "benefit", "contribution"
        ]

        # follow if URL contains relevant keyword
        if any(keyword in url_lower for keyword in relevant_keywords):
            return True

        # skip URLs with these patterns
        skip_keywords = [
            "gallery", "photo", "image", "video",
            "tender", "recruitment", "job", "career",
            "press-release", "news", "event", "archive",
            "hindi", "regional", "state-portal"
        ]

        if any(keyword in url_lower for keyword in skip_keywords):
            return False

        # by default follow links within allowed domains
        return True