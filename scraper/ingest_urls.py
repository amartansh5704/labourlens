# scraper/ingest_urls.py
# Scrapes specific URLs and saves text to correct folders
# Then the folder ingestion pipeline picks them up

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from bs4 import BeautifulSoup
import ftfy
import re
import time
from loguru import logger

# ─────────────────────────────────────────────────────────
# URLS TO SCRAPE WITH FULL METADATA
# ─────────────────────────────────────────────────────────
URLS_TO_SCRAPE = [
    {
        "url": "https://www.indiacode.nic.in/handle/123456789/13698?view_type=browse",
        "name": "Delhi Minimum Wages Notification",
        "save_to": "documents/delhi/minimum_wage/delhi_minimum_wages_notification.txt",
        "jurisdiction": "Delhi",
        "topic": "minimum_wage",
        "law_name": "Delhi Minimum Wages Notification",
        "agency": "Delhi Labour Department",
        "source": "India Code",
    },
    {
        "url": "https://karmikaspandana.karnataka.gov.in/64/minimum-wages-rates-for-the-year-2026-27/en",
        "name": "Karnataka Minimum Wages 2026-27",
        "save_to": "documents/karnataka/minimum_wage/karnataka_minimum_wages_2026_27.txt",
        "jurisdiction": "Karnataka",
        "topic": "minimum_wage",
        "law_name": "Karnataka Minimum Wages Notification 2026-27",
        "agency": "Karnataka Labour Department",
        "source": "Karnataka Govt",
    },
    {
        "url": "https://www.sgcms.com/regulatory-updates/minimumwage-tamil-nadu-2026/",
        "name": "Tamil Nadu Minimum Wages 2026",
        "save_to": "documents/tamil_nadu/minimum_wage/tamilnadu_minimum_wages_2026.txt",
        "jurisdiction": "Tamil Nadu",
        "topic": "minimum_wage",
        "law_name": "Tamil Nadu Minimum Wages Notification 2026",
        "agency": "Tamil Nadu Labour Department",
        "source": "SGCMS Regulatory Updates",
    },
    {
        "url": "https://www.sgcms.com/regulatory-updates/minimum-rate-of-wages-telangana-2/",
        "name": "Telangana Minimum Wages",
        "save_to": "documents/telangana/minimum_wage/telangana_minimum_wages.txt",
        "jurisdiction": "Telangana",
        "topic": "minimum_wage",
        "law_name": "Telangana Minimum Rate of Wages Notification",
        "agency": "Telangana Labour Department",
        "source": "SGCMS Regulatory Updates",
    },
    {
        "url": "https://www.indiacode.nic.in/handle/123456789/19710",
        "name": "Maharashtra Shops and Establishments Act",
        "save_to": "documents/maharashtra/leave_policy/maharashtra_shops_establishments_act.txt",
        "jurisdiction": "Maharashtra",
        "topic": "leave_policy",
        "law_name": "Maharashtra Shops and Establishments Act",
        "agency": "Maharashtra Labour Department",
        "source": "India Code",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
}


def extract_clean_text(html: str, url: str) -> str:
    """Extract clean text from HTML"""
    soup = BeautifulSoup(html, "html.parser")

    # remove noise tags
    for tag in soup([
        "script", "style", "nav", "footer",
        "header", "aside", "iframe", "noscript",
        "button", "form", "input", "select",
        "meta", "link"
    ]):
        tag.decompose()

    # try to find main content area
    main = (
        soup.find("main") or
        soup.find("div", {"id": "content"}) or
        soup.find("div", {"id": "main-content"}) or
        soup.find("div", {"class": "content"}) or
        soup.find("div", {"class": "entry-content"}) or
        soup.find("article") or
        soup.find("div", {"class": "post-content"}) or
        soup.find("div", {"role": "main"}) or
        soup.body or
        soup
    )

    text = main.get_text(separator="\n")
    text = ftfy.fix_text(text)

    # clean up
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 3:
            continue
        if re.match(r"^[\W\s]+$", line):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def scrape_url(item: dict) -> bool:
    """Scrape one URL and save to file"""
    url = item["url"]
    name = item["name"]
    save_path = item["save_to"]

    logger.info(f"Scraping: {name}")
    logger.info(f"URL: {url}")

    try:
        response = httpx.get(
            url,
            headers=HEADERS,
            timeout=30.0,
            follow_redirects=True,
            verify=False,
        )

        if response.status_code != 200:
            logger.error(
                f"HTTP {response.status_code} for {url}"
            )
            return False

        text = extract_clean_text(response.text, url)

        if len(text) < 200:
            logger.warning(
                f"Very little text extracted: {len(text)} chars"
            )
            logger.warning(
                "Site may require JavaScript - try Playwright"
            )
            return False

        # add metadata header to the text file
        # this gets read during ingestion
        content = f"""SOURCE_URL: {url}
LAW_NAME: {item['law_name']}
JURISDICTION: {item['jurisdiction']}
TOPIC: {item['topic']}
AGENCY: {item['agency']}
---CONTENT_START---
{text}"""

        # save to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            f"✅ Saved: {save_path} ({len(text):,} chars)"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return False


def scrape_with_playwright(item: dict) -> bool:
    """
    Fallback: use Playwright for JavaScript heavy sites
    """
    from playwright.sync_api import sync_playwright

    url = item["url"]
    name = item["name"]
    save_path = item["save_to"]

    logger.info(f"Trying Playwright for: {name}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=HEADERS["User-Agent"]
            )

            page.goto(url, wait_until="networkidle", timeout=30000)

            # wait for content to load
            page.wait_for_timeout(3000)

            # get rendered HTML
            html = page.content()
            browser.close()

        text = extract_clean_text(html, url)

        if len(text) < 200:
            logger.warning(
                f"Still little text with Playwright: {len(text)}"
            )
            return False

        content = f"""SOURCE_URL: {url}
LAW_NAME: {item['law_name']}
JURISDICTION: {item['jurisdiction']}
TOPIC: {item['topic']}
AGENCY: {item['agency']}
---CONTENT_START---
{text}"""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            f"✅ Playwright saved: {save_path} ({len(text):,} chars)"
        )
        return True

    except Exception as e:
        logger.error(f"Playwright also failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LaborLens - URL Scraper")
    print("=" * 60)

    success_count = 0
    failed_items = []

    for i, item in enumerate(URLS_TO_SCRAPE, 1):
        print(f"\n[{i}/{len(URLS_TO_SCRAPE)}] {item['name']}")

        # try normal HTTP first
        ok = scrape_url(item)

        # if failed, try Playwright
        if not ok:
            print(f"    Normal scraping failed, trying Playwright...")
            ok = scrape_with_playwright(item)

        if ok:
            success_count += 1
        else:
            failed_items.append(item)

        # be polite
        time.sleep(2)

    print("\n" + "=" * 60)
    print("URL SCRAPING COMPLETE")
    print("=" * 60)
    print(f"✅ Succeeded: {success_count}/{len(URLS_TO_SCRAPE)}")

    if failed_items:
        print(f"\n❌ Failed ({len(failed_items)}):")
        for item in failed_items:
            print(f"   - {item['name']}")
            print(f"     Save manually to: {item['save_to']}")
    else:
        print("\n🎉 All URLs scraped successfully!")

    print("\nNext: run python scraper/ingest_documents.py")