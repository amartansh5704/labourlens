# scraper/ingest_urls_deep.py
# Deep scraper that extracts EVERYTHING from a page
# Tables, accordions, nested divs, paginated content

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
from bs4 import BeautifulSoup, Tag
import ftfy
import re
import time
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from loguru import logger

# ─────────────────────────────────────────────────────────
# TARGET URLS WITH FULL CONFIG
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
        "method": "playwright",  # needs JS
        "wait_for": "body",
        "extra_pages": [],
    },
    {
        "url": "https://karmikaspandana.karnataka.gov.in/64/minimum-wages-rates-for-the-year-2026-27/en",
        "name": "Karnataka Minimum Wages 2026-27",
        "save_to": "documents/karnataka/minimum_wage/karnataka_minimum_wages_2026_27.txt",
        "jurisdiction": "Karnataka",
        "topic": "minimum_wage",
        "law_name": "Karnataka Minimum Wages Notification 2026-27",
        "agency": "Karnataka Labour Department",
        "method": "playwright",
        "wait_for": "body",
        "extra_pages": [],
    },
    {
        "url": "https://www.sgcms.com/regulatory-updates/minimumwage-tamil-nadu-2026/",
        "name": "Tamil Nadu Minimum Wages 2026",
        "save_to": "documents/tamil_nadu/minimum_wage/tamilnadu_minimum_wages_2026.txt",
        "jurisdiction": "Tamil Nadu",
        "topic": "minimum_wage",
        "law_name": "Tamil Nadu Minimum Wages Notification 2026",
        "agency": "Tamil Nadu Labour Department",
        "method": "playwright",
        "wait_for": "article",
        "extra_pages": [],
    },
    {
        "url": "https://www.sgcms.com/regulatory-updates/minimum-rate-of-wages-telangana-2/",
        "name": "Telangana Minimum Wages",
        "save_to": "documents/telangana/minimum_wage/telangana_minimum_wages.txt",
        "jurisdiction": "Telangana",
        "topic": "minimum_wage",
        "law_name": "Telangana Minimum Rate of Wages Notification",
        "agency": "Telangana Labour Department",
        "method": "playwright",
        "wait_for": "article",
        "extra_pages": [],
    },
    {
        "url": "https://www.indiacode.nic.in/handle/123456789/19710",
        "name": "Maharashtra Shops and Establishments Act",
        "save_to": "documents/maharashtra/leave_policy/maharashtra_shops_establishments_act.txt",
        "jurisdiction": "Maharashtra",
        "topic": "leave_policy",
        "law_name": "Maharashtra Shops and Establishments Act",
        "agency": "Maharashtra Labour Department",
        "method": "playwright",
        "wait_for": "body",
        "extra_pages": [],
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


# ─────────────────────────────────────────────────────────
# TABLE EXTRACTOR
# Converts HTML tables to readable text
# Critical for wage rate tables
# ─────────────────────────────────────────────────────────
def extract_table_as_text(table: Tag) -> str:
    """
    Convert an HTML table to readable plain text.
    Wage rate tables are CRITICAL - must not miss them.
    """
    rows = []

    # get headers
    headers = []
    header_row = table.find("thead")
    if header_row:
        for th in header_row.find_all(["th", "td"]):
            headers.append(th.get_text(strip=True))

    if headers:
        rows.append(" | ".join(headers))
        rows.append("-" * 60)

    # get body rows
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cell_text = td.get_text(separator=" ", strip=True)
            cell_text = re.sub(r"\s+", " ", cell_text)
            cells.append(cell_text)

        if any(c.strip() for c in cells):
            rows.append(" | ".join(cells))

    return "\n".join(rows)


# ─────────────────────────────────────────────────────────
# DEEP HTML EXTRACTOR
# ─────────────────────────────────────────────────────────
def deep_extract_html(html: str, url: str = "") -> str:
    """
    Deep extraction that gets EVERYTHING:
    - All text content
    - All tables converted to readable format
    - Nested content
    - Section headers preserved
    """
    soup = BeautifulSoup(html, "html.parser")

    # step 1: remove pure noise tags
    noise_tags = [
        "script", "style", "noscript", "iframe",
        "svg", "path", "symbol", "defs",
    ]
    for tag in soup(noise_tags):
        tag.decompose()

    # step 2: remove navigation and UI elements
    # but be CAREFUL - some sites put content in nav
    nav_selectors = [
        "nav", "header", "footer",
        "[class*='cookie']",
        "[class*='popup']",
        "[class*='modal']",
        "[class*='newsletter']",
        "[class*='subscribe']",
        "[class*='social']",
        "[class*='share']",
        "[id*='cookie']",
        "[id*='popup']",
        "[class*='breadcrumb']",
        "[class*='sidebar']",
        "[class*='widget']",
        "[class*='advertisement']",
        "[class*='ads']",
        "[class*='banner']",
    ]
    for selector in nav_selectors:
        for tag in soup.select(selector):
            tag.decompose()

    # step 3: find main content area
    # try multiple selectors in order of preference
    content_selectors = [
        "main",
        "article",
        "[role='main']",
        "#main-content",
        "#content",
        "#main",
        ".content",
        ".main-content",
        ".entry-content",
        ".post-content",
        ".page-content",
        ".article-content",
        ".container",
        "body",
    ]

    main_content = None
    for selector in content_selectors:
        found = soup.select_one(selector)
        if found:
            # check it has reasonable amount of text
            text_length = len(found.get_text(strip=True))
            if text_length > 200:
                main_content = found
                logger.debug(
                    f"Using selector '{selector}' "
                    f"({text_length} chars)"
                )
                break

    if not main_content:
        main_content = soup.body or soup

    # step 4: process content block by block
    # this preserves structure better than get_text()
    text_parts = []

    def process_element(element, depth=0):
        """Recursively process elements"""

        if not isinstance(element, Tag):
            # it is a text node
            text = str(element).strip()
            if text and len(text) > 1:
                text_parts.append(text)
            return

        tag_name = element.name.lower() if element.name else ""

        # skip hidden elements
        style = element.get("style", "")
        if "display:none" in style.replace(" ", "") or \
           "visibility:hidden" in style.replace(" ", ""):
            return

        # handle tables specially
        if tag_name == "table":
            table_text = extract_table_as_text(element)
            if table_text.strip():
                text_parts.append("\n[TABLE]\n")
                text_parts.append(table_text)
                text_parts.append("[/TABLE]\n")
            return

        # add spacing for block elements
        block_tags = [
            "p", "div", "section", "article",
            "li", "dd", "dt", "blockquote",
            "pre", "h1", "h2", "h3", "h4", "h5", "h6"
        ]

        is_block = tag_name in block_tags

        # add header markers
        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            header_text = element.get_text(
                separator=" ", strip=True
            )
            if header_text:
                text_parts.append(f"\n\n{'#' * int(tag_name[1])} {header_text}\n")
            return

        # add newline before block elements
        if is_block and text_parts and text_parts[-1] != "\n":
            text_parts.append("\n")

        # process children
        for child in element.children:
            process_element(child, depth + 1)

        # add newline after block elements
        if is_block:
            text_parts.append("\n")

    # process all children of main content
    for child in main_content.children:
        process_element(child)

    # step 5: join and clean
    raw_text = "".join(text_parts)

    # fix encoding
    raw_text = ftfy.fix_text(raw_text)

    # clean up
    raw_text = _clean_final_text(raw_text)

    return raw_text


def _clean_final_text(text: str) -> str:
    """Final cleaning pass"""

    # fix rupee symbol
    text = text.replace("₹", "Rs.")
    text = text.replace("â‚¹", "Rs.")

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove lines that are just whitespace or single chars
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()

        # keep empty lines (paragraph breaks)
        if not stripped:
            lines.append("")
            continue

        # skip lines that are just punctuation
        if re.match(r"^[\W_]{1,4}$", stripped):
            continue

        # skip very short random lines
        # but keep important short lines like "Rs. 500"
        if len(stripped) < 4 and not re.search(
            r"\d|Rs\.|%", stripped
        ):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────
# PLAYWRIGHT SCRAPER
# ─────────────────────────────────────────────────────────
def scrape_with_playwright(item: dict) -> str:
    """
    Full Playwright scraper with:
    - JavaScript rendering
    - Scroll to load lazy content
    - Click to expand accordions
    - Handle pagination
    - Wait for dynamic content
    """
    url = item["url"]
    wait_for = item.get("wait_for", "body")

    logger.info(f"Playwright scraping: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ]
            )

            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                viewport={"width": 1920, "height": 1080},
                locale="en-IN",
            )

            page = context.new_page()

            # navigate to page
            logger.info(f"Navigating to: {url}")
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=45000
            )

            # wait for main content
            try:
                page.wait_for_selector(wait_for, timeout=10000)
            except Exception:
                logger.warning("Selector wait timed out, continuing")

            # wait for dynamic content to load
            page.wait_for_timeout(3000)

            # scroll down slowly to trigger lazy loading
            logger.info("Scrolling to load all content...")
            _scroll_page(page)

            # try to expand accordions and collapsed sections
            _expand_collapsed_content(page)

            # wait after expanding
            page.wait_for_timeout(2000)

            # scroll again after expanding
            _scroll_page(page)

            # get final HTML
            html = page.content()

            # check for pagination and get all pages
            all_html = [html]
            extra_content = _handle_pagination(page, item)
            if extra_content:
                all_html.extend(extra_content)

            browser.close()

            # extract text from all pages
            all_text_parts = []
            for i, page_html in enumerate(all_html):
                page_text = deep_extract_html(page_html, url)
                if page_text:
                    if i > 0:
                        all_text_parts.append(
                            f"\n\n--- PAGE {i+1} ---\n\n"
                        )
                    all_text_parts.append(page_text)

            final_text = "\n\n".join(all_text_parts)
            logger.info(
                f"Playwright extracted: {len(final_text):,} chars"
            )
            return final_text

    except Exception as e:
        logger.error(f"Playwright failed: {e}")
        return ""


def _scroll_page(page):
    """Scroll page to load lazy content"""
    try:
        # get page height
        height = page.evaluate("document.body.scrollHeight")

        # scroll in steps
        scroll_step = 800
        current = 0

        while current < height:
            page.evaluate(f"window.scrollTo(0, {current})")
            page.wait_for_timeout(300)
            current += scroll_step

            # update height (page may have grown)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height > height:
                height = new_height

        # scroll back to top
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)

    except Exception as e:
        logger.warning(f"Scroll failed: {e}")


def _expand_collapsed_content(page):
    """
    Try to click on accordions, read more buttons,
    collapsed sections to reveal hidden content
    """
    expand_selectors = [
        # accordion buttons
        ".accordion-button",
        ".accordion-header",
        "[data-toggle='collapse']",
        "[data-bs-toggle='collapse']",
        # read more buttons
        "button[class*='read-more']",
        "a[class*='read-more']",
        ".read-more",
        # show more
        "button[class*='show-more']",
        "[class*='expand']",
        # tab buttons (click all tabs to get content)
        ".nav-tab",
        ".tab-button",
        "[role='tab']",
        # details/summary elements
        "details",
    ]

    for selector in expand_selectors:
        try:
            elements = page.query_selector_all(selector)
            for element in elements[:10]:  # max 10 clicks
                try:
                    element.click()
                    page.wait_for_timeout(500)
                except Exception:
                    pass
        except Exception:
            pass

    # also try to open details elements via JS
    try:
        page.evaluate("""
            document.querySelectorAll('details').forEach(d => {
                d.open = true;
            });
        """)
    except Exception:
        pass


def _handle_pagination(page, item: dict) -> list:
    """
    Check if page has pagination and get all pages.
    Returns list of HTML strings for additional pages.
    """
    extra_html = []

    try:
        # look for next page buttons
        next_selectors = [
            "a[aria-label='Next']",
            "a[class*='next']",
            ".pagination a[rel='next']",
            "a:has-text('Next')",
            "a:has-text('>')",
            ".next-page",
        ]

        max_pages = 5  # limit pagination

        for page_num in range(max_pages):
            next_button = None

            for selector in next_selectors:
                try:
                    btn = page.query_selector(selector)
                    if btn and btn.is_visible():
                        next_button = btn
                        break
                except Exception:
                    continue

            if not next_button:
                break

            logger.info(f"Found next page button, clicking...")
            next_button.click()
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(2000)
            _scroll_page(page)

            extra_html.append(page.content())
            logger.info(
                f"Got page {page_num + 2}"
            )

    except Exception as e:
        logger.warning(f"Pagination handling failed: {e}")

    return extra_html


# ─────────────────────────────────────────────────────────
# HTTPX FALLBACK
# For simple static sites
# ─────────────────────────────────────────────────────────
def scrape_with_httpx(item: dict) -> str:
    """Simple HTTP scraper for static sites"""
    url = item["url"]

    try:
        response = httpx.get(
            url,
            headers=HEADERS,
            timeout=30.0,
            follow_redirects=True,
            verify=False,
        )

        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code}")
            return ""

        text = deep_extract_html(response.text, url)
        logger.info(f"httpx extracted: {len(text):,} chars")
        return text

    except Exception as e:
        logger.error(f"httpx failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────
# SAVE TO FILE
# ─────────────────────────────────────────────────────────
def save_to_file(item: dict, text: str) -> bool:
    """Save scraped text with metadata header"""
    save_path = item["save_to"]

    content = f"""SOURCE_URL: {item['url']}
LAW_NAME: {item['law_name']}
JURISDICTION: {item['jurisdiction']}
TOPIC: {item['topic']}
AGENCY: {item['agency']}
SCRAPED_CHARS: {len(text)}
---CONTENT_START---
{text}"""

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            f"Saved to: {save_path} "
            f"({len(text):,} chars)"
        )
        return True

    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False


# ─────────────────────────────────────────────────────────
# QUALITY CHECK
# ─────────────────────────────────────────────────────────
def quality_check(text: str, item: dict) -> dict:
    """
    Check if extracted text is good quality.
    Returns report dict.
    """
    report = {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.split("\n")),
        "has_numbers": bool(re.search(r"\d+", text)),
        "has_rupee": bool(
            re.search(r"Rs\.|₹|rupee|wage|salary", text, re.I)
        ),
        "has_tables": "[TABLE]" in text,
        "quality": "unknown",
    }

    if report["char_count"] > 5000:
        report["quality"] = "EXCELLENT"
    elif report["char_count"] > 2000:
        report["quality"] = "GOOD"
    elif report["char_count"] > 500:
        report["quality"] = "ACCEPTABLE"
    else:
        report["quality"] = "POOR"

    return report


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print("LaborLens - Deep URL Scraper")
    print("=" * 65)
    print("Uses Playwright to render JavaScript")
    print("Extracts tables, accordions, all content\n")

    results = []

    for i, item in enumerate(URLS_TO_SCRAPE, 1):
        print(f"\n{'─' * 65}")
        print(f"[{i}/{len(URLS_TO_SCRAPE)}] {item['name']}")
        print(f"URL: {item['url'][:70]}")
        print(f"Saving to: {item['save_to']}")

        text = ""

        # try Playwright first
        print("  Trying Playwright (full JS rendering)...")
        text = scrape_with_playwright(item)

        # fallback to httpx if playwright got nothing
        if len(text) < 500:
            print(
                f"  Playwright got only {len(text)} chars, "
                f"trying httpx..."
            )
            text_httpx = scrape_with_httpx(item)
            if len(text_httpx) > len(text):
                text = text_httpx

        # quality check
        if text:
            report = quality_check(text, item)
            quality_emoji = {
                "EXCELLENT": "🟢",
                "GOOD": "🟡",
                "ACCEPTABLE": "🟠",
                "POOR": "🔴"
            }.get(report["quality"], "⚪")

            print(
                f"  {quality_emoji} Quality: {report['quality']} | "
                f"{report['char_count']:,} chars | "
                f"{report['word_count']:,} words | "
                f"Tables: {report['has_tables']} | "
                f"Has wages: {report['has_rupee']}"
            )

            # show sample of extracted text
            lines = [
                l.strip() for l in text.split("\n")
                if l.strip() and len(l.strip()) > 20
            ]
            if lines:
                print(f"\n  📝 Content preview:")
                for line in lines[2:6]:
                    print(f"     {line[:80]}")

            if report["quality"] != "POOR":
                saved = save_to_file(item, text)
                result_status = "SUCCESS" if saved else "SAVE_FAILED"
            else:
                print(
                    f"\n  ⚠️  Content too poor to save"
                )
                print(
                    f"  Manual action needed for: {item['name']}"
                )
                result_status = "POOR_CONTENT"
        else:
            print(f"  ❌ No content extracted")
            result_status = "FAILED"

        results.append({
            "name": item["name"],
            "status": result_status,
            "chars": len(text),
            "save_to": item["save_to"],
        })

        # be polite between requests
        if i < len(URLS_TO_SCRAPE):
            print(f"\n  Waiting 3 seconds before next URL...")
            time.sleep(3)

    # ── Final Report ───────────────────────────────────
    print(f"\n\n{'=' * 65}")
    print("SCRAPING COMPLETE - FINAL REPORT")
    print("=" * 65)

    success = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] != "SUCCESS"]

    print(f"\n✅ Succeeded ({len(success)}):")
    for r in success:
        print(f"   {r['name']:<45} {r['chars']:>8,} chars")

    if failed:
        print(f"\n❌ Failed/Poor ({len(failed)}):")
        for r in failed:
            print(
                f"   {r['name']:<45} "
                f"Status: {r['status']}"
            )
            print(
                f"   Save manually to: {r['save_to']}"
            )

    print(f"\nTotal content: "
          f"{sum(r['chars'] for r in success):,} chars "
          f"from {len(success)} sources")

    print("\nNext step:")
    print("  python scraper/ingest_documents.py")
    print("=" * 65)