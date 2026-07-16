# scraper/processors/html_parser.py
# Extracts clean text from HTML pages

from bs4 import BeautifulSoup
import ftfy
import re
from loguru import logger


# Tags that never contain useful content
TAGS_TO_REMOVE = [
    "script",    # JavaScript
    "style",     # CSS
    "nav",       # navigation menus
    "header",    # page header
    "footer",    # page footer
    "aside",     # sidebars
    "iframe",    # embedded frames
    "noscript",  # fallback JS content
    "form",      # search forms etc
    "button",    # buttons
    "input",     # form inputs
    "select",    # dropdowns
    "meta",      # meta tags
    "link",      # link tags
]


def extract_text_from_html(html: str) -> str:
    """
    Main function to extract clean text from HTML string.

    Steps:
    1. Parse HTML with BeautifulSoup
    2. Remove unwanted tags
    3. Extract text
    4. Fix encoding issues
    5. Clean whitespace
    6. Return clean text
    """

    if not html:
        return ""

    try:
        # parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # remove unwanted tags
        for tag_name in TAGS_TO_REMOVE:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # try to find main content area first
        # government sites often have a main div
        main_content = (
            soup.find("main") or
            soup.find("div", {"id": "content"}) or
            soup.find("div", {"id": "main-content"}) or
            soup.find("div", {"class": "content"}) or
            soup.find("article") or
            soup.body or
            soup
        )

        # extract text with newlines between tags
        text = main_content.get_text(separator="\n")

        # fix broken unicode characters
        text = ftfy.fix_text(text)

        # clean up the text
        text = _clean_text(text)

        return text

    except Exception as e:
        logger.error(f"HTML parsing failed: {e}")
        return ""


def extract_title_from_html(html: str) -> str:
    """Extract page title from HTML"""
    try:
        soup = BeautifulSoup(html, "html.parser")

        # try title tag first
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        # try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()

        return ""

    except Exception:
        return ""


def _clean_text(text: str) -> str:
    """
    Clean up extracted text.
    Removes noise while keeping legal content intact.
    """

    # remove lines that are just navigation/menu items
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # skip empty lines (will add back as paragraph breaks)
        if not line:
            cleaned_lines.append("")
            continue

        # skip very short lines that are likely menu items
        # but keep them if they look like section headers
        if len(line) < 5:
            continue

        # skip lines that are just numbers (page numbers)
        if re.match(r"^\d+$", line):
            continue

        # skip lines that are just special characters
        if re.match(r"^[\W]+$", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # remove leading/trailing whitespace
    text = text.strip()

    return text