# scraper/processors/cleaner.py
# Final text cleaning before chunking
# Handles Indian government website specific noise

import re
import ftfy
from loguru import logger


# Common noise patterns found on Indian govt websites
NOISE_PATTERNS = [
    # cookie notices
    r"This website uses cookies.*?(?=\n\n|\Z)",
    # skip to content links
    r"Skip to (?:main )?content",
    # screen reader text
    r"Screen Reader Access",
    # last updated dates
    r"Last Updated\s*:.*?\n",
    # visitor counters
    r"Visitor Counter\s*:?\s*\d+",
    # copyright lines
    r"©.*?(?:Government of India|Ministry|Department).*?\n",
    # website policies
    r"(?:Website Policy|Privacy Policy|Terms of Use).*?\n",
    # social media share buttons text
    r"Share on (?:Facebook|Twitter|WhatsApp|LinkedIn)",
    # print/download button text
    r"(?:Print|Download) (?:this )?(?:page|document)",
    # breadcrumb separators
    r"(?:Home\s*>|»|›)",
]


def clean_document_text(text: str, source_url: str = "") -> str:
    """
    Main cleaning function.
    Takes raw extracted text and returns clean text.

    Args:
        text: raw text from HTML or PDF parser
        source_url: URL where text came from (for logging)

    Returns:
        Clean text ready for chunking
    """

    if not text:
        return ""

    original_length = len(text)

    # fix encoding first
    text = ftfy.fix_text(text)

    # remove noise patterns
    text = _remove_noise_patterns(text)

    # normalize whitespace
    text = _normalize_whitespace(text)

    # fix common OCR/encoding errors in Indian legal docs
    text = _fix_legal_text(text)

    final_length = len(text)

    logger.debug(
        f"Cleaned text: {original_length} → {final_length} chars "
        f"({source_url[:40] if source_url else 'unknown'})"
    )

    return text


def detect_topic_from_text(text: str, url: str = "") -> str:
    """
    Try to detect topic from text content and URL.
    Returns best matching topic key.
    Falls back to 'general' if no match.
    """
    from shared.constants import TOPIC_KEYWORDS

    text_lower = (text[:2000] + url).lower()

    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(
            1 for keyword in keywords
            if keyword.lower() in text_lower
        )
        scores[topic] = score

    # get topic with highest score
    best_topic = max(scores, key=scores.get)

    # only return if we found at least one keyword match
    if scores[best_topic] > 0:
        return best_topic

    return "general"


def detect_topic_from_url(url: str) -> str:
    """
    Try to detect topic just from the URL.
    Faster than analyzing full text.
    """
    from shared.constants import TOPIC_KEYWORDS

    url_lower = url.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower().replace(" ", "-") in url_lower:
                return topic
            if keyword.lower().replace(" ", "_") in url_lower:
                return topic

    return "general"


def _remove_noise_patterns(text: str) -> str:
    """Remove common website noise"""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize all whitespace"""

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def _fix_legal_text(text: str) -> str:
    """
    Fix common issues in Indian legal document text.
    """

    # fix section symbol
    text = text.replace("§", "Section")
    text = text.replace("Sec.", "Section")

    # fix rupee symbol encoding issues
    text = text.replace("Rs.", "Rs.")
    text = text.replace("₹", "Rs.")

    # fix common abbreviations in Indian labor law
    replacements = {
        "w.e.f.": "with effect from",
        "w.r.t.": "with respect to",
        "i.e.": "that is",
        "e.g.": "for example",
        "govt.": "government",
        "dept.": "department",
        "min.": "minimum",
        "max.": "maximum",
        "p.m.": "per month",
        "p.a.": "per annum",
        "p.d.": "per day",
    }

    for short, full in replacements.items():
        text = re.sub(
            re.escape(short),
            full,
            text,
            flags=re.IGNORECASE
        )

    return text