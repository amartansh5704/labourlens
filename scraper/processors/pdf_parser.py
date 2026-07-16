# scraper/processors/pdf_parser.py
# Extracts text from PDF files
# Tries pdfplumber first, falls back to PyMuPDF

import pdfplumber
import fitz  # PyMuPDF
import ftfy
import io
import re
from loguru import logger


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Main function to extract text from PDF bytes.

    Tries pdfplumber first (better for structured PDFs)
    Falls back to PyMuPDF (better for scanned/complex PDFs)

    Args:
        pdf_bytes: raw PDF file content as bytes

    Returns:
        Extracted and cleaned text string
    """

    if not pdf_bytes:
        return ""

    # try pdfplumber first
    text = _extract_with_pdfplumber(pdf_bytes)

    # if pdfplumber got less than 100 chars, try PyMuPDF
    if len(text.strip()) < 100:
        logger.debug("pdfplumber got little text, trying PyMuPDF")
        text = _extract_with_pymupdf(pdf_bytes)

    # clean the extracted text
    if text:
        text = _clean_pdf_text(text)

    return text


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """
    Extract text using pdfplumber.
    Better for PDFs with clear text layers.
    """
    try:
        text_parts = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            logger.debug(f"PDF has {total_pages} pages")

            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()

                if page_text:
                    # add page marker to help with chunking later
                    text_parts.append(f"\n--- Page {i+1} ---\n")
                    text_parts.append(page_text)

        full_text = "\n".join(text_parts)
        full_text = ftfy.fix_text(full_text)

        logger.debug(
            f"pdfplumber extracted {len(full_text)} characters"
        )
        return full_text

    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        return ""


def _extract_with_pymupdf(pdf_bytes: bytes) -> str:
    """
    Extract text using PyMuPDF (fitz).
    Better fallback for complex PDFs.
    """
    try:
        text_parts = []

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for i, page in enumerate(doc):
            page_text = page.get_text()

            if page_text:
                text_parts.append(f"\n--- Page {i+1} ---\n")
                text_parts.append(page_text)

        doc.close()

        full_text = "\n".join(text_parts)
        full_text = ftfy.fix_text(full_text)

        logger.debug(
            f"PyMuPDF extracted {len(full_text)} characters"
        )
        return full_text

    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")
        return ""


def _clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDFs.
    PDFs often have extra artifacts.
    """

    # remove page markers we added (optional, can keep them)
    # text = re.sub(r"\n--- Page \d+ ---\n", "\n\n", text)

    # fix common PDF extraction issues

    # remove hyphenation at line breaks (word- \n word)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # remove lines with just dots (table of contents lines)
    text = re.sub(r"^[.\s]+$", "", text, flags=re.MULTILINE)

    text = text.strip()

    return text