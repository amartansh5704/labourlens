# scraper/pipelines.py
# Processes each scraped item after spider extracts it
# Pipeline 1: Validates the item
# Pipeline 2: Saves to SQLite database

from database.connection import get_db_session
from database.models import Document, ScrapeLog
from loguru import logger
from datetime import datetime


# ─────────────────────────────────────────────────────────
# PIPELINE 1 - Validation
# Checks if item has required fields
# Drops bad items before they reach database
# ─────────────────────────────────────────────────────────
class ValidationPipeline:

    def process_item(self, item, spider):
        """
        Validate each scraped item.
        Return item to continue pipeline.
        Raise DropItem to discard it.
        """
        from scrapy.exceptions import DropItem

        # must have URL
        if not item.get("url"):
            raise DropItem("Missing URL")

        # must have some text content
        text = item.get("raw_text", "")
        if not text or len(text.strip()) < 100:
            raise DropItem(
                f"Text too short ({len(text)} chars): {item.get('url')}"
            )

        # must have jurisdiction
        if not item.get("jurisdiction"):
            raise DropItem(f"Missing jurisdiction: {item.get('url')}")

        # must have topic
        if not item.get("topic"):
            raise DropItem(f"Missing topic: {item.get('url')}")

        logger.debug(f"Validation passed: {item.get('url')[:60]}")
        return item


# ─────────────────────────────────────────────────────────
# PIPELINE 2 - Database
# Saves validated items to SQLite
# Skips duplicates (same URL)
# ─────────────────────────────────────────────────────────
class DatabasePipeline:

    def open_spider(self, spider):
        """Called when spider starts"""
        logger.info(f"Spider started: {spider.name}")

    def close_spider(self, spider):
        """Called when spider finishes"""
        logger.info(f"Spider finished: {spider.name}")

    def process_item(self, item, spider):
        """Save item to database"""
        db = get_db_session()

        try:
            url = item.get("url")

            # check if URL already exists in database
            existing = db.query(Document).filter(
                Document.url == url
            ).first()

            if existing:
                # log as skipped
                self._log_scrape(
                    db=db,
                    url=url,
                    spider_name=spider.name,
                    jurisdiction=item.get("jurisdiction"),
                    status="skipped",
                    text_length=len(item.get("raw_text", ""))
                )
                logger.info(f"Skipped duplicate: {url[:60]}")
                return item

            # create new document record
            doc = Document(
                url=url,
                title=item.get("title", ""),
                raw_text=item.get("raw_text", ""),
                jurisdiction=item.get("jurisdiction"),
                topic=item.get("topic"),
                document_type=item.get("document_type", "guidance"),
                law_name=item.get("law_name", ""),
                agency=item.get("agency", ""),
                effective_date=item.get("effective_date", ""),
                file_type=item.get("file_type", "html"),
                scraped_at=datetime.utcnow(),
                is_indexed=False,
                chunk_count=0
            )

            db.add(doc)
            db.commit()

            # log success
            self._log_scrape(
                db=db,
                url=url,
                spider_name=spider.name,
                jurisdiction=item.get("jurisdiction"),
                status="success",
                http_status=item.get("http_status", 200),
                text_length=len(item.get("raw_text", ""))
            )

            logger.info(
                f"Saved: [{item.get('jurisdiction')}] "
                f"[{item.get('topic')}] "
                f"{url[:60]}"
            )

        except Exception as e:
            db.rollback()
            # log failure
            self._log_scrape(
                db=db,
                url=item.get("url", "unknown"),
                spider_name=spider.name,
                jurisdiction=item.get("jurisdiction"),
                status="failed",
                error_msg=str(e)
            )
            logger.error(f"Database save failed: {e}")

        finally:
            db.close()

        return item

    def _log_scrape(
        self,
        db,
        url,
        spider_name,
        jurisdiction,
        status,
        http_status=None,
        error_msg=None,
        text_length=0,
        chunks_created=0
    ):
        """Helper to save scrape log record"""
        try:
            log = ScrapeLog(
                url=url,
                spider_name=spider_name,
                jurisdiction=jurisdiction,
                status=status,
                http_status=http_status,
                error_msg=error_msg,
                text_length=text_length,
                chunks_created=chunks_created
            )
            db.add(log)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save scrape log: {e}")