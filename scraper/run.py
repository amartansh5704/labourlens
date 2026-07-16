# scraper/run.py
# Main entry point for the scraper service
# Run this manually when you want to scrape

import sys
import os

# add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from database.connection import init_db, get_db_session
from database.models import Document, IndexLog
from scraper.processors.chunker import create_chunks
from loguru import logger
from datetime import datetime


def run_spiders(spider_names: list = None):
    """
    Run Scrapy spiders to collect documents.

    Args:
        spider_names: list of spider names to run
                     None = run all spiders
    """
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    import scrapy.settings

    logger.info("=" * 50)
    logger.info("Starting LaborLens Scraper")
    logger.info("=" * 50)

    # initialize database first
    init_db()

    # load scrapy settings
    settings = scrapy.settings.Settings()
    settings.setmodule("scraper.settings")

    process = CrawlerProcess(settings)

    # import all spiders
    from scraper.spiders.central_spider import CentralLaborSpider

    # map of available spiders
    available_spiders = {
        "central_labor": CentralLaborSpider,
        # we will add more spiders in later phases
        # "delhi_labor": DelhiLaborSpider,
        # "maharashtra_labor": MaharashtraLaborSpider,
    }

    # decide which spiders to run
    if spider_names:
        spiders_to_run = {
            name: cls
            for name, cls in available_spiders.items()
            if name in spider_names
        }
    else:
        spiders_to_run = available_spiders

    if not spiders_to_run:
        logger.error(f"No valid spiders found in: {spider_names}")
        return

    # add each spider to the process
    for name, spider_class in spiders_to_run.items():
        logger.info(f"Adding spider: {name}")
        process.crawl(spider_class)

    logger.info(f"Running {len(spiders_to_run)} spider(s)...")
    logger.info("This may take several minutes...")
    logger.info("Press Ctrl+C to stop early")

    # start crawling (blocks until done)
    process.start()

    logger.info("Scraping complete")
    _print_scrape_summary()


def index_documents(jurisdiction: str = None):
    """
    Take scraped documents from SQLite
    and index them into Qdrant vector database.

    Args:
        jurisdiction: only index this jurisdiction
                     None = index all
    """
    from scraper.indexer.embedder import Embedder
    from scraper.indexer.qdrant_indexer import QdrantIndexer

    logger.info("=" * 50)
    logger.info("Starting Document Indexing")
    logger.info("=" * 50)

    db = get_db_session()
    embedder = Embedder()
    indexer = QdrantIndexer()

    try:
        # get unindexed documents
        query = db.query(Document).filter(
            Document.is_indexed == False,
            Document.raw_text != None,
        )

        if jurisdiction:
            query = query.filter(
                Document.jurisdiction == jurisdiction
            )

        documents = query.all()
        total = len(documents)

        if total == 0:
            logger.info("No unindexed documents found")
            logger.info("Run scraping first: python scraper/run.py scrape")
            return

        logger.info(f"Found {total} unindexed documents")

        success_count = 0
        fail_count = 0

        for i, doc in enumerate(documents, 1):
            logger.info(
                f"[{i}/{total}] Indexing: "
                f"[{doc.jurisdiction}] {doc.url[:50]}"
            )

            try:
                # create chunks from document text
                chunks = create_chunks(
                    text=doc.raw_text,
                    document_metadata={
                        "id": doc.id,
                        "url": doc.url,
                        "title": doc.title or "",
                        "jurisdiction": doc.jurisdiction,
                        "topic": doc.topic,
                        "law_name": doc.law_name or "",
                        "document_type": doc.document_type or "",
                        "agency": doc.agency or "",
                        "effective_date": doc.effective_date or "",
                        "file_type": doc.file_type or "html",
                    }
                )

                if not chunks:
                    logger.warning(f"No chunks created for: {doc.url}")
                    continue

                # generate embeddings and index in Qdrant
                chunk_count = indexer.index_chunks(
                    chunks=chunks,
                    embedder=embedder
                )

                # update document record
                doc.is_indexed = True
                doc.chunk_count = chunk_count
                db.commit()

                # save index log
                index_log = IndexLog(
                    document_id=doc.id,
                    document_url=doc.url,
                    status="success",
                    chunks_indexed=chunk_count,
                    indexed_at=datetime.utcnow()
                )
                db.add(index_log)
                db.commit()

                success_count += 1
                logger.info(
                    f"    ✅ Indexed {chunk_count} chunks"
                )

            except Exception as e:
                fail_count += 1
                logger.error(f"    ❌ Failed: {e}")

                # save failed index log
                index_log = IndexLog(
                    document_id=doc.id,
                    document_url=doc.url,
                    status="failed",
                    chunks_indexed=0,
                    error_msg=str(e),
                    indexed_at=datetime.utcnow()
                )
                db.add(index_log)
                db.commit()

    finally:
        db.close()

    logger.info("=" * 50)
    logger.info(f"Indexing complete")
    logger.info(f"Success: {success_count} documents")
    logger.info(f"Failed:  {fail_count} documents")
    logger.info("=" * 50)


def _print_scrape_summary():
    """Print summary of what was scraped"""
    from database.connection import get_db_stats
    stats = get_db_stats()

    logger.info("=" * 50)
    logger.info("Scrape Summary")
    logger.info("=" * 50)
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(f"By jurisdiction: {stats['by_jurisdiction']}")
    logger.info(f"By topic:        {stats['by_topic']}")


def show_status():
    """Show current database and index status"""
    from database.connection import get_db_stats

    stats = get_db_stats()

    print("\n" + "=" * 50)
    print("LaborLens Status")
    print("=" * 50)
    print(f"Total documents:    {stats['total_documents']}")
    print(f"Indexed documents:  {stats['indexed_documents']}")
    print(f"Unindexed docs:     {stats['unindexed_documents']}")
    print(f"Total chunks:       {stats['total_chunks']}")
    print(f"Failed scrapes:     {stats['failed_scrapes']}")
    print("\nBy Jurisdiction:")
    for j, count in stats['by_jurisdiction'].items():
        print(f"  {j:<20} {count} documents")
    print("\nBy Topic:")
    for t, count in stats['by_topic'].items():
        print(f"  {t:<25} {count} documents")
    print("=" * 50)


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # usage:
    # python scraper/run.py             → scrape + index everything
    # python scraper/run.py scrape      → only scrape
    # python scraper/run.py index       → only index
    # python scraper/run.py status      → show current status

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "scrape":
        run_spiders()

    elif mode == "index":
        run_index = sys.argv[2] if len(sys.argv) > 2 else None
        index_documents(jurisdiction=run_index)

    elif mode == "status":
        show_status()

    elif mode == "all":
        run_spiders()
        index_documents()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage:")
        print("  python scraper/run.py           # scrape + index")
        print("  python scraper/run.py scrape    # only scrape")
        print("  python scraper/run.py index     # only index")
        print("  python scraper/run.py status    # show status")