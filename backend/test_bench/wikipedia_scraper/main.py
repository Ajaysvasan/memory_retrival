#!/usr/bin/env python3
"""
Wikipedia Scraper for RAG-TCRL-X
Produces topic-aware, clean Wikipedia corpus
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from seeds import WikipediaSeeds
from crawler import WikipediaCrawler
from extractor import ContentExtractor
from cleaner import TextCleaner
from topic_assigner import TopicAssigner
from exporter import DataExporter


def setup_logging():
    """Setup logging configuration"""

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "wikipedia_scraper.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main scraping pipeline"""

    setup_logging()
    logger = logging.getLogger("Main")

    logger.info("=" * 80)
    logger.info("WIKIPEDIA SCRAPER FOR RAG-TCRL-X")
    logger.info("=" * 80)

    try:
        logger.info("Validating configuration...")
        Config.validate()
        WikipediaSeeds.validate()
        logger.info("✓ Configuration valid")

        crawler = WikipediaCrawler(Config)
        extractor = ContentExtractor(Config)
        cleaner = TextCleaner(Config)
        topic_assigner = TopicAssigner(Config)
        exporter = DataExporter(Config)

        all_processed_data = []

        topics = WikipediaSeeds.get_all_topics()
        logger.info(f"Topics to scrape: {len(topics)}")

        for topic_idx, topic_id in enumerate(topics, 1):
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"TOPIC {topic_idx}/{len(topics)}: {topic_id}")
            logger.info("=" * 80)

            seeds = WikipediaSeeds.get_seeds_for_topic(topic_id)

            raw_pages = crawler.crawl_topic(topic_id, seeds)

            if not raw_pages:
                logger.warning(f"No pages crawled for {topic_id}")
                continue

            logger.info(f"Processing {len(raw_pages)} pages for {topic_id}...")

            for page_data in raw_pages:
                try:
                    extracted = extractor.extract(page_data)

                    cleaned_sections = [
                        cleaner.clean_section(section)
                        for section in extracted["sections"]
                    ]

                    extracted["sections"] = [
                        s for s in cleaned_sections if s["text"].strip()
                    ]

                    if not extracted["sections"]:
                        logger.warning(
                            f"No valid sections for {extracted['title']}, skipping"
                        )
                        continue

                    assigned = topic_assigner.assign(extracted)

                    exporter.export(assigned, topic_id)

                    all_processed_data.append(assigned)

                except Exception as e:
                    logger.error(
                        f"Failed to process {page_data.get('title', 'unknown')}: {e}"
                    )
                    continue

            logger.info(f"✓ Completed {topic_id}: {len(raw_pages)} pages processed")

        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 80)

        summary_path = Config.OUTPUT_DIR / "scrape_summary.json"
        exporter.export_summary(all_processed_data, summary_path)

        logger.info(f"Total pages scraped: {len(all_processed_data)}")
        logger.info(f"Output directory: {Config.OUTPUT_DIR}")
        logger.info(f"Summary: {summary_path}")

        topic_counts = {}
        for data in all_processed_data:
            topic_id = data["primary_topic_id"]
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1

        logger.info("")
        logger.info("Pages per topic:")
        for topic_id, count in sorted(topic_counts.items()):
            logger.info(f"  {topic_id}: {count} pages")

        logger.info("")
        logger.info("=" * 80)
        logger.info("SCRAPING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Successfully scraped {len(all_processed_data)} Wikipedia pages")
        logger.info(f"✓ Data saved to: {Config.OUTPUT_DIR}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review generated PDFs in data/datasets/wikipedia_general/")
        logger.info("2. Run RAG-TCRL-X ingestion to index the Wikipedia corpus")
        logger.info("3. Test queries against the expanded knowledge base")

    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
