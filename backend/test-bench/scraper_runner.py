"""Wikipedia scraper runner"""
import sys
import logging
from pathlib import Path

# Add scraper directory to path
SCRAPER_DIR = Path(__file__).parent / "wikipedia_scraper"
if str(SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAPER_DIR))

try:
    from config import Config
    from seeds import WikipediaSeeds
    from crawler import WikipediaCrawler
    from extractor import ContentExtractor
    from cleaner import TextCleaner
    from topic_assigner import TopicAssigner
    from exporter import DataExporter
except ImportError as e:
    raise ImportError(f"Failed to import scraper modules: {e}")


def run_scraper(force: bool = False) -> Path:
    """Run Wikipedia scraper"""
    logger = logging.getLogger(__name__)
    
    output_dir = Config.OUTPUT_DIR
    if output_dir.exists() and list(output_dir.rglob("*.pdf")) and not force:
        logger.info(f"Data already exists at {output_dir}. Skipping scraping.")
        return output_dir
    
    logger.info("=" * 80)
    logger.info("RUNNING WIKIPEDIA SCRAPER")
    logger.info("=" * 80)
    
    try:
        Config.validate()
        WikipediaSeeds.validate()
        
        crawler = WikipediaCrawler(Config)
        extractor = ContentExtractor(Config)
        cleaner = TextCleaner(Config)
        topic_assigner = TopicAssigner(Config)
        exporter = DataExporter(Config)
        
        all_processed_data = []
        topics = WikipediaSeeds.get_all_topics()
        
        logger.info(f"Scraping {len(topics)} topics...")
        
        for topic_idx, topic_id in enumerate(topics, 1):
            logger.info(f"Processing topic {topic_idx}/{len(topics)}: {topic_id}")
            
            seeds = WikipediaSeeds.get_seeds_for_topic(topic_id)
            raw_pages = crawler.crawl_topic(topic_id, seeds)
            
            if not raw_pages:
                continue
            
            for page_data in raw_pages:
                try:
                    extracted = extractor.extract(page_data)
                    cleaned_sections = [cleaner.clean_section(s) for s in extracted["sections"]]
                    extracted["sections"] = [s for s in cleaned_sections if s["text"].strip()]
                    
                    if not extracted["sections"]:
                        continue
                    
                    assigned = topic_assigner.assign(extracted)
                    exporter.export(assigned, topic_id)
                    all_processed_data.append(assigned)
                    
                except Exception as e:
                    logger.warning(f"Failed to process page: {e}")
                    continue
        
        summary_path = Config.OUTPUT_DIR / "scrape_summary.json"
        exporter.export_summary(all_processed_data, summary_path)
        
        logger.info(f"✓ Scraped {len(all_processed_data)} pages")
        logger.info(f"✓ Data saved to: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        logger.critical(f"Scraper error: {e}", exc_info=True)
        raise
