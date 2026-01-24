#!/usr/bin/env python3
"""Main entry point for test-bench"""

import logging
import sys
from pathlib import Path

from orchestrator import TestBenchOrchestrator
from output_formatter import format_output
from scraper_runner import run_scraper

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("TEST-BENCH RAG SYSTEM")
    logger.info("=" * 80)

    # Determine data directory
    default_data_dir = Path(__file__).parent / "data" / "datasets" / "wikipedia_general"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--force-scrape":
            logger.info("Force scraping enabled...")
            data_dir = run_scraper(force=True)
        elif sys.argv[1] == "--skip-scrape":
            data_dir = default_data_dir
        else:
            data_dir = Path(sys.argv[1])
    else:
        # Check if data exists
        if not default_data_dir.exists() or not list(default_data_dir.rglob("*.pdf")):
            logger.info("Wikipedia data not found. Running scraper...")
            data_dir = run_scraper(force=False)
        else:
            logger.info(f"Using existing data at {default_data_dir}")
            data_dir = default_data_dir

    # Verify data exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    pdf_files = list(data_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found. Running scraper...")
        data_dir = run_scraper(force=True)
        pdf_files = list(data_dir.rglob("*.pdf"))
        if not pdf_files:
            logger.error("Scraper did not generate any PDF files.")
            sys.exit(1)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Found {len(pdf_files)} PDF files")

    # Initialize orchestrator
    try:
        orchestrator = TestBenchOrchestrator(data_dir)
        logger.info("Training all architectures (this may take a few minutes)...")
        orchestrator.train_all()
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Failed to train architectures: {e}", exc_info=True)
        sys.exit(1)

    # Interactive query loop
    logger.info("\n" + "=" * 80)
    logger.info("Ready to process queries. Type 'exit' or 'quit' to stop.")
    logger.info("=" * 80 + "\n")

    while True:
        try:
            query = input("Enter your query: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                logger.info("Exiting...")
                break

            # Process query through all architectures
            results = orchestrator.process_query(query)

            # Format and print output
            output = format_output(results)
            print("\n" + output)

        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
