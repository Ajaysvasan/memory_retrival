from pathlib import Path
from datetime import datetime


class Config:
    """Wikipedia scraper configuration"""

    # Output paths
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR.parent / "data" / "datasets" / "wikipedia_general"

    # Crawling parameters
    MAX_DEPTH = 2
    MAX_PAGES_PER_TOPIC = 50
    REQUEST_DELAY_SECONDS = 1.0

    # Wikipedia API
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    USER_AGENT = "RAG-TCRL-X-WikiScraper/1.0 (Educational/Research)"

    # Content extraction
    MIN_SECTION_LENGTH = 100
    MIN_PARAGRAPH_LENGTH = 50

    # PDF generation
    PDF_FONT_SIZE = 11
    PDF_TITLE_SIZE = 16
    PDF_SECTION_SIZE = 13
    PDF_MARGIN = 72

    # Metadata
    SOURCE_NAME = "wikipedia"
    SCRAPE_TIMESTAMP = datetime.now().isoformat()

    @classmethod
    def validate(cls):
        """Validate configuration"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if cls.MAX_DEPTH < 1:
            raise ValueError("MAX_DEPTH must be >= 1")

        if cls.MAX_PAGES_PER_TOPIC < 1:
            raise ValueError("MAX_PAGES_PER_TOPIC must be >= 1")

        if cls.REQUEST_DELAY_SECONDS < 0.5:
            raise ValueError("REQUEST_DELAY_SECONDS must be >= 0.5 for API politeness")
