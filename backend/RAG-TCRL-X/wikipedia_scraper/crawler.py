import requests
import time
from typing import List, Set, Dict, Optional
from urllib.parse import unquote
import logging


class WikipediaCrawler:
    """Wikipedia API-based crawler with depth limiting"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("WikiCrawler")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.visited: Set[str] = set()

    def crawl_topic(self, topic_id: str, seed_pages: List[str]) -> List[Dict]:
        """Crawl pages for a specific topic"""

        self.logger.info(f"Starting crawl for topic: {topic_id}")
        self.logger.info(f"Seed pages: {len(seed_pages)}")

        pages_data = []
        queue = [(page, 0) for page in seed_pages]
        visited_in_topic = set()

        while queue and len(pages_data) < self.config.MAX_PAGES_PER_TOPIC:
            page_title, depth = queue.pop(0)

            if page_title in visited_in_topic:
                continue

            if depth > self.config.MAX_DEPTH:
                continue

            visited_in_topic.add(page_title)

            try:
                page_data = self._fetch_page(page_title, topic_id, depth)

                if page_data:
                    pages_data.append(page_data)
                    self.logger.info(
                        f"[{topic_id}] Fetched: {page_title} "
                        f"(depth={depth}, total={len(pages_data)}/{self.config.MAX_PAGES_PER_TOPIC})"
                    )

                    if depth < self.config.MAX_DEPTH:
                        links = self._extract_links(page_data["content"])
                        for link in links[:10]:
                            if link not in visited_in_topic:
                                queue.append((link, depth + 1))

                time.sleep(self.config.REQUEST_DELAY_SECONDS)

            except Exception as e:
                self.logger.error(f"Failed to fetch {page_title}: {e}")
                continue

        self.logger.info(f"Completed crawl for {topic_id}: {len(pages_data)} pages")
        return pages_data

    def _fetch_page(self, title: str, topic_id: str, depth: int) -> Optional[Dict]:
        """Fetch single page from Wikipedia API"""

        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts|info",
            "exintro": False,
            "explaintext": False,
            "inprop": "url",
            "redirects": 1,
        }

        response = self.session.get(
            self.config.WIKIPEDIA_API_URL, params=params, timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Wikipedia API returned status {response.status_code} for {title}"
            )

        data = response.json()

        pages = data.get("query", {}).get("pages", {})

        if not pages:
            self.logger.warning(f"No pages returned for {title}")
            return None

        page_id = list(pages.keys())[0]

        if page_id == "-1":
            self.logger.warning(f"Page not found: {title}")
            return None

        page = pages[page_id]

        if "extract" not in page:
            self.logger.warning(f"No extract for {title}")
            return None

        return {
            "title": page.get("title", title),
            "pageid": page_id,
            "url": page.get("fullurl", ""),
            "content": page.get("extract", ""),
            "topic_id": topic_id,
            "depth": depth,
        }

    def _extract_links(self, content: str) -> List[str]:
        """Extract Wikipedia links from content"""

        params = {
            "action": "parse",
            "format": "json",
            "text": content[:5000],
            "prop": "links",
            "contentmodel": "wikitext",
        }

        try:
            response = self.session.get(
                self.config.WIKIPEDIA_API_URL, params=params, timeout=30
            )

            if response.status_code != 200:
                return []

            data = response.json()
            links = data.get("parse", {}).get("links", [])

            internal_links = [
                link["*"]
                for link in links
                if link.get("ns") == 0 and not link["*"].startswith("List of")
            ]

            return internal_links[:20]

        except Exception as e:
            self.logger.debug(f"Link extraction failed: {e}")
            return []
