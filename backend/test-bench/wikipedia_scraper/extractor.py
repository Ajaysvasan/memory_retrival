import re
from typing import Dict, List
import logging


class ContentExtractor:
    """Extract structured content from Wikipedia HTML/text"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Extractor")

    def extract(self, page_data: Dict) -> Dict:
        """Extract sections and paragraphs from page"""

        content = page_data["content"]
        title = page_data["title"]

        if not content or not content.strip():
            raise RuntimeError(f"No content to extract for {title}")

        sections = self._extract_sections(content)

        if not sections:
            sections = [{"heading": "Main Content", "text": content, "level": 1}]

        extracted = {
            "title": title,
            "url": page_data["url"],
            "topic_id": page_data["topic_id"],
            "depth": page_data["depth"],
            "sections": sections,
        }

        self.logger.debug(f"Extracted {len(sections)} sections from {title}")
        return extracted

    def _extract_sections(self, content: str) -> List[Dict]:
        """Extract sections from content"""

        lines = content.split("\n")
        sections = []
        current_section = None
        current_text = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            heading_match = re.match(r"^(={2,6})\s*(.+?)\s*\1$", line)

            if heading_match:
                if current_section and current_text:
                    current_section["text"] = "\n\n".join(current_text)
                    if len(current_section["text"]) >= self.config.MIN_SECTION_LENGTH:
                        sections.append(current_section)

                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                if self._is_valid_section(heading):
                    current_section = {"heading": heading, "text": "", "level": level}
                    current_text = []
                else:
                    current_section = None
                    current_text = []

            elif current_section is not None:
                if len(line) >= self.config.MIN_PARAGRAPH_LENGTH:
                    current_text.append(line)
            else:
                if len(line) >= self.config.MIN_PARAGRAPH_LENGTH:
                    if not sections:
                        current_section = {
                            "heading": "Introduction",
                            "text": "",
                            "level": 1,
                        }
                        current_text = [line]
                    else:
                        current_text.append(line)

        if current_section and current_text:
            current_section["text"] = "\n\n".join(current_text)
            if len(current_section["text"]) >= self.config.MIN_SECTION_LENGTH:
                sections.append(current_section)

        return sections

    def _is_valid_section(self, heading: str) -> bool:
        """Check if section heading is valid for inclusion"""

        invalid_sections = {
            "references",
            "external links",
            "see also",
            "notes",
            "bibliography",
            "further reading",
            "sources",
            "footnotes",
        }

        heading_lower = heading.lower().strip()

        return heading_lower not in invalid_sections
