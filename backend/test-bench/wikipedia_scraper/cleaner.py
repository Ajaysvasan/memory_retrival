import re
import logging


class TextCleaner:
    """Conservative text cleaning without summarization"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cleaner")

    def clean(self, text: str) -> str:
        """Clean text conservatively"""

        if not text:
            return ""

        cleaned = text

        cleaned = self._remove_citations(cleaned)
        cleaned = self._remove_wiki_markup(cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._preserve_sentence_boundaries(cleaned)

        return cleaned

    def _remove_citations(self, text: str) -> str:
        """Remove citation markers like [1], [2], [citation needed]"""

        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text)
        text = re.sub(r"\[clarification needed\]", "", text)
        text = re.sub(r"\[when\?\]", "", text)
        text = re.sub(r"\[verification needed\]", "", text)

        return text

    def _remove_wiki_markup(self, text: str) -> str:
        """Remove Wikipedia markup"""

        text = re.sub(r"\{\{[^}]+\}\}", "", text)
        text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)
        text = re.sub(r"'''([^']+)'''", r"\1", text)
        text = re.sub(r"''([^']+)''", r"\1", text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure"""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def _preserve_sentence_boundaries(self, text: str) -> str:
        """Ensure proper sentence boundaries"""

        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        return text

    def clean_section(self, section: dict) -> dict:
        """Clean a section dictionary"""

        return {
            "heading": self.clean(section["heading"]),
            "text": self.clean(section["text"]),
            "level": section["level"],
        }
