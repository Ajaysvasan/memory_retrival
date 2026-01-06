import unicodedata
from typing import Iterator


class DocumentLoader:
    """Faithfully load documents with structure preservation."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def load_document(filepath: str) -> Iterator[str]:
        """Stream paragraphs from document."""
        with open(filepath, "r", encoding="utf-8") as f:
            current_para = []
            for line in f:
                line = DocumentLoader.normalize_unicode(line.strip())
                if line:
                    current_para.append(line)
                elif current_para:
                    yield " ".join(current_para)
                    current_para = []
            if current_para:
                yield " ".join(current_para)
