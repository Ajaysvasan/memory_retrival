import hashlib
from typing import List

try:
    from config.thresholds import Chunk
except:
    pass


class Chunker:
    """Deterministic chunking with meaning-preserving overlap."""

    def __init__(self, min_size: int = 200, max_size: int = 400, overlap: int = 50):
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap

    def chunk_document(self, paragraphs: List[str]) -> List[Chunk]:
        """Create chunks from paragraphs."""
        chunks = []
        text_buffer = []
        word_count = 0

        for para in paragraphs:
            words = para.split()
            text_buffer.append(para)
            word_count += len(words)

            if word_count >= self.min_size:
                chunk_text = " ".join(text_buffer)
                if word_count <= self.max_size or not self._has_sentence_boundary(
                    chunk_text
                ):
                    chunks.append(self._create_chunk(chunk_text))
                    text_buffer = self._create_overlap(text_buffer)
                    word_count = sum(len(t.split()) for t in text_buffer)

        if text_buffer:
            chunks.append(self._create_chunk(" ".join(text_buffer)))

        return chunks

    def _create_chunk(self, text: str) -> Chunk:
        chunk_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return Chunk(chunk_id=chunk_id, text=text)

    def _create_overlap(self, text_buffer: List[str]) -> List[str]:
        """Create tail-based overlap."""
        words = " ".join(text_buffer).split()
        if len(words) > self.overlap:
            overlap_text = " ".join(words[-self.overlap :])
            return [overlap_text]
        return text_buffer

    @staticmethod
    def _has_sentence_boundary(text: str) -> bool:
        return any(text.rstrip().endswith(p) for p in [".", "!", "?"])


if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(sys.path)
