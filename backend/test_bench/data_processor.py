"""Data processor for Wikipedia scraped data"""

import json
import logging
from pathlib import Path
from typing import List

import pypdf

from bench_core.document import Document


class DataProcessor:
    """Processes Wikipedia scraped PDFs into Document objects"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def load_documents(self) -> List[Document]:
        """Load all documents from scraped data"""
        documents = []
        pdf_files = list(self.data_dir.rglob("*.pdf"))

        self.logger.info(f"Found {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            try:
                docs = self._load_pdf(pdf_path)
                documents.extend(docs)
            except Exception as e:
                self.logger.warning(f"Failed to load {pdf_path}: {e}")
                continue

        self.logger.info(f"Loaded {len(documents)} document chunks")
        return documents

    def _load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF file"""
        documents = []

        # Load metadata if available
        metadata_file = pdf_path.parent / f"{pdf_path.stem}_metadata.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")

        # Extract text from PDF
        try:
            with open(pdf_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)

                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"

                # Split into paragraphs (chunks)
                paragraphs = [
                    p.strip()
                    for p in full_text.split("\n\n")
                    if p.strip() and len(p.strip()) >= 100
                ]

                # Create documents
                for idx, para in enumerate(paragraphs):
                    doc_metadata = metadata.copy()
                    doc_metadata.update(
                        {
                            "source_file": str(pdf_path),
                            "paragraph_index": idx,
                            "title": metadata.get("title", pdf_path.stem),
                        }
                    )

                    doc_id = f"{pdf_path.stem}_para_{idx}"
                    documents.append(
                        Document(doc_id=doc_id, content=para, metadata=doc_metadata)
                    )
        except Exception as e:
            self.logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise

        return documents
