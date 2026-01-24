import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging


class DataExporter:
    """Export scraped data as PDFs with metadata"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Exporter")
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if PDF dependencies are available"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                PageBreak,
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER

            self.letter = letter
            self.getSampleStyleSheet = getSampleStyleSheet
            self.ParagraphStyle = ParagraphStyle
            self.inch = inch
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.PageBreak = PageBreak
            self.TA_LEFT = TA_LEFT
            self.TA_CENTER = TA_CENTER

        except ImportError:
            raise ImportError(
                "reportlab is required for PDF export. Install with: pip install reportlab"
            )

    def export(self, data: Dict, topic_id: str) -> Path:
        """Export page data as PDF with metadata"""

        topic_dir = self.config.OUTPUT_DIR / topic_id
        topic_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = self._sanitize_filename(data["title"])
        pdf_path = topic_dir / f"{safe_filename}.pdf"
        metadata_path = topic_dir / f"{safe_filename}_metadata.json"

        self._create_pdf(data, pdf_path)
        self._create_metadata(data, metadata_path)

        self.logger.info(f"Exported: {pdf_path.name}")

        return pdf_path

    def _create_pdf(self, data: Dict, pdf_path: Path):
        """Create PDF from page data"""

        doc = self.SimpleDocTemplate(
            str(pdf_path),
            pagesize=self.letter,
            rightMargin=self.config.PDF_MARGIN,
            leftMargin=self.config.PDF_MARGIN,
            topMargin=self.config.PDF_MARGIN,
            bottomMargin=self.config.PDF_MARGIN,
        )

        styles = self.getSampleStyleSheet()

        title_style = self.ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=self.config.PDF_TITLE_SIZE,
            alignment=self.TA_CENTER,
            spaceAfter=12,
        )

        section_style = self.ParagraphStyle(
            "CustomSection",
            parent=styles["Heading2"],
            fontSize=self.config.PDF_SECTION_SIZE,
            spaceAfter=6,
        )

        body_style = self.ParagraphStyle(
            "CustomBody",
            parent=styles["BodyText"],
            fontSize=self.config.PDF_FONT_SIZE,
            alignment=self.TA_LEFT,
            spaceAfter=12,
        )

        story = []

        title = self.Paragraph(data["title"], title_style)
        story.append(title)
        story.append(self.Spacer(1, 0.2 * self.inch))

        url_text = f"<i>Source: {data['url']}</i>"
        url_para = self.Paragraph(url_text, styles["Italic"])
        story.append(url_para)
        story.append(self.Spacer(1, 0.3 * self.inch))

        for section in data["sections"]:
            heading = self.Paragraph(section["heading"], section_style)
            story.append(heading)

            text = section["text"]
            paragraphs = text.split("\n\n")

            for para in paragraphs:
                if para.strip():
                    safe_para = (
                        para.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    p = self.Paragraph(safe_para, body_style)
                    story.append(p)

            story.append(self.Spacer(1, 0.2 * self.inch))

        doc.build(story)

    def _create_metadata(self, data: Dict, metadata_path: Path):
        """Create metadata JSON file"""

        metadata = {
            "title": data["title"],
            "wikipedia_url": data["url"],
            "primary_topic_id": data["primary_topic_id"],
            "secondary_topics": data["secondary_topics"],
            "crawl_depth": data["depth"],
            "retrieved_timestamp": self.config.SCRAPE_TIMESTAMP,
            "source": self.config.SOURCE_NAME,
            "num_sections": len(data["sections"]),
            "section_headings": [s["heading"] for s in data["sections"]],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as filename"""

        safe = title.replace("/", "_").replace("\\", "_")
        safe = safe.replace(":", "_").replace("*", "_")
        safe = safe.replace("?", "_").replace('"', "_")
        safe = safe.replace("<", "_").replace(">", "_")
        safe = safe.replace("|", "_")

        safe = safe[:200]

        return safe

    def export_summary(self, all_data: List[Dict], output_path: Path):
        """Export summary of all scraped data"""

        summary = {
            "scrape_timestamp": self.config.SCRAPE_TIMESTAMP,
            "total_pages": len(all_data),
            "topics": {},
            "files": [],
        }

        for data in all_data:
            topic_id = data["primary_topic_id"]

            if topic_id not in summary["topics"]:
                summary["topics"][topic_id] = {"count": 0, "pages": []}

            summary["topics"][topic_id]["count"] += 1
            summary["topics"][topic_id]["pages"].append(data["title"])

            safe_filename = self._sanitize_filename(data["title"])
            summary["files"].append(f"{topic_id}/{safe_filename}.pdf")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported summary to {output_path}")
