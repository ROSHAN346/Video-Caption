"""
PDF Generator Service

Converts Markdown reports to PDF format with embedded keyframe images.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)


def _sanitize_for_latin1(text: str) -> str:
    """Replace Unicode characters that are not in Latin-1 with ASCII equivalents."""
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u2022': '-',  # Bullet
        '\u00ab': '"',  # Left guillemet
        '\u00bb': '"',  # Right guillemet
        '\u2039': "'",  # Single left guillemet
        '\u203a': "'",  # Single right guillemet
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    # Fallback: encode to latin-1, replacing any remaining unknown chars
    return text.encode('latin-1', errors='replace').decode('latin-1')


class ReportPDF(FPDF):
    """Custom PDF class for report generation."""

    def header(self):
        """Add header to each page."""
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Video Keyframe Analysis Report", align="C")
        self.ln(10)

    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report_pdf(
    scene_id: int,
    style: str,
    report: str,
    keyframe_path: str = None,
    scene_data: dict = None,
    output_path: str = None
) -> Path:
    """
    Generate a PDF report.

    Args:
        scene_id: Scene ID
        style: Report style
        report: Report text
        keyframe_path: Optional path to keyframe image
        scene_data: Optional scene analysis data
        output_path: Optional output path

    Returns:
        Path to generated PDF
    """
    if output_path is None:
        output_path = f"reports/scene_{scene_id:03d}/{style}.pdf"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create PDF
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Add keyframe image if provided
    if keyframe_path and Path(keyframe_path).exists():
        _add_keyframe_page(pdf, keyframe_path, scene_data)

    # Add report content
    _add_report_content(pdf, scene_id, style, report, scene_data)

    # Save
    pdf.output(str(output_path))
    logger.info(f"Generated PDF: {output_path}")

    return output_path


def _add_keyframe_page(pdf: ReportPDF, keyframe_path: str, scene_data: dict = None):
    """Add keyframe image page."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Scene Keyframe", ln=True)
    pdf.ln(5)

    # Add image
    try:
        pdf.image(keyframe_path, x=10, w=190)
        pdf.ln(10)
    except Exception as e:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"[Image could not be embedded: {e}]", ln=True)

    # Add scene metadata if available
    if scene_data:
        _add_metadata_table(pdf, scene_data)

    pdf.add_page()


def _add_metadata_table(pdf: ReportPDF, scene_data: dict):
    """Add scene metadata table."""
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Scene Metadata", ln=True)

    pdf.set_font("Helvetica", "", 10)

    fields = [
        ("Scene ID", scene_data.get("scene_id", "N/A")),
        ("Scene Type", scene_data.get("scene_type", "N/A")),
        ("Location", scene_data.get("location", "N/A")),
        ("Risk Level", scene_data.get("risk_level", "N/A")),
        ("Confidence", f"{scene_data.get('confidence', 0):.2f}"),
    ]

    for label, value in fields:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 8, f"{label}:", border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, str(value), border=1, ln=True)

    pdf.ln(5)


def _add_report_content(
    pdf: ReportPDF,
    scene_id: int,
    style: str,
    report: str,
    scene_data: dict = None
):
    """Add report content."""
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Scene {scene_id} - {style.replace('_', ' ').title()} Report", ln=True)
    pdf.ln(5)

    # Metadata
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Style: {style}", ln=True)
    pdf.ln(5)

    # Report content
    pdf.set_font("Helvetica", "", 11)

    # Split report into paragraphs and add them
    paragraphs = report.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            paragraph = _sanitize_for_latin1(paragraph)
            # Check if it's a heading (starts with #)
            if paragraph.startswith("#"):
                pdf.set_font("Helvetica", "B", 13)
                paragraph = paragraph.lstrip("#").strip()
                pdf.multi_cell(0, 8, paragraph)
                pdf.set_font("Helvetica", "", 11)
            else:
                pdf.multi_cell(0, 6, paragraph)
            pdf.ln(3)


def markdown_to_pdf(markdown_text: str, output_path: str) -> Path:
    """
    Convert markdown text to PDF.

    Args:
        markdown_text: Markdown formatted text
        output_path: Output PDF path

    Returns:
        Path to generated PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Parse and add markdown content
    lines = markdown_text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            pdf.ln(3)
            continue

        line = _sanitize_for_latin1(line)

        # Handle headings
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, line[4:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, line[3:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 8, line[2:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("- "):
            # Bullet point
            pdf.cell(10, 6, chr(8226))
            pdf.multi_cell(0, 6, line[2:])
        else:
            pdf.multi_cell(0, 6, line)

    pdf.output(str(output_path))
    return output_path
