"""
PDF Rendering Engine — Phase 4.2

Converts a structured ResumeJSON dict into a professional PDF
using HTML/CSS templates rendered by WeasyPrint.

Supports 3 templates: modern, classic, minimalist.
"""

import logging
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# Template directory — adjacent to this file
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=True,
)

VALID_TEMPLATES = ("modern", "classic", "minimalist")
TemplateChoice = Literal["modern", "classic", "minimalist"]


def render_resume_pdf(
    resume_json: dict,
    template: TemplateChoice = "modern",
) -> bytes:
    """
    Render a ResumeJSON dict into a PDF byte-string.

    Args:
        resume_json: Validated ResumeJSON.model_dump() dict.
        template: One of 'modern', 'classic', 'minimalist'.

    Returns:
        PDF content as bytes.
    """
    if template not in VALID_TEMPLATES:
        raise ValueError(f"Invalid template '{template}'. Choose from: {VALID_TEMPLATES}")

    # Lazy import — weasyprint can be heavy and is only needed here
    try:
        from weasyprint import HTML
    except ImportError:
        logger.error("[PDF] weasyprint is not installed. Run: pip install weasyprint")
        raise RuntimeError("weasyprint is not installed")

    # Load and render Jinja2 template
    tpl = _jinja_env.get_template(f"{template}.html")
    html_content = tpl.render(resume=resume_json)

    logger.info(f"[PDF] Rendering '{template}' template ({len(html_content)} chars HTML)")

    # Convert HTML → PDF
    pdf_bytes = HTML(string=html_content).write_pdf()

    logger.info(f"[PDF] Generated PDF: {len(pdf_bytes)} bytes")
    return pdf_bytes


def render_resume_html(
    resume_json: dict,
    template: TemplateChoice = "modern",
) -> str:
    """
    Render a ResumeJSON dict into an HTML string (for preview).
    """
    if template not in VALID_TEMPLATES:
        raise ValueError(f"Invalid template '{template}'. Choose from: {VALID_TEMPLATES}")

    tpl = _jinja_env.get_template(f"{template}.html")
    return tpl.render(resume=resume_json)
