"""PDF rendering for quotations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from freight_budget_management.storage.repositories import QuotationLineItemRecord, QuotationRecord


def render_quotation_pdf(
    quotation: QuotationRecord,
    line_items: Iterable[QuotationLineItemRecord],
    output_path: Path,
    watermark_text: Optional[str] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    if watermark_text:
        pdf.saveState()
        pdf.setFont("Helvetica-Bold", 48)
        pdf.setFillGray(0.9)
        pdf.translate(width / 2, height / 2)
        pdf.rotate(35)
        pdf.drawCentredString(0, 0, watermark_text)
        pdf.restoreState()

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(inch, height - inch, "Freight Quotation")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(inch, height - 1.4 * inch, f"Quotation ID: {quotation.quotation_id}")
    pdf.drawString(inch, height - 1.6 * inch, f"Version: {quotation.version}")
    pdf.drawString(inch, height - 1.8 * inch, f"Status: {quotation.status}")
    pdf.drawString(inch, height - 2.0 * inch, f"Valid From: {quotation.valid_from}")
    pdf.drawString(inch, height - 2.2 * inch, f"Valid To: {quotation.valid_to}")
    pdf.drawString(inch, height - 2.4 * inch, f"Currency: {quotation.currency}")

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(inch, height - 2.9 * inch, "Line Items")
    pdf.setFont("Helvetica", 10)

    y_cursor = height - 3.2 * inch
    for item in line_items:
        pdf.drawString(inch, y_cursor, f"- {item.description} ({item.quantity} @ {item.unit_price})")
        pdf.drawRightString(width - inch, y_cursor, f"{quotation.currency} {item.amount:,.2f}")
        y_cursor -= 0.25 * inch
        if y_cursor < inch:
            pdf.showPage()
            y_cursor = height - inch

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawRightString(width - inch, y_cursor - 0.2 * inch, f"Total: {quotation.currency} {quotation.total_amount:,.2f}")

    pdf.showPage()
    pdf.save()
    return output_path
