"""Consolidated PDF report (reportlab). Content is whatever the caller computed - nothing is added here."""
from __future__ import annotations

import io
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ORANGE = colors.HexColor("#ff7a1a")


def _table(df: pd.DataFrame, max_rows: int = 20) -> Table:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
    data = [list(map(str, d.columns))] + [[escape(str(x))[:60] for x in r] for r in d.values.tolist()]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15203a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa7c0")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")])]))
    return t


def build_pdf(title: str, subtitle: str, sections: list[tuple[str, list]]) -> bytes:
    """sections: [(heading, [str | DataFrame | ('bullets', [str])])]"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title=title)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=ORANGE, alignment=0, fontSize=22)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#0b1220"), fontSize=13, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    story = [Paragraph(escape(title), h1), Paragraph(escape(subtitle), body), Spacer(1, 8)]
    for head, items in sections:
        story.append(Paragraph(escape(head), h2))
        for it in items:
            if isinstance(it, pd.DataFrame):
                story += [_table(it) if len(it) else Paragraph("No rows.", body), Spacer(1, 6)]
            elif isinstance(it, tuple) and it[0] == "bullets":
                story += [Paragraph("• " + escape(b), body) for b in it[1]]
            else:
                story.append(Paragraph(escape(str(it)), body))
    doc.build(story)
    return buf.getvalue()
