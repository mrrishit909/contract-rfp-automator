"""Turn a PDF, DOCX or scanned image into ordered lines of text with page numbers.

PDF pages with a text layer are read directly; pages without one (scans) are rendered and sent through Tesseract OCR.
Ruled tables are lifted out as their own lines ("cell | cell | cell") so they are not smeared into the paragraphs.
Lines that repeat on most pages (running headers, footers, page numbers) are dropped.
"""
from __future__ import annotations

import io
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import pdfplumber
import pytesseract
from docx import Document
from docx.table import Table
from PIL import Image

MAX_PAGES = 400
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


@dataclass(frozen=True)
class Line:
    text: str
    page: int
    kind: Literal["text", "table", "heading"] = "text"   # "heading" only when the file itself says so (DOCX styles)


class UnsupportedFile(ValueError):
    pass


def ocr_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr(image: Image.Image, page: int) -> list[Line]:
    if not ocr_available():
        raise UnsupportedFile("this file needs OCR, and Tesseract is not installed (set TESSERACT_CMD)")
    return [Line(t.strip(), page) for t in pytesseract.image_to_string(image).splitlines() if t.strip()]


def _table_lines(rows: list[list[str | None]], page: int) -> list[Line]:
    out = []
    for row in rows:
        cells = [" ".join((c or "").split()) for c in row]
        if any(cells):
            out.append(Line(" | ".join(cells), page, "table"))
    return out


def _pdf(content: bytes) -> list[Line]:
    lines: list[Line] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        if len(pdf.pages) > MAX_PAGES:
            raise UnsupportedFile(f"more than {MAX_PAGES} pages")
        for n, page in enumerate(pdf.pages, start=1):
            x0, y0, x1, y1 = page.bbox           # ignore "tables" whose box falls outside the page (broken PDFs)
            tables = [t for t in page.find_tables() if x0 <= t.bbox[0] < t.bbox[2] <= x1 and y0 <= t.bbox[1] < t.bbox[3] <= y1]
            body = page
            for t in tables:                       # read the text around the tables, then the tables themselves
                body = body.outside_bbox(t.bbox)
            text = body.extract_text() or ""
            if len(text.strip()) < 20 and not tables:
                lines += _ocr(page.to_image(resolution=200).original, n)
                continue
            lines += [Line(t.strip(), n) for t in text.splitlines() if t.strip()]
            for t in tables:
                lines += _table_lines(t.extract(), n)
    return _drop_running_lines(lines)


def _drop_running_lines(lines: list[Line]) -> list[Line]:
    """Remove page furniture: any line (digits ignored) that appears on at least half of the pages, and bare page numbers."""
    pages = {ln.page for ln in lines}

    def key(text: str) -> str:
        return re.sub(r"\d+", "#", text)

    seen = Counter(k for p in pages for k in {key(ln.text) for ln in lines if ln.page == p})
    furniture = {k for k, c in seen.items() if len(pages) >= 4 and c >= len(pages) / 2}
    return [ln for ln in lines if key(ln.text) not in furniture and not re.fullmatch(r"-?\s*\d{1,3}\s*-?", ln.text)]


def _docx(content: bytes) -> list[Line]:
    lines: list[Line] = []
    for item in Document(io.BytesIO(content)).iter_inner_content():   # paragraphs and tables in reading order
        if isinstance(item, Table):
            lines += _table_lines([[c.text for c in row.cells] for row in item.rows], 1)
        elif item.text.strip():
            style = (item.style.name or "") if item.style is not None else ""
            lines.append(Line(" ".join(item.text.split()), 1, "heading" if style.startswith(("Heading", "Title")) else "text"))
    return lines        # DOCX has no fixed pages: everything is reported as page 1


def parse(filename: str, content: bytes) -> list[Line]:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        lines = _pdf(content)
    elif ext == "docx":
        lines = _docx(content)
    elif ext in ("png", "jpg", "jpeg", "tif", "tiff"):
        lines = _ocr(Image.open(io.BytesIO(content)), 1)
    else:
        raise UnsupportedFile("supported files: .pdf, .docx, .png, .jpg, .tif")
    if not lines:
        raise UnsupportedFile("no text found in the file")
    return lines
