"""Semantic chunking: cut the document where the contract itself starts a new clause, not every N tokens.

A clause begins at a heading the drafters wrote: "ARTICLE IV", "Section 5.2 Payment.", "7.3 Limitation of Liability",
or a short stand-alone title line. Lettered sub-clauses "(a)", "(ii)" stay inside their parent clause.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from .parse import Line

MAX_CHARS = 6000         # a clause longer than this is split at paragraph breaks (keeps prompts bounded)
ARTICLE = re.compile(r"^(?:ARTICLE|Article)\s+([IVXLC]+|\d+)\b[.:\s–-]*(.*)$")
SECTION = re.compile(r"^(?:SECTION|Section)\s+(\d+(?:\.\d+)*)[.:]?\s*(.*)$")
NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?=[A-Z“\"(\[])(.*)$")
SUBCLAUSE = re.compile(r"^(\([a-z]{1,4}\)|\(\d{1,2}\)|[•·▪-])\s")
TOC_ENTRY = re.compile(r"^.{2,90}?[\s.]\s*(?:[A-Z]{1,2}-)?\d{1,3}$")


@dataclass
class Clause:
    id: str
    number: str        # "5.2", "IV", or "" for unnumbered headings
    title: str
    section: str       # the enclosing article / top-level section, for context
    text: str
    page: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _title_line(text: str) -> bool:
    """A short stand-alone heading such as "Termination for Convenience" or "PRELIMINARY STATEMENT"."""
    words = text.split()
    if not (1 <= len(words) <= 8) or len(text) > 70 or text[-1] in ".;:,” " or not text[0].isupper():
        return False
    if any(ch in text for ch in "|$@") or re.search(r"\d", text):
        return False
    small = {"of", "and", "the", "for", "to", "in", "or", "by", "on", "with", "a", "an", "&"}
    return text.isupper() or all(w[0].isupper() or w.lower() in small for w in words if w[0].isalpha())


def _heading(text: str) -> tuple[str, str, str] | None:
    """(level, number, rest-of-line) when the line opens a clause."""
    if m := ARTICLE.match(text):
        return "article", m.group(1), m.group(2).strip()
    if m := SECTION.match(text):
        return "clause", m.group(1), m.group(2).strip()
    if m := NUMBERED.match(text):
        return ("article" if "." not in m.group(1).rstrip("0").rstrip(".") else "clause"), m.group(1), m.group(2).strip()
    return None


def _roman(numeral: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    nums = [values[ch] for ch in numeral]
    return sum(-n if i + 1 < len(nums) and n < nums[i + 1] else n for i, n in enumerate(nums))


def _split_title(rest: str) -> tuple[str, str]:
    """"Payment Terms. Customer shall pay ..." -> ("Payment Terms", "Customer shall pay ...")."""
    m = re.match(r"^(.{3,80}?)[.:](?:\s+|$)(.*)$", rest)
    sentence = r"\b(shall|will|may|must|is|are|means|has|have|includes?)\b|[“”\"]"
    if m and len(m.group(1).split()) <= 10 and not re.search(sentence, m.group(1)):
        return m.group(1).strip(), m.group(2).strip()
    if len(rest) <= 80 and _title_line(rest):
        return rest, ""
    return "", rest


def _drop_toc(lines: list[Line]) -> list[Line]:
    """A table of contents is a run of heading-like lines that end in a page number: drop runs of four or more."""
    def toc(ln: Line) -> bool:
        return bool(TOC_ENTRY.match(ln.text)) and (_heading(ln.text) is not None or len(ln.text) < 70) or ln.text in ("Article", "Page", "Section")
    out: list[Line] = []
    run: list[Line] = []
    for ln in [*lines, Line("", 0)]:
        if ln.page and ln.kind == "text" and toc(ln):
            run.append(ln)
            continue
        if len(run) < 4:
            out += run
        run = []
        out.append(ln)
    return out[:-1]


def chunk(lines: list[Line]) -> list[Clause]:
    lines = _drop_toc(lines)
    width = median(len(ln.text) for ln in lines if ln.kind == "text") if lines else 80
    clauses: list[Clause] = []
    section = ""
    cur: dict[str, Any] = {"number": "", "title": "Preamble", "section": "", "page": lines[0].page if lines else 1, "paras": [[]]}

    def close() -> None:
        paras = [" ".join(p) for p in cur["paras"] if p]
        if not paras:
            return
        parts, size = [[]], 0   # type: ignore[var-annotated]
        for p in paras:
            if size + len(p) > MAX_CHARS and parts[-1]:
                parts.append([])
                size = 0
            parts[-1].append(p)
            size += len(p)
        for i, part in enumerate(parts):
            title = cur["title"] + (f" (part {i + 1})" if len(parts) > 1 else "")
            clauses.append(Clause(f"c{len(clauses) + 1:04d}", cur["number"], title, cur["section"], "\n".join(part), cur["page"]))

    last_article = 0
    for i, ln in enumerate(lines):
        nxt = lines[i + 1].text if i + 1 < len(lines) else ""
        prev = cur["paras"][-1][-1] if cur["paras"][-1] else ""
        article_waiting = cur["title"].startswith("Article ") and cur["title"] == f"Article {cur['number']}" and not prev
        if article_waiting and ln.kind == "text" and _heading(ln.text) is None and _title_line(ln.text):
            section = cur["title"] = cur["section"] = f"{cur['title']} {ln.text}"       # "ARTICLE II" + its title line
            continue
        head = _heading(ln.text) if ln.kind == "text" else None
        if head is not None and not ARTICLE.match(ln.text) and not SECTION.match(ln.text):
            # a bare number must continue the document's own sequence: this rejects "75 Park Plaza" and stray page
            # numbers. "5.4 ..." is also accepted when nothing was seen yet (a single scanned page from mid-document).
            major = int(head[1].split(".")[0])
            plain = "." not in head[1]
            in_sequence = 0 < major <= last_article + 2 or (last_article == 0 and not plain)
            if not in_sequence or (plain and not _split_title(head[2])[0]):
                head = None
        after_sentence = not prev or prev[-1] in ".;:”\")"
        if ln.kind == "heading" or (head is None and ln.kind == "text" and after_sentence and _title_line(ln.text) and len(nxt) > 60):
            head = ("clause", "", ln.text)
        if head is not None:
            level, number, rest = head
            title, body = _split_title(rest) if number else (rest, "")
            close()
            if level == "article":
                last_article = int(number.split(".")[0]) if number.split(".")[0].isdigit() else _roman(number)
                label = f"Article {number}" if ARTICLE.match(ln.text) else number
                section = f"{label} {title}".strip()
                cur = {"number": number, "title": section, "section": section, "page": ln.page, "paras": [[body] if body else []]}
            else:
                if number:
                    last_article = max(last_article, int(number.split(".")[0]))
                cur = {"number": number, "title": title or (f"Section {number}" if number else "Untitled"), "section": section,
                       "page": ln.page, "paras": [[body] if body else []]}
            continue
        new_para = ln.kind == "table" or bool(SUBCLAUSE.match(ln.text)) or (bool(prev) and len(prev) < 0.6 * width and prev[-1] in ".:;")
        if new_para and cur["paras"][-1]:
            cur["paras"].append([])
        cur["paras"][-1].append(ln.text)
    close()
    return clauses


DEFINITION = re.compile(r"[“\"]([A-Z][^”\"]{1,60})[”\"]\s+(?:shall mean|means|has the meaning|shall have the meaning|refers to)[^.]{5,400}\.")


def definitions(clauses: list[Clause]) -> dict[str, str]:
    """Defined terms anywhere in the document: {"Affiliate": "“Affiliate” means ..."}; the first definition wins."""
    found: dict[str, str] = {}
    for c in clauses:
        for m in DEFINITION.finditer(c.text):
            found.setdefault(m.group(1), m.group(0))
    return found
