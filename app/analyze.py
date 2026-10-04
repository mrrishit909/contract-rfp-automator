"""The pipeline: parse -> chunk by clause -> index -> review every clause against the playbook -> scorecard."""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, model_validator

from .chunk import Clause, chunk, definitions
from .diff import word_diff
from .index import HybridIndex
from .llm import LLM, LLMError
from .parse import parse
from .prompts import SCHEMA, review_messages

PLAYBOOK = Path(__file__).resolve().parent.parent / "playbook.json"
MIN_CHARS = 80          # shorter fragments (signature lines, stray headings) are not worth a model call
CONCURRENCY = 8


class Review(BaseModel):
    clause_type: str
    rule_id: str | None
    risk: Literal["Low", "Medium", "High"]
    explanation: str
    suggested_text: str | None

    @model_validator(mode="after")
    def _risky_needs_a_redraft(self) -> Review:
        if self.risk != "Low" and not (self.suggested_text or "").strip():
            raise ValueError("Medium and High reviews must include suggested_text")
        if self.risk == "Low":
            self.suggested_text = None
        return self


def load_playbook(path: Path = PLAYBOOK) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def rule_text(rule: dict[str, str]) -> str:
    return f"{rule['title']}. Standard: {rule['standard']} Red flags: {rule['red_flags']}"


async def review_clause(llm: LLM, index: HybridIndex, playbook: dict[str, Any], perspective: str, clause: Clause,
                        defined: dict[str, str]) -> dict[str, Any]:
    if len(clause.text) < MIN_CHARS:
        return Review(clause_type="other", rule_id=None, risk="Low", explanation="Too short to assess.", suggested_text=None).model_dump()
    rules = {r["id"]: r for r in playbook["rules"]}
    hits = await index.search("playbook", f"{clause.title}. {clause.text[:1500]}", k=3)
    candidates = [rules[i] for i, _ in hits if i in rules]
    used = {t: d[:400] for t, d in defined.items() if re.search(rf"\b{re.escape(t)}\b", clause.text) and d not in clause.text}
    messages = review_messages(perspective, {"number": clause.number, "title": clause.title, "section": clause.section, "text": clause.text},
                               candidates, dict(list(used.items())[:6]))
    error = ""
    for _ in range(2):          # the schema is enforced by the API; this guards the rules a schema cannot express
        try:
            review = Review.model_validate(await llm.chat_json(messages, SCHEMA, "clause_review"))
        except (ValidationError, LLMError) as exc:
            error = str(exc)[:200]
            continue
        if review.rule_id not in {c["id"] for c in candidates}:
            review.rule_id = None
        out = review.model_dump()
        if review.suggested_text:
            out["diff"] = word_diff(clause.text, review.suggested_text)
        return out
    return {"clause_type": "other", "rule_id": None, "risk": "Medium", "suggested_text": None, "failed": True,
            "explanation": f"Automatic review failed and this clause needs a human read. ({error})"}


def scorecard(clauses: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(c["review"]["risk"] for c in clauses if c.get("review"))
    by_rule: dict[str, str] = {}
    order = {"Low": 0, "Medium": 1, "High": 2}
    for c in clauses:
        r = c.get("review") or {}
        if r.get("rule_id") and order[r["risk"]] >= order[by_rule.get(r["rule_id"], "Low")]:
            by_rule[r["rule_id"]] = r["risk"]
    return {"High": counts["High"], "Medium": counts["Medium"], "Low": counts["Low"],
            "overall": "High" if counts["High"] else "Medium" if counts["Medium"] else "Low", "by_rule": by_rule}


async def ensure_playbook(index: HybridIndex, playbook: dict[str, Any]) -> None:
    if not index._corpus("playbook"):
        await index.add("playbook", {r["id"]: rule_text(r) for r in playbook["rules"]})


async def analyze(doc_id: str, filename: str, content: bytes, llm: LLM, index: HybridIndex, perspective: str | None = None,
                  playbook: dict[str, Any] | None = None) -> dict[str, Any]:
    playbook = playbook or load_playbook()
    perspective = perspective or playbook["perspective"]
    lines = await asyncio.to_thread(parse, filename, content)          # parsing and OCR are CPU work: keep the loop free
    clauses = chunk(lines)
    await ensure_playbook(index, playbook)
    await index.add(doc_id, {c.id: f"{c.title}. {c.text}" for c in clauses})
    defined = definitions(clauses)
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(c: Clause) -> dict[str, Any]:
        async with gate:
            return {**c.as_dict(), "review": await review_clause(llm, index, playbook, perspective, c, defined), "applied": False}

    reviewed = list(await asyncio.gather(*(one(c) for c in clauses)))
    return {"pages": max(ln.page for ln in lines), "clauses": reviewed, "scorecard": scorecard(reviewed),
            "perspective": perspective, "playbook": {"name": playbook["name"], "rules": playbook["rules"]}}


def export_text(doc: dict[str, Any]) -> str:
    """The contract as plain text with every applied redraft in place."""
    parts = []
    for c in doc["clauses"]:
        head = " ".join(x for x in (c["number"], c["title"]) if x)
        body = c["review"]["suggested_text"] if c["applied"] and c["review"].get("suggested_text") else c["text"]
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts) + "\n"
