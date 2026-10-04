"""Prompt templates. The few-shot examples are part of the prompt: they show the model the exact JSON shape,
the tone of the explanation and what a usable redraft looks like, which keeps outputs consistent across clauses."""
from __future__ import annotations

import json
from typing import Any

SYSTEM = """You are a contracts reviewer. You compare ONE clause of a contract or RFP with the company's playbook rules \
and report the risk to the company. You act for: {perspective}.

Rules for your answer:
- Use only the candidate playbook rules given with the clause. If none of them is about this clause's subject, set \
rule_id to null and risk to "Low", and say that no playbook rule covers it.
- risk: "Low" = meets the rule's standard (or no rule applies); "Medium" = departs from the standard but is commonly \
accepted or easy to negotiate; "High" = matches a red flag or leaves the company materially exposed.
- explanation: two or three plain sentences. Quote the words in the clause that drive the rating. Do not invent facts \
that are not in the clause.
- suggested_text: for Medium and High, a complete replacement clause that would meet the standard, keeping the \
clause's defined terms, numbering and style. For Low, null.
- Definitions shown under "defined_terms" are context only; do not rate them."""

RULE_A = {"id": "payment_terms", "title": "Payment terms", "standard": "Undisputed invoices are payable 30 days or more after receipt.", "red_flags": "Payment due in under 30 days; late interest above 1% per month."}
RULE_B = {"id": "governing_law", "title": "Governing law and venue", "standard": "Delaware or New York law.", "red_flags": "Law or venue outside the United States is a high risk."}

FEW_SHOT: list[tuple[dict[str, Any], dict[str, Any]]] = [
    ({"clause": {"number": "4.2", "title": "Payment", "text": "Customer shall pay each invoice within ten (10) days of the invoice date. Overdue amounts bear interest at 2% per month."},
      "candidate_rules": [RULE_A], "defined_terms": {}},
     {"clause_type": "payment_terms", "rule_id": "payment_terms", "risk": "High",
      "explanation": "The clause requires payment \"within ten (10) days of the invoice date\", well under the 30-day standard, and charges \"2% per month\" on late amounts, double the 1% limit. There is no right to withhold disputed amounts.",
      "suggested_text": "Customer shall pay each undisputed invoice within thirty (30) days after receipt. Customer may withhold any amount it disputes in good faith. Overdue undisputed amounts bear interest at 1% per month."}),
    ({"clause": {"number": "12.1", "title": "Governing Law", "text": "This Agreement is governed by the laws of the State of New York, without regard to its conflict of laws rules."},
      "candidate_rules": [RULE_B, RULE_A], "defined_terms": {}},
     {"clause_type": "governing_law", "rule_id": "governing_law", "risk": "Low",
      "explanation": "The clause chooses \"the laws of the State of New York\", which is one of the two laws the playbook accepts.",
      "suggested_text": None}),
    ({"clause": {"number": "12.9", "title": "Counterparts", "text": "This Agreement may be executed in counterparts, each of which is an original."},
      "candidate_rules": [RULE_B], "defined_terms": {}},
     {"clause_type": "other", "rule_id": None, "risk": "Low",
      "explanation": "This is a standard execution clause. No playbook rule covers it.", "suggested_text": None}),
]

SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["clause_type", "rule_id", "risk", "explanation", "suggested_text"],
    "properties": {
        "clause_type": {"type": "string", "description": "The playbook rule id this clause is about, or 'definitions' or 'other'."},
        "rule_id": {"type": ["string", "null"]},
        "risk": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "explanation": {"type": "string"},
        "suggested_text": {"type": ["string", "null"]},
    },
}


def review_messages(perspective: str, clause: dict[str, str], rules: list[dict[str, str]], defined: dict[str, str]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM.format(perspective=perspective)}]
    for question, answer in FEW_SHOT:
        messages += [{"role": "user", "content": json.dumps(question, ensure_ascii=False)},
                     {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
    messages.append({"role": "user", "content": json.dumps(
        {"clause": clause, "candidate_rules": rules, "defined_terms": defined}, ensure_ascii=False)})
    return messages
