"""Fakes shared by the tests: no test needs an API key, a network or a running Qdrant server."""
from __future__ import annotations

import json
import math
import os
import zlib
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from app.index import HybridIndex, tokens

os.environ["APP_API_TOKEN"] = "test-token"
FIXTURE = Path(__file__).parent / "fixtures" / "reynolds_tsa_52_pages.pdf"
DIM = 256


class FakeLLM:
    """Deterministic stand-in for OpenAI: hashed bag-of-words embeddings and keyword-driven reviews."""

    def __init__(self) -> None:
        self.chats: list[list[dict[str, str]]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * DIM
            for t in tokens(text):
                v[zlib.crc32(t.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    async def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any], name: str) -> dict[str, Any]:
        self.chats.append(messages)
        asked = json.loads(messages[-1]["content"])
        text, rules = asked["clause"]["text"].lower(), asked["candidate_rules"]
        rule = rules[0]["id"] if rules else None
        if rule and "willful misconduct" in text:
            return {"clause_type": rule, "rule_id": rule, "risk": "High", "explanation": "Liable only for willful misconduct.",
                    "suggested_text": asked["clause"]["text"].replace("WILLFUL MISCONDUCT", "NEGLIGENCE OR WILLFUL MISCONDUCT").replace("willful misconduct", "negligence or willful misconduct")}
        if rule and "sole discretion" in text:
            return {"clause_type": rule, "rule_id": rule, "risk": "Medium", "explanation": "One party decides alone.",
                    "suggested_text": asked["clause"]["text"].replace("sole discretion", "reasonable discretion")}
        return {"clause_type": rule or "other", "rule_id": rule, "risk": "Low", "explanation": "Meets the standard.", "suggested_text": None}


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def index(llm: FakeLLM) -> HybridIndex:
    return HybridIndex(QdrantClient(":memory:"), llm, DIM)


@pytest.fixture(scope="session")
def pdf() -> bytes:
    return FIXTURE.read_bytes()
