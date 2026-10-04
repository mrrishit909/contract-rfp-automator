"""Async OpenAI client with structured (JSON-schema) outputs, retries and a running cost tally.

Tests replace this class with a fake that has the same two methods, so no test needs a key or a network.
Prices are $ per 1M tokens from developers.openai.com/api/docs/pricing, checked 2026-10-01.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol

import httpx

EMBED_MODEL, CHAT_MODEL, EMBED_DIM = "text-embedding-3-small", "gpt-5.4-mini", 1536
PRICES = {EMBED_MODEL: (0.02, 0.0), CHAT_MODEL: (0.75, 4.50)}


class LLM(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any], name: str) -> dict[str, Any]: ...


class LLMError(RuntimeError):
    pass


class OpenAI:
    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None, backoff: float = 1.0) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.client = httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=120, transport=transport,
                                        headers={"Authorization": f"Bearer {key}"})
        self.backoff = backoff
        self.spent = {"usd": 0.0, "in_tokens": 0, "out_tokens": 0, "calls": 0}

    async def _post(self, path: str, body: dict[str, Any], model: str, tries: int = 5) -> dict[str, Any]:
        for attempt in range(tries):
            r = await self.client.post(path, json=body)
            if r.status_code in (429, 500, 502, 503) and attempt < tries - 1:
                await asyncio.sleep(self.backoff * min(2 ** attempt, 20))
                continue
            if r.status_code != 200:
                raise LLMError(f"OpenAI API error {r.status_code}: {r.text[:300]}")
            data: dict[str, Any] = r.json()
            usage = data.get("usage", {})
            i, o = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            self.spent["in_tokens"] += i
            self.spent["out_tokens"] += o
            self.spent["calls"] += 1
            self.spent["usd"] += (i * PRICES[model][0] + o * PRICES[model][1]) / 1e6
            return data
        raise AssertionError("unreachable")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for k in range(0, len(texts), 96):
            data = await self._post("/embeddings", {"model": EMBED_MODEL, "input": [t[:8000] for t in texts[k:k + 96]]}, EMBED_MODEL)
            out += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return out

    async def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any], name: str) -> dict[str, Any]:
        """One call that must come back as JSON matching `schema` (OpenAI structured outputs, strict mode)."""
        data = await self._post("/chat/completions", {
            "model": CHAT_MODEL, "messages": messages, "reasoning_effort": "low",
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        }, CHAT_MODEL)
        try:
            parsed: dict[str, Any] = json.loads(data["choices"][0]["message"]["content"])
            return parsed
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("the model did not return JSON") from exc
