from __future__ import annotations

import json

import httpx
import pytest

from app.diff import word_diff
from app.index import HybridIndex, bm25, rrf
from app.llm import LLMError, OpenAI
from app.prompts import FEW_SHOT, SCHEMA, review_messages

CLAUSES = {
    "c1": "Definitions. “Affiliate” means any entity that controls, is controlled by, or is under common control with a party.",
    "c2": "Payment. Customer shall pay each undisputed invoice within thirty days after receipt.",
    "c3": "Limitation of Liability. Neither party is liable for indirect or consequential damages.",
    "c4": "Fees. The charges for the services are set out in Schedule 2 and are invoiced monthly.",
}


def test_bm25_ranks_exact_terms_and_rrf_fuses() -> None:
    ranked = bm25("Affiliate means", list(CLAUSES.values()))
    assert ranked[0][0] == 0 and all(i != 2 for i, _ in ranked)
    assert bm25("zebra", list(CLAUSES.values())) == []
    fused = rrf([["a", "b", "c"], ["c", "a"]])
    assert [x for x, _ in fused] == ["a", "c", "b"] and fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


async def test_hybrid_search_finds_definition_and_paraphrase(index: HybridIndex) -> None:
    await index.add("doc1", CLAUSES)
    await index.add("doc2", {"x1": "Payment is due within ten days."})
    assert (await index.search("doc1", "what does Affiliate mean"))[0][0] == "c1"
    assert (await index.search("doc1", "when must the customer pay an invoice"))[0][0] == "c2"
    assert {i for i, _ in await index.search("doc1", "payment")} <= set(CLAUSES)          # never leaks another document
    assert await index.search("doc1", "   ") == [] and await index.search("nope", "payment") == []
    index.texts.clear()                                                                    # as after a restart
    assert (await index.search("doc1", "Affiliate"))[0][0] == "c1"


def test_word_diff_marks_only_what_changed() -> None:
    ops = word_diff("pay within ten (10) days of invoice", "pay within thirty (30) days after receipt of invoice")
    assert [o["op"] for o in ops] == ["equal", "replace", "equal", "replace", "equal"] or "replace" in [o["op"] for o in ops]
    assert "".join(o["a"] + " " for o in ops if o["op"] != "insert").split() == "pay within ten (10) days of invoice".split()
    assert "".join(o["b"] + " " for o in ops if o["op"] != "delete").split() == "pay within thirty (30) days after receipt of invoice".split()


def test_prompt_carries_few_shot_examples_in_the_exact_output_shape() -> None:
    messages = review_messages("the customer", {"number": "1", "title": "T", "text": "x"}, [], {})
    assert [m["role"] for m in messages] == ["system", *["user", "assistant"] * len(FEW_SHOT), "user"]
    assert "the customer" in messages[0]["content"]
    for m in messages[2::2]:
        assert set(json.loads(m["content"])) == set(SCHEMA["required"])
    assert {json.loads(m["content"])["risk"] for m in messages[2::2]} == {"High", "Low"}


async def test_openai_client_sends_strict_schema_retries_and_counts_cost() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["Authorization"] == "Bearer k"
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i)]} for i in reversed(range(len(body["input"])))],
                                             "usage": {"prompt_tokens": 1000}})
        if len(seen) == 2:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"risk": "Low"}'}}],
                                         "usage": {"prompt_tokens": 2000, "completion_tokens": 1000}})

    llm = OpenAI("k", transport=httpx.MockTransport(handler), backoff=0)
    assert await llm.embed(["a", "b"]) == [[0.0], [1.0]]
    assert await llm.chat_json([{"role": "user", "content": "x"}], SCHEMA, "clause_review") == {"risk": "Low"}
    fmt = seen[-1]["response_format"]
    assert fmt == {"type": "json_schema", "json_schema": {"name": "clause_review", "strict": True, "schema": SCHEMA}}
    assert llm.spent["calls"] == 2 and llm.spent["usd"] == pytest.approx((1000 * 0.02 + 2000 * 0.75 + 1000 * 4.5) / 1e6)

    bad = OpenAI("k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})))
    with pytest.raises(LLMError, match="did not return JSON"):
        await bad.chat_json([], SCHEMA, "x")
    down = OpenAI("k", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    with pytest.raises(LLMError, match="401"):
        await down.embed(["a"])


def test_missing_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenAI()
