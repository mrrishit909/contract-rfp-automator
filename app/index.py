"""Hybrid retrieval: vector similarity (Qdrant) + keyword match (BM25), fused with reciprocal rank fusion.

Vectors find clauses that mean the same thing in different words; BM25 finds the exact defined term or section number.
Qdrant runs embedded on disk by default, or as a server when QDRANT_URL is set (see docker-compose.yml).
"""
from __future__ import annotations

import math
import os
import re
import uuid
from collections import Counter
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from .llm import LLM

NAMESPACE = uuid.UUID("6f1d2c1e-8a57-4a0b-9d3e-2b7c5f0a9e11")
STOP = {"the", "of", "and", "to", "a", "in", "or", "by", "for", "is", "be", "shall", "any", "such", "with", "as", "that", "this", "its"}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()) if t not in STOP]


def bm25(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Okapi BM25 over a small in-memory corpus. Returns (doc index, score) for docs that share a term, best first."""
    toks = [tokens(d) for d in docs]
    avg = sum(map(len, toks)) / max(1, len(toks))
    df = Counter(t for d in toks for t in set(d))
    scores = []
    for i, d in enumerate(toks):
        tf = Counter(d)
        s = sum(math.log(1 + (len(docs) - df[q] + 0.5) / (df[q] + 0.5)) * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(d) / avg))
                for q in set(tokens(query)) if q in tf)
        if s > 0:
            scores.append((i, s))
    return sorted(scores, key=lambda x: (-x[1], x[0]))


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: an item's score is the sum of 1/(k + rank) over the lists it appears in."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1 / (k + rank)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


def make_client() -> QdrantClient:
    if os.environ.get("QDRANT_URL"):
        return QdrantClient(url=os.environ["QDRANT_URL"])
    return QdrantClient(path=os.path.join(os.environ.get("DATA_DIR", "data"), "qdrant"))


class HybridIndex:
    """One Qdrant collection; items are grouped by `group` (a document id, or "playbook")."""

    def __init__(self, client: QdrantClient, llm: LLM, dim: int, collection: str = "clauses") -> None:
        self.client, self.llm, self.collection = client, llm, collection
        self.texts: dict[str, dict[str, str]] = {}      # group -> {item id: text}, the BM25 corpus
        if not client.collection_exists(collection):
            client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    async def add(self, group: str, items: dict[str, str]) -> None:
        vectors = await self.llm.embed(list(items.values()))
        self.client.upsert(self.collection, points=[
            PointStruct(id=str(uuid.uuid5(NAMESPACE, f"{group}:{item}")), vector=v, payload={"group": group, "item": item, "text": text})
            for (item, text), v in zip(items.items(), vectors)])
        self.texts[group] = dict(items)

    def _corpus(self, group: str) -> dict[str, str]:
        if group not in self.texts:                       # after a restart: rebuild the keyword corpus from Qdrant
            points, _ = self.client.scroll(self.collection, limit=10_000, with_payload=True, scroll_filter=Filter(
                must=[FieldCondition(key="group", match=MatchValue(value=group))]))
            self.texts[group] = {str(p.payload["item"]): str(p.payload["text"]) for p in points if p.payload}
        return self.texts[group]

    async def search(self, group: str, query: str, k: int = 5) -> list[tuple[str, float]]:
        corpus = self._corpus(group)
        if not corpus or not query.strip():
            return []
        ids = list(corpus)
        vector = (await self.llm.embed([query]))[0]
        hits = self.client.query_points(self.collection, query=vector, limit=20, query_filter=Filter(
            must=[FieldCondition(key="group", match=MatchValue(value=group))])).points
        semantic = [str(h.payload["item"]) for h in hits if h.payload]
        keyword = [ids[i] for i, _ in bm25(query, list(corpus.values()))[:20]]
        return rrf([semantic, keyword])[:k]
