"""Async API. Upload returns at once with a document id; the analysis runs in the background and the client polls."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analyze import analyze, export_text, scorecard
from .index import HybridIndex, make_client
from .llm import EMBED_DIM, LLM, OpenAI

MAX_BYTES = 25 * 1024 * 1024
ALLOWED = (".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
app = FastAPI(title="Contract & RFP automator")
_tasks: set[asyncio.Task[None]] = set()
_state: dict[str, Any] = {}


def data_dir() -> Path:
    d = Path(os.environ.get("DATA_DIR", "data")) / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def services() -> tuple[LLM, HybridIndex]:
    """Created on first use so tests can install fakes with configure()."""
    if "llm" not in _state:
        llm = OpenAI()
        configure(llm, HybridIndex(make_client(), llm, EMBED_DIM))
    return _state["llm"], _state["index"]


def configure(llm: LLM, index: HybridIndex) -> None:
    _state["llm"], _state["index"] = llm, index


def auth(authorization: Annotated[str, Header()] = "") -> None:
    if not hmac.compare_digest(authorization.encode(), f"Bearer {os.environ['APP_API_TOKEN']}".encode()):
        raise HTTPException(401, "missing or wrong API token")


Auth = Annotated[None, Depends(auth)]


# ponytail: one JSON file per document, single process. Move to Postgres when there are several users or workers.
def _path(doc_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", doc_id):
        raise HTTPException(404, "no such document")
    return data_dir() / f"{doc_id}.json"


def load(doc_id: str) -> dict[str, Any]:
    path = _path(doc_id)
    if not path.exists():
        raise HTTPException(404, "no such document")
    doc: dict[str, Any] = json.loads(path.read_text())
    return doc


def save(doc: dict[str, Any]) -> None:
    path = _path(doc["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False))
    tmp.replace(path)


async def _run(doc: dict[str, Any], content: bytes, perspective: str | None) -> None:
    llm, index = services()
    try:
        doc.update(await analyze(doc["id"], doc["filename"], content, llm, index, perspective), status="ready")
        spent = getattr(llm, "spent", None)
        if spent:
            doc["usage"] = dict(spent)
    except Exception as exc:          # the client must see a failed status, not a document stuck on "processing"
        doc.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
    save(doc)


@app.post("/api/documents", status_code=202)
async def upload(file: UploadFile, _: Auth, perspective: Annotated[str | None, Form(max_length=300)] = None) -> dict[str, str]:
    name = Path(file.filename or "").name
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(415, f"supported files: {', '.join(ALLOWED)}")
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "file larger than 25 MB")
    doc = {"id": uuid.uuid4().hex, "filename": name, "status": "processing", "created_at": datetime.now(UTC).isoformat()}
    save(doc)
    task = asyncio.create_task(_run(doc, content, perspective))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"id": doc["id"], "status": "processing"}


@app.get("/api/documents")
async def documents(_: Auth) -> list[dict[str, Any]]:
    docs = [json.loads(p.read_text()) for p in data_dir().glob("*.json")]
    return sorted(({k: d.get(k) for k in ("id", "filename", "status", "created_at", "pages", "scorecard")} for d in docs),
                  key=lambda d: d["created_at"], reverse=True)


@app.get("/api/documents/{doc_id}")
async def document(doc_id: str, _: Auth) -> dict[str, Any]:
    return load(doc_id)


class Apply(BaseModel):
    applied: bool


@app.post("/api/documents/{doc_id}/clauses/{clause_id}/apply")
async def apply_draft(doc_id: str, clause_id: str, body: Apply, _: Auth) -> dict[str, Any]:
    """Accept (or undo) the AI redraft of one clause."""
    doc = load(doc_id)
    clause = next((c for c in doc.get("clauses", []) if c["id"] == clause_id), None)
    if clause is None or not clause["review"].get("suggested_text"):
        raise HTTPException(409, "this clause has no AI draft to apply")
    clause["applied"] = body.applied
    save(doc)
    return {"id": clause_id, "applied": body.applied}


@app.get("/api/documents/{doc_id}/search")
async def search(doc_id: str, q: str, _: Auth) -> list[dict[str, Any]]:
    load(doc_id)
    index = services()[1]
    return [{"id": cid, "score": round(score, 4)} for cid, score in await index.search(doc_id, q[:500], k=8)]


@app.get("/api/documents/{doc_id}/export", response_class=PlainTextResponse)
async def export(doc_id: str, _: Auth) -> str:
    doc = load(doc_id)
    if doc["status"] != "ready":
        raise HTTPException(409, "document is not ready")
    return export_text(doc)


__all__ = ["app", "configure", "scorecard"]
_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():  # pragma: no cover - only in the built image
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
