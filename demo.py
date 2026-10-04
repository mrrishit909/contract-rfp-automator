"""Run the real pipeline (OpenAI + Qdrant) on the 52-page agreement and record the result for the static demo.

    OPENAI_API_KEY in .env   ->   python demo.py   ->   web/public/data/document.json   (cost printed at the end)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from app.analyze import analyze
from app.index import HybridIndex, make_client
from app.llm import CHAT_MODEL, EMBED_DIM, EMBED_MODEL, OpenAI

HERE = Path(__file__).parent
PDF = HERE / "tests" / "fixtures" / "reynolds_tsa_52_pages.pdf"
PERSPECTIVE = "Reynolds Consumer Products Inc. (the “Company” or “RCP”), which mainly receives the transition services"


async def main() -> None:
    for line in (HERE / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            os.environ.setdefault(*line.split("=", 1))
    llm = OpenAI()
    index = HybridIndex(make_client(), llm, EMBED_DIM)
    started = time.time()
    doc = await analyze("0" * 32, PDF.name, PDF.read_bytes(), llm, index, PERSPECTIVE)
    doc.update(id="0" * 32, filename=PDF.name, status="ready", usage=llm.spent, seconds=round(time.time() - started, 1),
               models={"chat": CHAT_MODEL, "embedding": EMBED_MODEL})
    out = HERE / "web" / "public" / "data" / "document.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False))
    failed = sum(1 for c in doc["clauses"] if c["review"].get("failed"))
    print(json.dumps({"pages": doc["pages"], "clauses": len(doc["clauses"]), "scorecard": doc["scorecard"], "failed": failed,
                      "seconds": doc["seconds"], "usage": llm.spent}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
