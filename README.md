# Contract & RFP automator

Upload a contract or RFP (PDF, DOCX or a scanned image). The service splits it into clauses, compares each clause with a
company "playbook" of acceptable terms, and returns a risk score (Low / Medium / High), the reason, and a suggested
replacement clause with a side-by-side diff. One click applies the draft; the revised text can be exported.

Built from the "AI-Driven Contract & RFP Automator" blueprint as an MVP. **Not legal advice.**
**Live demo (recorded run, read-only):** https://mrrishit909.github.io/projects/contract-rfp-automator/demo/

## Run it

```bash
printf 'OPENAI_API_KEY=%s\nAPP_API_TOKEN=%s\n' "sk-...your key..." "$(openssl rand -hex 24)" > .env   # never committed
docker compose up --build -d          # API + workspace on http://localhost:8000, Qdrant vector database
docker compose run --rm test          # 16 tests, 97% coverage; no key or network needed (the model is faked)
```

Without Docker: `python3 -m venv venv && venv/bin/pip install -e ".[dev]"`, then `venv/bin/pytest`,
`venv/bin/uvicorn app.api:app` (Qdrant then runs embedded on disk under `data/`), and `python demo.py` to re-record the demo.
OCR needs Tesseract (`TESSERACT_CMD=/path/to/tesseract` if it is not on the PATH).

## Pipeline

| Step | File | What it does |
|---|---|---|
| Parse | `app/parse.py` | PDF text and ruled tables with pdfplumber; DOCX paragraphs, heading styles and tables in reading order; pages with no text layer and image files go through Tesseract OCR. Running headers, footers and page numbers are removed. |
| Chunk by clause | `app/chunk.py` | Cuts where the contract starts a clause ("ARTICLE IV", "Section 5.2 Payment.", "7.3 ...", a stand-alone title line), never at a fixed token count. Drops the table of contents; keeps "(a)", "(ii)" sub-clauses with their parent; splits only clauses over 6,000 characters, at paragraph breaks. Also collects defined terms ("“Affiliate” means ..."). |
| Index | `app/index.py` | `text-embedding-3-small` vectors in Qdrant plus BM25 keyword scores, fused by reciprocal rank fusion. Used to pick the playbook rules relevant to a clause and for search inside the document. |
| Review | `app/analyze.py`, `app/prompts.py` | One model call per clause with the three best-matching playbook rules and the definitions of the defined terms it uses. The reply is forced into a JSON schema (OpenAI structured outputs, strict), shown by three few-shot examples, then validated with pydantic (a Medium/High rating must carry a redraft; a rule id must be one of the candidates). A reply that fails twice marks the clause "needs a human read". |
| Diff | `app/diff.py` | Word-level diff between the clause and the redraft. |
| API | `app/api.py` | FastAPI, async. Upload returns `202` and an id; the analysis runs in the background (8 clauses at a time) and the client polls. Bearer token on every route; file type and size limits. |
| Workspace | `web/` | React + TypeScript, three panes: contents with risk dots, search and filter · the document with colour-coded clauses · the review with diff and "Apply AI draft". |
| Playbook | `playbook.json` | 15 invented example rules (payment terms, liability cap, governing law, ...), written from a services customer's side. Replace with your own. |

## What one real run produced

Document: the 52-page Transition Services Agreement between Reynolds Group Holdings and Reynolds Consumer Products
(SEC exhibit 10.22, January 2020; from CUAD v1, The Atticus Project, CC BY 4.0), reviewed for Reynolds Consumer Products.
Models: `gpt-5.4-mini` and `text-embedding-3-small`. Recorded by `python demo.py` on 2026-10-04:

- 52 pages → 74 clauses in **32 seconds**, 150 API calls, **$0.18**.
- **21 High, 11 Medium, 42 Low.** Examples: 5.4 Payment (due by month-end rather than 30 days after receipt),
  9.4 Limitations (cap is the fees for the one affected service, not 12 months of fees), 10.8 Governing Law (Illinois: Medium).
- No clause failed validation.

What that does **not** show: accuracy. Nobody with legal training checked these 74 reviews, and there is no labelled test set
behind them. Things seen on a read-through:

- In 5.4 the model says interest at "prime plus 2%" exceeds the playbook's 1% per month. That is an annual rate (under 1% a
  month at 2020 prime rates), so that part of the explanation is wrong.
- 8 of the 21 High ratings are fragments of the fee schedules, each flagged for the same "pass-through of actual third-party
  costs" with no cap. The flag matches the playbook, but it is one issue counted eight times.
- A second run of the same file (through Docker) gave 20 High instead of 21: the model is not perfectly repeatable.

## Not done, and caveats

- Clause detection is rule-based and was tuned on five SEC contracts. It handles numbered and "Article/Section" layouts well;
  contracts with a table of contents that has no page numbers, or with unusual numbering, come out with some wrong headings.
- Schedules that are tables without ruling lines are read as running text, not as tables.
- Storage is one JSON file per document and one process; no user accounts (one shared API token), no audit trail,
  no document versions beyond "draft applied / not applied". Export is plain text, not a redlined DOCX.
- OCR was tested on a rendered page of the same PDF (over 90% of words recovered), not on poor photocopies.
- Only OpenAI is wired in. Contract text is sent to the OpenAI API: do not upload documents you may not share with a third party.
