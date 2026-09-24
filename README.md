# local-knowledge-assistant

Enterprise AI-Powered Personal Knowledge Engine (PKE) — a local-first AI assistant
that ingests PDFs, Markdown, HTML, DOCX and text files, builds a lightweight
knowledge graph alongside a vector index, and answers questions with a
hybrid RAG pipeline. Full design in `SRS of PKE-1.pdf`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # or: pip install -r requirements.txt

cp .env.example .env           # edit as needed; .env is never committed
python -m app.db.init_db       # creates data/pke.sqlite3 with all tables + indexes

pytest                          # run the test suite
```

## Status

- [x] Day 1 — Environment configuration & database initialization
      (`app/config.py`, `app/db/base.py`, `app/models/`, `app/db/init_db.py`)
- [x] Day 1 — Multi-format async ingestion & header-aware chunking
      (`app/ingestion/` — parsers for PDF/MD/HTML/DOCX/TXT, layout stripping,
      512-token header-aware chunker; `pytest tests/test_ingestion.py`, 15 passing)
- [ ] Day 1 — Dual SHA-256 deduplication
- [ ] Day 1 — Vector indexing & spaCy graph construction
- [ ] Day 2 — Hybrid search, graph expansion, two-pipeline architecture
- [ ] Day 3 — API, UI, benchmarking, deployment
