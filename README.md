# local-knowledge-assistant

Enterprise AI-Powered Personal Knowledge Engine (PKE) — a local-first AI assistant
that ingests PDFs, Markdown, HTML, DOCX and text files, builds a lightweight
knowledge graph alongside a vector index, and answers questions with a
hybrid RAG pipeline. Full design in `SRS of PKE-1.pdf`.

## How to run

### 1. One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"            # installs the app + dev/test tools, and registers the `kengine` command

cp .env.example .env               # edit as needed (e.g. OLLAMA_MODEL, LLM_BACKEND) — .env is never committed
python -m app.db.init_db           # creates data/pke.sqlite3 with all tables, indexes, and the FTS5 search index
```

Two independent swappable backends, each defaulting to local:

- **LLM** (`LLM_BACKEND` in `.env`): `local` (Ollama, default) or `groq` (cloud, needs `GROQ_API_KEY`).
- **Embeddings** (`EMBEDDING_BACKEND` in `.env`): `ollama` (local, default) or `cohere` (cloud, needs `COHERE_API_KEY`).
  Groq doesn't offer an embeddings endpoint, which is why this is a separate setting from the LLM one.

If you're using the local backends (the default for both), make sure Ollama is running and has the models pulled:

```bash
ollama serve                       # if not already running (the desktop app usually does this for you)
ollama pull llama3.2:3b            # or whatever OLLAMA_MODEL is set to
ollama pull nomic-embed-text       # only needed if EMBEDDING_BACKEND=ollama
```

If you'd rather not run Ollama at all, set both to their cloud options in `.env`:
```
LLM_BACKEND=groq
GROQ_API_KEY=...
EMBEDDING_BACKEND=cohere
COHERE_API_KEY=...
```
Trade-off, stated plainly: chunk text leaves the machine at both embed time and generation time once you do this —
the opposite of this project's local-first default. Fine if your documents aren't sensitive; not fine if they are.

### 2. Run the server + dashboard

```bash
uvicorn app.main:app --reload
```

- **Dashboard:** http://127.0.0.1:8000/ui/ — drag-and-drop file upload, a chat box with live streaming answers, and
  an interactive graph view.
- **Swagger UI:** http://127.0.0.1:8000/docs — try every endpoint directly; **ReDoc:** http://127.0.0.1:8000/redoc.

**Or the Streamlit chat frontend** (a second, separate frontend — not a replacement; only the graph view is
dashboard-only) — needs the API server above running first, then in another terminal:

```bash
pip install -e ".[streamlit]"
streamlit run streamlit_app.py
```

Opens at http://localhost:8501 — real chat bubbles, live token streaming, login/register, file upload sidebar,
citations and grounding-coverage shown under each answer. It's a plain script talking to the API over HTTP
(`requests`), exactly like the JS dashboard does — set `PKE_API_BASE_URL` if your API isn't at the default
`http://127.0.0.1:8000`.

**Or start both at once, one command, one terminal:**
```bash
./run_dev.sh
```
Still two separate processes underneath (Streamlit has no backend of its own) — this just launches the API in the
background and the frontend in the foreground, and Ctrl+C stops both together. Creates the database automatically
on first run if it doesn't exist yet.

### 3. Authenticate

Every API endpoint except `/api/auth/register` and `/api/auth/login` requires a `Bearer` access token.

```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"a long passphrase","full_name":"Your Name"}'
# -> {"access_token": "...", "access_token_expires_at": "...", "refresh_token": "...",
#     "refresh_token_expires_at": "...", "user_id": 1, "email": "you@example.com"}

curl -X POST http://127.0.0.1:8000/api/query \
  -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" \
  -d '{"query":"what does Ollama run models on?"}'
```

Two tokens come back: a short-lived **access token** (`ACCESS_TOKEN_EXPIRE_MINUTES`, default 60) sent on every
request, and a long-lived **refresh token** (`REFRESH_TOKEN_EXPIRE_DAYS`, default 30) sent only to
`POST /api/auth/refresh` to mint a new pair once the access token lapses — no full re-login needed until the
refresh token itself expires or is invalidated. The refresh token rotates on every use (the old one stops working
the moment a new one is issued), so a leaked refresh token is only usable once before the real client's next
refresh reveals the problem. Logging in issues a fresh pair and invalidates whatever was issued before it (one
active session per user — see `app/auth/auth_service.py`). Both the dashboard (`/ui/`) and the Streamlit frontend
keep both tokens client-side (`localStorage` and `st.session_state` respectively) and transparently call
`/api/auth/refresh` the moment a request comes back 401, retrying the original call once before ever dropping back
to the login screen. The CLI doesn't use any of this — see below.

### 4. Or use the CLI (no server needed)

```bash
kengine ingest ./some-notes.md
kengine query "what does Ollama run models on?"
kengine evaluate ./eval_cases.json --k 5        # retrieval quality (Recall@k, Precision@k, MRR) against labeled cases
kengine evaluate-context-precision ./golden.json --k 5   # Context Precision (LLM-judged) against {question, ground_truth} pairs
kengine benchmark-latency --n-chunks 250000     # vector query latency at scale
```

`eval_cases.json` format: `[{"query": "...", "relevant_chunk_ids": [12, 47]}, ...]` — you have to hand-label which
chunk_ids actually answer each query; there's no way around that, see the note under Benchmarking below.

The CLI bypasses HTTP auth entirely — it talks to the DB directly and runs against one local user, created
automatically on first use (`local@kengine`). The reasoning: on a personal, single-user, local-first tool, having
shell access to the machine already *is* the trust boundary; asking you to log in to your own CLI would add
friction for no real security gain. Pass `--user-id N` to any command to target a different (already-registered)
user instead.

### 5. Run the test suite

```bash
pytest                                              # all tests (~180), a few seconds, no Ollama needed — everything
                                                     # that would otherwise call a model is tested against fakes/mocks
pytest --cov=app --cov-report=term-missing          # same, with a per-file coverage report
pytest tests/test_hallucination_guardrails.py -v    # just the hallucination-guardrail validation suite
```

The suite never needs a live Ollama server or a real GROQ_API_KEY: the LLM and embedder are swapped for
deterministic test doubles (`FakeLLM`/`FakeEmbedder` in `tests/conftest.py`) everywhere except the two tests that
specifically check "what happens when nothing is listening" (`test_llm_backends.py`,
`test_embedder_raises_clear_error_when_ollama_unreachable`).

## Known limitations

- Citations link to the exact cited chunk's text, not the original uploaded
  file — ingestion never stores the original file bytes, only extracted
  text, so there's no original document to serve back.
- One active session per user: logging in again invalidates whatever
  access/refresh token pair was issued before it, rather than supporting
  multiple concurrent devices/sessions. A deliberate simplicity trade-off
  (see `app/auth/auth_service.py`), not an oversight — multi-session
  support would mean a sessions table instead of the single token pair
  stored directly on `UserAuth`.
- No password reset flow, no email verification, no rate limiting on
  login attempts. Real gaps for anything beyond personal/trusted use.

## Status

- [x] Day 1 — Environment configuration & database initialization
      (`app/config.py`, `app/db/base.py`, `app/models/`, `app/db/init_db.py`)
- [x] Day 1 — Multi-format async ingestion & header-aware chunking
      (`app/ingestion/` — parsers for PDF/MD/HTML/DOCX/TXT, layout stripping,
      512-token header-aware chunker)
- [x] Day 1 — Dual SHA-256 deduplication
      (`app/hashing.py`, `app/services/ingest_service.py` — file-level gate vs
      `Document.hash_checksum`, chunk-level gate vs `Chunk.chunk_hash`)
- [x] Day 1 — Vector indexing & spaCy graph construction
      (`app/embeddings/` — Ollama nomic-embed-text client + ChromaDB store;
      `app/graph/` — spaCy entity extraction + NetworkX co-occurrence graph;
      `app/services/embedding_service.py`, `graph_service.py`, `pipeline_service.py`)
- [x] Day 2 — Hybrid search, graph expansion, RRF fusion & grounding gate
      (`app/db/fts.py` — SQLite FTS5 index + sync triggers for BM25;
      `app/search/` — bm25_search, vector_search, rrf_fusion, grounding_gate;
      `app/graph/graph_expansion.py` — 1-hop NetworkX neighbor traversal;
      `app/services/retrieval_service.py` — Pipeline 1: runs all three
      sources concurrently, fuses with RRF, halts before any LLM call if
      context isn't sufficiently grounded)
- [x] Day 2 — Pipeline 2: Grounding Verifier & Local LLM Integration
      (`app/llm/` — OllamaLLM + GroqLLM backends behind one `LLMBackend`
      interface, fact-restricted prompt templates, post-generation
      sentence-level grounding verifier, interactive source citations;
      `app/services/generation_service.py` — refuses to touch the LLM at all
      if Pipeline 1 already refused)
- [x] Day 3 — FastAPI REST surface, OpenAPI/Swagger, `kengine` CLI, retrieval evaluation
      (`POST /api/ingest`, `POST /api/query` + `/query/stream`, `GET /api/graph` +
      `/graph/neighbors`; full OpenAPI spec + Swagger UI at `/docs`; real
      `kengine` console command — `ingest`, `query`, `evaluate`,
      `benchmark-latency` — registered via `[project.scripts]`;
      `app/evaluation/` — Recall@k, Precision@k, MRR against a labeled query
      set, run through the real BM25+vector+graph+RRF pipeline)
- [x] Day 3 — Web UI Dashboard & Network Graph Visualization
      (`app/static/` — single-page vanilla HTML/CSS/JS, no build step, served
      at `/ui` via FastAPI's `StaticFiles`; drag-and-drop multi-file upload;
      chat box streaming tokens live via hand-parsed SSE over a `fetch()`
      `ReadableStream` — `EventSource` can't send the POST body this endpoint
      needs; clickable citations resolving to the exact cited chunk's text;
      D3.js force-directed graph, draggable/zoomable, click a node for its
      1-hop neighbors; verified with a real `uvicorn` server over live HTTP,
      not just the ASGI TestClient — which is how a real, non-mocked
      DB-schema bug was actually caught)
- [x] Day 3 — Benchmarking, Test Suites & Latency Optimization
      (**95% statement coverage** across `app/` — `pytest --cov=app`, well
      above the 85% target, closing real gaps (the non-streaming `/query`
      endpoint had zero coverage before this pass, found by actually reading
      the report rather than assuming); **vector query latency at 250,000
      chunks**, actually run end-to-end (not estimated) via `kengine
      benchmark-latency --n-chunks 250000` — inserted 250,000 real vectors
      into Chroma (844.9s insert time) and queried 50 times:
      **mean 8.1ms, p50 7.7ms, p95 11.5ms, p99 25.7ms, max 25.7ms** —
      **PASS** against the NFR-2 target of <400ms, with over 15x headroom
      even at p99; **hallucination guardrails formally validated**
      end-to-end in `tests/test_hallucination_guardrails.py` — adversarial
      cases for both guardrails: Pipeline 1's grounding gate refusing before
      the LLM is touched (empty retrieval, and the embedding-false-positive
      case where retrieved content scores well but shares no vocabulary with
      the query), and Pipeline 2's post-generation verifier flagging a
      model's answer that ignores good context — including a mixed-answer
      case proving it flags only the fabricated sentence, not the whole
      response. One real bug was found and fixed in the process: the latency
      benchmark's p99 calculation (`statistics.quantiles(n=100)`) could
      mathematically report a value *above* the observed maximum when run
      with far fewer than 100 samples — replaced with a nearest-rank
      percentile that can only ever return an actually-observed latency.)
- [x] Authentication & per-user data isolation
      (`app/auth/` — bcrypt password hashing, opaque access + refresh tokens
      hashed with SHA-256 for fast lookup (the two deliberately use
      different hash functions — see the module docstrings for why). Access
      token is short-lived (`UserAuth.access_token_expires_at`, default 60
      minutes via `ACCESS_TOKEN_EXPIRE_MINUTES`); refresh token is
      long-lived (default 30 days via `REFRESH_TOKEN_EXPIRE_DAYS`), sent
      only to `/api/auth/refresh`, and rotates on every use — the old one
      stops working the instant a new pair is issued, so a replayed stolen
      refresh token fails loudly on the legitimate client's own next
      refresh. `POST /api/auth/register`, `/login`, `/refresh`, `/logout`;
      `GET /api/documents` lists a user's own uploaded files with
      per-document chunk counts. `get_current_user`
      dependency gates every data-touching route — `user_id` now comes only
      from the verified token, never from a client-supplied field, closing
      a real gap where any caller could previously claim to be any user.
      The bigger fix: retrieval itself is now scoped by `user_id` end to
      end — `bm25_search` filters in SQL via a `documents` join,
      `vector_search` filters via Chroma's native metadata `where` clause
      (requires `embedding_service.py` to write `user_id` into every
      vector's metadata at embed time), and `graph_expansion` filters chunk
      ownership *before* the top-k cut (the co-occurrence graph itself
      stays shared/global on purpose — only the chunk text behind a node is
      per-user). `chunk_routes.py`'s citation lookup also gained an
      ownership check it never had before. Dashboard and Streamlit frontend
      both gained a real login/register UI, store both tokens client-side,
      and transparently call `/api/auth/refresh` on a 401 before ever
      forcing a re-login. Verified with a real `uvicorn` server over live
      HTTP — register, duplicate-register (409), login, old-token-
      invalidated-by-new-login (401), protected endpoint without a token
      (401), cross-user chunk access (404), logout, post-logout token
      (401), refresh issuing a working new pair, a replayed (already-used)
      refresh token rejected (401), an access token refreshed mid-session
      without a full login — all behaved exactly as designed;
      `pytest tests/test_auth_service.py tests/test_api_auth.py
      tests/test_api_documents.py` plus cross-user-isolation tests added to
      every retrieval test file, 210 total passing)
- [x] Reranking (`app/search/reranker.py` — a cross-encoder second relevance
      pass over RRF's fused shortlist, run right before the grounding gate.
      RRF only knows rank *position* per source; a cross-encoder reads the
      (query, chunk) pair together in one forward pass and scores actual
      relevance, closer to what a true reranker is supposed to do than
      fusion alone. Default model `cross-encoder/ms-marco-MiniLM-L-6-v2`
      (~17M params, MS MARCO-trained) — free, Apache-2.0, runs on CPU,
      downloaded once from Hugging Face and cached locally; **no API key
      needed, no per-call cost.** Loaded once at startup (`app/main.py`
      lifespan), same pattern as the LLM/embedder backends; `RERANK_ENABLED`
      (default true) turns it off entirely for faster/offline startup,
      falling back to RRF's own order. Wired through
      `retrieve_and_verify()` (query endpoints, CLI `query`) and
      `evaluate_retrieval()` (CLI `evaluate`), so `kengine evaluate` can
      directly measure whether reranking actually moves Recall@k/
      Precision@k/MRR, not just assert that it runs. Verified against the
      real (not mocked) model: downloaded once, correctly scored an
      actually-relevant passage at ~1.0 relevance versus ~0.0 for two
      unrelated ones, then confirmed end-to-end through a live `uvicorn`
      server — ingest a mixed-topic file, query it, get the right grounded
      citation back. Test suite stubs the model (`tests/test_reranker.py`,
      `tests/test_main_lifespan.py`) so `pytest` never downloads anything
      or depends on network access; 218 total passing)
- [x] Context Precision evaluation (LLM-as-judge)
      (`app/evaluation/evaluate_context_precision.py` — a second evaluation
      metric alongside Recall@k/Precision@k/MRR, for the case those can't
      cover: a golden file of plain `{id, question, ground_truth}` triples
      with no pre-labeled `relevant_chunk_ids`. Runs each question through
      the real retrieval pipeline (BM25+vector+graph+RRF+reranker, same as
      `evaluate_retrieval`), then sends the question, the ground truth, and
      the full ordered list of retrieved chunks to whichever LLM is already
      configured (`LLM_BACKEND` — Ollama or Groq, **no new API key**),
      asking it to mark each chunk relevant or not in one call per question
      (not per chunk — a 20-question golden file costs 20 judge calls, not
      100+). `context_precision_at_k()` (`app/evaluation/retrieval_metrics.py`)
      implements the standard RAGAS rank-weighted formula: a relevant chunk
      at rank 1 scores higher than the same chunk buried at rank 10, unlike
      plain Precision@k which only counts hits, not position. Citation
      artifacts like `[cite: 10]` are stripped from `ground_truth` on load
      (`load_golden_cases`) so they don't confuse the judge. New CLI command
      `kengine evaluate-context-precision <golden.json> --k 5`. Verified
      live against the real SRS document and the real Groq backend — 3
      real questions, mean Context Precision@5 = 0.935, individual scores
      matching the actual relevant-chunk counts the judge returned. Tests
      use `FakeLLM` for the judge (`tests/test_evaluate_context_precision.py`)
      so `pytest` makes no real LLM calls; 229 total passing)
- [x] Swappable cloud embedding backend (Cohere)
      (`app/embeddings/cohere_embedder.py` — Embed v2 API, verified against
      Cohere's actual documented request/response shape, not guessed;
      `EmbeddingBackend` Protocol (`app/embeddings/embedder.py`) is now the
      type every caller depends on instead of `OllamaEmbedder` directly,
      mirroring how `LLMBackend` already works for generation;
      `get_embedder_backend()` factory picks `ollama` (default) or `cohere`
      from `EMBEDDING_BACKEND` in `.env`, same pattern as `LLM_BACKEND`;
      `input_type` ("search_document" at index time, "search_query" at
      query time) threaded through both real call sites — Cohere's API
      actually uses this for retrieval quality, Ollama ignores it;
      `pytest tests/test_cohere_embedder.py`, 10 passing, 100% coverage)
- [x] Streamlit chat frontend
      (`streamlit_app.py` — a second, separate frontend process, talks to
      the FastAPI backend purely over HTTP via `requests`, same as the JS
      dashboard; `st.chat_message`/`st.chat_input`/`st.write_stream` for a
      real chat-bubble UI with live token streaming off the same
      `POST /api/query/stream` SSE endpoint; login/register forms, file
      upload sidebar, citations + grounding-coverage rendered under each
      answer. Deliberately doesn't reimplement the graph view — Streamlit
      has no equivalent to the dashboard's D3 force layout; booted for
      real and confirmed serving (not just import-checked) before calling
      this done)
- [ ] Day 3 — packaging & deployment (Docker, offline bundle)
