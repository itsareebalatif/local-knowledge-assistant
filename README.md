# Personal Knowledge Engine (PKE)

A local-first, multi-tenant RAG assistant that answers questions from your own documents — grounded, cited, and built to refuse rather than guess. Ingests PDF, Markdown, HTML, DOCX, and plain text; retrieves with a hybrid pipeline (keyword search, dense vectors, and a lightweight knowledge graph); and generates answers that are instructed to cite every claim and verified afterward to make sure they actually stayed inside the retrieved context.

## Features

**Retrieval**
- Hybrid search — BM25 (SQLite FTS5), dense vector search (Chroma), and knowledge-graph expansion (spaCy entities + NetworkX co-occurrence) run concurrently and are merged with Reciprocal Rank Fusion
- Cross-encoder reranking and MMR (diversity-aware) selection over the fused candidates
- A grounding gate that refuses before ever calling an LLM if nothing relevant enough was found

**Generation**
- Fact-restricted prompting — numbered context passages, mandatory inline citations, explicit "say you don't know" instruction
- Streaming responses over Server-Sent Events
- A post-generation verifier that checks each sentence of the answer against the retrieved passages and flags anything unsupported
- Citations reflect only what the model actually cited, not everything retrieval happened to find

**Conversation**
- Persisted chat history (per-user conversations and messages, not just browser session state)
- Query routing — a follow-up question is classified as standalone, a document follow-up needing context to resolve pronouns, or a question about the conversation itself (e.g. "what did I ask earlier?") and answered from the appropriate source

**Multi-tenancy & auth**
- Every retrieval path (BM25, vector, graph) is scoped to the authenticated user — never user-supplied, always resolved from the auth token
- Opaque, rotating access/refresh tokens; bcrypt password hashing; forgot-password flow with transactional email
- Per-chunk content deduplication across users with dedup-safe document deletion (shared embeddings are never removed while any surviving chunk still needs them)

**Interfaces**
- REST API (FastAPI) with OpenAPI/Swagger docs
- A dependency-free HTML/JS dashboard with drag-and-drop upload, streaming chat, and an interactive D3 knowledge-graph view
- A Streamlit chat frontend
- A `kengine` CLI for ingesting, querying, and evaluating retrieval quality without running a server

**Swappable backends**
- LLM: local (Ollama), Groq, Gemini, or OpenAI — one shared implementation for every OpenAI-compatible provider; switching is a config change, never new code
- Embeddings: local (Ollama) or Cohere

## Architecture

```
question
  │
  ├──► BM25 (keywords)        ─┐
  ├──► Dense vector (meaning) ─┼──► RRF fusion ──► rerank ──► MMR ──► grounding gate
  └──► Graph expansion        ─┘                                         │
                                                                   refuse honestly
                                                                   if not grounded
                                                                           │
                                                                           ▼
                                                        prompt (numbered passages,
                                                        cite every claim) ──► stream
                                                        the answer ──► verify grounding
                                                        sentence-by-sentence ──► cite
                                                        only what was actually used
```

Two independent stages, each allowed to say no on its own: retrieval decides whether there's enough to answer from *before* any model is called; generation independently checks afterward whether the model's answer actually stayed inside what it was given.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI, Uvicorn |
| Database | SQLite (SQLAlchemy ORM), FTS5 for BM25 |
| Vector store | Chroma |
| Knowledge graph | NetworkX + spaCy (named entity extraction) |
| Reranking | `sentence-transformers` cross-encoder |
| LLM providers | Ollama (local), Groq, Gemini, OpenAI |
| Embedding providers | Ollama (local), Cohere |
| Email | Resend |
| Observability | Langfuse (optional) |
| Frontends | Static HTML/JS dashboard, Streamlit |

## Getting started

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"            # installs the app + dev/test tools, registers the `kengine` command

cp .env.example .env               # edit as needed — .env is never committed
python -m app.db.init_db           # creates data/pke.sqlite3 with all tables and the FTS5 index
```

### 2. Configure a backend

Two independent, swappable backends, each defaulting to local:

- **LLM** (`LLM_BACKEND` in `.env`): `local` (Ollama) | `groq` | `gemini` | `openai`
- **Embeddings** (`EMBEDDING_BACKEND` in `.env`): `ollama` (local) | `cohere`

For local (the default for both), make sure Ollama is running and has the models pulled:

```bash
ollama serve                       # if not already running
ollama pull llama3.2:3b            # or whatever OLLAMA_MODEL is set to
ollama pull nomic-embed-text       # only needed if EMBEDDING_BACKEND=ollama
```

To use a cloud provider instead, set the matching backend and API key in `.env`, e.g.:

```
LLM_BACKEND=openai
OPENAI_API_KEY=...
```

Trade-off, stated plainly: document text leaves the machine at generation time (and at embed time, if you also switch `EMBEDDING_BACKEND`) once you do this — the opposite of this project's local-first default. Fine if your documents aren't sensitive; not fine if they are.

### 3. Run

```bash
uvicorn app.main:app --reload
```

- **Dashboard:** http://127.0.0.1:8000/ui/
- **Swagger UI:** http://127.0.0.1:8000/docs · **ReDoc:** http://127.0.0.1:8000/redoc

**Or the Streamlit frontend** (a second, separate UI — needs the API server above running first):

```bash
pip install -e ".[streamlit]"
streamlit run streamlit_app.py       # opens at http://localhost:8501
```

**Or both at once, one command:**

```bash
./run_dev.sh
```

## Authentication

Every endpoint except `/api/auth/register` and `/api/auth/login` requires a `Bearer` access token.

```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"a long passphrase","full_name":"Your Name"}'

curl -X POST http://127.0.0.1:8000/api/query \
  -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" \
  -d '{"query":"what does Ollama run models on?"}'
```

A short-lived access token is sent on every request; a long-lived refresh token (sent only to `/api/auth/refresh`) mints a new pair and rotates on every use, so a stolen refresh token is only usable once before the legitimate client's next refresh reveals the theft. Both the dashboard and Streamlit frontend handle this automatically.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/auth/register` | Create an account |
| `POST` | `/api/auth/login` | Log in |
| `POST` | `/api/auth/refresh` | Rotate an access/refresh token pair |
| `POST` | `/api/auth/logout` | Invalidate the current session |
| `POST` | `/api/auth/forgot-password` | Request a password-reset email |
| `POST` | `/api/auth/reset-password` | Reset a password with a reset token |
| `POST` | `/api/ingest` | Upload and index a document |
| `GET` | `/api/documents` | List your uploaded documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document and its index data |
| `GET` | `/api/chunks/{chunk_id}` | View a cited chunk's source text |
| `POST` | `/api/query` | Ask a question, get one JSON response |
| `POST` | `/api/query/stream` | Ask a question, stream the answer over SSE |
| `GET` | `/api/conversations` | List your conversations |
| `GET` | `/api/conversations/{conversation_id}` | Fetch a conversation's full message history |
| `DELETE` | `/api/conversations/{conversation_id}` | Delete a conversation |
| `GET` | `/api/graph` | Fetch the knowledge graph |
| `GET` | `/api/graph/neighbors` | Fetch a node's 1-hop neighbors |

Full request/response schemas: `/docs` (Swagger) or `/redoc`.

## CLI

The CLI talks to the database directly — no server, no HTTP auth — and runs against one local user created automatically on first use.

```bash
kengine ingest ./some-notes.md
kengine query "what does Ollama run models on?"
kengine evaluate ./eval_cases.json --k 5                  # Recall@k, Precision@k, MRR against labeled cases
kengine evaluate-context-precision ./golden.json --k 5    # LLM-judged Context Precision against {question, ground_truth} pairs
kengine benchmark-latency --n-chunks 250000                # vector query latency at scale
```

## Testing

```bash
pytest                                        # full suite — no live LLM/Ollama needed, backends are swapped for test doubles
pytest --cov=app --cov-report=term-missing    # with a per-file coverage report
ruff check .                                  # lint
```

## Project structure

```
app/
├── api/            REST route handlers
├── auth/           Password hashing, tokens, auth service
├── db/             Engine setup, migrations, FTS5 index
├── embeddings/      Embedder backends + Chroma vector store
├── evaluation/       Retrieval/context-precision evaluation harness
├── graph/          Entity extraction, graph building, graph expansion
├── ingestion/       Format parsers, chunking, cleaning
├── llm/            LLM backends, prompts, citations, grounding verifier
├── models/         SQLAlchemy models
├── notifications/    Transactional email
├── observability/   Langfuse tracing
├── search/         BM25, vector search, RRF fusion, reranking, MMR, grounding gate
├── services/        Orchestration layer (ingest, retrieval, generation, etc.)
└── static/         Dashboard (HTML/CSS/JS)
tests/              pytest suite
streamlit_app.py    Streamlit frontend
```

## Known limitations

- Citations link to the exact cited chunk's text, not the original uploaded file — ingestion never stores the original file bytes, only extracted text.
- One active session per user: logging in again invalidates whatever token pair was issued before it, rather than supporting multiple concurrent devices.
- No rate limiting on login attempts or LLM calls.
