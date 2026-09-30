"""Sample-file generators used by the ingestion tests. Each returns raw bytes
for a real file of that format, built with the same libraries the parsers
consume, so tests exercise real parsing rather than hand-rolled mocks."""

from __future__ import annotations

import io

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def db_session():
    """A fresh, isolated in-memory SQLite DB per test, with FK enforcement
    on (SQLite ignores foreign keys unless told otherwise, same as the real
    engine in app/db/base.py). Importing app.models registers every table on
    Base.metadata before create_all runs."""
    from app.db.base import Base
    from app.db.fts import create_fts_index
    from app import models  # noqa: F401 - registers tables on Base.metadata

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    create_fts_index(engine)  # matches real init_db.py — chunks_fts always exists alongside the schema
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def user(db_session):
    from app.models import User, UserAuth

    u = User(email="mohsin@example.com", full_name="Mohsin", role="ADMIN")
    db_session.add(u)
    db_session.flush()
    db_session.add(UserAuth(user_id=u.user_id, password_hash="hashed"))
    db_session.commit()
    return u


@pytest.fixture
def sample_txt_bytes() -> bytes:
    text = (
        "This is the first paragraph of a plain text file. It has a couple "
        "of sentences to make sure paragraph joining works.\n\n"
        "This is the second paragraph, completely separate from the first."
    )
    return text.encode("utf-8")


@pytest.fixture
def sample_markdown_bytes() -> bytes:
    text = """# Personal Knowledge Engine

Intro paragraph explaining the project in a sentence or two.

## Architecture

The system uses a vector store and a graph store together.

### Vector Store

Chunks are embedded with nomic-embed-text and stored in ChromaDB.

## Deployment

| Component | Tool |
| --- | --- |
| Vector DB | ChromaDB |
| Graph DB | NetworkX |

```python
def hello():
    return "world"
```
"""
    return text.encode("utf-8")


@pytest.fixture
def sample_html_bytes() -> bytes:
    html = """<!doctype html>
<html><head><title>Test Doc</title>
<style>body { color: red; }</style>
</head>
<body>
<nav>Site Nav Should Be Stripped</nav>
<header>Header Should Be Stripped</header>
<h1>Main Title</h1>
<p>Introductory paragraph with some content about the project.</p>
<h2>Sub Section</h2>
<p>Sub-section paragraph text goes here for testing purposes.</p>
<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
<footer>Footer Should Be Stripped</footer>
<script>console.log('should be stripped')</script>
</body></html>
"""
    return html.encode("utf-8")


@pytest.fixture
def sample_docx_bytes() -> bytes:
    import docx

    document = docx.Document()
    document.add_heading("Main Title", level=1)
    document.add_paragraph("Introductory paragraph with some content about the project.")
    document.add_heading("Sub Section", level=2)
    document.add_paragraph("Sub-section paragraph text goes here for testing purposes.")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "A"
    table.rows[0].cells[1].text = "B"
    table.rows[1].cells[0].text = "1"
    table.rows[1].cells[1].text = "2"
    document.add_paragraph("A closing paragraph after the table.")

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(content_width, 12, "Main Title")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(content_width, 8, "Introductory paragraph with some content about the project, long enough to wrap.")
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(content_width, 10, "Sub Section")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(content_width, 8, "Sub-section paragraph text goes here for testing purposes.")
    return bytes(pdf.output())


@pytest.fixture
def oversized_paragraph_text() -> str:
    # ~900 words, no punctuation breaks -> forces the hard-wrap fallback path.
    return " ".join(f"word{i}" for i in range(900))


class FakeEmbedder:
    """Deterministic stand-in for OllamaEmbedder — no network, no Ollama
    needed. Vector = [len(text), call_count] so tests can assert both
    content and call counts precisely."""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), float(len(self.calls))] for t in texts]


class FakeVectorStore:
    """In-memory stand-in for ChromaVectorStore — same interface, no disk."""

    def __init__(self):
        self.records: dict[str, dict] = {}

    def add(self, ids, embeddings, documents, metadatas):
        for i, vec, doc, meta in zip(ids, embeddings, documents, metadatas):
            self.records[i] = {"embedding": vec, "document": doc, "metadata": meta}

    def query(self, embedding, top_k=5):
        return list(self.records.values())[:top_k]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()
