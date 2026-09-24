"""Sample-file generators used by the ingestion tests. Each returns raw bytes
for a real file of that format, built with the same libraries the parsers
consume, so tests exercise real parsing rather than hand-rolled mocks."""

from __future__ import annotations

import io

import pytest


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
