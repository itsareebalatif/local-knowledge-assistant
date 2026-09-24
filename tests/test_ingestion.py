from __future__ import annotations

import pytest

from app.ingestion.chunker import HeaderAwareChunker
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser
from app.ingestion.pipeline import ingest_bytes, ingest_many
from app.ingestion.tokenizer import count_tokens
from app.ingestion.types import Block


# --------------------------------------------------------------------------
# Per-format parsers
# --------------------------------------------------------------------------


async def test_txt_parser_splits_paragraphs(sample_txt_bytes):
    doc = await TxtParser().parse(sample_txt_bytes, "notes.txt")
    assert [b.type for b in doc.blocks] == ["paragraph", "paragraph"]
    assert "first paragraph" in doc.blocks[0].text


async def test_markdown_parser_extracts_headings_table_and_code(sample_markdown_bytes):
    doc = await MarkdownParser().parse(sample_markdown_bytes, "readme.md")
    types = [b.type for b in doc.blocks]
    assert "heading" in types and "table" in types and "code" in types

    headings = [b for b in doc.blocks if b.type == "heading"]
    assert [h.level for h in headings] == [1, 2, 3, 2]
    assert headings[0].text == "Personal Knowledge Engine"
    assert headings[2].text == "Vector Store"

    table_block = next(b for b in doc.blocks if b.type == "table")
    assert "ChromaDB" in table_block.text

    code_block = next(b for b in doc.blocks if b.type == "code")
    assert "def hello" in code_block.text


async def test_html_parser_strips_layout_noise_and_keeps_structure(sample_html_bytes):
    doc = await HtmlParser().parse(sample_html_bytes, "page.html")
    full_text = " ".join(b.text for b in doc.blocks)
    for noise in ("Site Nav", "Header Should", "Footer Should", "console.log"):
        assert noise not in full_text

    assert doc.meta["title"] == "Test Doc"
    headings = [b for b in doc.blocks if b.type == "heading"]
    assert [h.level for h in headings] == [1, 2]
    table_block = next(b for b in doc.blocks if b.type == "table")
    assert "1 | 2" in table_block.text


async def test_docx_parser_preserves_heading_paragraph_table_order(sample_docx_bytes):
    doc = await DocxParser().parse(sample_docx_bytes, "report.docx")
    types = [b.type for b in doc.blocks]
    # heading, paragraph, heading, paragraph, table, paragraph
    assert types == ["heading", "paragraph", "heading", "paragraph", "table", "paragraph"]
    assert doc.blocks[0].level == 1
    assert doc.blocks[2].level == 2
    assert "A | B" in doc.blocks[4].text


async def test_pdf_parser_detects_headings_by_font_size(sample_pdf_bytes):
    doc = await PdfParser().parse(sample_pdf_bytes, "doc.pdf")
    headings = [b for b in doc.blocks if b.type == "heading"]
    assert len(headings) >= 2
    assert any("Main Title" in h.text for h in headings)
    assert any("Sub Section" in h.text for h in headings)
    paragraphs = [b for b in doc.blocks if b.type == "paragraph"]
    assert any("Introductory paragraph" in p.text for p in paragraphs)


async def test_pdf_parser_falls_back_gracefully_on_garbage_bytes():
    from app.ingestion.exceptions import ParsingError

    with pytest.raises(ParsingError):
        await PdfParser().parse(b"not a real pdf", "broken.pdf")


# --------------------------------------------------------------------------
# Chunker: strict 512-token boundary + header-awareness
# --------------------------------------------------------------------------


def test_chunker_respects_token_budget_and_never_exceeds_it():
    blocks = [
        Block(type="heading", text="Section One", level=1),
        *[Block(type="paragraph", text=f"Paragraph {i} with some words in it.") for i in range(40)],
        Block(type="heading", text="Section Two", level=1),
        *[Block(type="paragraph", text=f"Other paragraph {i} with more words here too.") for i in range(40)],
    ]
    chunks = HeaderAwareChunker(max_tokens=100).chunk(blocks)
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.token_count <= 100
        assert chunk.token_count == count_tokens(chunk.content)


def test_chunker_breadcrumbs_reflect_nested_headings():
    blocks = [
        Block(type="heading", text="Intro", level=1),
        Block(type="paragraph", text="Top-level content."),
        Block(type="heading", text="Scope", level=2),
        Block(type="paragraph", text="Nested content under Scope."),
    ]
    chunks = HeaderAwareChunker(max_tokens=512).chunk(blocks)
    assert chunks[0].heading_path == ["Intro"]
    assert chunks[1].heading_path == ["Intro", "Scope"]


def test_chunker_pops_sibling_headings_correctly():
    blocks = [
        Block(type="heading", text="A", level=1),
        Block(type="heading", text="A.1", level=2),
        Block(type="paragraph", text="under A.1"),
        Block(type="heading", text="A.2", level=2),
        Block(type="paragraph", text="under A.2, sibling of A.1"),
    ]
    chunks = HeaderAwareChunker(max_tokens=512).chunk(blocks)
    assert chunks[0].heading_path == ["A", "A.1"]
    assert chunks[1].heading_path == ["A", "A.2"]


def test_chunker_hard_wraps_a_single_oversized_paragraph(oversized_paragraph_text):
    blocks = [Block(type="paragraph", text=oversized_paragraph_text)]
    chunks = HeaderAwareChunker(max_tokens=100).chunk(blocks)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 100
    # No words lost or duplicated across the split.
    rebuilt = " ".join(c.content for c in chunks).split()
    assert rebuilt == oversized_paragraph_text.split()


def test_chunker_rejects_too_small_a_budget():
    with pytest.raises(ValueError):
        HeaderAwareChunker(max_tokens=8)


# --------------------------------------------------------------------------
# Pipeline: end-to-end + NFR-7 error tolerance
# --------------------------------------------------------------------------


async def test_pipeline_end_to_end_for_markdown(sample_markdown_bytes):
    result = await ingest_bytes(sample_markdown_bytes, "readme.md")
    assert result.ok is True
    assert result.file_type == "MD"
    assert len(result.chunks) > 0
    assert all(c.token_count <= 512 for c in result.chunks)


async def test_pipeline_reports_unsupported_extension_without_raising():
    result = await ingest_bytes(b"whatever", "archive.zip")
    assert result.ok is False
    assert "No parser registered" in result.error


async def test_pipeline_reports_corrupt_file_without_raising():
    result = await ingest_bytes(b"not a real pdf", "broken.pdf")
    assert result.ok is False
    assert result.file_type == "PDF"
    assert result.error


async def test_ingest_many_keeps_going_after_one_bad_file(sample_txt_bytes, sample_markdown_bytes):
    results = await ingest_many(
        [
            (sample_txt_bytes, "notes.txt"),
            (b"garbage", "broken.pdf"),
            (sample_markdown_bytes, "readme.md"),
        ]
    )
    assert [r.ok for r in results] == [True, False, True]
