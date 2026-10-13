from __future__ import annotations

import pytest

from app.ingestion.chunker import HeaderAwareChunker
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser
from app.ingestion.pipeline import ingest_bytes, ingest_many
from app.ingestion.tokenizer import count_tokens
from app.ingestion.types import Block


async def test_txt_parser_splits_paragraphs(sample_txt_bytes):
    doc = await TxtParser().parse(sample_txt_bytes, "notes.txt")
    assert [b.type for b in doc.blocks] == ["paragraph", "paragraph"]
    assert "first paragraph" in doc.blocks[0].text


# async def test_markdown_parser_extracts_headings_table_and_code(sample_markdown_bytes):
#     doc = await MarkdownParser().parse(sample_markdown_bytes, "readme.md")
#     types = [b.type for b in doc.blocks]
#     assert "heading" in types and "table" in types and "code" in types

#     headings = [b for b in doc.blocks if b.type == "heading"]
#     assert [h.level for h in headings] == [1, 2, 3, 2]
#     assert headings[0].text == "Personal Knowledge Engine"
#     assert headings[2].text == "Vector Store"

#     table_block = next(b for b in doc.blocks if b.type == "table")
#     assert "ChromaDB" in table_block.text

#     code_block = next(b for b in doc.blocks if b.type == "code")
#     assert "def hello" in code_block.text


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


def test_pdf_row_text_inserts_a_space_at_a_positional_word_gap():
    # Regression: many PDFs (ebook conversions especially) position words
    # with a horizontal gap instead of an explicit space glyph in
    # page.chars — real measured gaps from an actual affected file: ~0.001x
    # size within a word, ~0.26x size between words.
    def _char(text, x0, x1, size=12.0):
        return {"text": text, "x0": x0, "x1": x1, "size": size}

    row = [
        _char("H", 100.0, 108.0),
        _char("i", 108.0, 110.0),
        _char("t", 113.1, 116.0),  # gap 3.1 / size 12 = 0.26x -> new word
        _char("h", 116.0, 120.0),
        _char("e", 120.0, 124.0),
    ]
    assert PdfParser._row_text(row) == "Hi the"


def test_pdf_row_text_does_not_double_space_a_real_space_character():
    def _char(text, x0, x1, size=12.0):
        return {"text": text, "x0": x0, "x1": x1, "size": size}

    row = [_char("H", 0.0, 8.0), _char(" ", 8.0, 12.0), _char("i", 12.0, 14.0)]
    assert PdfParser._row_text(row) == "H i"


def test_pdf_row_text_treats_tight_kerning_as_the_same_word():
    def _char(text, x0, x1, size=12.0):
        return {"text": text, "x0": x0, "x1": x1, "size": size}

    row = [_char("r", 0.0, 10.0), _char("r", 10.01, 20.0)]  # 0.01/12 ≈ 0.001x, pure kerning
    assert PdfParser._row_text(row) == "rr"


def _line(text, top, bottom=None):
    from app.ingestion.parsers.pdf_parser import _Line

    return _Line(text=text, size=12.0, bold=False, top=top, bottom=bottom or top + 10)


def test_strip_positional_running_headers_removes_a_header_whose_text_changes_but_position_does_not():
    # Regression: a real running header's text changes every page (chapter
    # title, page number) so strip_repeated_page_furniture's exact-text
    # matching can't catch it, but its vertical position is identical on
    # every page it appears on — confirmed live (top=23.4pt on 18/29 real
    # pages, 4.0% of page height, distinct from every other page type).
    page_heights = [600.0] * 10
    page_lines = [
        [_line(f"Chapter Title {i}", top=24.0), _line("Real paragraph content here.", top=50.0)] for i in range(10)
    ]
    cleaned = PdfParser._strip_positional_running_headers(page_lines, page_heights)
    for lines in cleaned:
        assert all("Chapter Title" not in ln.text for ln in lines)
        assert any("Real paragraph" in ln.text for ln in lines)


def test_strip_positional_running_headers_leaves_a_chapter_opener_page_alone():
    # A chapter-opener's title sits at a DIFFERENT position than the
    # running header (confirmed live: 11.6% vs 4.0% of page height) and
    # only appears once, not on a recurring majority of pages — must not
    # be mistaken for the header and stripped.
    page_heights = [600.0] * 10
    page_lines = [
        [_line(f"Chapter Title {i}", top=24.0), _line("Real paragraph content here.", top=50.0)] for i in range(9)
    ]
    page_lines.append([_line("— CHAPTER TWO —", top=70.0), _line("The Vanishing Glass", top=90.0)])
    cleaned = PdfParser._strip_positional_running_headers(page_lines, page_heights)
    assert cleaned[9][0].text == "— CHAPTER TWO —"


def test_pdf_parser_keeps_a_line_containing_an_nfkc_expanding_character(tmp_path):
    # Regression for a real, confirmed-live bug: strip_repeated_page_furniture
    # runs normalize_text() (NFKC) on its output before handing back which
    # lines survived, but the parser was comparing that against each line's
    # RAW (un-normalized) text. NFKC expands some characters into more than
    # one (e.g. "½" -> "1⁄2"), so any line containing one never matched —
    # and got silently dropped from the whole document. Found via a real
    # PDF where "…" expanded the same way and ate an entire quoted sentence.
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(3):
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 8, "Shared Running Header")
        pdf.multi_cell(
            pdf.w - pdf.l_margin - pdf.r_margin,
            8,
            f"On page {i}, the recipe needs 1½ cups of flour to finish properly.",
        )
    file_bytes = bytes(pdf.output())

    doc = PdfParser().parse_sync(file_bytes, "fractions.pdf")
    full_text = " ".join(b.text for b in doc.blocks)
    assert "Shared Running Header" not in full_text  # the genuinely-repeated line should still be stripped
    assert "finish properly" in full_text  # but the unique line (with the NFKC-expanding char) must survive


def test_strip_positional_running_headers_does_nothing_without_a_recurring_pattern():
    # A headerless document's first lines land at varying, unrelated
    # positions — nothing should be stripped just because a line happens
    # to be short and near the top of its own page.
    page_heights = [600.0] * 5
    page_lines = [[_line(f"Unrelated short line {i}", top=10.0 + i * 7)] for i in range(5)]
    cleaned = PdfParser._strip_positional_running_headers(page_lines, page_heights)
    assert cleaned == page_lines



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
    assert chunks[0].heading_path == ["A"]
    assert chunks[1].heading_path == ["A", "A.1"]
    assert chunks[2].heading_path == ["A", "A.2"]


def test_chunker_heading_with_no_body_still_produces_a_chunk():
    # Regression: an all-headings outline used to silently produce zero chunks.
    blocks = [
        Block(type="heading", text="Day 1 Agenda", level=1),
        Block(type="heading", text="Day 2 Agenda", level=1),
        Block(type="heading", text="Day 3 Agenda", level=1),
    ]
    chunks = HeaderAwareChunker(max_tokens=512).chunk(blocks)
    assert len(chunks) == 3
    assert [c.content for c in chunks] == ["Day 1 Agenda", "Day 2 Agenda", "Day 3 Agenda"]
    assert [c.heading_path for c in chunks] == [["Day 1 Agenda"], ["Day 2 Agenda"], ["Day 3 Agenda"]]


def test_chunker_prefixes_chunk_content_with_its_own_heading():
    blocks = [Block(type="heading", text="Scope", level=1), Block(type="paragraph", text="Body text here.")]
    chunks = HeaderAwareChunker(max_tokens=512).chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].content == "Scope\n\nBody text here."


def test_chunker_hard_wraps_a_single_oversized_paragraph(oversized_paragraph_text):
    blocks = [Block(type="paragraph", text=oversized_paragraph_text)]
    chunks = HeaderAwareChunker(max_tokens=100).chunk(blocks)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 100
    rebuilt = " ".join(c.content for c in chunks).split()
    assert rebuilt == oversized_paragraph_text.split()


def test_chunker_rejects_too_small_a_budget():
    with pytest.raises(ValueError):
        HeaderAwareChunker(max_tokens=8)



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
