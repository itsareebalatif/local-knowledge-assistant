from __future__ import annotations

from app.ingestion.cleaning import is_page_number_line, normalize_text, strip_repeated_page_furniture


def test_is_page_number_line_matches_common_formats():
    assert is_page_number_line("12")
    assert is_page_number_line("Page 3")
    assert is_page_number_line("3 / 40")
    assert not is_page_number_line("Introduction")
    assert not is_page_number_line("Section 12: Overview")


def test_strip_repeated_page_furniture_removes_running_header_and_footer():
    pages = [
        "CONFIDENTIAL — Acme Corp\nIntroduction to the product.\nPage 1",
        "CONFIDENTIAL — Acme Corp\nArchitecture overview goes here.\nPage 2",
        "CONFIDENTIAL — Acme Corp\nConclusion and next steps.\nPage 3",
    ]

    cleaned = strip_repeated_page_furniture(pages)

    assert len(cleaned) == 3
    for page in cleaned:
        assert "CONFIDENTIAL" not in page
        assert "Page" not in page
    assert "Introduction to the product." in cleaned[0]
    assert "Architecture overview goes here." in cleaned[1]


def test_strip_repeated_page_furniture_keeps_content_that_is_not_repeated():
    pages = [f"Unique content {i} with nothing in common." for i in range(4)]
    cleaned = strip_repeated_page_furniture(pages)
    for original, result in zip(pages, cleaned):
        assert original in result


def test_strip_repeated_page_furniture_skips_dedup_under_three_pages():
    pages =["Shared line\nReal content A", "Shared line\nReal content B"]
    cleaned = strip_repeated_page_furniture(pages)
    assert "Shared line" in cleaned[0]
    assert "Shared line" in cleaned[1]


def test_strip_repeated_page_furniture_matches_a_running_header_despite_its_changing_page_number():
    # Regression: a real running header's page number differs every time
    # ("Chapter Title 21" vs "Chapter Title 23"), so exact-line matching
    # never recognized it as repeated furniture — confirmed live against a
    # real PDF where this left the header glued onto the next paragraph.
    pages = [
        "Chapter Title 21\nFirst page of real content here.",
        "Chapter Title 23\nSecond page of real content here.",
        "Chapter Title 27\nThird page of real content here.",
    ]
    cleaned = strip_repeated_page_furniture(pages)
    for page in cleaned:
        assert "Chapter Title" not in page
    assert "First page of real content here." in cleaned[0]
    assert "Third page of real content here." in cleaned[2]


def test_normalize_text_collapses_whitespace_and_control_chars():
    messy = "Hello    world\r\n\r\n\r\nThis  is\ta  test.\x00\x07"
    result = normalize_text(messy)
    assert "\x00" not in result
    assert "\r" not in result
    assert "   " not in result
    assert "\n\n\n" not in result
