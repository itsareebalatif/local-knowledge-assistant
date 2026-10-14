from __future__ import annotations

from app.search.keyword_utils import _simple_stem, is_broad_query, significant_terms, term_coverage


def test_simple_stem_strips_common_plural_suffixes():
    assert _simple_stem("models") == "model"
    assert _simple_stem("entities") == "entity"
    assert _simple_stem("boxes") == "box"  # real "-es" plural: strip both letters
    assert _simple_stem("dishes") == "dish"  # same: base ends in a sibilant


def test_simple_stem_does_not_overstrip_a_word_ending_in_e_plus_s():
    # "tables" is "table" + "s", not a true "-es" plural — stripping both
    # letters would wrongly produce "tabl" instead of "table". This is a
    # heuristic, not a real lemmatizer: it can't distinguish every "-es"
    # case (e.g. "phases" vs a hypothetical base ending in a sibilant like
    # "phas") without a dictionary. "tables" is the real case this project
    # hit; that's the one guaranteed correct here.
    assert _simple_stem("tables") == "table"


def test_simple_stem_leaves_non_plurals_and_short_words_alone():
    assert _simple_stem("model") == "model"
    assert _simple_stem("glass") == "glass"  # ends in "ss", not a plural suffix
    assert _simple_stem("gas") == "gas"  # too short for the "s"-stripping rule


def test_significant_terms_drops_stopwords_and_short_tokens():
    terms = significant_terms("What is the capital of France?")
    assert terms == ["capital", "france"]


def test_significant_terms_splits_snake_case_on_underscore():
    # Regression: \w+ treats underscore as a word character, so
    # "freelancer_marketplace" used to tokenize as one indivisible blob
    # that could never match "Freelancer Marketplace" (two separate words)
    # as it actually appears in real document text — a confirmed cause of
    # false refusals via low_keyword_coverage.
    terms = significant_terms("what tables does freelancer_marketplace need")
    assert "freelancer" in terms
    assert "marketplace" in terms
    assert "freelancer_marketplace" not in terms


def test_term_coverage_matches_a_formerly_underscored_term_against_spaced_text():
    terms = significant_terms("freelancer_marketplace tables")
    coverage = term_coverage(terms, "The Freelancer Marketplace has several tables.")
    assert coverage == 1.0


def test_term_coverage_is_whole_word_not_substring():
    assert term_coverage(["cat"], "concatenate") == 0.0
    assert term_coverage(["cat"], "the cat sat") == 1.0


def test_term_coverage_vacuously_perfect_for_no_significant_terms():
    assert term_coverage([], "anything") == 1.0


def test_term_coverage_matches_singular_query_term_against_plural_document_text():
    # Regression: a real refusal where the document said "lightweight
    # models (such as qwen2.5:3b)" (plural) but the query term was
    # "model" (singular) — pure whole-word regex matching missed it even
    # though the chunk was clearly the right answer.
    coverage = term_coverage(["model"], "This uses lightweight models such as qwen2.5:3b.")
    assert coverage == 1.0


def test_term_coverage_matches_plural_query_term_against_singular_document_text():
    coverage = term_coverage(["skills"], "Each freelancer profile lists one primary skill.")
    assert coverage == 1.0


def test_significant_terms_splits_simple_camel_case():
    # Regression: a query like "DataFlowDiagram" (no separator at all) used
    # to tokenize as one indivisible blob that could never match "Data Flow
    # Diagram" (three separate words) as it actually appears in real
    # document text — confirmed live, same failure mode as the underscore
    # case, different separator (none).
    terms = significant_terms("how the DataFlowDiagram work")
    assert "data" in terms
    assert "flow" in terms
    assert "diagram" in terms
    assert "dataflowdiagram" not in terms


def test_significant_terms_splits_acronym_camel_case():
    # The acronym case needs a second rule: "SQLDatabase" has no
    # lowercase-to-uppercase transition before "Database" starts (SQLD is
    # all-caps), so only the upper-run -> upper-then-lower boundary catches it.
    terms = significant_terms("the SQLDatabase design")
    assert "sql" in terms
    assert "database" in terms


def test_term_coverage_matches_a_formerly_camel_case_term_against_spaced_text():
    terms = significant_terms("DataFlowDiagram")
    coverage = term_coverage(terms, "4.2 Data Flow Diagram (DFD Level 1)")
    assert coverage == 1.0


def test_is_broad_query_recognizes_list_every_and_list_all():
    assert is_broad_query("List every animal that Harry encounters in both chapters.")
    assert is_broad_query("List every gift Dudley received on his birthday")
    assert is_broad_query("List all the ingredients")


def test_is_broad_query_recognizes_all_the_plural_phrasing():
    assert is_broad_query("What are all the ingredients in this recipe?")
    assert is_broad_query("summarize all documents")


def test_is_broad_query_recognizes_every_single_and_each_of_the():
    assert is_broad_query("What does every single character say about Voldemort?")
    assert is_broad_query("Tell me about each of the Hogwarts houses")


def test_is_broad_query_recognizes_everything_about():
    assert is_broad_query("Tell me everything about Hagrid")


def test_is_broad_query_false_for_ordinary_narrow_questions():
    # Regression guard for the detector itself: these should NOT trigger the
    # wider, slower retrieval path — narrow single-fact questions are the
    # common case and must stay on the default top-k.
    assert not is_broad_query("Who is Professor McGonagall?")
    assert not is_broad_query("Where did he get it?")
    assert not is_broad_query("Tell me about Hagrid")
    assert not is_broad_query("What was Hagrid wearing?")
