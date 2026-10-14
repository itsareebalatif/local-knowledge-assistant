from __future__ import annotations

import re

# [^\W_]+ excludes underscore (unlike \w+) so snake_case terms tokenize as separate words.
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

# Inserts camelCase boundaries before lowercasing; acronym rule must run after the simple one.
_CAMEL_BOUNDARY_SIMPLE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _tokenize(text: str) -> list[str]:
    text = _CAMEL_BOUNDARY_SIMPLE.sub(" ", text)
    text = _CAMEL_BOUNDARY_ACRONYM.sub(" ", text)
    return _TOKEN_PATTERN.findall(text.lower())

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "whom", "this", "that",
    "these", "those", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "by", "from", "as", "it", "its", "i", "you", "he",
    "she", "we", "they", "them", "his", "her", "their", "our", "your",
    "not", "no", "so", "if", "then", "than", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "have", "has", "had",
}


def significant_terms(text: str) -> list[str]:
    tokens = _tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _simple_stem(word: str) -> str:
    # Heuristic plural stripping (model/models, table/tables) — not a real lemmatizer.
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        # "-es" only strips a real s/x/z/ch/sh plural; otherwise just drop the trailing "s".
        stripped = word[:-2]
        if stripped.endswith(("s", "x", "z", "ch", "sh")):
            return stripped
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def term_coverage(terms: list[str], reference_text: str) -> float:
    if not terms:
        return 1.0
    haystack_words = {_simple_stem(w) for w in _tokenize(reference_text)}
    matched = sum(1 for term in terms if _simple_stem(term) in haystack_words)
    return matched / len(terms)


# A narrow question ("who is Hagrid?") only needs the single best-matching
# chunk; ranked top-k retrieval is built for that. A broad/enumerative
# question ("list every animal Harry sees") needs every chunk that mentions
# a relevant fact, and a detail scattered in a chunk that ranks 7th gets cut
# by a top-k of 5 even though it's genuinely relevant, not noise. Confirmed
# live: the real failure wasn't retrieval ranking the right chunk low, it
# just never made it into the fixed-size candidate pool at all.
_BROAD_QUERY_PATTERN = re.compile(
    r"\b(list\s+(every|all)|all\s+(of\s+)?(the\s+)?\w+s\b|every\s+single|each\s+(one\s+)?of\s+the|everything\s+(about|that))",
    re.IGNORECASE,
)


def is_broad_query(query: str) -> bool:
    return bool(_BROAD_QUERY_PATTERN.search(query))
