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
