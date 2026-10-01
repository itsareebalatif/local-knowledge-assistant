"""Shared keyword-coverage helpers, used by both the pre-retrieval grounding
gate (app/search/grounding_gate.py — does the query's vocabulary show up in
retrieved chunks?) and the post-generation grounding verifier
(app/llm/grounding_verifier.py — does each generated sentence's vocabulary
show up in the retrieved context?). Same underlying question, two different
places it needs asking, so the logic lives once.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Small, deliberately unsurprising stopword list — good enough to strip noise
# words from a short query/sentence without pulling in a whole NLP dependency.
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
    """Lowercased content words: stopwords and very short tokens dropped."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def term_coverage(terms: list[str], reference_text: str) -> float:
    """Fraction of `terms` that appear as whole words in `reference_text`.
    1.0 (not penalized) when `terms` is empty — coverage can't meaningfully
    be judged against zero significant words."""
    if not terms:
        return 1.0
    haystack = reference_text.lower()
    matched = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", haystack))
    return matched / len(terms)
