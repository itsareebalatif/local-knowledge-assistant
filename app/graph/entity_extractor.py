
from __future__ import annotations

from functools import lru_cache

import spacy

_LABEL_MAP: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "NORP": "Concept",
    "PRODUCT": "Concept",
    "EVENT": "Concept",
    "WORK_OF_ART": "Concept",
    "LAW": "Concept",
    "LANGUAGE": "Concept",
}

_IGNORED_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


@lru_cache(maxsize=1)
def _load_pipeline():
    return spacy.load("en_core_web_sm")


class SpacyEntityExtractor:
    def __init__(self):
        self._nlp = _load_pipeline()

    def extract(self, text: str) -> list[tuple[str, str]]:
        
        if not text or not text.strip():
            return []
        doc = self._nlp(text)
        seen_names: set[str] = set()
        entities: list[tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ in _IGNORED_LABELS:
                continue
            entity_type = _LABEL_MAP.get(ent.label_)
            if entity_type is None:
                continue
            name = ent.text.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            entities.append((name, entity_type))
        return entities
