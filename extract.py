"""
extract.py - Entity and relation extraction pipeline.

Approach:
  1. Use spaCy for Named Entity Recognition (NER).
  2. Attempt LLM-based relation extraction if an API key is configured.
  3. Fall back to heuristic co-occurrence-based relation extraction.

Output: list of Triple dicts with keys:
  subject, relation, object, source_chunk, source_document
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import List, Optional

import config
from utils import get_logger

logger = get_logger(__name__)


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_chunk: str       # chunk_id
    source_document: str    # file name

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "source_chunk": self.source_chunk,
            "source_document": self.source_document,
        }


# ─── spaCy loader ────────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model (downloads if needed)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy  # type: ignore
        try:
            _nlp = spacy.load(config.SPACY_MODEL)
            logger.info("Loaded spaCy model: %s", config.SPACY_MODEL)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Downloading...", config.SPACY_MODEL
            )
            import subprocess
            import sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", config.SPACY_MODEL],
                check=True,
            )
            _nlp = spacy.load(config.SPACY_MODEL)
        return _nlp
    except Exception as exc:
        logger.error("Failed to load spaCy: %s", exc)
        return None


# ─── Entity extraction (spaCy NER) ───────────────────────────────────────────

_RELEVANT_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART",
    "EVENT", "LAW", "LANGUAGE", "NORP", "FAC",
}


def extract_entities(text: str) -> List[dict]:
    """Return list of {text, label} dicts for named entities in text."""
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        if ent.label_ in _RELEVANT_ENTITY_TYPES:
            key = (ent.text.strip().lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({"text": ent.text.strip(), "label": ent.label_})
    return entities


# ─── Relation extraction – LLM-based ─────────────────────────────────────────

_RELATION_SYSTEM_PROMPT = """You are an information extraction assistant.
Extract a list of (subject, relation, object) triples from the given text.
- subject and object must be named entities (persons, organisations, places, products, etc.)
- relation should be a short verb phrase
- Return ONLY a JSON array of objects with keys "subject", "relation", "object"
- If no relations are found, return an empty array []
- Do not include explanations."""


def _extract_relations_llm(text: str, entities: List[dict]) -> List[dict]:
    """Use OpenAI to extract triples. Returns list of dicts."""
    if not config.OPENAI_API_KEY:
        return []
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=config.OPENAI_API_KEY)

        entity_names = [e["text"] for e in entities[:20]]
        prompt = (
            f"Entities present: {', '.join(entity_names)}\n\n"
            f"Text:\n{text[:2000]}\n\n"
            "Extract triples as JSON array:"
        )
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _RELATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw)
        import json
        return json.loads(raw)
    except Exception as exc:
        logger.debug("LLM relation extraction failed: %s", exc)
        return []


# ─── Relation extraction – heuristic fallback ────────────────────────────────

# Patterns: (regex on sentence, relation label)
_PATTERNS = [
    (re.compile(r"\b(\w[\w\s]+)\s+(?:is|was|are|were)\s+(?:the\s+)?(?:ceo|founder|director|president|author|creator|head)\s+of\s+(\w[\w\s]+)", re.I), "is_head_of"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:founded|created|established|launched|started)\s+(\w[\w\s]+)", re.I), "founded"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:acquired|purchased|bought)\s+(\w[\w\s]+)", re.I), "acquired"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:works?\s+(?:at|for)|is\s+employed\s+(?:at|by))\s+(\w[\w\s]+)", re.I), "works_at"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:is\s+located\s+in|is\s+based\s+in|headquartered\s+in)\s+(\w[\w\s]+)", re.I), "located_in"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:is\s+a\s+(?:subsidiary|division|branch|part)\s+of)\s+(\w[\w\s]+)", re.I), "subsidiary_of"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:partnered\s+with|collaborated\s+with|worked\s+with)\s+(\w[\w\s]+)", re.I), "partnered_with"),
    (re.compile(r"\b(\w[\w\s]+)\s+(?:invented|developed|designed|built|wrote)\s+(\w[\w\s]+)", re.I), "created"),
]


def _extract_relations_heuristic(text: str, entities: List[dict]) -> List[dict]:
    """
    Lightweight heuristic extractor.
    - Applies regex patterns on each sentence.
    - Also generates co-occurrence triples for entity pairs in the same sentence.
    """
    entity_set = {e["text"].lower() for e in entities}
    triples: List[dict] = []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        # Pattern-based
        for pattern, relation in _PATTERNS:
            for m in pattern.finditer(sent):
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                if subj.lower() in entity_set or obj.lower() in entity_set:
                    triples.append({"subject": subj, "relation": relation, "object": obj})

        # Co-occurrence for entity pairs in the same sentence
        sent_entities = [
            e["text"] for e in entities if e["text"].lower() in sent.lower()
        ]
        if len(sent_entities) >= 2:
            for a, b in itertools.combinations(sent_entities[:6], 2):
                triples.append({
                    "subject": a,
                    "relation": "related_to",
                    "object": b,
                })

    # Deduplicate
    seen = set()
    unique: List[dict] = []
    for t in triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_triples_from_chunk(chunk) -> List[Triple]:
    """
    Extract triples from a single Chunk object.
    Uses LLM if available, otherwise heuristic fallback.
    """
    text = chunk.text
    entities = extract_entities(text)

    if not entities:
        logger.debug("No entities found in chunk %s", chunk.chunk_id)
        return []

    # Try LLM first; fall back to heuristic
    raw_triples = _extract_relations_llm(text, entities)
    if not raw_triples:
        raw_triples = _extract_relations_heuristic(text, entities)

    triples = []
    for t in raw_triples:
        subj = str(t.get("subject", "")).strip()
        rel = str(t.get("relation", "")).strip()
        obj = str(t.get("object", "")).strip()
        if subj and rel and obj and subj.lower() != obj.lower():
            triples.append(
                Triple(
                    subject=subj,
                    relation=rel,
                    object=obj,
                    source_chunk=chunk.chunk_id,
                    source_document=chunk.source_file,
                )
            )

    logger.debug(
        "Chunk %s → %d entities, %d triples",
        chunk.chunk_id, len(entities), len(triples),
    )
    return triples


def extract_all(chunks: list) -> List[Triple]:
    """Extract triples from all chunks. Robust to per-chunk failures."""
    all_triples: List[Triple] = []
    for chunk in chunks:
        try:
            all_triples.extend(extract_triples_from_chunk(chunk))
        except Exception as exc:
            logger.error("Extraction failed for chunk %s: %s", chunk.chunk_id, exc)
    logger.info("Total triples extracted: %d", len(all_triples))
    return all_triples
