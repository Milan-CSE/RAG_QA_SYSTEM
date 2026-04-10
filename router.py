"""
router.py - Query classification and routing.

Decides whether a query should go to:
  - "graph"  : relational / entity-centric questions
  - "vector" : semantic / conceptual questions
  - "hybrid" : combined approach (default for ambiguous queries)

Strategy:
  1. Rule-based keyword matching (fast, no API call).
  2. Optional LLM override for nuanced classification.
"""

from __future__ import annotations

import re
from typing import Literal

import config
from utils import get_logger

logger = get_logger(__name__)

RouteType = Literal["graph", "vector", "hybrid"]


# ── Rule-based classifier ────────────────────────────────────────────────────

_GRAPH_WORDS = set(config.GRAPH_KEYWORDS)
_VECTOR_WORDS = set(config.VECTOR_KEYWORDS)

# Patterns that strongly indicate a graph (relational) query
_GRAPH_PATTERNS = [
    re.compile(r"\bwho\b", re.I),
    re.compile(r"\bwhom\b", re.I),
    re.compile(r"\bwhose\b", re.I),
    re.compile(r"\bwhere\s+(?:is|are|was|were)\b", re.I),
    re.compile(r"\brelat(?:ion|ed|ionship)\b", re.I),
    re.compile(r"\bconnect(?:ion|ed)?\b", re.I),
    re.compile(r"\bfound(?:ed|er)\b", re.I),
    re.compile(r"\bcreated?\s+by\b", re.I),
    re.compile(r"\bworks?\s+(?:at|for)\b", re.I),
    re.compile(r"\bheadquartered\b", re.I),
    re.compile(r"\blocated\b", re.I),
    re.compile(r"\bbased\s+in\b", re.I),
    re.compile(r"\bsubsidiar(?:y|ies)\b", re.I),
    re.compile(r"\bacquired?\b", re.I),
    re.compile(r"\bpartner(?:ed)?\s+with\b", re.I),
    re.compile(r"\bbetween\b.{0,40}\band\b", re.I),
]

_VECTOR_PATTERNS = [
    re.compile(r"\bexplain\b", re.I),
    re.compile(r"\bdescribe\b", re.I),
    re.compile(r"\bsummariz(?:e|ation)\b", re.I),
    re.compile(r"\boverview\b", re.I),
    re.compile(r"\bdefin(?:e|ition)\b", re.I),
    re.compile(r"\bwhat\s+(?:is|are|was|were|does|do|did)\b", re.I),
    re.compile(r"\bhow\s+(?:does|do|did|to|can|is|are)\b", re.I),
    re.compile(r"\bwhy\b", re.I),
    re.compile(r"\btell\s+me\s+about\b", re.I),
    re.compile(r"\binformation\s+(?:on|about)\b", re.I),
    re.compile(r"\bdetails?\s+(?:on|about)\b", re.I),
    re.compile(r"\blimitation\b", re.I),
    re.compile(r"\badvantage\b", re.I),
    re.compile(r"\bdisadvantage\b", re.I),
    re.compile(r"\bmeaning\s+of\b", re.I),
]


def _rule_based_route(query: str) -> RouteType:
    graph_score = sum(1 for p in _GRAPH_PATTERNS if p.search(query))
    vector_score = sum(1 for p in _VECTOR_PATTERNS if p.search(query))

    logger.debug(
        "Router scores — graph: %d, vector: %d | query: '%s'",
        graph_score, vector_score, query[:80],
    )

    # Strong signal in one direction with no competing signal
    if graph_score > 0 and vector_score == 0:
        return "graph"
    if vector_score > 0 and graph_score == 0:
        return "vector"
    if graph_score > 0 and vector_score > 0:
        return "hybrid"
    # Default to hybrid for maximum recall on ambiguous queries
    return "hybrid"


# ── LLM-based classifier (optional) ─────────────────────────────────────────

_LLM_ROUTER_PROMPT = """You are a query routing assistant for a hybrid QA system.
Classify the user's question into exactly one of three categories:
  - "graph"  : The question asks about specific entities, relations, connections, or facts between named entities (who, where, founded by, works at, etc.)
  - "vector" : The question asks for explanations, definitions, summaries, or conceptual information.
  - "hybrid" : The question requires both relational facts AND semantic context.

Respond with only the single word: graph, vector, or hybrid."""


def _llm_route(query: str) -> RouteType | None:
    """Return LLM classification or None on failure."""
    if not config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _LLM_ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip().lower()
        if raw in ("graph", "vector", "hybrid"):
            return raw  # type: ignore[return-value]
        return None
    except Exception as exc:
        logger.debug("LLM router failed: %s", exc)
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def classify_query(query: str, use_llm: bool = False) -> RouteType:
    """
    Classify a query and return the routing decision.

    Args:
        query:   The natural language question.
        use_llm: If True, attempt LLM classification first.

    Returns:
        "graph", "vector", or "hybrid"
    """
    if use_llm:
        result = _llm_route(query)
        if result:
            logger.info("LLM router → %s | '%s'", result, query[:60])
            return result

    result = _rule_based_route(query)
    logger.info("Rule-based router → %s | '%s'", result, query[:60])
    return result
