"""
qa.py - Answer generation with citation support.

Combines graph facts and vector chunks into an LLM prompt and synthesises
a grounded answer. Falls back to a context-only answer if no LLM key is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import config
from utils import get_logger, truncate

logger = get_logger(__name__)


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    answer: str
    route: str
    graph_facts: List[dict]
    vector_chunks: List[dict]
    sources: List[str]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "route": self.route,
            "graph_facts": self.graph_facts,
            "vector_chunks": self.vector_chunks,
            "sources": self.sources,
        }


# ─── Prompt builder ──────────────────────────────────────────────────────────

def _build_context(
    graph_facts: List[dict],
    vector_chunks: List[dict],
) -> str:
    parts = []

    if graph_facts:
        facts_text = "\n".join(
            f"  • {f['subject']} —[{f['relation']}]→ {f['object']}"
            f"  (from: {f.get('source_document', '?')})"
            for f in graph_facts[:20]
        )
        parts.append(f"GRAPH FACTS:\n{facts_text}")

    if vector_chunks:
        chunks_text = "\n\n".join(
            f"[Chunk {i+1} — {c.get('source_file', '?')} "
            f"(score: {c.get('score', 0):.2f})]\n{truncate(c['text'], 400)}"
            for i, c in enumerate(vector_chunks[:5])
        )
        parts.append(f"DOCUMENT CONTEXT:\n{chunks_text}")

    return "\n\n".join(parts) if parts else "No supporting evidence found."


_SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided context (graph facts and document chunks).
- Be factual and concise.
- If the context does not contain enough information, say so clearly — do NOT speculate.
- Cite the source documents when relevant.
- Use bullet points for complex answers."""


def _llm_answer(query: str, context: str) -> str:
    """Generate an answer using OpenAI."""
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=config.OPENAI_TEMPERATURE,
        max_tokens=config.OPENAI_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def _fallback_answer(query: str, context: str) -> str:
    """No-LLM fallback: returns the raw context with a note."""
    if context == "No supporting evidence found.":
        return (
            f"⚠️ No relevant information was found in the indexed documents "
            f"for your query: \"{query}\". "
            "Please index documents first or rephrase the question."
        )
    return (
        "ℹ️ **Note:** No OpenAI API key configured — showing raw retrieved context.\n\n"
        "---\n"
        + context
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def answer_query(
    query: str,
    route: str,
    graph_store,
    vector_store,
    top_k: int = config.TOP_K_RETRIEVAL,
) -> QAResult:
    """
    Main entry point.

    1. Retrieves graph facts and/or vector chunks based on the route.
    2. Builds a combined context.
    3. Generates a grounded answer.

    Args:
        query:        The user's question.
        route:        "graph", "vector", or "hybrid".
        graph_store:  GraphStore instance.
        vector_store: VectorStore instance.
        top_k:        Number of vector chunks to retrieve.

    Returns:
        QAResult with answer, evidence, and sources.
    """
    graph_facts: List[dict] = []
    vector_chunks: List[dict] = []

    # ── Graph retrieval ──────────────────────────────────────────────────────
    if route in ("graph", "hybrid"):
        # Extract likely entity mentions from the query
        candidate_nodes = graph_store.search_nodes(query, max_results=5)
        if not candidate_nodes:
            # Try individual words
            for word in query.split():
                if len(word) > 3:
                    candidate_nodes.extend(
                        graph_store.search_nodes(word, max_results=2)
                    )
            candidate_nodes = list(dict.fromkeys(candidate_nodes))  # dedup

        for node in candidate_nodes[:3]:
            graph_facts.extend(graph_store.query_neighbors(node))

        # Deduplicate graph facts
        seen = set()
        unique_facts = []
        for f in graph_facts:
            key = (f["subject"], f["relation"], f["target"])
            if key not in seen:
                seen.add(key)
                unique_facts.append({
                    "subject": f["subject"],
                    "relation": f["relation"],
                    "object": f["target"],
                    "source_document": f.get("source_document", ""),
                    "source_chunk": f.get("source_chunk", ""),
                })
        graph_facts = unique_facts[:20]

        logger.info("Graph retrieval: %d facts for route '%s'", len(graph_facts), route)

    # ── Vector retrieval ─────────────────────────────────────────────────────
    if route in ("vector", "hybrid"):
        vector_chunks = vector_store.search(query, top_k=top_k)
        logger.info("Vector retrieval: %d chunks for route '%s'", len(vector_chunks), route)

    # ── Build context ────────────────────────────────────────────────────────
    context = _build_context(graph_facts, vector_chunks)

    # ── Generate answer ──────────────────────────────────────────────────────
    if config.OPENAI_API_KEY:
        try:
            answer = _llm_answer(query, context)
        except Exception as exc:
            logger.error("LLM answer generation failed: %s", exc)
            answer = _fallback_answer(query, context)
    else:
        answer = _fallback_answer(query, context)

    # ── Collect sources ──────────────────────────────────────────────────────
    sources: List[str] = []
    for fact in graph_facts:
        s = fact.get("source_document", "")
        if s and s not in sources:
            sources.append(s)
    for chunk in vector_chunks:
        s = chunk.get("source_file", "")
        if s and s not in sources:
            sources.append(s)

    return QAResult(
        answer=answer,
        route=route,
        graph_facts=graph_facts,
        vector_chunks=vector_chunks,
        sources=sources,
    )
