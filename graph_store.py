"""
graph_store.py - Knowledge Graph layer using NetworkX.

Interface is designed so Neo4j can be swapped in later without changing
callers. All public methods are the same regardless of the backend.

Usage:
    gs = GraphStore()
    gs.add_triples(triples)        # List[Triple]
    gs.query_neighbors("OpenAI")   # list of (relation, neighbor) tuples
    gs.search_nodes("open")        # fuzzy node search
    gs.get_stats()                 # dict with node/edge counts
    gs.save() / gs.load()
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

import config
from utils import get_logger

logger = get_logger(__name__)


class GraphStore:
    """NetworkX-backed knowledge graph with a Neo4j-compatible interface."""

    def __init__(self, persist_path: str = config.GRAPH_PERSIST_PATH):
        self._persist_path = Path(persist_path)
        self._graph = None  # lazy import
        self._init_graph()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_graph(self):
        import networkx as nx  # type: ignore
        self._nx = nx
        if self._persist_path.exists():
            self.load()
            logger.info(
                "Loaded graph: %d nodes, %d edges",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        else:
            self._graph = nx.MultiDiGraph()
            logger.info("Initialized empty knowledge graph.")

    # ── Mutations ────────────────────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        relation: str,
        object_: str,
        source_chunk: str = "",
        source_document: str = "",
    ) -> None:
        """Add a single (subject, relation, object) triple to the graph."""
        s = subject.strip()
        o = object_.strip()
        r = relation.strip()
        if not (s and r and o):
            return

        # Ensure nodes exist with at minimum an empty attribute dict
        if not self._graph.has_node(s):
            self._graph.add_node(s, docs=set())
        if not self._graph.has_node(o):
            self._graph.add_node(o, docs=set())

        # Track which documents mention each node
        self._graph.nodes[s]["docs"].add(source_document)
        self._graph.nodes[o]["docs"].add(source_document)

        self._graph.add_edge(
            s, o,
            relation=r,
            source_chunk=source_chunk,
            source_document=source_document,
        )

    def add_triples(self, triples) -> int:
        """Add a list of Triple objects. Returns count added."""
        added = 0
        for t in triples:
            try:
                self.add_triple(
                    subject=t.subject,
                    relation=t.relation,
                    object_=t.object,
                    source_chunk=t.source_chunk,
                    source_document=t.source_document,
                )
                added += 1
            except Exception as exc:
                logger.warning("Skipping triple due to error: %s", exc)
        logger.info("Added %d triples to graph.", added)
        return added

    def remove_document(self, document_name: str) -> None:
        """Remove all nodes and edges associated with a document."""
        nodes_to_remove = [
            n for n, d in self._graph.nodes(data=True)
            if document_name in d.get("docs", set())
        ]
        # Only remove node if it's exclusively from this document
        for node in nodes_to_remove:
            docs = self._graph.nodes[node].get("docs", set())
            docs.discard(document_name)
            if not docs:
                self._graph.remove_node(node)

        # Remove orphaned edges for this document
        edges_to_remove = [
            (u, v, k)
            for u, v, k, d in self._graph.edges(data=True, keys=True)
            if d.get("source_document") == document_name
        ]
        for u, v, k in edges_to_remove:
            if self._graph.has_edge(u, v, key=k):
                self._graph.remove_edge(u, v, key=k)

    # ── Queries ──────────────────────────────────────────────────────────────

    def query_neighbors(self, entity: str, max_hops: int = 1) -> List[dict]:
        """
        Return all direct neighbors of an entity.
        Each result is a dict: {source, relation, target, source_document}.
        """
        results = []
        entity_lower = entity.lower()

        # Find matching node (case-insensitive)
        matched_node: Optional[str] = None
        for node in self._graph.nodes:
            if node.lower() == entity_lower:
                matched_node = node
                break

        if matched_node is None:
            return []

        # Outgoing edges
        for _, target, data in self._graph.out_edges(matched_node, data=True):
            results.append({
                "source": matched_node,
                "relation": data.get("relation", ""),
                "target": target,
                "source_document": data.get("source_document", ""),
                "source_chunk": data.get("source_chunk", ""),
            })

        # Incoming edges
        for source, _, data in self._graph.in_edges(matched_node, data=True):
            results.append({
                "source": source,
                "relation": data.get("relation", ""),
                "target": matched_node,
                "source_document": data.get("source_document", ""),
                "source_chunk": data.get("source_chunk", ""),
            })

        return results

    def search_nodes(self, query: str, max_results: int = 10) -> List[str]:
        """Fuzzy node search – returns nodes whose names contain the query."""
        q = query.lower()
        return [n for n in self._graph.nodes if q in n.lower()][:max_results]

    def get_all_triples(self) -> List[dict]:
        """Return every triple in the graph as a list of dicts."""
        triples = []
        for u, v, data in self._graph.edges(data=True):
            triples.append({
                "subject": u,
                "relation": data.get("relation", ""),
                "object": v,
                "source_document": data.get("source_document", ""),
                "source_chunk": data.get("source_chunk", ""),
            })
        return triples

    def get_stats(self) -> dict:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "is_connected": self._nx.is_weakly_connected(self._graph)
            if self._graph.number_of_nodes() > 0
            else False,
        }

    def get_graph(self):
        """Return the underlying NetworkX graph (for visualization)."""
        return self._graph

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            data = pickle.dumps(self._graph)
            checksum = hashlib.sha256(data).hexdigest()
            f.write(data)
        checksum_path = self._persist_path.with_suffix(".sha256")
        checksum_path.write_text(checksum, encoding="utf-8")
        logger.info("Graph saved to %s", self._persist_path)

    def load(self) -> None:
        try:
            data = self._persist_path.read_bytes()
            checksum_path = self._persist_path.with_suffix(".sha256")
            if checksum_path.exists():
                expected = checksum_path.read_text(encoding="utf-8").strip()
                actual = hashlib.sha256(data).hexdigest()
                if actual != expected:
                    raise ValueError(
                        f"Graph file checksum mismatch — file may be corrupted or tampered. "
                        f"Expected {expected}, got {actual}."
                    )
            self._graph = pickle.loads(data)  # noqa: S301
        except Exception as exc:
            logger.error("Failed to load graph: %s — starting fresh.", exc)
            import networkx as nx
            self._graph = nx.MultiDiGraph()

    def clear(self) -> None:
        import networkx as nx
        self._graph = nx.MultiDiGraph()
        if self._persist_path.exists():
            self._persist_path.unlink()
        checksum_path = self._persist_path.with_suffix(".sha256")
        if checksum_path.exists():
            checksum_path.unlink()
        logger.info("Graph cleared.")
