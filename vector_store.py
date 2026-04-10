"""
vector_store.py - Vector storage and retrieval using ChromaDB.

Embeds document chunks with sentence-transformers and stores them in a
persistent ChromaDB collection for similarity search.
"""

from __future__ import annotations

from typing import List, Optional

import config
from utils import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION,
        embedding_model: str = config.EMBEDDING_MODEL,
    ):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._embedder = None
        self._client = None
        self._collection = None
        self._init()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._embedder = SentenceTransformer(self._embedding_model_name)
                logger.info("Loaded embedding model: %s", self._embedding_model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model '{self._embedding_model_name}'. "
                    "Ensure you have internet access on first run so the model can be "
                    f"downloaded, or set EMBEDDING_MODEL to a cached local path. "
                    f"Original error: {exc}"
                ) from exc
        return self._embedder

    def _init(self):
        import chromadb  # type: ignore
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s': %d docs",
            self._collection_name,
            self._collection.count(),
        )

    # ── Mutations ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list) -> int:
        """
        Embed and store a list of Chunk objects.
        Skips duplicates (by chunk_id).
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        embedder = self._get_embedder()
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        # Remove duplicates that already exist
        existing_ids = set(self._collection.get(ids=ids)["ids"])
        new_chunks = [
            (t, i, m)
            for t, i, m in zip(texts, ids, metadatas)
            if i not in existing_ids
        ]

        if not new_chunks:
            logger.info("All chunks already in collection, nothing to add.")
            return 0

        new_texts, new_ids, new_metas = zip(*new_chunks)
        embeddings = embedder.encode(list(new_texts), show_progress_bar=False).tolist()

        self._collection.add(
            documents=list(new_texts),
            embeddings=embeddings,
            ids=list(new_ids),
            metadatas=list(new_metas),
        )
        logger.info("Added %d chunks to vector store.", len(new_ids))
        return len(new_ids)

    def remove_document(self, source_file: str) -> None:
        """Remove all chunks from a specific source document."""
        results = self._collection.get(
            where={"source_file": source_file},
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info(
                "Removed %d chunks for document '%s'.", len(results["ids"]), source_file
            )

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Semantic similarity search.
        Returns list of dicts: {text, source_file, chunk_index, score, chunk_id}.
        """
        if self._collection.count() == 0:
            logger.warning("Vector store is empty.")
            return []

        embedder = self._get_embedder()
        q_embedding = embedder.encode(query, show_progress_bar=False).tolist()

        where_clause = {"source_file": source_filter} if source_filter else None

        kwargs = {
            "query_embeddings": [q_embedding],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            kwargs["where"] = where_clause

        results = self._collection.query(**kwargs)

        hits = []
        for doc, meta, dist, id_ in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            hits.append({
                "text": doc,
                "source_file": meta.get("source_file", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "score": round(1 - dist, 4),   # cosine distance → similarity
                "chunk_id": id_,
            })
        return hits

    # ── Stats ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def get_stats(self) -> dict:
        return {"total_chunks": self._collection.count()}

    # ── Reset ────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.")
