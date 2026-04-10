"""
tests/test_core.py - Core unit tests for the Hybrid QA System.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Ingestion ─────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_load_txt(self, tmp_path):
        from ingest import load_document
        f = tmp_path / "test.txt"
        f.write_text("Alice works at Acme Corp in New York. Bob founded Widgets Inc.")
        chunks = load_document(str(f))
        assert len(chunks) >= 1
        assert chunks[0].source_file == "test.txt"
        assert "Alice" in chunks[0].text

    def test_load_md(self, tmp_path):
        from ingest import load_document
        f = tmp_path / "test.md"
        f.write_text("# Title\n\n**Bold text** and `code` here.\n\nSome content.")
        chunks = load_document(str(f))
        assert len(chunks) >= 1

    def test_unsupported_extension(self, tmp_path):
        from ingest import load_document
        f = tmp_path / "test.docx"
        f.write_bytes(b"fake docx")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(str(f))

    def test_chunk_ids_are_unique(self, tmp_path):
        from ingest import load_document
        long_text = " ".join([f"Sentence {i} about OpenAI and AI research." for i in range(200)])
        f = tmp_path / "big.txt"
        f.write_text(long_text)
        chunks = load_document(str(f))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_empty_file(self, tmp_path):
        from ingest import load_document
        f = tmp_path / "empty.txt"
        f.write_text("")
        chunks = load_document(str(f))
        assert chunks == []


# ── Entity Extraction ────────────────────────────────────────────────────────

class TestExtraction:
    def test_extract_entities(self):
        from extract import extract_entities
        entities = extract_entities("OpenAI was founded by Sam Altman in San Francisco.")
        texts = [e["text"] for e in entities]
        assert any("OpenAI" in t for t in texts)
        assert any("Sam Altman" in t for t in texts)

    def test_extract_entities_empty(self):
        from extract import extract_entities
        entities = extract_entities("")
        assert entities == []

    def test_extract_all_returns_list(self, tmp_path):
        from ingest import load_document
        from extract import extract_all
        f = tmp_path / "test.txt"
        f.write_text(
            "OpenAI was founded by Sam Altman. "
            "Google was created by Larry Page and Sergey Brin. "
            "Microsoft is headquartered in Redmond."
        )
        chunks = load_document(str(f))
        triples = extract_all(chunks)
        assert isinstance(triples, list)

    def test_triple_fields(self, tmp_path):
        from ingest import load_document
        from extract import extract_all
        f = tmp_path / "test.txt"
        f.write_text("Apple was founded by Steve Jobs. Steve Jobs worked at Apple.")
        chunks = load_document(str(f))
        triples = extract_all(chunks)
        for t in triples:
            assert hasattr(t, "subject")
            assert hasattr(t, "relation")
            assert hasattr(t, "object")
            assert hasattr(t, "source_chunk")
            assert hasattr(t, "source_document")


# ── Graph Store ───────────────────────────────────────────────────────────────

class TestGraphStore:
    def _make_store(self, tmp_path):
        from graph_store import GraphStore
        return GraphStore(str(tmp_path / "graph.pkl"))

    def test_add_and_query(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("Alice", "works_at", "Acme", source_document="doc1.txt")
        gs.add_triple("Alice", "located_in", "Paris", source_document="doc1.txt")
        neighbors = gs.query_neighbors("Alice")
        assert len(neighbors) == 2

    def test_search_nodes(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("OpenAI", "founded_by", "Sam Altman")
        gs.add_triple("OpenAI Research", "part_of", "OpenAI")
        results = gs.search_nodes("openai")
        assert any("OpenAI" in r for r in results)

    def test_get_stats(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("A", "rel", "B")
        gs.add_triple("B", "rel", "C")
        stats = gs.get_stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 2

    def test_save_and_load(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("Foo", "knows", "Bar")
        gs.save()

        from graph_store import GraphStore
        gs2 = GraphStore(str(tmp_path / "graph.pkl"))
        assert gs2.get_stats()["edges"] == 1

    def test_clear(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("X", "y", "Z")
        gs.clear()
        assert gs.get_stats()["nodes"] == 0

    def test_add_triples_bulk(self, tmp_path):
        from extract import Triple
        gs = self._make_store(tmp_path)
        triples = [
            Triple("A", "rel1", "B", "chunk1", "doc.txt"),
            Triple("B", "rel2", "C", "chunk1", "doc.txt"),
            Triple("C", "rel3", "A", "chunk1", "doc.txt"),
        ]
        added = gs.add_triples(triples)
        assert added == 3
        assert gs.get_stats()["nodes"] == 3

    def test_empty_triple_skipped(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("", "rel", "B")  # empty subject should be skipped
        assert gs.get_stats()["nodes"] == 0

    def test_get_all_triples(self, tmp_path):
        gs = self._make_store(tmp_path)
        gs.add_triple("X", "rel", "Y", source_document="doc.txt")
        all_triples = gs.get_all_triples()
        assert len(all_triples) == 1
        assert all_triples[0]["subject"] == "X"
        assert all_triples[0]["object"] == "Y"


# ── Router ────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_who_query_is_graph(self):
        from router import classify_query
        assert classify_query("Who founded OpenAI?") == "graph"

    def test_what_is_query_is_vector(self):
        from router import classify_query
        assert classify_query("What is retrieval-augmented generation?") == "vector"

    def test_explain_query_is_vector(self):
        from router import classify_query
        assert classify_query("Explain how embeddings work.") == "vector"

    def test_summarize_query_is_vector(self):
        from router import classify_query
        assert classify_query("Summarize the document.") == "vector"

    def test_related_query_is_hybrid(self):
        from router import classify_query
        assert classify_query("How is Google related to DeepMind and what does it do?") == "hybrid"

    def test_empty_query_defaults_to_hybrid(self):
        from router import classify_query
        assert classify_query("") == "hybrid"

    def test_founded_by_is_graph(self):
        from router import classify_query
        assert classify_query("Who is the founder of Tesla?") == "graph"

    def test_where_headquartered_is_graph(self):
        from router import classify_query
        assert classify_query("Where is Anthropic headquartered?") == "graph"

    def test_why_is_vector(self):
        from router import classify_query
        assert classify_query("Why is transformer architecture important?") == "vector"

    def test_limitations_is_vector(self):
        from router import classify_query
        assert classify_query("What are the limitations of RAG?") == "vector"


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_clean_text(self):
        from utils import clean_text
        result = clean_text("Hello   \t world\n\n")
        assert result == "Hello world"

    def test_truncate(self):
        from utils import truncate
        long = "a" * 500
        result = truncate(long, max_chars=100)
        assert len(result) <= 104  # 100 chars + " …"
        assert result.endswith("…")

    def test_truncate_short(self):
        from utils import truncate
        short = "hello"
        assert truncate(short) == "hello"

    def test_sanitize_filename(self):
        from utils import sanitize_filename
        assert sanitize_filename("my file/name.txt") == "my file_name.txt"

    def test_file_extension(self):
        from utils import file_extension
        assert file_extension("document.PDF") == ".pdf"
        assert file_extension("notes.md") == ".md"
        assert file_extension("data.txt") == ".txt"
