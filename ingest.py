"""
ingest.py - Document ingestion pipeline.

Supports PDF, TXT, and Markdown files.
Produces a list of Chunk dataclasses, each with text, metadata, and a unique ID.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import config
from utils import clean_text, file_extension, get_logger

logger = get_logger(__name__)


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk derived from a source document."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_md(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Strip common markdown syntax for cleaner NLP processing.
    # Note: handles standard markdown patterns; complex/nested or non-standard
    # markdown may not be fully stripped. For production use, consider replacing
    # this with a dedicated markdown parser (e.g. `markdown` or `mistletoe`).
    text = re.sub(r"#{1,6}\s*", "", text)        # headings
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", " ", text, flags=re.S)  # code
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)  # images
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)  # links
    return text


def _load_pdf(path: Path) -> str:
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed; falling back to raw read for %s", path)
        return path.read_bytes().decode("latin-1", errors="replace")
    except Exception as exc:
        logger.error("PDF load failed for %s: %s", path, exc)
        return ""


_LOADERS = {
    ".txt": _load_txt,
    ".md":  _load_md,
    ".pdf": _load_pdf,
}


# ─── Chunker ─────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    Tries to split at sentence boundaries first.
    """
    # Split into sentences roughly
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if current_len + len(words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Keep the overlap portion
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words + words
            current_len = len(current_words)
        else:
            current_words.extend(words)
            current_len += len(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _make_chunk_id(source_file: str, index: int, text: str) -> str:
    key = f"{source_file}:{index}:{text}"
    return hashlib.md5(key.encode()).hexdigest()


# ─── Public API ──────────────────────────────────────────────────────────────

def load_document(path: str | Path) -> List[Chunk]:
    """
    Load a single document and return its chunks.
    Raises ValueError for unsupported file types.
    """
    path = Path(path)
    ext = file_extension(str(path))

    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {list(_LOADERS.keys())}"
        )

    logger.info("Loading document: %s", path.name)
    raw_text = _LOADERS[ext](path)
    raw_text = clean_text(raw_text)

    if not raw_text.strip():
        logger.warning("Document %s produced empty text.", path.name)
        return []

    raw_chunks = _chunk_text(raw_text)
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_id = _make_chunk_id(path.name, i, chunk_text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=path.name,
                chunk_index=i,
                metadata={"file_path": str(path), "extension": ext},
            )
        )

    logger.info("  → %d chunks from '%s'", len(chunks), path.name)
    return chunks


def load_documents(paths: List[str | Path]) -> List[Chunk]:
    """Load multiple documents and return all chunks."""
    all_chunks: List[Chunk] = []
    for path in paths:
        try:
            all_chunks.extend(load_document(path))
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
    return all_chunks
