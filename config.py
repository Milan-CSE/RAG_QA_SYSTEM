"""
config.py - Central configuration management for the Hybrid QA System.
Loads settings from environment variables and provides sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
GRAPHS_DIR = BASE_DIR / "graphs"
LOGS_DIR = BASE_DIR / "logs"

for _d in (DATA_DIR, STORAGE_DIR, GRAPHS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── LLM / OpenAI ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

# ─── Document Ingestion ───────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".txt", ".md")

# ─── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ─── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = str(STORAGE_DIR / "chroma")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "hybrid_qa")
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# ─── Graph Store ─────────────────────────────────────────────────────────────
GRAPH_PERSIST_PATH: str = str(STORAGE_DIR / "knowledge_graph.pkl")

# Neo4j (optional — only used if NEO4J_URI is set)
NEO4J_URI: str = os.getenv("NEO4J_URI", "")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

# ─── spaCy ───────────────────────────────────────────────────────────────────
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ─── Router ───────────────────────────────────────────────────────────────────
# Keyword sets used by the rule-based router
GRAPH_KEYWORDS: list = [
    "who", "whom", "whose", "which", "where", "relation", "related",
    "connection", "link", "between", "founded", "created", "built",
    "located", "owned", "works", "employed", "partner", "subsidiary",
    "ceo", "author", "inventor", "president", "director",
]
VECTOR_KEYWORDS: list = [
    "what", "how", "why", "explain", "describe", "summarize",
    "overview", "definition", "meaning", "tell me about",
    "details about", "information on",
]

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = str(LOGS_DIR / "app.log")
