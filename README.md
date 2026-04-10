# 🧠 Hybrid Knowledge Graph + Vector RAG QA System

A complete, production-ready MVP that combines **Knowledge Graph** and **Vector Retrieval** for question answering over uploaded documents.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Document Ingestion** | PDF, TXT, Markdown with overlap chunking |
| **Entity Extraction** | spaCy NER + LLM-based relation extraction |
| **Knowledge Graph** | NetworkX (Neo4j-ready interface) |
| **Vector Store** | ChromaDB with sentence-transformer embeddings |
| **Query Routing** | Rule-based + optional LLM classifier (graph / vector / hybrid) |
| **Answer Generation** | OpenAI GPT with grounded citations (fallback mode without API key) |
| **Graph Visualisation** | Interactive PyVis network in the browser |
| **Streamlit UI** | Clean, tab-based interface |

---

## 🗂 Project Structure

```
hybrid-qa-system/
├── app.py            # Streamlit UI
├── ingest.py         # Document loading + chunking
├── extract.py        # NER + relation extraction
├── graph_store.py    # Knowledge graph (NetworkX)
├── vector_store.py   # ChromaDB vector store
├── router.py         # Query classification + routing
├── qa.py             # LLM answer generation
├── config.py         # Centralised configuration
├── utils.py          # Logging, text helpers
├── requirements.txt
├── .env.example
├── data/             # Place your documents here
│   ├── sample_ai_companies.txt
│   └── sample_rag_overview.md
├── storage/          # Persisted vector and graph data (auto-created)
├── graphs/           # Graph exports (auto-created)
└── logs/             # Application logs (auto-created)
```

---

## 🚀 Quick Start

### 1. Clone and enter the directory

```bash
git clone https://github.com/Milan-CSE/RAG_QA_SYSTEM.git
cd RAG_QA_SYSTEM
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional but recommended)
```

> **Note:** The system works without an OpenAI key in *fallback mode*, where it shows raw retrieved context instead of a synthesised answer.

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Usage Walkthrough

1. **Upload documents** using the sidebar file uploader (PDF, TXT, or MD).
2. Click **Index Documents** to run the full pipeline:
   - Documents are loaded and chunked.
   - Entities and relations are extracted.
   - Knowledge graph is built and saved.
   - Vector embeddings are created and stored.
3. **Ask questions** in the Q&A tab:
   - The router decides whether to use graph, vector, or hybrid retrieval.
   - The answer is generated and shown with supporting evidence.
4. **Explore the graph** in the Graph Visualisation tab.
5. **Inspect system internals** in the Debug/Stats tab.

---

## 🔧 Configuration

All settings can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for answer generation |
| `CHUNK_SIZE` | `500` | Target words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of vector chunks to retrieve |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NER model |

---

## 🧪 Sample Questions

Use the included sample documents to test:

- *"Who founded OpenAI?"* → **graph route**
- *"What is Retrieval-Augmented Generation?"* → **vector route**
- *"How is Google related to DeepMind and what does DeepMind do?"* → **hybrid route**
- *"Which companies are based in San Francisco?"* → **graph route**
- *"What are the limitations of RAG systems?"* → **vector route**

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router   │  ← rule-based keywords + optional LLM
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌────────┐
│Graph │  │Vector  │
│Store │  │Store   │
│(NX)  │  │(Chroma)│
└──┬───┘  └───┬────┘
   │           │
   └─────┬─────┘
         │ context
         ▼
┌─────────────────┐
│  LLM (OpenAI)   │
│  Answer Gen     │
└─────────────────┘
         │
         ▼
    Final Answer
    + Citations
```

---

## 🔌 Extending the System

### Swap in Neo4j

Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in `.env` and replace `GraphStore` with a Neo4j implementation following the same interface (same public methods).

### Add REBEL relation extraction

Install `rebel-base` from Hugging Face and add a `_extract_relations_rebel()` function in `extract.py`, inserting it before the LLM call in `extract_triples_from_chunk()`.

### Use a different embedding model

Change `EMBEDDING_MODEL` in `.env` to any model name supported by `sentence-transformers`.

---

## 📋 Requirements

- Python 3.10+
- 4 GB+ RAM (for embedding model)
- Internet access for first-time model downloads

---

## 📄 License

MIT
