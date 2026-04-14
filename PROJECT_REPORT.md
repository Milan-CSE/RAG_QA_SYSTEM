# Gujarat Technological University
## Chandkheda, Ahmedabad

---

# SHRI LABHUBHAI TRIVEDI INSTITUTE OF ENGINEERING AND TECHNOLOGY
### (An Autonomous Institution)

---

## DEPARTMENT OF COMPUTER ENGINEERING

---

# A Project Report on

# "Hybrid Knowledge Graph + Vector RAG QA System"

### (RAG_QA_SYSTEM)

---

**Under partial fulfilment of Major Project / Mini Project Course**
**Academic Year: 2025–2026**

---

### Submitted By

**\[YOUR NAME\]**
Enrollment No.: \[YOUR ENROLLMENT NO.\] | Semester: 8th (4th Year) | Branch: CSE

---

### Internal Guide

**Prof. \[GUIDE NAME\]**
Dept. of Computer Engineering, SLTIET

### Head of Department

**Prof. \[HOD NAME\]**
Dept. of Computer Engineering, SLTIET

---
---

## DEPARTMENT OF COMPUTER ENGINEERING

---

## CERTIFICATE

This is to certify that the Project work entitled **"Hybrid Knowledge Graph + Vector RAG QA System (RAG_QA_SYSTEM)"** has been satisfactorily completed by **\[YOUR NAME\]** (Enrollment No.: \[YOUR ENROLLMENT NO.\]) of 8th Semester (4th Year), Branch: Computer Science & Engineering, under my guidance in partial fulfilment of the Major Project during the academic year 2025–2026.

&nbsp;

**Internal Guide**
Prof. \[GUIDE NAME\]
Dept. of Computer Engineering
SLTIET, Ahmedabad

**Head of Department**
Prof. \[HOD NAME\]
Dept. of Computer Engineering
SLTIET, Ahmedabad

---

## ACKNOWLEDGEMENT

I would like to express my sincere gratitude to Shri Labhubhai Trivedi Institute of Engineering and Technology for providing me with the opportunity to undertake and complete this project titled **"Hybrid Knowledge Graph + Vector RAG QA System (RAG_QA_SYSTEM)"**. I am deeply thankful to my internal guide, **Prof. \[GUIDE NAME\]**, for his/her valuable guidance, encouragement, and constructive feedback throughout the project.

The development of this system has provided me with practical hands-on experience in advanced Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), Knowledge Graph construction, and full-stack AI application development using modern Python frameworks and libraries.

This project has greatly enhanced my understanding of large language model (LLM) integration, vector database technologies, graph-based information retrieval, and real-world software engineering practices such as modular design, persistent storage, and interactive web application development. I extend my gratitude to all faculty members and peers who offered their suggestions and support during this endeavour.

Finally, I wish to thank my family and friends for their constant motivation and encouragement throughout this journey.

**\[YOUR NAME\]**
Enrollment No.: \[YOUR ENROLLMENT NO.\]

---

## TABLE OF CONTENTS

- Certificate ............................................................... 2
- Acknowledgement ........................................................ 3
- Table of Contents ....................................................... 4
- Chapter 1: Introduction ................................................ 5
  - 1.1 Project Overview ................................................. 5
  - 1.2 Motivation & Problem Statement ................................... 5
  - 1.3 Project Objectives ................................................ 5
- Chapter 2: Technology Stack & Development Areas ........................ 6
  - 2.1 Frontend Development (Streamlit UI) ............................... 6
  - 2.2 Document Ingestion Pipeline ....................................... 6
  - 2.3 Natural Language Processing & Entity Extraction ................... 6
  - 2.4 Knowledge Graph Store ............................................. 7
  - 2.5 Vector Store & Semantic Search .................................... 7
  - 2.6 Query Routing ..................................................... 7
  - 2.7 Answer Generation ................................................. 7
- Chapter 3: Project Details – RAG_QA_SYSTEM ............................. 8
  - 3.1 Project Overview & Objectives ..................................... 8
  - 3.2 System Architecture ................................................ 8
  - 3.3 Module 1: Document Ingestion (ingest.py) .......................... 9
  - 3.4 Module 2: Entity & Relation Extraction (extract.py) ............... 9
  - 3.5 Module 3: Knowledge Graph Store (graph_store.py) .................. 10
  - 3.6 Module 4: Vector Store (vector_store.py) .......................... 10
  - 3.7 Module 5: Query Router (router.py) ................................ 11
  - 3.8 Module 6: Answer Generation (qa.py) ............................... 11
  - 3.9 Configuration & Utilities ......................................... 12
  - 3.10 Streamlit Web Application (app.py) ............................... 12
- Chapter 4: Overall Experience & Learning ............................... 13
  - 4.1 Technical Skills Gained ........................................... 13
  - 4.2 Professional Skills Developed ..................................... 13
  - 4.3 Challenges & How They Were Overcome ............................... 13
- Conclusion .............................................................. 14
- References .............................................................. 15

---

## Chapter 1: Introduction

### 1.1 Project Overview

The **RAG_QA_SYSTEM** is a production-ready, end-to-end Hybrid Knowledge Graph and Vector Retrieval-Augmented Generation (RAG) Question Answering system. The project demonstrates how combining two complementary retrieval techniques — structured Knowledge Graph traversal and unstructured dense vector search — leads to a significantly more robust and accurate question answering system compared to using either approach in isolation.

The system is built as an interactive web application using the Streamlit framework, enabling users to upload documents (PDF, TXT, Markdown), index them through a full NLP pipeline, and then pose natural language questions to receive grounded, cited answers. The system intelligently routes each query to the most appropriate retrieval strategy: graph-based, vector-based, or a hybrid of both.

### 1.2 Motivation & Problem Statement

Traditional Retrieval-Augmented Generation systems rely exclusively on dense vector similarity search. While effective for open-domain semantic questions, they struggle with relational and entity-centric questions such as:
- *"Who founded OpenAI?"*
- *"Which companies are related to DeepMind?"*
- *"Where is Anthropic headquartered?"*

These questions require structured relational knowledge, which is precisely what Knowledge Graphs are designed to store and query. At the same time, purely graph-based systems lack the ability to answer open-ended conceptual questions like *"What are the limitations of RAG systems?"* — questions that require understanding broad semantic context from unstructured text.

This project addresses both limitations by developing a **hybrid system** that intelligently routes queries to the most suitable retrieval backend based on the nature of the question.

### 1.3 Project Objectives

- Build a full-stack AI-powered document question answering system from scratch.
- Implement a complete document ingestion pipeline supporting PDF, TXT, and Markdown formats with overlap-based chunking.
- Develop a Named Entity Recognition (NER) and relation extraction pipeline to automatically construct a Knowledge Graph from uploaded documents.
- Implement a persistent Knowledge Graph using NetworkX with a Neo4j-compatible interface.
- Build a semantic vector store using ChromaDB and sentence-transformer embeddings.
- Design and implement a rule-based query router (with optional LLM-based override) that classifies queries as graph, vector, or hybrid.
- Integrate OpenAI GPT for grounded answer generation with source citations.
- Deploy the complete system as an interactive Streamlit web application with graph visualisation.
- Ensure system extensibility, maintainability, and production readiness through modular design and clean code practices.

---

## Chapter 2: Technology Stack & Development Areas

The RAG_QA_SYSTEM was built using a modern, industry-relevant Python technology stack. The following sections describe the core technologies and frameworks employed throughout the project.

### 2.1 Frontend Development (Streamlit UI)

The user interface was built using **Streamlit**, a Python-native web application framework designed for data science and AI applications. The UI provides a clean, tab-based, responsive interface with no frontend JavaScript or HTML coding required.

- Tab-based layout: Q&A, Graph Visualisation, and Debug/Stats tabs
- Sidebar document uploader supporting PDF, TXT, and MD files with multi-file selection
- Session-state management for persistent stores across page reruns
- Real-time progress bars during document indexing
- Interactive graph visualisation using PyVis rendered inside Streamlit
- System stats dashboard showing graph node/edge counts and vector chunk count

### 2.2 Document Ingestion Pipeline

The ingestion pipeline (`ingest.py`) handles the loading and preprocessing of documents into text chunks suitable for downstream NLP processing and embedding.

- Supports three file formats: `.pdf` (via pypdf), `.txt` (UTF-8 text), `.md` (Markdown with syntax stripping)
- Overlap-based word-level chunking: configurable `CHUNK_SIZE` (default 500 words) with `CHUNK_OVERLAP` (default 50 words)
- Sentence-boundary-aware splitting to avoid mid-sentence truncation
- Deterministic MD5-based chunk ID generation for deduplication
- `Chunk` dataclass encapsulating: `chunk_id`, `text`, `source_file`, `chunk_index`, and `metadata`

### 2.3 Natural Language Processing & Entity Extraction

The extraction pipeline (`extract.py`) uses a multi-stage approach to extract structured knowledge (subject–relation–object triples) from document chunks.

- **spaCy NER**: Named Entity Recognition using `en_core_web_sm` to identify entities of types: PERSON, ORG, GPE, LOC, PRODUCT, WORK_OF_ART, EVENT, LAW, LANGUAGE, NORP, FAC
- **LLM-based relation extraction**: When an OpenAI API key is configured, GPT is prompted to extract (subject, relation, object) triples as a structured JSON array
- **Heuristic fallback**: Eight regex-based patterns covering common relation types (founded, acquired, located_in, works_at, subsidiary_of, partnered_with, created, is_head_of) plus co-occurrence-based triple generation for entity pairs appearing in the same sentence
- `Triple` dataclass encapsulating: `subject`, `relation`, `object`, `source_chunk`, `source_document`

### 2.4 Knowledge Graph Store

The Knowledge Graph layer (`graph_store.py`) stores and queries the extracted triples as a directed multigraph.

- Built on **NetworkX** `MultiDiGraph` with a Neo4j-compatible public interface for future backend swapping
- Persistent storage via Python `pickle` serialisation to `storage/knowledge_graph.pkl`
- Node-level document provenance tracking: each node stores the set of source documents it appears in
- Case-insensitive fuzzy node search for flexible entity lookup
- Neighbour queries returning both incoming and outgoing edges with full metadata
- Graph statistics: node count, edge count, weak connectivity
- Support for document-level removal of nodes and edges

### 2.5 Vector Store & Semantic Search

The vector layer (`vector_store.py`) provides dense semantic search over document chunks using embeddings.

- Built on **ChromaDB** `PersistentClient` with cosine similarity space (`hnsw:space: cosine`)
- Embeddings generated by **sentence-transformers** `all-MiniLM-L6-v2` (384-dimensional)
- Automatic deduplication by `chunk_id` on insertion
- Similarity scores computed as `1 − cosine_distance` for intuitive relevance ranking
- Configurable `TOP_K_RETRIEVAL` (default: 5 chunks per query)
- Optional source-file filter for targeted retrieval

### 2.6 Query Routing

The query router (`router.py`) classifies each natural language query and decides which retrieval backend(s) to invoke.

- **Rule-based classifier**: scores queries against 16 graph-signal patterns (who, founded, located, relation, acquired, etc.) and 16 vector-signal patterns (explain, describe, summarize, what is, how does, why, etc.)
- Routing logic: graph-only if only graph signals, vector-only if only vector signals, hybrid if both signals present or no signals
- **Optional LLM override**: OpenAI GPT can classify queries into `graph`, `vector`, or `hybrid` with a structured system prompt when enabled via UI toggle
- Configurable keyword lists via `config.py`

### 2.7 Answer Generation

The QA module (`qa.py`) synthesises grounded natural language answers from retrieved evidence.

- Dual-source context builder: formats graph facts as `SUBJECT —[RELATION]→ OBJECT (from: source)` and vector chunks as labelled text blocks with similarity scores
- **LLM answer generation**: OpenAI GPT (`gpt-4o-mini` by default) prompted with a strict grounding instruction to answer only from provided context
- **No-LLM fallback**: returns raw retrieved context with a clear notice when no API key is configured
- `QAResult` dataclass encapsulating: `answer`, `route`, `graph_facts`, `vector_chunks`, `sources`
- Source citation collection from both graph facts and vector chunks

---

## Chapter 3: Project Details – RAG_QA_SYSTEM

### 3.1 Project Overview & Objectives

| Field | Details |
|---|---|
| **Project Title** | Hybrid Knowledge Graph + Vector RAG QA System |
| **Domain** | Natural Language Processing / AI / Full-Stack Development |
| **Project Type** | Major Project |
| **Semester** | Semester 8 |
| **Keywords** | RAG, Knowledge Graph, Vector Store, NER, LLM, Streamlit, ChromaDB, NetworkX |
| **Repository** | https://github.com/Milan-CSE/RAG_QA_SYSTEM |
| **Internal Guide** | Prof. \[GUIDE NAME\] |

**Project Abstract:**

RAG_QA_SYSTEM is a hybrid document question answering system that combines Knowledge Graph-based relational retrieval with dense vector semantic search. The system allows users to upload documents of any format (PDF, TXT, Markdown), runs them through a full NLP pipeline to extract entities and relations, constructs a persistent Knowledge Graph alongside a ChromaDB vector index, and then answers natural language questions using an intelligent query router and OpenAI GPT. A clean Streamlit web interface provides document upload, Q&A, interactive knowledge graph visualisation, and system debugging capabilities.

### 3.2 System Architecture

The RAG_QA_SYSTEM follows a five-layer pipeline architecture:

```
┌───────────────────────────────────────────────────────┐
│              Streamlit Web Application (app.py)        │
│   [Upload] → [Index] → [Ask] → [Visualise] → [Debug]  │
└──────────────────────┬────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     Indexing Pipeline      │
         │  ingest.py → extract.py    │
         │  GraphStore ← → VectorStore│
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │       Query Router         │
         │        router.py           │
         └──────┬──────────┬──────────┘
                │          │
          ┌─────▼──┐   ┌───▼──────┐
          │ Graph  │   │  Vector  │
          │ Store  │   │  Store   │
          │(NX/pkl)│   │(Chroma)  │
          └──┬─────┘   └─────┬────┘
             │               │
             └───────┬───────┘
                     │ context
           ┌─────────▼─────────┐
           │   LLM (OpenAI)    │
           │   Answer Gen      │
           └─────────┬─────────┘
                     │
                Final Answer
                + Citations
```

**Architecture Layers:**
- **Presentation Layer**: Streamlit web application with sidebar, multi-tab layout, and PyVis graph rendering
- **Ingestion & Extraction Layer**: Document loading, chunking, NER, and relation extraction pipeline
- **Storage Layer**: Dual-backend — NetworkX Knowledge Graph (pickle-persisted) + ChromaDB vector collection (disk-persisted)
- **Routing Layer**: Dual-strategy query classifier (rule-based + optional LLM)
- **Generation Layer**: OpenAI GPT answer synthesis with context grounding and citation collection

### 3.3 Module 1: Document Ingestion (`ingest.py`)

The ingestion module is the entry point of the data pipeline. It loads raw documents, preprocesses and cleans the text, and splits it into overlapping chunks for downstream processing.

**Supported Formats & Loaders:**

| Format | Library | Processing |
|---|---|---|
| `.txt` | Built-in `pathlib` | Direct UTF-8 read |
| `.md` | Built-in `re` | Strip headings, bold, code, images, links |
| `.pdf` | pypdf | Extract text from all pages |

**Chunking Algorithm:**
- Sentence-boundary-aware splitting using regex `(?<=[.!?])\s+`
- Word-count-based target chunk size (default: 500 words)
- Configurable overlap (default: 50 words) retained between adjacent chunks to preserve cross-boundary context
- Chunk IDs generated as MD5 hash of `source_file:index:text_prefix` for deterministic deduplication

**Key Data Structure:**
```python
@dataclass
class Chunk:
    chunk_id: str        # MD5 hash identifier
    text: str            # Cleaned chunk text
    source_file: str     # Original filename
    chunk_index: int     # Position in document
    metadata: dict       # Extension, file path
```

### 3.4 Module 2: Entity & Relation Extraction (`extract.py`)

This module forms the knowledge extraction backbone of the system, converting unstructured text chunks into structured (subject, relation, object) triples that populate the Knowledge Graph.

**Entity Recognition:**
- Uses spaCy `en_core_web_sm` model for Named Entity Recognition
- Extracts entities of 11 types: PERSON, ORG, GPE, LOC, PRODUCT, WORK_OF_ART, EVENT, LAW, LANGUAGE, NORP, FAC
- Lazy model loading with auto-download fallback

**Relation Extraction — Three-Tier Strategy:**

| Tier | Method | Trigger |
|---|---|---|
| 1st | LLM-based (OpenAI GPT) | OpenAI API key configured |
| 2nd | Regex pattern matching | LLM unavailable or returns empty |
| 3rd | Co-occurrence triples | Supplementary to pattern matching |

**Heuristic Relation Patterns:**

| Pattern | Relation Label |
|---|---|
| "X founded/created/established Y" | `founded` |
| "X is CEO/director/head of Y" | `is_head_of` |
| "X acquired/purchased/bought Y" | `acquired` |
| "X works at/for Y" | `works_at` |
| "X is located/based in Y" | `located_in` |
| "X is a subsidiary/division of Y" | `subsidiary_of` |
| "X partnered/collaborated with Y" | `partnered_with` |
| "X invented/developed/built Y" | `created` |

**Key Data Structure:**
```python
@dataclass
class Triple:
    subject: str          # Head entity
    relation: str         # Relation type / verb phrase
    object: str           # Tail entity
    source_chunk: str     # chunk_id provenance
    source_document: str  # Filename provenance
```

### 3.5 Module 3: Knowledge Graph Store (`graph_store.py`)

The graph store provides a persistent, queryable Knowledge Graph layer built on NetworkX's `MultiDiGraph`, designed with a Neo4j-compatible interface for future backend migration.

**Graph Model:**
- **Nodes**: Named entities extracted from documents; each node stores a set of source document names
- **Edges**: Directed relations between entity pairs with metadata (relation type, source chunk, source document)
- **MultiDiGraph**: Allows multiple distinct edges between the same node pair (different relations or sources)

**Key Operations:**

| Operation | Method | Description |
|---|---|---|
| Add triple | `add_triple()` | Add a single (subject, relation, object) edge |
| Bulk add | `add_triples()` | Add a list of Triple objects |
| Neighbour query | `query_neighbors()` | Retrieve all in/out edges of a named entity |
| Fuzzy search | `search_nodes()` | Case-insensitive substring node search |
| All triples | `get_all_triples()` | Return every triple as a list of dicts |
| Statistics | `get_stats()` | Node count, edge count, connectivity |
| Persistence | `save()` / `load()` | Pickle serialisation to disk |
| Reset | `clear()` | Wipe graph and delete persisted file |

### 3.6 Module 4: Vector Store (`vector_store.py`)

The vector store provides dense semantic search over document chunks using sentence embeddings and ChromaDB.

**Embedding Model:**
- `sentence-transformers/all-MiniLM-L6-v2`: lightweight (22M parameters), 384-dimensional embeddings, strong semantic similarity performance
- Lazy initialisation with error-recovery guidance for offline environments

**ChromaDB Configuration:**
- `PersistentClient` with on-disk storage at `storage/chroma/`
- Cosine similarity space (`hnsw:space: cosine`) for normalized embedding comparison
- Collection: `hybrid_qa` (configurable via `CHROMA_COLLECTION` env variable)

**Key Operations:**

| Operation | Method | Description |
|---|---|---|
| Add chunks | `add_chunks()` | Embed and store chunk list; skips duplicates |
| Semantic search | `search()` | Cosine similarity search, returns top-K hits |
| Remove document | `remove_document()` | Delete all chunks from a specific source file |
| Stats | `get_stats()` | Total chunk count |
| Reset | `clear()` | Delete and recreate collection |

**Search Result Format:**
Each search hit returns: `text`, `source_file`, `chunk_index`, `score` (1 − cosine distance), `chunk_id`.

### 3.7 Module 5: Query Router (`router.py`)

The query router classifies incoming natural language queries and decides which retrieval backend(s) to invoke.

**Rule-Based Classification:**

| Query Signal | Route | Example |
|---|---|---|
| Graph patterns only | `graph` | "Who founded OpenAI?" |
| Vector patterns only | `vector` | "What is retrieval-augmented generation?" |
| Both signal types | `hybrid` | "How is Google related to DeepMind and what does it do?" |
| No signal matched | `hybrid` | Ambiguous or very short queries |

**Graph Signal Patterns (16):** `\bwho\b`, `\bwhom\b`, `\bwhose\b`, founded, relation, connection, located, headquartered, acquired, subsidiary, partnered, `between…and`, etc.

**Vector Signal Patterns (16):** explain, describe, summarize, overview, define, `what is/are`, `how does/do`, why, limitations, advantages, `tell me about`, etc.

**LLM Override (Optional):**
When enabled via the Streamlit UI toggle, OpenAI GPT is first asked to classify the query with a structured system prompt, and the result is used if valid; otherwise the rule-based result is used as fallback.

### 3.8 Module 6: Answer Generation (`qa.py`)

The QA module orchestrates the full retrieval and generation pipeline to produce a final grounded answer.

**Retrieval Logic by Route:**

| Route | Graph Retrieval | Vector Retrieval |
|---|---|---|
| `graph` | ✅ Top-3 entities, neighbour query | ❌ |
| `vector` | ❌ | ✅ Top-K semantic chunks |
| `hybrid` | ✅ Top-3 entities, neighbour query | ✅ Top-K semantic chunks |

**Graph Entity Matching:**
- `graph_store.search_nodes(query)` is used first to find the top-5 matching entities
- If no match, individual query words (length > 3) are searched as fallback
- Top-3 matched entities are used for neighbour queries, capped at 20 deduplicated facts

**Context Building:**
```
GRAPH FACTS:
  • OpenAI —[founded_by]→ Sam Altman  (from: sample_ai_companies.txt)
  ...

DOCUMENT CONTEXT:
[Chunk 1 — sample_rag_overview.md (score: 0.87)]
Retrieval-Augmented Generation (RAG) is a technique that...
```

**Answer Modes:**

| Mode | Trigger | Output |
|---|---|---|
| LLM-grounded | `OPENAI_API_KEY` set | GPT-generated answer with citations |
| Fallback | No API key | Raw retrieved context with informational note |

**Key Data Structure:**
```python
@dataclass
class QAResult:
    answer: str            # Final answer text
    route: str             # "graph" / "vector" / "hybrid"
    graph_facts: List[dict]
    vector_chunks: List[dict]
    sources: List[str]     # Deduplicated source filenames
```

### 3.9 Configuration & Utilities

**`config.py` — Centralised Configuration:**
All system parameters are configurable via environment variables (`.env` file), with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | OpenAI API key for LLM features |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for extraction, routing, and answering |
| `CHUNK_SIZE` | `500` | Target words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between adjacent chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of vector chunks per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NER model |
| `NEO4J_URI` | *(none)* | Optional Neo4j URI for graph backend swap |

**`utils.py` — Shared Utilities:**
- `get_logger()`: Dual-sink logger (console + file) with timestamp and level formatting
- `clean_text()`: Strips non-printable characters and normalises whitespace
- `truncate()`: Safe string truncation for display with ellipsis
- `sanitize_filename()`: Removes filesystem-unsafe characters
- `file_extension()`: Case-normalised extension extraction

### 3.10 Streamlit Web Application (`app.py`)

The Streamlit application ties all modules together into an interactive web UI accessible at `http://localhost:8501`.

**Sidebar:**
- File uploader (PDF, TXT, MD, multi-file)
- LLM router toggle
- "Index Documents" primary action button
- Real-time system stats metrics (graph nodes, edges, vector chunks)
- "Clear All Data" reset button
- API key status indicator

**Tab 1 — Question & Answer:**
- Question text input with route override selector (auto / graph / vector / hybrid)
- Top-K slider (1–10)
- Route badge display (Graph 🔵, Vector 🟢, Hybrid 🟡)
- Answer display with markdown rendering
- Source citation line
- Side-by-side evidence display: graph facts table and vector chunk expanders
- Sample questions reference expander

**Tab 2 — Graph Visualisation:**
- Node search with neighbour traversal display
- Full interactive PyVis graph: physics-enabled, directed edges, degree-based node colouring (steel-blue → orange-red gradient), node size proportional to degree
- Configurable: shows up to 200 nodes (top-100 by degree if larger)
- All triples table (pandas DataFrame) in expander

**Tab 3 — Debug / Stats:**
- System configuration JSON display
- Graph statistics JSON (with top-10 nodes by degree)
- Vector store statistics JSON
- Retrieval preview: test vector retrieval with top-3 results
- Router test: classify any query and see the routing decision

---

## Chapter 4: Overall Experience & Learning

### 4.1 Technical Skills Gained

Developing the RAG_QA_SYSTEM provided deep, practical exposure to several advanced areas of AI and software engineering:

- **RAG Architecture**: Hands-on design and implementation of a production-quality Retrieval-Augmented Generation system combining two distinct retrieval paradigms
- **Knowledge Graph Engineering**: Building, querying, and persisting a directed multigraph for structured knowledge representation using NetworkX with a Neo4j-extensible interface
- **Vector Database & Embeddings**: Working with ChromaDB for persistent semantic search and sentence-transformers for generating high-quality text embeddings
- **NLP Pipeline Development**: Implementing a multi-stage NLP pipeline covering text cleaning, chunking, Named Entity Recognition (spaCy), and relation extraction (LLM + heuristic)
- **LLM Integration**: Integrating OpenAI GPT for relation extraction, query routing, and grounded answer generation with structured prompting techniques
- **Streamlit Application Development**: Building a full-featured, multi-tab AI web application with real-time UI feedback, session state management, and interactive graph visualisation (PyVis)
- **Modular Python Software Design**: Applying clean separation of concerns across dedicated modules (`ingest`, `extract`, `graph_store`, `vector_store`, `router`, `qa`, `config`, `utils`)
- **Configuration Management**: Using environment-variable-driven configuration with `python-dotenv` for twelve-factor app compliance
- **Testing**: Writing comprehensive unit tests with `pytest` covering ingestion, extraction, graph operations, vector store, routing, and utilities (45+ test cases)
- **Version Control**: Using Git with meaningful commit messages and clean project structure

### 4.2 Professional Skills Developed

- **System Design**: Designing a multi-component AI system with well-defined interfaces and extensibility points (e.g., Neo4j swap-in, REBEL relation extraction)
- **Technical Documentation**: Comprehensive docstrings across all modules, a detailed README with architecture diagrams, and this project report
- **Problem Decomposition**: Breaking down a complex AI problem (hybrid QA) into independently testable and replaceable modules
- **Research & Implementation**: Reading and implementing concepts from academic literature on RAG, Knowledge Graph-enhanced RAG (KG-RAG), and hybrid retrieval systems
- **Performance Considerations**: Implementing chunking with overlap, cosine similarity scoring, graph display limits, and deduplication to ensure practical performance
- **Fallback Design**: Designing graceful degradation (no API key → fallback mode; spaCy missing → auto-download; PDF library missing → raw read fallback)

### 4.3 Challenges & How They Were Overcome

| Challenge | Solution / Learning |
|---|---|
| LLM relation extraction returning inconsistent JSON | Added markdown code-fence stripping with regex and wrapped in try/except with fallback to heuristic extractor |
| Graph node matching across case variants | Implemented case-insensitive node matching in `query_neighbors()` and `search_nodes()` for robust entity lookup |
| ChromaDB duplicate chunk insertion on re-indexing | Added pre-insertion duplicate check using `collection.get(ids=ids)` to fetch existing IDs before adding |
| PyVis graph rendering inside Streamlit | Used `tempfile` to write HTML, read it back as a string, then rendered with `st.components.v1.html()` |
| Graph becoming too large to visualise | Added automatic subgraph extraction: limit to top-100 nodes by degree when graph exceeds 200 nodes |
| spaCy model not available offline on first run | Implemented auto-download via `subprocess` call to `python -m spacy download` with a clear warning log |
| Ambiguous query routing (both graph and vector signals) | Defaulted to `hybrid` route to maximise recall; added optional LLM override for more precise classification |
| PDF libraries producing garbled text | Used pypdf as primary PDF loader with graceful exception handling and latin-1 raw bytes fallback |

---

## Conclusion

The **RAG_QA_SYSTEM** project has been an exceptionally enriching experience in designing and building a production-ready AI system that pushes beyond the standard single-backend RAG architecture. Over the course of this project, I successfully designed, implemented, and delivered a fully functional hybrid question answering system that intelligently combines Knowledge Graph relational retrieval with dense vector semantic search.

The project encompassed the complete software development lifecycle — from requirements analysis, system architecture design, and module implementation, through to comprehensive testing and a polished interactive web deployment. Building real-world components including a multi-format document ingestion pipeline, a multi-stage NER and relation extraction engine, a persistent NetworkX Knowledge Graph with Neo4j-extensible interface, a ChromaDB-backed semantic vector store, a dual-strategy query router, and an OpenAI GPT-powered answer generation system has significantly deepened my capabilities as an AI/ML engineer and Python developer.

The system's modular architecture ensures that each component can be independently tested, replaced, or extended — for example, swapping NetworkX for Neo4j, adding REBEL-based relation extraction, or upgrading to a more powerful embedding model — without requiring changes to dependent modules. This design discipline reflects industry-standard software engineering principles applied to AI system development.

The skills and knowledge gained during this project — spanning NLP pipeline design, Knowledge Graph engineering, vector database technology, LLM integration, query routing, Streamlit web development, and unit testing — form a strong and practical foundation for a career in AI/ML engineering and NLP research. The RAG_QA_SYSTEM stands as a comprehensive demonstration of advanced AI system development and reflects the depth of learning achieved through this project.

---

## References

1. Lewis, P., Perez, E., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
2. GitHub Repository – RAG_QA_SYSTEM: https://github.com/Milan-CSE/RAG_QA_SYSTEM
3. spaCy Documentation: https://spacy.io/usage
4. ChromaDB Documentation: https://docs.trychroma.com/
5. Sentence Transformers Documentation: https://www.sbert.net/
6. NetworkX Documentation: https://networkx.org/documentation/stable/
7. OpenAI API Documentation: https://platform.openai.com/docs/
8. Streamlit Documentation: https://docs.streamlit.io/
9. PyVis Documentation: https://pyvis.readthedocs.io/
10. pypdf Documentation: https://pypdf.readthedocs.io/
11. Hugging Face – all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
12. OWASP Security Best Practices: https://owasp.org/
13. Python pytest Documentation: https://docs.pytest.org/
14. python-dotenv Documentation: https://pypi.org/project/python-dotenv/
