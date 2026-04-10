# Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation (RAG) is a natural language processing technique that combines information retrieval with language model generation. It was introduced by researchers at Facebook AI Research (now Meta AI) in a 2020 paper by Patrick Lewis, Ethan Perez, Aleksandra Piktus, and others.

## How RAG Works

RAG operates in two main phases:

**Retrieval Phase:** When a user submits a query, the system searches a knowledge base or document corpus to find the most relevant passages. This is typically done using dense vector embeddings and approximate nearest-neighbour search. The retriever scores documents by semantic similarity to the query.

**Generation Phase:** The retrieved documents are concatenated with the original query and passed as context to a language model. The language model then generates an answer grounded in the retrieved context rather than relying solely on parametric knowledge.

## Knowledge Graphs in RAG

Knowledge Graph-enhanced RAG (KG-RAG) extends the standard approach by incorporating structured relational knowledge. A knowledge graph represents entities as nodes and relationships as edges. This structured representation is particularly useful for answering relational questions such as "Who founded company X?" or "What is the relationship between A and B?"

Hybrid RAG systems combine dense vector retrieval with knowledge graph traversal. Vector retrieval handles open-domain semantic questions, while graph traversal handles entity-centric relational queries. The query router determines which approach to use based on the question type.

## Vector Stores

Common vector store implementations include:
- **ChromaDB**: An open-source embedding database with a simple Python API.
- **FAISS**: Facebook AI Similarity Search, a highly efficient library for dense vector similarity search.
- **Pinecone**: A managed vector database service.
- **Weaviate**: An open-source vector database with hybrid search capabilities.

## Embedding Models

Popular sentence embedding models include:
- **all-MiniLM-L6-v2**: A lightweight but effective model from Sentence Transformers.
- **text-embedding-ada-002**: OpenAI's embedding model.
- **E5**: A family of text embeddings from Microsoft.
- **BGE**: BAAI General Embeddings, known for strong multilingual performance.

## Evaluation Metrics

RAG systems are commonly evaluated using:
- **Faithfulness**: Are the generated answers grounded in the retrieved context?
- **Answer Relevance**: Does the answer address the query?
- **Context Recall**: Were the relevant documents retrieved?
- **Context Precision**: How many of the retrieved documents were relevant?

## Limitations

RAG systems have several known limitations. The quality of answers depends heavily on the quality of the retrieval step. If relevant documents are not retrieved, the generator cannot produce a correct answer. Chunking strategy significantly affects retrieval quality. Very long documents may be split in ways that destroy important context.
