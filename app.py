"""
app.py - Streamlit UI for the Hybrid Knowledge Graph + Vector RAG QA System.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Hybrid KG + Vector QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports (after page config) ─────────────────────────────────────────────
import config
from ingest import load_document, Chunk
from extract import extract_all
from graph_store import GraphStore
from vector_store import VectorStore
from router import classify_query
from qa import answer_query
from utils import get_logger, truncate

logger = get_logger(__name__)


# ─── Session-state helpers ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_stores():
    """Load or initialise persistent stores (cached across reruns)."""
    gs = GraphStore()
    vs = VectorStore()
    return gs, vs


def _reset_stores():
    """Clear all data and reload."""
    gs, vs = get_stores()
    gs.clear()
    vs.clear()
    st.cache_resource.clear()
    st.success("All data cleared.")


# ─── Graph visualisation helper ───────────────────────────────────────────────

def render_graph_html(graph_store: GraphStore, height: int = 500) -> str:
    """Build a PyVis network HTML string from the knowledge graph."""
    try:
        from pyvis.network import Network  # type: ignore
        import networkx as nx  # type: ignore

        G = graph_store.get_graph()
        if G.number_of_nodes() == 0:
            return "<p style='color:gray;'>Graph is empty – index some documents first.</p>"

        # Limit to a manageable sub-graph for display
        if G.number_of_nodes() > 200:
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:100]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        net = Network(
            height=f"{height}px",
            width="100%",
            directed=True,
            bgcolor="#1e1e2e",
            font_color="#cdd6f4",
        )
        net.from_nx(G)

        # Colour nodes by degree
        max_deg = max(dict(G.degree()).values(), default=1)
        for node in net.nodes:
            deg = G.degree(node["id"])
            # Gradient from steel blue (low) to orange-red (high)
            ratio = deg / max_deg
            r = int(30 + ratio * 225)
            green = int(144 - ratio * 100)
            b = int(255 - ratio * 200)
            node["color"] = f"rgb({r},{green},{b})"
            node["title"] = node["id"]
            node["size"] = 10 + deg * 2

        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 120
            },
            "minVelocity": 0.75
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
            "color": { "color": "#45475a" },
            "smooth": { "type": "curvedCW", "roundness": 0.2 }
          }
        }
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            html = Path(f.name).read_text(encoding="utf-8")
        os.unlink(f.name)
        return html

    except ImportError:
        return (
            "<p style='color:orange;'>Install <code>pyvis</code> to enable "
            "graph visualisation.</p>"
        )
    except Exception as exc:
        logger.error("Graph render error: %s", exc)
        return f"<p style='color:red;'>Graph render error: {exc}</p>"


# ─── Indexing pipeline ────────────────────────────────────────────────────────

def index_files(uploaded_files, graph_store: GraphStore, vector_store: VectorStore):
    progress = st.progress(0, text="Starting indexing…")
    total = len(uploaded_files)
    all_chunks: list[Chunk] = []

    for i, uf in enumerate(uploaded_files):
        progress.progress((i) / total, text=f"Loading {uf.name} …")
        suffix = Path(uf.name).suffix.lower()
        if suffix not in config.SUPPORTED_EXTENSIONS:
            st.warning(f"Skipping unsupported file: {uf.name}")
            continue

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="upload_"
        ) as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name

        try:
            chunks = load_document(tmp_path)
            # Override source_file with the original filename
            for c in chunks:
                c.source_file = uf.name
            all_chunks.extend(chunks)
        except Exception as exc:
            st.error(f"Failed to load {uf.name}: {exc}")
        finally:
            os.unlink(tmp_path)

    if not all_chunks:
        progress.empty()
        st.warning("No chunks extracted. Check your files.")
        return

    progress.progress(0.5, text="Extracting entities and relations…")
    triples = extract_all(all_chunks)

    progress.progress(0.7, text="Building knowledge graph…")
    graph_store.add_triples(triples)
    graph_store.save()

    progress.progress(0.85, text="Building vector index…")
    vector_store.add_chunks(all_chunks)

    progress.progress(1.0, text="Done!")
    progress.empty()

    stats_g = graph_store.get_stats()
    stats_v = vector_store.get_stats()

    st.success(
        f"✅ Indexed **{len(uploaded_files)}** file(s) → "
        f"**{len(all_chunks)}** chunks | "
        f"**{len(triples)}** triples extracted | "
        f"Graph: **{stats_g['nodes']}** nodes / **{stats_g['edges']}** edges | "
        f"Vector store: **{stats_v['total_chunks']}** chunks"
    )


# ─── Main UI ─────────────────────────────────────────────────────────────────

def main():
    graph_store, vector_store = get_stores()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/?size=100&id=63777&format=png",
            width=60,
        )
        st.title("Hybrid KG + RAG QA")
        st.caption("Knowledge Graph ⊕ Vector Retrieval")

        st.divider()
        st.subheader("📂 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        use_llm_router = st.toggle(
            "Use LLM for query routing",
            value=False,
            help="Uses OpenAI to classify the query type. Requires API key.",
        )

        if uploaded_files:
            if st.button("🚀 Index Documents", type="primary", use_container_width=True):
                index_files(uploaded_files, graph_store, vector_store)
                st.rerun()

        st.divider()
        stats_g = graph_store.get_stats()
        stats_v = vector_store.get_stats()

        st.subheader("📊 System Stats")
        col1, col2 = st.columns(2)
        col1.metric("Graph Nodes", stats_g["nodes"])
        col2.metric("Graph Edges", stats_g["edges"])
        st.metric("Vector Chunks", stats_v["total_chunks"])

        st.divider()
        if st.button("🗑️ Clear All Data", use_container_width=True):
            _reset_stores()
            st.rerun()

        st.divider()
        st.caption("💡 API key status: " + (
            "✅ OpenAI configured" if config.OPENAI_API_KEY
            else "⚠️ No OpenAI key (fallback mode)"
        ))

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_qa, tab_graph, tab_debug = st.tabs(
        ["💬 Question & Answer", "🕸️ Graph Visualisation", "🔍 Debug / Stats"]
    )

    # ── Q&A Tab ───────────────────────────────────────────────────────────────
    with tab_qa:
        st.header("Ask a Question")

        with st.form("qa_form"):
            query = st.text_input(
                "Your question",
                placeholder="e.g. Who founded OpenAI? / What is RAG?",
            )
            route_override = st.selectbox(
                "Route override (auto = let the router decide)",
                ["auto", "hybrid", "graph", "vector"],
            )
            top_k = st.slider("Top-K vector results", 1, 10, config.TOP_K_RETRIEVAL)
            submitted = st.form_submit_button("🔎 Ask", type="primary")

        if submitted and query.strip():
            if stats_v["total_chunks"] == 0 and stats_g["nodes"] == 0:
                st.warning(
                    "⚠️ No documents indexed yet. "
                    "Please upload and index documents first."
                )
            else:
                with st.spinner("Thinking…"):
                    route = (
                        classify_query(query, use_llm=use_llm_router)
                        if route_override == "auto"
                        else route_override
                    )
                    result = answer_query(
                        query=query,
                        route=route,
                        graph_store=graph_store,
                        vector_store=vector_store,
                        top_k=top_k,
                    )

                # Answer display
                route_badge = {
                    "graph": "🔵 Graph",
                    "vector": "🟢 Vector",
                    "hybrid": "🟡 Hybrid",
                }.get(result.route, result.route)
                st.info(f"**Route:** {route_badge}", icon="🧭")

                st.subheader("📝 Answer")
                st.markdown(result.answer)

                if result.sources:
                    st.caption("📚 Sources: " + " · ".join(result.sources))

                # Evidence columns
                col_g, col_v = st.columns(2)

                with col_g:
                    st.subheader(f"🔗 Graph Facts ({len(result.graph_facts)})")
                    if result.graph_facts:
                        for f in result.graph_facts[:15]:
                            st.markdown(
                                f"**{f['subject']}** —`{f['relation']}`→ **{f['object']}**  \n"
                                f"<small>📄 {f.get('source_document', '?')}</small>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No graph facts retrieved.")

                with col_v:
                    st.subheader(f"📄 Document Chunks ({len(result.vector_chunks)})")
                    if result.vector_chunks:
                        for chunk in result.vector_chunks:
                            with st.expander(
                                f"📄 {chunk.get('source_file','?')} "
                                f"(score: {chunk.get('score',0):.2f})"
                            ):
                                st.write(chunk["text"])
                    else:
                        st.caption("No chunks retrieved.")

        # Sample questions
        with st.expander("💡 Sample Questions"):
            samples = [
                "Who founded the company described in the document?",
                "What is the main topic of the uploaded documents?",
                "Which organisations are mentioned and how are they related?",
                "Summarise the key points from the documents.",
                "What technology is described and how does it work?",
            ]
            for q in samples:
                st.code(q, language=None)

    # ── Graph Tab ─────────────────────────────────────────────────────────────
    with tab_graph:
        st.header("Knowledge Graph Visualisation")

        col_search, col_refresh = st.columns([3, 1])
        with col_search:
            node_search = st.text_input("Search node", placeholder="e.g. OpenAI")
        with col_refresh:
            st.write("")
            do_search = st.button("🔍 Find neighbours")

        if do_search and node_search.strip():
            neighbours = graph_store.query_neighbors(node_search.strip())
            if neighbours:
                st.subheader(f"Neighbours of '{node_search}'")
                for n in neighbours:
                    st.markdown(
                        f"**{n['source']}** —`{n['relation']}`→ **{n['target']}**  "
                        f"<small>({n.get('source_document','?')})</small>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"No neighbours found for '{node_search}'.")

        st.divider()
        if stats_g["nodes"] > 0:
            with st.spinner("Rendering graph…"):
                html = render_graph_html(graph_store, height=600)
            st.components.v1.html(html, height=620, scrolling=False)
        else:
            st.info(
                "Graph is empty. Upload and index documents to populate it.",
                icon="🕸️",
            )

        # Triple table
        with st.expander(f"📋 All Triples ({stats_g['edges']} total)"):
            triples = graph_store.get_all_triples()
            if triples:
                import pandas as pd
                df = pd.DataFrame(triples)[
                    ["subject", "relation", "object", "source_document"]
                ]
                st.dataframe(df, use_container_width=True)
            else:
                st.caption("No triples yet.")

    # ── Debug Tab ─────────────────────────────────────────────────────────────
    with tab_debug:
        st.header("Debug & Evaluation")

        st.subheader("System Information")
        debug_info = {
            "openai_configured": bool(config.OPENAI_API_KEY),
            "openai_model": config.OPENAI_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "spacy_model": config.SPACY_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "graph_persist_path": config.GRAPH_PERSIST_PATH,
            "chroma_persist_dir": config.CHROMA_PERSIST_DIR,
        }
        st.json(debug_info)

        st.subheader("Graph Statistics")
        g_stats = graph_store.get_stats()
        g_stats["top_nodes_by_degree"] = []
        try:
            G = graph_store.get_graph()
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
            g_stats["top_nodes_by_degree"] = [
                {"node": n, "degree": d} for n, d in top_nodes
            ]
        except Exception:
            pass
        st.json(g_stats)

        st.subheader("Vector Store Statistics")
        st.json(vector_store.get_stats())

        st.subheader("Retrieval Preview")
        preview_query = st.text_input(
            "Test vector retrieval",
            placeholder="Enter any query to preview retrieved chunks…",
            key="preview_q",
        )
        if preview_query:
            hits = vector_store.search(preview_query, top_k=3)
            if hits:
                for h in hits:
                    st.markdown(
                        f"**Score:** {h['score']:.3f} | **Source:** {h['source_file']}"
                    )
                    st.text(truncate(h["text"], 250))
                    st.divider()
            else:
                st.info("No results.")

        st.subheader("Router Test")
        test_query = st.text_input(
            "Test router",
            placeholder="Enter a query to see how the router classifies it…",
            key="router_q",
        )
        if test_query:
            route = classify_query(test_query)
            st.success(f"Route decision: **{route}**")


if __name__ == "__main__":
    main()
