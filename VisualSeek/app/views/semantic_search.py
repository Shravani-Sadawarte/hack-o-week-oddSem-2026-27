import streamlit as st
from typing import Optional, List
from pathlib import Path

from src.config import AppConfig
from src.search.search_engine import SearchEngine, SearchResult
from src.models.base import BaseEncoder
from app.components.image_grid import render_image_grid

def execute_semantic_search(
    query_text: str,
    config: AppConfig,
    engine: SearchEngine,
    encoder: BaseEncoder,
    top_k: int,
    engine_mode: str
):
    """Executes semantic search and updates session state."""
    clean_query = query_text.strip()
    if not clean_query:
        st.warning("Please enter a valid description to search.")
        return

    with st.spinner(f"Generating CLIP text embedding & searching vector index..."):
        try:
            results, latency_ms = engine.search_by_text(
                text_query=clean_query,
                encoder=encoder,
                top_k=top_k,
                engine=engine_mode
            )
            st.session_state["semantic_active_query"] = clean_query
            st.session_state["semantic_results"] = results
            st.session_state["semantic_latency_ms"] = latency_ms
        except Exception as e:
            st.error(f"Search failed: {e}")

def render_semantic_search(
    config: AppConfig,
    engine: Optional[SearchEngine] = None,
    encoder: Optional[BaseEncoder] = None,
    *args,
    **kwargs
):
    st.markdown('<h1 style="color: #0F172A; font-weight: 800; margin-bottom: 4px;">💬 Semantic Natural Language Search</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #475569; font-size: 1rem; margin-bottom: 24px;">'
        'Search the image collection using descriptive natural language queries. '
        'OpenAI CLIP projects the text prompt into the shared 512-dimensional vector space to retrieve matching dog images.'
        '</p>',
        unsafe_allow_html=True
    )

    if encoder is None and config is not None:
        try:
            from src.models.clip_encoder import CLIPEncoder
            encoder = CLIPEncoder(config.model)
        except Exception:
            encoder = None

    if engine is None or encoder is None:
        st.error("⚠️ Search index or model encoder is unavailable. Please run the setup pipeline first:")
        st.code("python scripts/generate_embeddings.py --limit 1000\npython scripts/build_index.py", language="bash")
        return

    top_k = st.session_state.get("top_k", 10)
    search_engine_mode = st.session_state.get("search_engine", "faiss")

    # 1. State Initialization
    if "semantic_input_query" not in st.session_state:
        st.session_state["semantic_input_query"] = "white fluffy dog in grass"
    if "semantic_active_query" not in st.session_state:
        st.session_state["semantic_active_query"] = None
    if "semantic_results" not in st.session_state:
        st.session_state["semantic_results"] = None
    if "semantic_latency_ms" not in st.session_state:
        st.session_state["semantic_latency_ms"] = 0.0

    # 2. Quick Search Suggestions
    st.markdown('<div style="font-size: 0.88rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Quick Suggestions</div>', unsafe_allow_html=True)
    quick_suggestions = [
        "white fluffy small dog",
        "golden retriever sitting outdoors",
        "black and tan hound dog",
        "spotted dalmatian puppy"
    ]

    btn_cols = st.columns(4)
    for col, suggestion in zip(btn_cols, quick_suggestions):
        with col:
            if st.button(f'🐾 {suggestion}', use_container_width=True, key=f"quick_{suggestion}"):
                st.session_state["semantic_input_query"] = suggestion
                execute_semantic_search(
                    query_text=suggestion,
                    config=config,
                    engine=engine,
                    encoder=encoder,
                    top_k=top_k,
                    engine_mode=search_engine_mode
                )
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Main Search Form
    with st.form("semantic_search_form", clear_on_submit=False):
        col_input, col_submit = st.columns([4, 1])
        with col_input:
            text_input_val = st.text_input(
                "Describe the dog you're looking for:",
                value=st.session_state["semantic_input_query"],
                key="semantic_text_box",
                placeholder="e.g. fluffy white dog running in grass, brown terrier puppy...",
                label_visibility="collapsed"
            )
        with col_submit:
            submit_search = st.form_submit_button("🔎 Search", type="primary", use_container_width=True)

    if submit_search:
        st.session_state["semantic_input_query"] = text_input_val
        execute_semantic_search(
            query_text=text_input_val,
            config=config,
            engine=engine,
            encoder=encoder,
            top_k=top_k,
            engine_mode=search_engine_mode
        )
        st.rerun()

    # 4. Display Results
    active_query = st.session_state.get("semantic_active_query")
    results = st.session_state.get("semantic_results")
    latency_ms = st.session_state.get("semantic_latency_ms", 0.0)

    if active_query and results is not None:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">Results for: </span>
                    <span style="font-size: 1.25rem; font-weight: 800; color: #2563EB;">"{active_query}"</span>
                </div>
                <div style="font-size: 0.85rem; color: #64748B;">
                    Found <strong style="color: #0F172A;">{len(results)}</strong> matches · 
                    Latency: <strong style="color: #059669;">{latency_ms:.2f} ms</strong> ({search_engine_mode.upper()})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if len(results) > 0:
            render_image_grid(results, columns_per_row=4)
        else:
            # Empty state
            st.markdown(
                """
                <div style="
                    border: 1px dashed #CBD5E1;
                    border-radius: 12px;
                    padding: 32px;
                    text-align: center;
                    background: #F8FAFC;
                    margin: 20px 0;
                ">
                    <h3 style="color: #0F172A; margin-bottom: 8px;">No matching images found</h3>
                    <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 12px;">Try adjusting your search criteria:</p>
                    <ul style="color: #334155; text-align: left; display: inline-block; font-size: 0.9rem;">
                        <li>Use broader descriptions (e.g. "large brown dog" instead of specific rare traits)</li>
                        <li>Focus on visible characteristics like color, fur texture, or posture</li>
                        <li>Increase the Top-K setting in the sidebar</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Developer Diagnostic View
        with st.expander("🛠️ Developer Diagnostics (Retrieval Verification)", expanded=False):
            st.markdown(f"**Active Query**: `{active_query}`")
            st.markdown(f"**Query Embedding Dimension**: `{encoder.embedding_dim}`")
            st.markdown(f"**Indexed Corpus Size**: `{engine.num_items}` images")
            st.markdown(f"**Search Latency**: `{latency_ms:.3f} ms`")
            if results:
                st.markdown(f"**Top Result ID**: `{results[0].image_id}` | Breed: `{results[0].breed}` | Cosine Similarity: `{results[0].similarity_score:.4f}`")
                st.markdown(f"**Top Image Path**: `{results[0].image_path}`")
    elif active_query is None:
        st.info("💡 Enter a description above or click one of the quick suggestions to search.")
