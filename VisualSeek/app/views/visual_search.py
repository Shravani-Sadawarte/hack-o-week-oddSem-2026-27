import streamlit as st
from PIL import Image
from pathlib import Path
from typing import Optional, List

from src.config import AppConfig
from src.search.search_engine import SearchEngine, SearchResult
from src.models.base import BaseEncoder
from app.components.image_grid import render_image_grid

def execute_visual_search(
    query_image: Image.Image,
    engine: SearchEngine,
    encoder: BaseEncoder,
    top_k: int,
    engine_mode: str
):
    """Encodes image query and stores results in session state."""
    with st.spinner("Generating CLIP visual embedding & searching vector index..."):
        try:
            results, latency_ms = engine.search_by_image(
                image_input=query_image,
                encoder=encoder,
                top_k=top_k,
                engine=engine_mode
            )
            st.session_state["visual_results"] = results
            st.session_state["visual_latency_ms"] = latency_ms
        except Exception as e:
            st.error(f"Visual search failed: {e}")

def render_visual_search(
    config: AppConfig,
    engine: Optional[SearchEngine] = None,
    encoder: Optional[BaseEncoder] = None,
    *args,
    **kwargs
):
    st.markdown('<h1 style="color: #0F172A; font-weight: 800; margin-bottom: 4px;">🔎 Visual Similarity Search</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #475569; font-size: 1rem; margin-bottom: 24px;">'
        'Find visually and semantically similar dogs using deep vision embeddings. '
        'Upload any dog photo or select a demo sample to query the 512-dimensional vector space.'
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

    # State initialization
    if "visual_active_image" not in st.session_state:
        st.session_state["visual_active_image"] = None
    if "visual_active_source" not in st.session_state:
        st.session_state["visual_active_source"] = None
    if "visual_results" not in st.session_state:
        st.session_state["visual_results"] = None
    if "visual_latency_ms" not in st.session_state:
        st.session_state["visual_latency_ms"] = 0.0

    col_upload, col_preview = st.columns([1.3, 1])

    with col_upload:
        st.markdown('<div style="font-size: 0.88rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select a clear photo of a dog to search for nearest visual neighbors.",
            label_visibility="collapsed"
        )

        # Quick demo sample selector
        st.markdown('<div style="font-size: 0.85rem; font-weight: 600; color: #475569; margin-top: 14px; margin-bottom: 6px;">Or choose a quick demo image:</div>', unsafe_allow_html=True)
        sample_cols = st.columns(3)
        if engine.metadata_records and len(engine.metadata_records) >= 3:
            s_recs = [
                engine.metadata_records[0],
                engine.metadata_records[len(engine.metadata_records)//2],
                engine.metadata_records[-1]
            ]
            for sc, r in zip(sample_cols, s_recs):
                with sc:
                    if st.button(f"🐕 {r['breed']}", use_container_width=True, key=f"vsample_{r['image_id']}"):
                        try:
                            img = Image.open(r["file_path"]).convert("RGB")
                            st.session_state["visual_active_image"] = img
                            st.session_state["visual_active_source"] = f"Dataset: {r['breed']} ({r['image_id']})"
                            execute_visual_search(img, engine, encoder, top_k, search_engine_mode)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cannot load sample: {e}")

    # Process uploaded file if provided
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            # Only trigger search if this is a newly uploaded file
            file_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_key") != file_key:
                st.session_state["last_uploaded_key"] = file_key
                st.session_state["visual_active_image"] = img
                st.session_state["visual_active_source"] = f"Uploaded: {uploaded_file.name}"
                execute_visual_search(img, engine, encoder, top_k, search_engine_mode)
                st.rerun()
        except Exception as e:
            st.error(f"Failed to open uploaded image: {e}")

    # Preview Column
    with col_preview:
        active_img = st.session_state.get("visual_active_image")
        active_src = st.session_state.get("visual_active_source")
        if active_img is not None:
            st.markdown('<div style="font-size: 0.88rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Query Preview</div>', unsafe_allow_html=True)
            st.image(active_img, caption=active_src or "Query Image", width=260)

    # 4. Display Results
    results = st.session_state.get("visual_results")
    latency_ms = st.session_state.get("visual_latency_ms", 0.0)

    if results is not None and active_img is not None:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">Top-{len(results)} Visually Similar Images</span>
                </div>
                <div style="font-size: 0.85rem; color: #64748B;">
                    Retrieved in <strong style="color: #059669;">{latency_ms:.2f} ms</strong> using <strong style="color: #2563EB;">{search_engine_mode.upper()}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        render_image_grid(results, columns_per_row=4)

        # Developer Diagnostic View
        with st.expander("🛠️ Developer Diagnostics (Retrieval Verification)", expanded=False):
            st.markdown(f"**Query Source**: `{active_src}`")
            st.markdown(f"**Image Dimensions**: `{active_img.size[0]} x {active_img.size[1]} px`")
            st.markdown(f"**Encoder Model**: `{encoder.name}` ({encoder.embedding_dim}-dim)")
            st.markdown(f"**Search Engine**: `{search_engine_mode.upper()}` · Latency: `{latency_ms:.3f} ms`")
            if results:
                st.markdown(f"**Top Result ID**: `{results[0].image_id}` | Breed: `{results[0].breed}` | Score: `{results[0].similarity_score:.4f}`")
                st.markdown(f"**Top Image Path**: `{results[0].image_path}`")
    elif active_img is None:
        st.info("💡 Upload an image or select a quick demo image above to begin searching.")
