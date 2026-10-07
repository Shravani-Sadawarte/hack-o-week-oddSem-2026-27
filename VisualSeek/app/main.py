import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config, AppConfig
from src.embeddings.storage import EmbeddingStorage
from src.search.faiss_index import FaissIndexManager
from src.search.search_engine import SearchEngine
from src.models.model_factory import ModelFactory
from src.models.base import BaseEncoder
from app.components.metric_card import render_status_badge

# Page Configuration
st.set_page_config(
    page_title="VisualSeek — AI Visual Similarity & Semantic Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast SaaS UI Styling (Modern White / Light Theme)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Backgrounds */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    .sidebar-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 18px;
        margin-bottom: 6px;
    }

    /* Brand Logo / Header */
    .brand-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 16px;
    }

    /* Divider */
    .sidebar-divider {
        border-top: 1px solid #E2E8F0;
        margin: 16px 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #0F172A;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #2563EB;
        background-color: #EFF6FF;
        color: #1D4ED8;
    }

    /* Primary submit button */
    button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.3) !important;
    }
    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1E40AF !important;
    }

    /* Form Inputs */
    .stTextInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* File uploader container */
    div[data-testid="stFileUploader"] {
        background-color: #F8FAFC;
        border: 1px dashed #CBD5E1;
        border-radius: 10px;
        padding: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def get_app_config() -> AppConfig:
    return load_config()

@st.cache_resource
def get_model_encoder(model_name: str, pretrained: str, device: str) -> BaseEncoder:
    return ModelFactory.create(model_name=model_name, pretrained=pretrained, device=device)

@st.cache_resource
def get_search_engine(tag: str = "default"):
    config = get_app_config()
    storage = EmbeddingStorage(base_dir=Path(config.paths.artifacts_dir))
    
    if not storage.exists(tag):
        return None, None, None
        
    embeddings, metadata, header = storage.load(tag)
    
    # Try loading FAISS index
    index_path = Path(config.paths.indexes_dir) / f"{tag}_faiss.index"
    faiss_mgr = None
    if index_path.exists():
        try:
            faiss_mgr = FaissIndexManager.load(index_path)
        except Exception:
            faiss_mgr = None

    engine = SearchEngine(
        corpus_embeddings=embeddings,
        metadata_records=metadata,
        faiss_manager=faiss_mgr
    )
    return engine, metadata, header

def main():
    config = get_app_config()
    engine, metadata, header = get_search_engine(tag="default")
    encoder = get_model_encoder(config.model.name, config.model.pretrained, config.model.device)

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.markdown('<div class="brand-title">VISUALSEEK</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-subtitle">AI Visual Similarity & Semantic Search Engine</div>', unsafe_allow_html=True)
        
        # Structured Navigation as required in Section 26
        st.markdown('<div class="sidebar-section-header">Navigation</div>', unsafe_allow_html=True)
        nav_options = [
            "🔎 Visual Search",
            "💬 Semantic Search",
            "🧭 Embedding Explorer",
            "📊 Analytics",
            "ℹ️ About"
        ]
        
        selected_page = st.radio(
            "Navigation Menu",
            nav_options,
            label_visibility="collapsed"
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">Search Controls</div>', unsafe_allow_html=True)
        
        top_k = st.selectbox(
            "Top K Results",
            options=[5, 10, 15, 20],
            index=1,  # Default 10
            help="Number of nearest neighbors to retrieve."
        )
        
        search_engine_mode = st.radio(
            "Vector Search Backend",
            ["FAISS (Vector Index)", "Brute Force (Exact Cosine)"],
            index=0 if config.search.engine == "faiss" else 1,
            help="Switch between sub-millisecond FAISS indexing and exact pairwise dot-product."
        )
        chosen_engine = "faiss" if "FAISS" in search_engine_mode else "brute_force"

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">Dataset & Hardware</div>', unsafe_allow_html=True)
        
        st.markdown(f"**Dataset**: Stanford Dogs")
        if metadata:
            num_imgs = len(metadata)
            is_partial = num_imgs < 20580
            st.markdown(f"**Indexed Images**: `{num_imgs:,}`")
            if is_partial:
                st.caption("🟡 Development Mode (Stratified Sample)")
            else:
                st.caption("🟢 Full Dataset Indexed")
        else:
            st.markdown("**Indexed Images**: `0 (Not indexed yet)`")

        st.markdown(f"**Encoder**: `{config.model.name}`")
        st.markdown(f"**Device**: `{'CUDA (GPU)' if config.model.device == 'cuda' else 'CPU (16 Cores)'}`")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">System Health</div>', unsafe_allow_html=True)
        is_ready = (engine is not None and engine.num_items > 0)
        render_status_badge(is_ready, "Index Ready", "Index Not Built")

    # Sync Session State
    st.session_state["top_k"] = top_k
    st.session_state["search_engine"] = chosen_engine

    # Page Routing via views
    if selected_page == "🔎 Visual Search":
        from app.views.visual_search import render_visual_search
        render_visual_search(config, engine, encoder)
    elif selected_page == "💬 Semantic Search":
        from app.views.semantic_search import render_semantic_search
        render_semantic_search(config, engine, encoder)
    elif selected_page == "🧭 Embedding Explorer":
        from app.views.embedding_explorer import render_embedding_explorer
        render_embedding_explorer(config, engine, metadata)
    elif selected_page == "📊 Analytics":
        from app.views.analytics import render_analytics
        render_analytics(config, engine, metadata, header)
    elif selected_page == "ℹ️ About":
        from app.views.about import render_about
        render_about(config)

if __name__ == "__main__":
    main()
