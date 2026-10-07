import streamlit as st
from src.config import AppConfig

def render_about(config: AppConfig, *args, **kwargs):
    st.markdown('<h1 style="color: #0F172A; font-weight: 800; margin-bottom: 4px;">ℹ️ Architecture & Technical Documentation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #475569; font-size: 1rem; margin-bottom: 24px;">'
        'System design, algorithmic principles, evaluation methodology, and architectural decisions behind VisualSeek.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <h3 style="color: #0F172A; margin-top: 0; font-size: 1.25rem; font-weight: 700;">What is VisualSeek?</h3>
        <p style="color: #334155; line-height: 1.6; margin-bottom: 0;">
            <strong>VisualSeek</strong> is a full-stack computer vision and retrieval engine engineered to index, explore, 
            and retrieve visually and semantically similar dog breeds in real-time across high-dimensional latent vector spaces. 
            Designed for technical interviews and portfolio demonstration, VisualSeek features a strict separation between 
            the Streamlit presentation layer, core ML domain logic, FAISS vector search, and offline evaluation pipelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(r"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 24px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #2563EB; margin-top: 0; font-size: 1.1rem; font-weight: 700;">🔄 How the Retrieval Pipeline Works</h4>
            <pre style="background: #F8FAFC; color: #0F172A; padding: 14px; border-radius: 8px; font-size: 0.85rem; line-height: 1.5; border: 1px solid #E2E8F0; font-family: monospace;">
Query Input (Image or Text Prompt)
           │
           ▼
Preprocessing (Bicubic Crop, Mean/Std Norm)
           │
           ▼
Encoder Model (OpenAI CLIP ViT-B/32)
           │
           ▼
L2 Vector Normalization (Unit 512-dim Vector)
           │
           ▼
Vector Search (FAISS IndexFlatIP / Dot Product)
           │
           ▼
Top-K Ranked Results & Image Metadata
            </pre>
            <h4 style="color: #2563EB; margin-top: 16px; font-size: 1.1rem; font-weight: 700;">⚡ Why Vector Search (FAISS)?</h4>
            <p style="color: #334155; font-size: 0.9rem; line-height: 1.5;">
                Exact pairwise cosine similarity between high-dimensional vectors scales with $O(N \cdot D)$. 
                FAISS utilizes SIMD vectorization, cache-friendly memory layouts, and BLAS inner product routines 
                to reduce query latencies from hundreds of milliseconds to <strong>sub-millisecond</strong> execution 
                on standard CPU hardware.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 24px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #2563EB; margin-top: 0; font-size: 1.1rem; font-weight: 700;">📉 Why Principal Component Analysis (PCA)?</h4>
            <p style="color: #334155; font-size: 0.9rem; line-height: 1.5;">
                PCA is an orthogonal linear transformation that maps 512-dimensional vectors into coordinate systems 
                of decreasing variance. In VisualSeek, the first 50 principal components capture <strong>over 71% of total variance</strong>, 
                making PCA both a fast global exploratory view and an essential pre-filter for non-linear manifold algorithms.
            </p>
            <h4 style="color: #2563EB; margin-top: 16px; font-size: 1.1rem; font-weight: 700;">🌐 Why t-SNE Manifold Learning?</h4>
            <p style="color: #334155; font-size: 0.9rem; line-height: 1.5;">
                While PCA prioritizes large pairwise distances and global variance, <strong>t-SNE</strong> minimizes the Kullback-Leibler 
                divergence between joint probabilities in the high-dimensional space and low-dimensional representation. 
                This preserves fine-grained local neighborhoods, naturally clustering dogs with similar facial structures, 
                coloration, and coat types into visually cohesive groups.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 24px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #2563EB; margin-top: 0; font-size: 1.1rem; font-weight: 700;">📊 Evaluation Methodology</h4>
            <p style="color: #334155; font-size: 0.9rem; line-height: 1.5;">
                We benchmarked retrieval quality using strict leave-one-out testing on 50 sample queries across 500 images.
                Same-breed matching served as a ground-truth proxy signal.
            </p>
            <ul style="color: #334155; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0;">
                <li><strong>Baseline (Color Histograms)</strong>: Precision@1 = <strong>2.0%</strong>, MRR = <strong>0.0356</strong>.</li>
                <li><strong>CLIP ViT-B/32</strong>: Precision@1 = <strong>10.0%</strong> (5.0x improvement), MRR = <strong>0.2116</strong> (5.9x improvement).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px 24px; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #2563EB; margin-top: 0; font-size: 1.1rem; font-weight: 700;">⚠️ Known Limitations & Design Trade-offs</h4>
            <ul style="color: #334155; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0;">
                <li><strong>Breed vs. Semantic Similarity</strong>: Two distinct breeds (e.g. West Highland Terrier and Maltese) sharing small white coats will have high cosine similarity (~0.85). Visual similarity reflects visual appearance, not pure genetic taxonomy.</li>
                <li><strong>Background Sensitivity</strong>: Complex outdoor backgrounds (green grass, snow) slightly influence embeddings alongside the dog subject.</li>
                <li><strong>Development Sizing</strong>: Running on 1,000 stratified images enables near-instant sub-second iteration on CPU. Full dataset indexing (20,580 images) can be toggled via configuration.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
