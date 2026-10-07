import streamlit as st
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import json
from PIL import Image

from src.config import AppConfig
from src.search.search_engine import SearchEngine
from src.reduction.tsne import TSNEReducer
from app.components.charts import (
    render_scree_plot,
    render_reduction_scatter_2d,
    render_reduction_scatter_3d
)

def render_embedding_explorer(
    config: AppConfig,
    engine: Optional[SearchEngine] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    *args,
    **kwargs
):
    st.markdown('<h1 style="color: #0F172A; font-weight: 800; margin-bottom: 4px;">🧭 Latent Embedding Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #475569; font-size: 1rem; margin-bottom: 24px;">'
        'Explore how dog breeds are distributed across the high-dimensional latent space. '
        'Interact with 2D and 3D manifold projections computed via <strong>PCA</strong> and <strong>t-SNE</strong>.'
        '</p>',
        unsafe_allow_html=True
    )

    reductions_path = Path(config.paths.reductions_dir) / "default_reductions.npz"
    meta_path = Path(config.paths.reductions_dir) / "default_reductions_meta.json"

    if not reductions_path.exists() or not meta_path.exists():
        st.warning("⚠️ Latent projections have not been computed yet. Please run:")
        st.code("python scripts/run_reduction.py", language="bash")
        return

    data = np.load(reductions_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_info = json.load(f)

    all_breeds = [r["breed"] for r in metadata] if metadata else []
    all_image_ids = [r["image_id"] for r in metadata] if metadata else []
    total_available = len(all_breeds)

    tab_vis, tab_variance, tab_inspect = st.tabs([
        "🌌 Manifold Projections (2D & 3D)",
        "📈 PCA Scree & Explained Variance",
        "🔍 Point & Neighbor Inspection"
    ])

    with tab_vis:
        # Controls Row
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            method = st.selectbox(
                "Projection Method",
                ["PCA (2D)", "PCA (3D)", "t-SNE (2D)"] + (["UMAP (2D)"] if "umap_2d" in data else []),
                index=0
            )
        with c2:
            color_mode = st.radio("Color Palette", ["Color by Breed", "Uniform Color"], horizontal=True)
        with c3:
            num_points = st.select_slider(
                "Points to Display",
                options=[100, 250, 500, total_available],
                value=total_available
            )

        # Slice data
        sub_breeds = all_breeds[:num_points]
        sub_ids = all_image_ids[:num_points]
        color_labels = sub_breeds if color_mode == "Color by Breed" else ["All Dogs"] * len(sub_breeds)

        # Method specific stats
        if "PCA" in method:
            var_ratios = meta_info.get("pca_explained_variance_ratio", [])
            p1_str = f"{var_ratios[0]*100:.1f}%" if len(var_ratios) > 0 else "N/A"
            p2_str = f"{var_ratios[1]*100:.1f}%" if len(var_ratios) > 1 else "N/A"
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 12px 18px; border-radius: 8px; margin: 10px 0 16px 0; font-size: 0.88rem; color: #0F172A; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <strong>PCA Variance Preserved</strong>: PC1 explains <span style="color: #2563EB; font-weight: 700;">{p1_str}</span> · 
                    PC2 explains <span style="color: #2563EB; font-weight: 700;">{p2_str}</span> · 
                    Combined 2D variance: <span style="color: #059669; font-weight: 700;">{(var_ratios[0]+var_ratios[1])*100:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        if method == "PCA (2D)":
            fig = render_reduction_scatter_2d(data["pca_2d"][:num_points], color_labels, sub_ids, method_name="PCA")
            st.plotly_chart(fig, use_container_width=True)
        elif method == "PCA (3D)":
            fig = render_reduction_scatter_3d(data["pca_3d"][:num_points], color_labels, sub_ids)
            st.plotly_chart(fig, use_container_width=True)
        elif method == "t-SNE (2D)":
            fig = render_reduction_scatter_2d(data["tsne_2d"][:num_points], color_labels, sub_ids, method_name="t-SNE")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("⚙️ t-SNE Parameter Controls & Recomputation", expanded=False):
                col_p, col_iter, col_seed = st.columns(3)
                with col_p:
                    tsne_perp = st.slider("Perplexity", min_value=5.0, max_value=50.0, value=float(config.tsne.perplexity), step=5.0)
                with col_iter:
                    tsne_iter = st.slider("Max Iterations", min_value=250, max_value=1500, value=int(config.tsne.max_iter), step=250)
                with col_seed:
                    tsne_seed = st.number_input("Random Seed", value=int(config.tsne.random_state), step=1)

                if st.button("🔄 Recompute t-SNE Projections", type="secondary"):
                    with st.spinner(f"Computing non-linear t-SNE projection for {total_available} points..."):
                        if engine is not None:
                            reducer = TSNEReducer(
                                perplexity=tsne_perp,
                                max_iter=tsne_iter,
                                random_state=int(tsne_seed)
                            )
                            new_coords = reducer.fit_transform(engine.corpus_embeddings)
                            # Update in-memory and save
                            np.savez_compressed(
                                reductions_path,
                                pca_2d=data["pca_2d"],
                                pca_3d=data["pca_3d"],
                                tsne_2d=new_coords,
                                pca_var_ratio=data["pca_var_ratio"],
                                pca_cum_var=data["pca_cum_var"]
                            )
                            st.success("t-SNE successfully recomputed and cached!")
                            st.rerun()
        elif method == "UMAP (2D)" and "umap_2d" in data:
            fig = render_reduction_scatter_2d(data["umap_2d"][:num_points], color_labels, sub_ids, method_name="UMAP")
            st.plotly_chart(fig, use_container_width=True)

    with tab_variance:
        st.markdown('<h3 style="color: #0F172A; font-weight: 700; margin-bottom: 8px;">PCA Scree Plot & Cumulative Variance</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: #475569; font-size: 0.95rem;">'
            'Principal Component Analysis identifies the orthogonal axes of maximum variance in the 512-dimensional embedding space. '
            'This scree plot visualizes the exact computed eigenvalues and cumulative variance coverage.'
            '</p>',
            unsafe_allow_html=True
        )

        var_ratios = meta_info.get("pca_explained_variance_ratio", [])
        cum_var = meta_info.get("pca_cumulative_variance", [])

        if var_ratios and cum_var:
            scree_fig = render_scree_plot(var_ratios, cum_var)
            st.plotly_chart(scree_fig, use_container_width=True)

    with tab_inspect:
        st.markdown('<h3 style="color: #0F172A; font-weight: 700; margin-bottom: 8px;">Point & Nearest Neighbor Inspector</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: #475569; font-size: 0.95rem;">'
            'Select any indexed image to view its photo, metadata, and its Top-3 nearest neighbors in the latent embedding space.'
            '</p>',
            unsafe_allow_html=True
        )

        if metadata and engine is not None:
            inspect_idx = st.selectbox(
                "Choose an image to inspect:",
                range(len(metadata)),
                format_func=lambda i: f"#{i+1}: {metadata[i]['breed']} ({metadata[i]['image_id']})"
            )

            target_rec = metadata[inspect_idx]
            col_target, col_neighbors = st.columns([1, 2.5])

            with col_target:
                st.markdown('<h4 style="color: #0F172A; font-size: 1rem;">Target Image</h4>', unsafe_allow_html=True)
                p = Path(target_rec["file_path"])
                if p.exists():
                    with Image.open(p) as img:
                        st.image(img, use_container_width=True)
                st.markdown(f"**Breed**: `{target_rec['breed']}`")
                st.markdown(f"**Image ID**: `{target_rec['image_id']}`")

            with col_neighbors:
                st.markdown('<h4 style="color: #0F172A; font-size: 1rem;">Top-3 Nearest Neighbors in Latent Space</h4>', unsafe_allow_html=True)
                # Query nearest neighbors
                target_vec = engine.corpus_embeddings[inspect_idx]
                results, _ = engine.search_by_vector(target_vec, top_k=4, engine="faiss")

                # Filter out target itself
                neighbors = [r for r in results if r.image_id != target_rec["image_id"]][:3]

                ncol = st.columns(len(neighbors))
                for col, nb in zip(ncol, neighbors):
                    with col:
                        np_path = Path(nb.image_path)
                        if np_path.exists():
                            with Image.open(np_path) as n_img:
                                st.image(n_img, use_container_width=True)
                        st.markdown(f"**{nb.breed}**")
                        st.caption(f"Similarity: `{nb.similarity_score:.3f}` · ID: `{nb.image_id}`")
