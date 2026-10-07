import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    font=dict(family="Inter, sans-serif", color="#0F172A"),
    title_font=dict(size=16, color="#0F172A", family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
        tickfont=dict(color="#475569", size=11),
        title_font=dict(color="#0F172A", size=13)
    ),
    yaxis=dict(
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
        tickfont=dict(color="#475569", size=11),
        title_font=dict(color="#0F172A", size=13)
    ),
    legend=dict(
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(color="#0F172A", size=11)
    )
)

def render_scree_plot(variance_ratios: List[float], cumulative_variance: List[float]) -> go.Figure:
    """Renders PCA individual and cumulative explained variance interactive chart."""
    components = list(range(1, len(variance_ratios) + 1))

    fig = go.Figure()

    # Individual explained variance (Bars)
    fig.add_trace(go.Bar(
        x=components,
        y=[v * 100 for v in variance_ratios],
        name="Individual Variance (%)",
        marker_color="#2563EB",
        opacity=0.75,
        hovertemplate="Component %{x}<br>Individual Variance: %{y:.2f}%<extra></extra>"
    ))

    # Cumulative explained variance (Line)
    fig.add_trace(go.Scatter(
        x=components,
        y=[v * 100 for v in cumulative_variance],
        name="Cumulative Variance (%)",
        mode="lines+markers",
        line=dict(color="#059669", width=3),
        marker=dict(size=6, color="#10B981"),
        hovertemplate="Component %{x}<br>Cumulative Variance: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title="PCA Explained Variance Scree Plot",
        xaxis_title="Principal Component Number",
        yaxis_title="Explained Variance (%)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def render_reduction_scatter_2d(
    coords: np.ndarray,
    breeds: List[str],
    image_ids: List[str],
    method_name: str = "PCA"
) -> go.Figure:
    """Renders 2D interactive scatter plot of latent embeddings."""
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Breed": breeds,
        "Image ID": image_ids
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Breed",
        hover_data=["Breed", "Image ID"],
        title=f"{method_name} 2D Latent Space Projection",
        labels={"x": f"{method_name} Dimension 1", "y": f"{method_name} Dimension 2"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.85, line=dict(width=0.75, color="#FFFFFF"))
    )

    fig.update_layout(
        **CHART_LAYOUT,
        margin=dict(l=30, r=30, t=50, b=30),
        height=620,
        legend=dict(
            itemsizing="constant",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10, color="#334155")
        )
    )

    return fig

def render_reduction_scatter_3d(
    coords: np.ndarray,
    breeds: List[str],
    image_ids: List[str]
) -> go.Figure:
    """Renders 3D interactive scatter plot for PCA."""
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "Breed": breeds,
        "Image ID": image_ids
    })

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="Breed",
        hover_data=["Breed", "Image ID"],
        title="PCA 3D Manifold Projection",
        labels={"x": "Principal Component 1", "y": "Principal Component 2", "z": "Principal Component 3"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    fig.update_traces(marker=dict(size=4, opacity=0.85))
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        height=650,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

def render_benchmark_bar_chart(benchmark_data: List[Dict[str, Any]]) -> go.Figure:
    """Renders grouped bar chart with text labels for Baseline vs CLIP evaluation."""
    metrics = ["Precision@1", "Precision@5", "Precision@10"]
    keys = ["precision_at_1", "precision_at_5", "precision_at_10"]

    fig = go.Figure()
    colors = ["#94A3B8", "#2563EB"]  # Slate for Baseline, Royal Blue for CLIP

    for idx, item in enumerate(benchmark_data):
        m_name = item.get("model_name", f"Model {idx+1}")
        scores_pct = [round(item.get(k, 0.0) * 100, 1) for k in keys]
        text_labels = [f"{s:.1f}%" for s in scores_pct]

        fig.add_trace(go.Bar(
            x=metrics,
            y=scores_pct,
            name=m_name,
            text=text_labels,
            textposition="auto",
            textfont=dict(size=12, color="#FFFFFF"),
            marker_color=colors[idx % len(colors)]
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        title="Retrieval Precision@K: Baseline Color Histogram vs. CLIP ViT-B/32",
        yaxis_title="Precision Score (%)",
        xaxis_title="Retrieval Metric",
        yaxis=dict(ticksuffix="%", range=[0, max(15, max([item.get('precision_at_1', 0)*100 for item in benchmark_data] + [10]) * 1.3)]),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
