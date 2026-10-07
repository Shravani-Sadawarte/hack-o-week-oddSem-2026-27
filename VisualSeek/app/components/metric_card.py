import streamlit as st
from typing import Optional

def render_metric_card(label: str, value: str, subtitle: Optional[str] = None):
    """Renders a high-contrast, polished KPI metric card in light SaaS style."""
    sub_html = f'<div style="font-size: 0.82rem; color: #64748B; margin-top: 6px; font-weight: 500;">{subtitle}</div>' if subtitle else ''
    html = f"""
    <div style="
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 4px 12px rgba(0, 0, 0, 0.03);
        text-align: left;
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.76rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
            {label}
        </div>
        <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; line-height: 1.1; letter-spacing: -0.02em;">
            {value}
        </div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(is_ready: bool, ready_text: str = "Ready", not_ready_text: str = "Not Ready"):
    """Renders an inline high-contrast status indicator for light theme."""
    color = "#15803D" if is_ready else "#B91C1C"
    bg = "#DCFCE7" if is_ready else "#FEE2E2"
    border = "#86EFAC" if is_ready else "#FCA5A5"
    text = ready_text if is_ready else not_ready_text
    
    st.markdown(
        f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            background: {bg};
            border: 1px solid {border};
            font-size: 0.84rem;
            font-weight: 700;
            color: {color};
            margin-top: 4px;
        ">
            <span style="font-size: 0.85rem;">●</span> {text}
        </div>
        """,
        unsafe_allow_html=True
    )
