import streamlit as st
from PIL import Image
from pathlib import Path
import base64
import io
from src.search.search_engine import SearchResult

def _image_to_base64(img_path: str, max_size: int = 320) -> str:
    """Loads and converts an image to a base64 JPEG data URL with thumbnail resizing for speed."""
    try:
        path = Path(img_path)
        if not path.exists():
            return ""
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return ""

def render_result_card(result: SearchResult):
    """Renders a uniform, high-contrast, card-based visual search result in light SaaS style."""
    img_data_url = _image_to_base64(result.image_path)

    # Color code similarity badge based on cosine similarity
    score = result.similarity_score
    if score >= 0.85:
        badge_bg = "#ECFDF5"
        badge_border = "#6EE7B7"
        badge_color = "#047857"
    elif score >= 0.70:
        badge_bg = "#EFF6FF"
        badge_border = "#93C5FD"
        badge_color = "#1D4ED8"
    else:
        badge_bg = "#FFFBEB"
        badge_border = "#FCD34D"
        badge_color = "#B45309"

    img_html = f"""
    <div style="
        width: 100%;
        height: 180px;
        background-color: #F1F5F9;
        border-radius: 10px 10px 0 0;
        overflow: hidden;
        position: relative;
    ">
        <img src="{img_data_url}" alt="{result.breed}" style="
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        "/>
        <div style="
            position: absolute;
            top: 10px;
            left: 10px;
            background: #2563EB;
            color: #FFFFFF;
            font-size: 0.78rem;
            font-weight: 800;
            padding: 3px 9px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        ">#{result.rank}</div>
    </div>
    """ if img_data_url else f"""
    <div style="
        width: 100%;
        height: 180px;
        background-color: #F8FAFC;
        border-radius: 10px 10px 0 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94A3B8;
        font-size: 0.85rem;
        border-bottom: 1px solid #E2E8F0;
    ">Image Unavailable</div>
    """

    card_html = f"""
    <div style="
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    ">
        {img_html}
        <div style="padding: 14px 16px; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="
                    font-size: 1.05rem;
                    font-weight: 700;
                    color: #0F172A;
                    line-height: 1.3;
                    margin-bottom: 6px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                " title="{result.breed}">{result.breed}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 12px; font-family: monospace;">
                    ID: {result.image_id}
                </div>
            </div>
            
            <div style="
                border-top: 1px solid #F1F5F9;
                padding-top: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">
                    Similarity
                </div>
                <div style="
                    background: {badge_bg};
                    border: 1px solid {badge_border};
                    color: {badge_color};
                    font-size: 0.88rem;
                    font-weight: 800;
                    padding: 3px 10px;
                    border-radius: 6px;
                " title="Cosine similarity derived from embedding distance; not a calibrated probability.">
                    {result.similarity_score:.3f}
                </div>
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)
