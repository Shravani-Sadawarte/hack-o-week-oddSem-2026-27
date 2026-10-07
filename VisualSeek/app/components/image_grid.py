import streamlit as st
from typing import List
from src.search.search_engine import SearchResult
from app.components.result_card import render_result_card

def render_image_grid(results: List[SearchResult], columns_per_row: int = 4):
    """Renders a responsive multi-column grid of search result cards."""
    if not results:
        st.info("No results to display.")
        return

    # Chunk into rows
    for i in range(0, len(results), columns_per_row):
        row_items = results[i : i + columns_per_row]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                render_result_card(item)
