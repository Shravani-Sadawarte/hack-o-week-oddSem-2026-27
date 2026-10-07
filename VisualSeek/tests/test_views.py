import inspect
import pytest
from src.config import load_config
from app.views.visual_search import render_visual_search
from app.views.semantic_search import render_semantic_search
from app.views.embedding_explorer import render_embedding_explorer
from app.views.analytics import render_analytics
from app.views.about import render_about

def test_view_signatures_flexible():
    for fn in [render_visual_search, render_semantic_search, render_embedding_explorer, render_analytics, render_about]:
        sig = inspect.signature(fn)
        assert any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in sig.parameters.values())

def test_view_call_signatures_no_type_error(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, 'markdown', lambda *a, **kw: None)
    monkeypatch.setattr(st, 'error', lambda *a, **kw: None)
    monkeypatch.setattr(st, 'code', lambda *a, **kw: None)
    config = load_config()
    render_visual_search(config, None)
    render_visual_search(config, None, None)
    render_semantic_search(config, None)
    render_semantic_search(config, None, None)
    render_about(config)
