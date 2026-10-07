import pytest
import numpy as np
from pathlib import Path

from src.config import load_config
from src.embeddings.storage import EmbeddingStorage
from src.search.faiss_index import FaissIndexManager
from src.search.search_engine import SearchEngine
from src.models.clip_encoder import CLIPEncoder

@pytest.fixture(scope="module")
def search_system():
    config = load_config()
    storage = EmbeddingStorage(base_dir=Path(config.paths.artifacts_dir))
    if not storage.exists("default"):
        pytest.skip("Default embeddings artifact not present. Run generate_embeddings.py first.")
    
    embeddings, metadata, header = storage.load("default")
    index_path = Path(config.paths.indexes_dir) / "default_faiss.index"
    faiss_mgr = FaissIndexManager.load(index_path) if index_path.exists() else None
    
    engine = SearchEngine(
        corpus_embeddings=embeddings,
        metadata_records=metadata,
        faiss_manager=faiss_mgr
    )
    encoder = CLIPEncoder(model_name=config.model.name, pretrained=config.model.pretrained, device="cpu")
    return engine, encoder, metadata, embeddings

def test_semantic_query_distinct_results(search_system):
    """Test that Query A ('white fluffy dog in grass') and Query B ('black and tan hound dog')
    yield distinct result sets with different top matches and no stale state leakage.
    """
    engine, encoder, _, _ = search_system

    query_a = "white fluffy dog in grass"
    results_a, lat_a = engine.search_by_text(query_a, encoder, top_k=5, engine="faiss")

    query_b = "black and tan hound dog"
    results_b, lat_b = engine.search_by_text(query_b, encoder, top_k=5, engine="faiss")

    assert len(results_a) == 5
    assert len(results_b) == 5

    ids_a = [r.image_id for r in results_a]
    ids_b = [r.image_id for r in results_b]

    # The top retrieved image ID for Query A must not equal top retrieved image for Query B
    assert ids_a[0] != ids_b[0], "Query A and Query B produced identical top results!"
    assert ids_a != ids_b, "Results for distinct queries are identical!"

def test_quick_suggestion_transition(search_system):
    """Test that switching through quick suggestions produces the expected semantic shifts."""
    engine, encoder, _, _ = search_system

    suggestions = [
        "golden retriever sitting outdoors",
        "spotted dalmatian puppy"
    ]
    all_results = []
    for s in suggestions:
        res, _ = engine.search_by_text(s, encoder, top_k=3, engine="faiss")
        all_results.append(res)

    # Both suggestions must return valid non-empty results
    assert len(all_results[0]) == 3
    assert len(all_results[1]) == 3

    # Breed or image distributions must differ
    breeds_0 = [r.breed for r in all_results[0]]
    breeds_1 = [r.breed for r in all_results[1]]
    assert breeds_0 != breeds_1

def test_metadata_embedding_exact_alignment(search_system):
    """Verify that embedding[i] maps exactly to metadata[i] and querying vector[i]
    returns item[i] at Rank 1 with similarity score approx 1.0.
    """
    engine, _, metadata, embeddings = search_system

    # Test across multiple distinct indices
    test_indices = [0, 50, 100, 250]
    for idx in test_indices:
        target_vec = embeddings[idx]
        target_meta = metadata[idx]

        results, _ = engine.search_by_vector(target_vec, top_k=3, engine="faiss")
        top_match = results[0]

        assert top_match.rank == 1
        assert top_match.image_id == target_meta["image_id"]
        assert top_match.breed == target_meta["breed"]
        assert pytest.approx(top_match.similarity_score, abs=1e-4) == 1.0

def test_empty_and_zero_query_safety(search_system):
    """Ensure search handles empty query lists safely by returning empty array."""
    engine, encoder, _, _ = search_system

    empty_embs = encoder.encode_text([])
    assert empty_embs.shape == (0, 512)
    assert empty_embs.dtype == np.float32

def test_score_and_distance_math(search_system):
    """Verify similarity score and distance arithmetic."""
    engine, _, _, embeddings = search_system
    
    query = embeddings[10]
    results, _ = engine.search_by_vector(query, top_k=5, engine="faiss")
    
    for r in results:
        # Distance should equal 1 - similarity_score (within float precision)
        assert pytest.approx(r.distance, abs=1e-3) == round(1.0 - r.similarity_score, 4)
        assert 0.0 <= r.similarity_score <= 1.0001
