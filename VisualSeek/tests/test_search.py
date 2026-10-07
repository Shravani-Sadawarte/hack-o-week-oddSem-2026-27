import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.search.similarity import cosine_similarity_search
from src.search.faiss_index import FaissIndexManager
from src.search.search_engine import SearchEngine

def test_cosine_similarity_exact_matches():
    corpus = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7071, 0.7071, 0.0]
    ], dtype=np.float32)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    indices, scores = cosine_similarity_search(query, corpus, top_k=2)
    assert indices[0] == 0
    assert pytest.approx(scores[0], rel=1e-4) == 1.0
    assert indices[1] == 3
    assert pytest.approx(scores[1], rel=1e-3) == 0.7071

def test_faiss_index_manager_and_serialization():
    dim = 16
    n = 50
    rng = np.random.RandomState(42)
    data = rng.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    mgr = FaissIndexManager(dimension=dim)
    mgr.build(data)
    assert mgr.num_vectors == n

    query = data[5]  # Query with item 5
    indices, scores = mgr.search(query, top_k=3)
    assert indices[0] == 5
    assert pytest.approx(scores[0], abs=1e-5) == 1.0

    # Test serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test.index"
        mgr.save(save_path)
        assert save_path.exists()

        loaded_mgr = FaissIndexManager.load(save_path)
        assert loaded_mgr.num_vectors == n
        l_indices, l_scores = loaded_mgr.search(query, top_k=3)
        np.testing.assert_array_equal(indices, l_indices)
        np.testing.assert_allclose(scores, l_scores, atol=1e-5)

def test_search_engine_facade():
    dim = 8
    corpus = np.eye(dim, dtype=np.float32)
    metadata = [{"image_id": f"img_{i}", "breed": f"Breed_{i}", "file_path": f"/p/{i}.jpg"} for i in range(dim)]

    engine = SearchEngine(corpus_embeddings=corpus, metadata_records=metadata)

    # Search with query vector 2
    query = np.zeros(dim, dtype=np.float32)
    query[2] = 1.0

    results_faiss, lat_faiss = engine.search_by_vector(query, top_k=3, engine="faiss")
    results_bf, lat_bf = engine.search_by_vector(query, top_k=3, engine="brute_force")

    assert len(results_faiss) == 3
    assert len(results_bf) == 3

    assert results_faiss[0].image_id == "img_2"
    assert results_faiss[0].similarity_percent == 100.0
    assert results_bf[0].image_id == "img_2"
    assert results_bf[0].similarity_percent == 100.0
