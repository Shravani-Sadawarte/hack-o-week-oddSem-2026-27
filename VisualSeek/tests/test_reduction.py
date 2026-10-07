import pytest
import numpy as np

from src.reduction.pca import PCAReducer
from src.reduction.tsne import TSNEReducer

def test_pca_reducer():
    n_samples = 40
    n_features = 20
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_features).astype(np.float32)

    reducer = PCAReducer(n_components=10, random_state=42)
    transformed, var_ratio, cum_var = reducer.fit_transform(data)

    assert transformed.shape == (n_samples, 10)
    assert len(var_ratio) == 10
    assert len(cum_var) == 10
    assert 0.0 < cum_var[-1] <= 1.0
    assert np.all(np.diff(cum_var) >= 0)  # monotonically non-decreasing

    p2d = reducer.get_2d(transformed)
    assert p2d.shape == (n_samples, 2)

    p3d = reducer.get_3d(transformed)
    assert p3d.shape == (n_samples, 3)

def test_tsne_reducer():
    n_samples = 30
    n_features = 15
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_features).astype(np.float32)

    reducer = TSNEReducer(perplexity=5.0, max_iter=250, random_state=42)
    coords = reducer.fit_transform(data)

    assert coords.shape == (n_samples, 2)
    assert coords.dtype == np.float32
