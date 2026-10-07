from typing import Optional
import numpy as np

from src.utils.logging import setup_logger

logger = setup_logger("visualseek.reduction.umap")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

class UMAPReducer:
    """Computes UMAP manifold projection if umap-learn is available."""

    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    @property
    def is_available(self) -> bool:
        return UMAP_AVAILABLE

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP is not installed in the environment. "
                "Install via 'pip install umap-learn' to enable UMAP projections."
            )
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        )
        return reducer.fit_transform(embeddings).astype(np.float32)
