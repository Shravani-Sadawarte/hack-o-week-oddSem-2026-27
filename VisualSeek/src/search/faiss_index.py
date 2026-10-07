from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import faiss

from src.utils.logging import setup_logger
from src.utils.paths import ensure_dir

logger = setup_logger("visualseek.search.faiss")

class FaissIndexManager:
    """Manages creation, serialization, deserialization, and querying of FAISS vector indices."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        # IndexFlatIP calculates inner product, which equals cosine similarity for L2-normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.num_vectors = 0

    def build(self, embeddings: np.ndarray) -> None:
        """Adds normalized float32 embeddings to the index."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        # Reset and add
        self.index.reset()
        self.index.add(embeddings)
        self.num_vectors = self.index.ntotal
        logger.info(f"Built FAISS IndexFlatIP with {self.num_vectors} vectors ({self.dimension}-dim)")

    def save(self, file_path: Path) -> None:
        """Serializes FAISS index to disk."""
        ensure_dir(file_path.parent)
        faiss.write_index(self.index, str(file_path))
        logger.info(f"Saved FAISS index to {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> "FaissIndexManager":
        """Loads FAISS index from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        index = faiss.read_index(str(path))
        manager = cls(dimension=index.d)
        manager.index = index
        manager.num_vectors = index.ntotal
        logger.info(f"Loaded FAISS index with {manager.num_vectors} vectors ({manager.dimension}-dim) from {path}")
        return manager

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Queries the index and returns (indices, scores)."""
        if self.num_vectors == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        if query_vector.ndim == 1:
            q = query_vector.reshape(1, -1).astype(np.float32)
        else:
            q = query_vector.astype(np.float32)

        # Normalize query vector if needed
        faiss.normalize_L2(q)

        k = min(top_k, self.num_vectors)
        scores, indices = self.index.search(q, k)
        return indices[0], scores[0]
