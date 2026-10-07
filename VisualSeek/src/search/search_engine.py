import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np

from src.utils.logging import setup_logger
from src.search.similarity import cosine_similarity_search
from src.search.faiss_index import FaissIndexManager
from src.models.base import BaseEncoder

logger = setup_logger("visualseek.search.engine")

@dataclass
class SearchResult:
    rank: int
    image_id: str
    image_path: str
    breed: str
    breed_folder: str
    similarity_score: float
    similarity_percent: float
    distance: float
    width: Optional[int] = None
    height: Optional[int] = None

class SearchEngine:
    """Unified retrieval service supporting both FAISS indexing and exact Brute-Force Cosine Search."""

    def __init__(
        self,
        corpus_embeddings: np.ndarray,
        metadata_records: List[Dict[str, Any]],
        faiss_manager: Optional[FaissIndexManager] = None
    ):
        self.corpus_embeddings = corpus_embeddings
        self.metadata_records = metadata_records
        self.faiss_manager = faiss_manager
        self.num_items = len(metadata_records)

        # Build FAISS index if not passed
        if self.faiss_manager is None and self.num_items > 0:
            dim = corpus_embeddings.shape[1]
            self.faiss_manager = FaissIndexManager(dimension=dim)
            self.faiss_manager.build(corpus_embeddings)

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        engine: str = "faiss"
    ) -> Tuple[List[SearchResult], float]:
        """Searches for Top-K similar vectors. Returns (results, latency_ms)."""
        start_time = time.perf_counter()

        if self.num_items == 0:
            return [], 0.0

        if engine.lower() == "faiss" and self.faiss_manager is not None:
            indices, scores = self.faiss_manager.search(query_vector, top_k=top_k)
        else:
            indices, scores = cosine_similarity_search(query_vector, self.corpus_embeddings, top_k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        results: List[SearchResult] = []
        for rank_idx, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(self.metadata_records):
                continue
            rec = self.metadata_records[idx]
            sim_score = float(score)
            # Clip cosine similarity to [0.0, 1.0] for percentage presentation
            sim_pct = round(max(0.0, min(1.0, sim_score)) * 100, 1)
            dist = round(float(1.0 - sim_score), 4)

            result = SearchResult(
                rank=rank_idx,
                image_id=rec.get("image_id", f"img_{idx}"),
                image_path=rec.get("file_path", ""),
                breed=rec.get("breed", "Unknown Breed"),
                breed_folder=rec.get("breed_folder", ""),
                similarity_score=round(sim_score, 4),
                similarity_percent=sim_pct,
                distance=dist,
                width=rec.get("width"),
                height=rec.get("height")
            )
            results.append(result)

        return results, latency_ms

    def search_by_image(
        self,
        image_input: Union[str, Path, Image.Image],
        encoder: BaseEncoder,
        top_k: int = 10,
        engine: str = "faiss"
    ) -> Tuple[List[SearchResult], float]:
        """Encodes an image query and retrieves Top-K results."""
        q_vec = encoder.encode_image([image_input], batch_size=1, normalize=True)[0]
        return self.search_by_vector(q_vec, top_k=top_k, engine=engine)

    def search_by_text(
        self,
        text_query: str,
        encoder: BaseEncoder,
        top_k: int = 10,
        engine: str = "faiss"
    ) -> Tuple[List[SearchResult], float]:
        """Encodes a text prompt query and retrieves Top-K results."""
        if not encoder.supports_text:
            raise NotImplementedError(f"Encoder '{encoder.name}' does not support natural language text search.")
        q_vec = encoder.encode_text([text_query], batch_size=1, normalize=True)[0]
        return self.search_by_vector(q_vec, top_k=top_k, engine=engine)
