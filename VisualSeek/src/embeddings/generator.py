from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import time
import numpy as np

from src.utils.logging import setup_logger
from src.data.loader import StanfordDogsDataset, ImageRecord
from src.models.base import BaseEncoder
from src.embeddings.storage import EmbeddingStorage

logger = setup_logger("visualseek.embeddings.generator")

class EmbeddingPipeline:
    """Orchestrates dataset loading, feature extraction via BaseEncoder, and caching to disk."""

    def __init__(
        self,
        encoder: BaseEncoder,
        storage: EmbeddingStorage
    ):
        self.encoder = encoder
        self.storage = storage

    def run(
        self,
        dataset: StanfordDogsDataset,
        tag: str = "default",
        batch_size: int = 32,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Runs the embedding pipeline or loads cached artifacts if available."""
        if not force_recompute and self.storage.exists(tag):
            logger.info(f"Cached embeddings found for tag '{tag}'. Loading from storage...")
            embeddings, metadata, _ = self.storage.load(tag)
            return embeddings, metadata

        logger.info(f"Generating embeddings for {len(dataset)} images using '{self.encoder.name}' (batch_size={batch_size})...")
        start_time = time.perf_counter()

        image_paths = [r.file_path for r in dataset.records]
        embeddings = self.encoder.encode_image(
            image_paths,
            batch_size=batch_size,
            normalize=True
        )

        elapsed = time.perf_counter() - start_time
        ms_per_image = (elapsed / max(1, len(dataset))) * 1000
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s ({ms_per_image:.1f}ms/image)")

        metadata = dataset.to_metadata_list()

        # Save to storage
        extra_info = {
            "elapsed_seconds": round(elapsed, 3),
            "ms_per_image": round(ms_per_image, 2),
            "dataset_limit": dataset.limit,
            "total_images": len(dataset)
        }
        self.storage.save(
            tag=tag,
            embeddings=embeddings,
            metadata=metadata,
            model_name=self.encoder.name,
            extra_info=extra_info
        )

        return embeddings, metadata
