import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.utils.logging import setup_logger
from src.utils.paths import ensure_dir

logger = setup_logger("visualseek.embeddings.storage")

class EmbeddingStorage:
    """Manages persistence and retrieval of computed vector embeddings and corresponding image metadata."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.embeddings_dir = ensure_dir(self.base_dir / "embeddings")
        self.metadata_dir = ensure_dir(self.base_dir / "metadata")

    def _get_paths(self, tag: str) -> Tuple[Path, Path]:
        emb_path = self.embeddings_dir / f"{tag}_embeddings.npz"
        meta_path = self.metadata_dir / f"{tag}_metadata.json"
        return emb_path, meta_path

    def exists(self, tag: str) -> bool:
        emb_path, meta_path = self._get_paths(tag)
        return emb_path.exists() and meta_path.exists()

    def save(
        self,
        tag: str,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        model_name: str,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Saves embeddings in compressed .npz format and metadata in JSON."""
        emb_path, meta_path = self._get_paths(tag)

        # 1. Save embeddings
        np.savez_compressed(
            emb_path,
            embeddings=embeddings.astype(np.float32),
            model_name=model_name
        )

        # 2. Save metadata
        payload = {
            "tag": tag,
            "model_name": model_name,
            "num_records": len(metadata),
            "embedding_dim": int(embeddings.shape[1]),
            "extra": extra_info or {},
            "records": metadata
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved {len(embeddings)} embeddings ({embeddings.shape[1]}-dim) to {emb_path} and metadata to {meta_path}")

    def load(self, tag: str) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Loads embeddings, metadata records, and header info."""
        emb_path, meta_path = self._get_paths(tag)

        if not (emb_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Embedding or metadata artifact not found for tag '{tag}' in {self.base_dir}")

        with np.load(emb_path) as data:
            embeddings = data["embeddings"]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        records = meta_data["records"]
        header = {
            "tag": meta_data.get("tag", tag),
            "model_name": meta_data.get("model_name", "unknown"),
            "num_records": meta_data.get("num_records", len(records)),
            "embedding_dim": meta_data.get("embedding_dim", embeddings.shape[1]),
            "extra": meta_data.get("extra", {})
        }

        logger.info(f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-dim) for tag '{tag}'")
        return embeddings, records, header
