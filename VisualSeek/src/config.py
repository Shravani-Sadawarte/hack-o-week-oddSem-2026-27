import os
from pathlib import Path
from typing import List, Optional, Any, Dict
import yaml
from pydantic import BaseModel, Field

from src.utils.paths import get_project_root, get_artifacts_dir, ensure_dir

class ProjectConfig(BaseModel):
    name: str = "VisualSeek"
    version: str = "1.0.0"
    description: str = "AI-Powered Visual Similarity & Semantic Search Engine"
    dev_mode: bool = True

class DatasetConfig(BaseModel):
    name: str = "stanford_dogs"
    path: Optional[str] = None
    limit: Optional[int] = 1000
    seed: int = 42
    supported_extensions: List[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])

class ModelConfig(BaseModel):
    name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "auto"
    batch_size: int = 32
    image_size: int = 224

class EmbeddingConfig(BaseModel):
    normalize: bool = True
    dimension: int = 512

class SearchConfig(BaseModel):
    engine: str = "faiss"
    top_k: int = 10
    supported_k: List[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    metric: str = "cosine"

class PCAConfig(BaseModel):
    n_components: int = 50
    random_state: int = 42

class TSNEConfig(BaseModel):
    perplexity: float = 30.0
    max_iter: int = 1000
    random_state: int = 42
    early_exaggeration: float = 12.0

class PathsConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    embeddings_dir: str = "artifacts/embeddings"
    indexes_dir: str = "artifacts/indexes"
    reductions_dir: str = "artifacts/visualizations"
    metadata_dir: str = "artifacts/metadata"
    cache_dir: str = "artifacts/cache"

class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    pca: PCAConfig = Field(default_factory=PCAConfig)
    tsne: TSNEConfig = Field(default_factory=TSNEConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def resolve_paths(self, root: Optional[Path] = None) -> None:
        """Resolves relative artifact paths against project root and ensures directories exist."""
        root = root or get_project_root()
        art_dir = get_artifacts_dir(self.paths.artifacts_dir)
        self.paths.artifacts_dir = str(art_dir)
        
        for attr in ["embeddings_dir", "indexes_dir", "reductions_dir", "metadata_dir", "cache_dir"]:
            val = getattr(self.paths, attr)
            p = Path(val) if Path(val).is_absolute() else root / val
            ensure_dir(p)
            setattr(self.paths, attr, str(p))

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads configuration from YAML with environment variable overrides."""
    root = get_project_root()
    path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    
    raw_data: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

    config = AppConfig(**raw_data)

    # Environment variable overrides
    if "VISUALSEEK_DATASET_DIR" in os.environ:
        config.dataset.path = os.environ["VISUALSEEK_DATASET_DIR"]
    
    if "VISUALSEEK_ARTIFACTS_DIR" in os.environ:
        config.paths.artifacts_dir = os.environ["VISUALSEEK_ARTIFACTS_DIR"]

    if "VISUALSEEK_MODEL" in os.environ:
        config.model.name = os.environ["VISUALSEEK_MODEL"]

    if "VISUALSEEK_DEVICE" in os.environ:
        config.model.device = os.environ["VISUALSEEK_DEVICE"]

    if "VISUALSEEK_DATASET_LIMIT" in os.environ:
        limit_val = os.environ["VISUALSEEK_DATASET_LIMIT"].strip().lower()
        if limit_val in ["none", "null", "all", "0"]:
            config.dataset.limit = None
        else:
            try:
                config.dataset.limit = int(limit_val)
            except ValueError:
                pass

    config.resolve_paths(root)
    return config
