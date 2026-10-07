from typing import Tuple, Optional
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.utils.logging import setup_logger

logger = setup_logger("visualseek.reduction.tsne")

class TSNEReducer:
    """Computes 2D t-SNE non-linear projection using PCA pre-reduction."""

    def __init__(
        self,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        pca_components: int = 50,
        random_state: int = 42,
        early_exaggeration: float = 12.0
    ):
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.pca_components = pca_components
        self.random_state = random_state
        self.early_exaggeration = early_exaggeration

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Projects high-dimensional embeddings to 2D coordinates [N, 2]."""
        n_samples, n_features = embeddings.shape

        # Step 1: PCA pre-reduction if dimension is high
        if n_features > self.pca_components and n_samples > self.pca_components:
            logger.info(f"Applying PCA pre-reduction from {n_features} to {self.pca_components} dims for t-SNE stability...")
            pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            reduced_inputs = pca.fit_transform(embeddings)
        else:
            reduced_inputs = embeddings

        # Adjust perplexity if sample count is small
        effective_perp = min(self.perplexity, max(1.0, (n_samples - 1) / 3.0))

        logger.info(f"Running t-SNE (perplexity={effective_perp:.1f}, max_iter={self.max_iter}, samples={n_samples})...")
        
        # In sklearn, max_iter is standard
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perp,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_exaggeration=self.early_exaggeration,
            init="pca",
            learning_rate="auto"
        )
        
        coords_2d = tsne.fit_transform(reduced_inputs).astype(np.float32)
        logger.info("t-SNE 2D projection completed successfully.")
        return coords_2d
