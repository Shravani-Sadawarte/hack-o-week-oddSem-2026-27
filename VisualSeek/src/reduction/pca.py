from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.decomposition import PCA

from src.utils.logging import setup_logger

logger = setup_logger("visualseek.reduction.pca")

class PCAReducer:
    """Computes Principal Component Analysis for 2D/3D visualization and variance analysis."""

    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca_model: Optional[PCA] = None

    def fit_transform(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits PCA and transforms embeddings.
        
        Returns:
            reduced_data: shape [N, n_components]
            explained_variance_ratio: shape [n_components]
            cumulative_variance: shape [n_components]
        """
        n_samples, n_features = embeddings.shape
        k = min(self.n_components, n_samples, n_features)
        
        logger.info(f"Fitting PCA with {k} components on matrix shape {embeddings.shape}...")
        self.pca_model = PCA(n_components=k, random_state=self.random_state)
        transformed = self.pca_model.fit_transform(embeddings).astype(np.float32)

        var_ratio = self.pca_model.explained_variance_ratio_.astype(np.float32)
        cum_var = np.cumsum(var_ratio).astype(np.float32)

        logger.info(f"PCA fit complete. Top-2 components explain {cum_var[1]*100:.2f}% of variance. Total explained: {cum_var[-1]*100:.2f}%")
        return transformed, var_ratio, cum_var

    def get_2d(self, transformed: np.ndarray) -> np.ndarray:
        """Returns the first 2 principal components [N, 2]."""
        return transformed[:, :2]

    def get_3d(self, transformed: np.ndarray) -> np.ndarray:
        """Returns the first 3 principal components [N, 3]."""
        if transformed.shape[1] >= 3:
            return transformed[:, :3]
        raise ValueError("Transformed data has fewer than 3 components.")
