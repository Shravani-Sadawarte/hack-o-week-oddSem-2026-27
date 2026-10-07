from typing import List, Union
from pathlib import Path
from PIL import Image
import numpy as np

from src.models.base import BaseEncoder

class BaselineColorHistogramEncoder(BaseEncoder):
    """Simple baseline feature extractor using 3D RGB and HSV color histograms.
    
    Serves as an empirical baseline to compare against deep semantic representations.
    
    Limitations:
    - Purely color and tone dependent; lacks structural, shape, and semantic understanding.
    - Fails to distinguish between different dog breeds with similar fur colors.
    - Completely lacks cross-modal capability (cannot encode text).
    """

    def __init__(self, bins_rgb: int = 4, bins_hsv: int = 4):
        self.bins_rgb = bins_rgb
        self.bins_hsv = bins_hsv
        # 4*4*4 = 64 for RGB + 4*4*4 = 64 for HSV -> 128 dimensions total
        self._dim = (bins_rgb ** 3) + (bins_hsv ** 3)

    @property
    def name(self) -> str:
        return "Color-Histogram-Baseline"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def supports_text(self) -> bool:
        return False

    def _extract_single_vector(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # Resize for consistent histogram computation
        img = img.resize((128, 128), Image.Resampling.BILINEAR)
        rgb_arr = np.array(img, dtype=np.float32)

        # 1. 3D RGB Histogram
        rgb_hist, _ = np.histogramdd(
            rgb_arr.reshape(-1, 3),
            bins=(self.bins_rgb, self.bins_rgb, self.bins_rgb),
            range=[(0, 256), (0, 256), (0, 256)]
        )
        rgb_vec = rgb_hist.flatten()

        # 2. 3D HSV Histogram
        hsv_img = img.convert("HSV")
        hsv_arr = np.array(hsv_img, dtype=np.float32)
        hsv_hist, _ = np.histogramdd(
            hsv_arr.reshape(-1, 3),
            bins=(self.bins_hsv, self.bins_hsv, self.bins_hsv),
            range=[(0, 256), (0, 256), (0, 256)]
        )
        hsv_vec = hsv_hist.flatten()

        feat = np.concatenate([rgb_vec, hsv_vec]).astype(np.float32)
        return feat

    def encode_image(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        vectors = []
        for img in images:
            vec = self._extract_single_vector(img)
            vectors.append(vec)

        matrix = np.stack(vectors, axis=0)

        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            matrix = matrix / norms

        return matrix
