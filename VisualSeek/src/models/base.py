from abc import ABC, abstractmethod
from typing import List, Union, Optional
from pathlib import Path
from PIL import Image
import numpy as np

class BaseEncoder(ABC):
    """Abstract base class for all image and text embedding encoders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name/identifier of this model."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the dimension of the embedding vectors produced by this model."""
        pass

    @property
    @abstractmethod
    def supports_text(self) -> bool:
        """Indicates whether this model supports encoding natural language text into the same vector space."""
        pass

    @abstractmethod
    def encode_image(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encodes a list of images into normalized float32 embedding vectors of shape [N, D]."""
        pass

    def encode_text(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encodes a list of text queries into normalized float32 embedding vectors of shape [N, D].
        Raises NotImplementedError if supports_text is False.
        """
        if not self.supports_text:
            raise NotImplementedError(f"Model {self.name} does not support text encoding.")
        raise NotImplementedError
