from typing import List, Union, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import open_clip

from src.models.base import BaseEncoder
from src.data.preprocessing import load_and_preprocess_image
from src.utils.logging import setup_logger

logger = setup_logger("visualseek.models.clip")

class CLIPEncoder(BaseEncoder):
    """Vision-Language dual encoder using OpenCLIP (e.g. ViT-B-32 / openai).
    
    Extracts L2-normalized 512-dimensional embeddings for both images and text queries
    projected into a shared semantic latent space.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto"
    ):
        self._model_name = model_name
        self.pretrained = pretrained

        # Hardware resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading CLIP model '{model_name}' (pretrained='{pretrained}') on device: {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Determine embedding dimension
        with torch.no_grad():
            dummy_text = self.tokenizer(["a dog"]).to(self.device)
            dummy_emb = self.model.encode_text(dummy_text)
            self._dim = dummy_emb.shape[-1]
            
        logger.info(f"CLIP encoder '{model_name}' initialized successfully. Embedding dimension: {self._dim}")

    @property
    def name(self) -> str:
        return f"CLIP-{self._model_name}"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def supports_text(self) -> bool:
        return True

    def encode_image(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encodes a list of images in batches and returns normalized float32 numpy array [N, D]."""
        if not images:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_inputs = images[i : i + batch_size]
                tensors = []
                for item in batch_inputs:
                    if isinstance(item, (str, Path)):
                        img = Image.open(item).convert("RGB")
                    else:
                        img = item.convert("RGB")
                    tensors.append(self.preprocess(img))

                batch_tensor = torch.stack(tensors, dim=0).to(self.device)
                features = self.model.encode_image(batch_tensor)

                if normalize:
                    features = features / features.norm(dim=-1, keepdim=True)

                embeddings_list.append(features.cpu().numpy().astype(np.float32))

        return np.concatenate(embeddings_list, axis=0)

    def encode_text(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encodes natural language queries into the shared latent space."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                text_tokens = self.tokenizer(batch_texts).to(self.device)
                features = self.model.encode_text(text_tokens)

                if normalize:
                    features = features / features.norm(dim=-1, keepdim=True)

                embeddings_list.append(features.cpu().numpy().astype(np.float32))

        return np.concatenate(embeddings_list, axis=0)
