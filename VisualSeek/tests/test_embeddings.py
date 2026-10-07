import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

from src.models.baseline import BaselineColorHistogramEncoder
from src.embeddings.storage import EmbeddingStorage

def test_baseline_encoder_dimensions_and_normalization():
    encoder = BaselineColorHistogramEncoder(bins_rgb=4, bins_hsv=4)
    assert encoder.embedding_dim == 128
    assert encoder.supports_text is False

    # Create synthetic images
    img1 = Image.new("RGB", (64, 64), color="blue")
    img2 = Image.new("RGB", (64, 64), color="yellow")

    embs = encoder.encode_image([img1, img2], batch_size=2, normalize=True)
    assert embs.shape == (2, 128)
    assert embs.dtype == np.float32

    # Check L2 normalization
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

def test_embedding_storage_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = EmbeddingStorage(base_dir=Path(tmp_dir))
        
        tag = "test_tag"
        dummy_embs = np.random.randn(10, 64).astype(np.float32)
        # Normalize
        dummy_embs /= np.linalg.norm(dummy_embs, axis=1, keepdims=True)
        dummy_meta = [{"image_id": f"img_{i}", "breed": f"breed_{i % 2}"} for i in range(10)]

        assert not storage.exists(tag)

        storage.save(tag, dummy_embs, dummy_meta, model_name="test_model")
        assert storage.exists(tag)

        loaded_embs, loaded_meta, header = storage.load(tag)
        np.testing.assert_allclose(loaded_embs, dummy_embs, atol=1e-6)
        assert len(loaded_meta) == 10
        assert header["model_name"] == "test_model"
        assert header["embedding_dim"] == 64
