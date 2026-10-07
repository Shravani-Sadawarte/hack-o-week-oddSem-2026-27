from typing import Union, List, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import torch

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def get_image_transform(image_size: int = 224):
    """Returns standard preprocessing pipeline for image encoders."""
    if TORCHVISION_AVAILABLE:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
    else:
        # Fallback pure-Pillow + NumPy transform
        return None

def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image],
    image_size: int = 224,
    transform = None
) -> torch.Tensor:
    """Loads an image from file path or PIL instance, converts to RGB, and returns a preprocessed 4D tensor [1, C, H, W]."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if transform is not None:
        tensor = transform(img)
    elif TORCHVISION_AVAILABLE:
        t = get_image_transform(image_size)
        tensor = t(img)
    else:
        # Manual fallback
        img_resized = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.array(img_resized, dtype=np.float32) / 255.0  # [H, W, C]
        arr = (arr - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # [C, H, W]

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]

    return tensor

def preprocess_image_batch(
    images: List[Union[str, Path, Image.Image]],
    image_size: int = 224,
    transform = None
) -> torch.Tensor:
    """Processes a batch of images into a single stacked tensor [B, C, H, W]."""
    tensors = [load_and_preprocess_image(img, image_size, transform).squeeze(0) for img in images]
    return torch.stack(tensors, dim=0)
