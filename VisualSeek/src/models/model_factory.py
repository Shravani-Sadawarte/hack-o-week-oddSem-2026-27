from typing import Optional
from src.models.base import BaseEncoder
from src.models.baseline import BaselineColorHistogramEncoder
from src.models.clip_encoder import CLIPEncoder
from src.config import ModelConfig

class ModelFactory:
    """Factory to instantiate embedding encoders based on configuration."""

    @staticmethod
    def create(
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto"
    ) -> BaseEncoder:
        model_name_lower = model_name.lower()
        if "baseline" in model_name_lower or "histogram" in model_name_lower or "color" in model_name_lower:
            return BaselineColorHistogramEncoder()
        else:
            # Default to CLIP
            return CLIPEncoder(
                model_name=model_name,
                pretrained=pretrained,
                device=device
            )

    @staticmethod
    def from_config(config: ModelConfig) -> BaseEncoder:
        return ModelFactory.create(
            model_name=config.name,
            pretrained=config.pretrained,
            device=config.device
        )
