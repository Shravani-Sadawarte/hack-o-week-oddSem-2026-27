import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.loader import StanfordDogsDataset
from src.models.model_factory import ModelFactory
from src.embeddings.storage import EmbeddingStorage
from src.embeddings.generator import EmbeddingPipeline
from src.utils.logging import setup_logger

logger = setup_logger("scripts.generate_embeddings")

def main():
    parser = argparse.ArgumentParser(description="Generate image embeddings for VisualSeek")
    parser.add_argument("--limit", type=str, default=None, help="Dataset sample limit (integer or 'all')")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--tag", type=str, default="default", help="Artifact tag name")
    parser.add_argument("--force", action="store_true", help="Force recomputing embeddings even if cached")
    args = parser.parse_args()

    config = load_config()

    # Determine limit
    if args.limit:
        limit_val = args.limit.strip().lower()
        limit = None if limit_val in ["all", "none", "0"] else int(limit_val)
    else:
        limit = config.dataset.limit

    batch_size = args.batch_size or config.model.batch_size

    print("=" * 60)
    print(" VISUALSEEK — EMBEDDING GENERATION")
    print("=" * 60)
    print(f"Model:       {config.model.name} (pretrained='{config.model.pretrained}')")
    print(f"Device:      {config.model.device}")
    print(f"Batch Size:  {batch_size}")
    print(f"Limit:       {limit if limit else 'ALL (Full 20,580 images)'}")
    print(f"Mode:        {'DEVELOPMENT (Partial Index)' if limit else 'FULL DATASET'}")
    print("=" * 60)

    print("\n[1/3] Loading dataset...")
    dataset = StanfordDogsDataset(
        images_dir=config.dataset.path,
        limit=limit,
        seed=config.dataset.seed
    )
    print(f"Dataset ready: {len(dataset)} images selected.")

    print("\n[2/3] Initializing embedding encoder...")
    encoder = ModelFactory.from_config(config.model)

    print("\n[3/3] Running embedding pipeline...")
    storage = EmbeddingStorage(base_dir=Path(config.paths.artifacts_dir))
    pipeline = EmbeddingPipeline(encoder=encoder, storage=storage)

    embeddings, metadata = pipeline.run(
        dataset=dataset,
        tag=args.tag,
        batch_size=batch_size,
        force_recompute=args.force
    )

    print("\n" + "=" * 60)
    print(f" SUCCESS: {len(embeddings)} embeddings ({embeddings.shape[1]}-dim) cached under:")
    print(f" {config.paths.artifacts_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
