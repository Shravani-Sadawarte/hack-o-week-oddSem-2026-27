import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.embeddings.storage import EmbeddingStorage
from src.search.faiss_index import FaissIndexManager

def main():
    parser = argparse.ArgumentParser(description="Build and save FAISS index for VisualSeek")
    parser.add_argument("--tag", type=str, default="default", help="Embedding tag name")
    args = parser.parse_args()

    config = load_config()
    print("=" * 60)
    print(" VISUALSEEK — BUILD FAISS INDEX")
    print("=" * 60)

    storage = EmbeddingStorage(base_dir=Path(config.paths.artifacts_dir))
    if not storage.exists(args.tag):
        print(f"ERROR: Embeddings for tag '{args.tag}' do not exist.")
        print("Please run: python scripts/generate_embeddings.py first.")
        sys.exit(1)

    print("\n[1/2] Loading cached embeddings...")
    embeddings, metadata, header = storage.load(args.tag)
    dim = embeddings.shape[1]
    print(f"Loaded {len(embeddings)} vectors with dimension {dim}.")

    print("\n[2/2] Building FAISS IndexFlatIP (Cosine Similarity)...")
    faiss_mgr = FaissIndexManager(dimension=dim, metric=config.search.metric)
    faiss_mgr.build(embeddings)

    index_save_path = Path(config.paths.indexes_dir) / f"{args.tag}_faiss.index"
    faiss_mgr.save(index_save_path)

    print("\n" + "=" * 60)
    print(f" SUCCESS: FAISS index saved to {index_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
