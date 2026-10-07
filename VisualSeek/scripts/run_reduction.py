import argparse
import sys
import json
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.embeddings.storage import EmbeddingStorage
from src.reduction.pca import PCAReducer
from src.reduction.tsne import TSNEReducer
from src.reduction.umap import UMAPReducer, UMAP_AVAILABLE

def main():
    parser = argparse.ArgumentParser(description="Precompute PCA and t-SNE projections for VisualSeek")
    parser.add_argument("--tag", type=str, default="default", help="Artifact tag name")
    args = parser.parse_args()

    config = load_config()
    print("=" * 60)
    print(" VISUALSEEK — DIMENSIONALITY REDUCTION PIPELINE")
    print("=" * 60)

    storage = EmbeddingStorage(base_dir=Path(config.paths.artifacts_dir))
    if not storage.exists(args.tag):
        print(f"ERROR: Embeddings for tag '{args.tag}' do not exist.")
        print("Please run: python scripts/generate_embeddings.py first.")
        sys.exit(1)

    print("\n[1/3] Loading embeddings...")
    embeddings, metadata, header = storage.load(args.tag)
    print(f"Loaded {len(embeddings)} vectors ({embeddings.shape[1]}-dim).")

    print("\n[2/3] Computing PCA (2D, 3D, and explained variance ratios)...")
    pca_reducer = PCAReducer(n_components=config.pca.n_components, random_state=config.pca.random_state)
    pca_transformed, var_ratio, cum_var = pca_reducer.fit_transform(embeddings)
    pca_2d = pca_reducer.get_2d(pca_transformed)
    pca_3d = pca_reducer.get_3d(pca_transformed)

    print("\n[3/3] Computing t-SNE 2D non-linear projection...")
    tsne_reducer = TSNEReducer(
        perplexity=config.tsne.perplexity,
        max_iter=config.tsne.max_iter,
        random_state=config.tsne.random_state,
        early_exaggeration=config.tsne.early_exaggeration
    )
    tsne_2d = tsne_reducer.fit_transform(embeddings)

    # Optional UMAP
    umap_2d = None
    if UMAP_AVAILABLE:
        try:
            print("\nComputing optional UMAP 2D projection...")
            umap_reducer = UMAPReducer()
            umap_2d = umap_reducer.fit_transform(embeddings)
        except Exception as e:
            print(f"UMAP skipped: {e}")

    # Save projections
    reductions_dir = Path(config.paths.reductions_dir)
    reductions_dir.mkdir(parents=True, exist_ok=True)
    npz_path = reductions_dir / f"{args.tag}_reductions.npz"

    save_dict = {
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "tsne_2d": tsne_2d,
        "pca_var_ratio": var_ratio,
        "pca_cum_var": cum_var
    }
    if umap_2d is not None:
        save_dict["umap_2d"] = umap_2d

    np.savez_compressed(npz_path, **save_dict)

    meta_json_path = reductions_dir / f"{args.tag}_reductions_meta.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_points": len(embeddings),
            "pca_explained_variance_ratio": var_ratio.tolist(),
            "pca_cumulative_variance": cum_var.tolist(),
            "tsne_perplexity": config.tsne.perplexity,
            "has_umap": (umap_2d is not None)
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f" SUCCESS: Projections saved to {npz_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
