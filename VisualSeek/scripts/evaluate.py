import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.loader import StanfordDogsDataset
from src.models.baseline import BaselineColorHistogramEncoder
from src.models.clip_encoder import CLIPEncoder
from src.evaluation.benchmark import ModelBenchmarkSuite

def main():
    parser = argparse.ArgumentParser(description="Evaluate and benchmark retrieval performance for VisualSeek")
    parser.add_argument("--queries", type=int, default=50, help="Number of query images to evaluate")
    parser.add_argument("--limit", type=int, default=500, help="Dataset sample limit for evaluation")
    args = parser.parse_args()

    config = load_config()
    print("=" * 60)
    print(" VISUALSEEK — RETRIEVAL EVALUATION & BENCHMARK")
    print("=" * 60)

    print(f"\n[1/3] Loading evaluation dataset (limit={args.limit})...")
    dataset = StanfordDogsDataset(
        images_dir=config.dataset.path,
        limit=args.limit,
        seed=42
    )
    print(f"Loaded {len(dataset)} images across {len(dataset.get_breed_distribution())} breeds.")

    suite = ModelBenchmarkSuite(dataset)

    print("\n[2/3] Benchmarking Baseline (Color & Texture Histograms)...")
    base_encoder = BaselineColorHistogramEncoder()
    base_res = suite.benchmark_model(base_encoder, num_eval_queries=args.queries)
    print(f"  P@1:  {base_res['precision_at_1']*100:.1f}%")
    print(f"  P@5:  {base_res['precision_at_5']*100:.1f}%")
    print(f"  P@10: {base_res['precision_at_10']*100:.1f}%")
    print(f"  MRR:  {base_res['mrr']:.4f}")
    print(f"  FAISS Latency: {base_res['mean_search_latency_ms']:.2f} ms")

    print("\n[3/3] Benchmarking CLIP (ViT-B/32)...")
    clip_encoder = CLIPEncoder(model_name=config.model.name, pretrained=config.model.pretrained, device=config.model.device)
    clip_res = suite.benchmark_model(clip_encoder, num_eval_queries=args.queries)
    print(f"  P@1:  {clip_res['precision_at_1']*100:.1f}%")
    print(f"  P@5:  {clip_res['precision_at_5']*100:.1f}%")
    print(f"  P@10: {clip_res['precision_at_10']*100:.1f}%")
    print(f"  MRR:  {clip_res['mrr']:.4f}")
    print(f"  FAISS Latency: {clip_res['mean_search_latency_ms']:.2f} ms")

    # Export
    art_dir = Path(config.paths.artifacts_dir)
    json_path = art_dir / "benchmark_results.json"
    csv_path = art_dir / "benchmark_results.csv"
    suite.export(json_path=json_path, csv_path=csv_path)

    print("\n" + "=" * 60)
    print(" SUMMARY COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Baseline (Histogram)':<22} | {'CLIP (ViT-B/32)':<18}")
    print("-" * 72)
    print(f"{'Embedding Dimension':<25} | {base_res['embedding_dim']:<22} | {clip_res['embedding_dim']:<18}")
    print(f"{'Precision@1':<25} | {base_res['precision_at_1']*100:>20.1f}% | {clip_res['precision_at_1']*100:>16.1f}%")
    print(f"{'Precision@5':<25} | {base_res['precision_at_5']*100:>20.1f}% | {clip_res['precision_at_5']*100:>16.1f}%")
    print(f"{'Precision@10':<25} | {base_res['precision_at_10']*100:>20.1f}% | {clip_res['precision_at_10']*100:>16.1f}%")
    print(f"{'MRR':<25} | {base_res['mrr']:>21.4f} | {clip_res['mrr']:>17.4f}")
    print(f"{'FAISS Latency (ms)':<25} | {base_res['mean_search_latency_ms']:>21.2f} | {clip_res['mean_search_latency_ms']:>17.2f}")
    print("=" * 60)
    print(f"Results exported to {json_path}")

if __name__ == "__main__":
    main()
