import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.utils.logging import setup_logger
from src.data.loader import StanfordDogsDataset
from src.models.base import BaseEncoder
from src.models.baseline import BaselineColorHistogramEncoder
from src.models.clip_encoder import CLIPEncoder
from src.evaluation.metrics import evaluate_retrieval_performance

logger = setup_logger("visualseek.evaluation.benchmark")

class ModelBenchmarkSuite:
    """Benchmarks and compares multiple feature representations."""

    def __init__(self, dataset: StanfordDogsDataset):
        self.dataset = dataset
        self.results: List[Dict[str, Any]] = []

    def benchmark_model(
        self,
        encoder: BaseEncoder,
        batch_size: int = 32,
        num_eval_queries: int = 50
    ) -> Dict[str, Any]:
        """Runs full benchmark on a given encoder."""
        logger.info(f"Starting benchmark for encoder: {encoder.name}...")
        image_paths = [r.file_path for r in self.dataset.records]

        # 1. Measure embedding generation
        t0 = time.perf_counter()
        embeddings = encoder.encode_image(image_paths, batch_size=batch_size, normalize=True)
        gen_time_sec = time.perf_counter() - t0
        ms_per_image = (gen_time_sec / max(1, len(image_paths))) * 1000

        # 2. Measure storage size estimation
        storage_kb = (embeddings.nbytes) / 1024.0

        # 3. Measure retrieval performance & latency
        meta_records = self.dataset.to_metadata_list()
        eval_metrics = evaluate_retrieval_performance(
            corpus_embeddings=embeddings,
            metadata_records=meta_records,
            num_query_samples=num_eval_queries,
            ks=[1, 5, 10],
            use_faiss=True
        )

        record = {
            "model_name": encoder.name,
            "supports_text": encoder.supports_text,
            "embedding_dim": encoder.embedding_dim,
            "total_images": len(self.dataset),
            "generation_time_sec": round(gen_time_sec, 2),
            "generation_ms_per_image": round(ms_per_image, 2),
            "storage_size_kb": round(storage_kb, 2),
            "mean_search_latency_ms": eval_metrics.get("mean_latency_ms", 0.0),
            "precision_at_1": eval_metrics.get("precision_at_1", 0.0),
            "precision_at_5": eval_metrics.get("precision_at_5", 0.0),
            "precision_at_10": eval_metrics.get("precision_at_10", 0.0),
            "mrr": eval_metrics.get("mrr", 0.0)
        }
        self.results.append(record)
        return record

    def export(self, json_path: Path, csv_path: Path) -> None:
        """Exports benchmark comparison to JSON and CSV."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        if self.results:
            keys = list(self.results[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)

        logger.info(f"Exported benchmark results to {json_path} and {csv_path}")
