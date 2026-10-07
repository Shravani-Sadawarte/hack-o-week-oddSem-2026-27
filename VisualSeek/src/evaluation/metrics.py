from typing import List, Dict, Any, Tuple
import numpy as np
import time

from src.search.similarity import cosine_similarity_search
from src.search.faiss_index import FaissIndexManager
from src.utils.logging import setup_logger

logger = setup_logger("visualseek.evaluation.metrics")

def precision_at_k(query_breed: str, retrieved_breeds: List[str], k: int) -> float:
    """Calculates Precision@K: the fraction of retrieved items in Top-K matching query_breed."""
    sub = retrieved_breeds[:k]
    if not sub:
        return 0.0
    matches = sum(1 for b in sub if b == query_breed)
    return matches / len(sub)

def reciprocal_rank(query_breed: str, retrieved_breeds: List[str]) -> float:
    """Calculates Reciprocal Rank: 1 / (rank of first matching breed), or 0.0 if not found."""
    for rank, b in enumerate(retrieved_breeds, start=1):
        if b == query_breed:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval_performance(
    corpus_embeddings: np.ndarray,
    metadata_records: List[Dict[str, Any]],
    num_query_samples: int = 100,
    ks: List[int] = [1, 5, 10],
    seed: int = 42,
    use_faiss: bool = True
) -> Dict[str, Any]:
    """Runs leave-one-out retrieval evaluation over a sample of corpus vectors.
    
    Caveat: Same-breed matching is used as an objective quantitative proxy.
    Semantic visual similarity frequently bridges across distinct breeds
    (e.g., color, texture, pose, or background).
    """
    n_samples = len(metadata_records)
    if n_samples < 2:
        return {"error": "Insufficient dataset records for evaluation."}

    rng = np.random.RandomState(seed)
    max_k = max(ks) + 1  # +1 because query itself will be retrieved at rank 1

    sample_indices = rng.choice(n_samples, size=min(num_query_samples, n_samples), replace=False)

    faiss_mgr = None
    if use_faiss:
        faiss_mgr = FaissIndexManager(dimension=corpus_embeddings.shape[1])
        faiss_mgr.build(corpus_embeddings)

    precisions: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: List[float] = []
    latencies: List[float] = []

    for q_idx in sample_indices:
        q_vec = corpus_embeddings[q_idx]
        q_breed = metadata_records[q_idx]["breed"]

        t0 = time.perf_counter()
        if faiss_mgr:
            ret_indices, _ = faiss_mgr.search(q_vec, top_k=max_k)
        else:
            ret_indices, _ = cosine_similarity_search(q_vec, corpus_embeddings, top_k=max_k)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        # Leave-one-out: filter out the exact query image itself
        retrieved_breeds = [
            metadata_records[idx]["breed"]
            for idx in ret_indices
            if idx != q_idx
        ]

        for k in ks:
            precisions[k].append(precision_at_k(q_breed, retrieved_breeds, k))

        mrrs.append(reciprocal_rank(q_breed, retrieved_breeds))

    results = {
        "num_evaluated_queries": len(sample_indices),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "mrr": round(float(np.mean(mrrs)), 4),
        "notes": "Same-breed retrieval proxy evaluation (leave-one-out methodology)."
    }

    for k in ks:
        results[f"precision_at_{k}"] = round(float(np.mean(precisions[k])), 4)

    logger.info(f"Evaluation complete on {len(sample_indices)} queries: P@1={results.get('precision_at_1')}, P@5={results.get('precision_at_5')}, P@10={results.get('precision_at_10')}, Latency={results['mean_latency_ms']}ms")
    return results
