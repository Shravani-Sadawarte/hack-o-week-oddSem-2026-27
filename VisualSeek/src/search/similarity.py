from typing import List, Tuple
import numpy as np

def cosine_similarity_search(
    query_vector: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes exact brute-force cosine similarity between a 1D/2D query and corpus embeddings.
    
    Args:
        query_vector: shape [D] or [1, D]
        corpus_embeddings: shape [N, D] (assumed L2-normalized)
        top_k: number of top results to return
        
    Returns:
        indices: shape [top_k]
        scores: shape [top_k] (cosine similarities in [-1.0, 1.0])
    """
    if query_vector.ndim == 1:
        query = query_vector.reshape(1, -1)
    else:
        query = query_vector

    # Ensure query is normalized
    q_norm = np.linalg.norm(query, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1e-10
    query_normed = query / q_norm

    # Dot product with normalized corpus equals cosine similarity
    similarities = np.dot(corpus_embeddings, query_normed.T).flatten()

    # Get Top-K indices sorted descending
    k = min(top_k, len(similarities))
    # argpartition is O(N) followed by sort of top k
    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
        
    candidate_indices = np.argpartition(similarities, -k)[-k:]
    sorted_sub_indices = np.argsort(similarities[candidate_indices])[::-1]
    top_indices = candidate_indices[sorted_sub_indices]
    top_scores = similarities[top_indices]

    return top_indices, top_scores
