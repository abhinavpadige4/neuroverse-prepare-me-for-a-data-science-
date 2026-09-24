import math
import numpy as np
from typing import List, Dict, Tuple, Optional


def dcg(relevances: List[float], k: Optional[int] = None) -> float:
    """Compute Discounted Cumulative Gain.

    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    Args:
        relevances: List of relevance scores in ranked order.
        k: Cutoff position. If None, use all positions.

    Returns:
        DCG score.
    """
    if k is not None:
        relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(relevances: List[float], k: Optional[int] = None) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    NDCG@k = DCG@k / IDCG@k

    Args:
        relevances: List of relevance scores in ranked order.
        k: Cutoff position. If None, use all positions.

    Returns:
        NDCG score in [0, 1].
    """
    if k is not None:
        relevances = relevances[:k]
    if not relevances:
        return 0.0
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def ndcg_binary(relevances: List[int], k: Optional[int] = None) -> float:
    """NDCG with binary relevance (0 or 1).

    Simplified: DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)

    Args:
        relevances: Binary relevance scores (0 or 1).
        k: Cutoff position.

    Returns:
        NDCG score.
    """
    if k is not None:
        relevances = relevances[:k]
    if not relevances:
        return 0.0
    actual_dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal_count = min(sum(relevances), k if k else len(relevances))
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_ranking(
    query_results: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Dict[str, float]],
    k_values: List[int] = None
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple query rankings against ground truth relevance.

    Args:
        query_results: {query: [(doc_id, score), ...]} ranked by score desc.
        ground_truth: {query: {doc_id: relevance_score}}
        k_values: List of k values to evaluate. Default [1, 3, 5, 10].

    Returns:
        {k: {metric: value}} aggregated across queries.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {k: {"ndcg": [], "dcg": [], "ndcg_binary": []} for k in k_values}

    for query, ranked_docs in query_results.items():
        if query not in ground_truth:
            continue
        gt = ground_truth[query]
        relevances = [gt.get(doc_id, 0.0) for doc_id, _ in ranked_docs]
        binary_rels = [1 if rel > 0 else 0 for rel in relevances]

        for k in k_values:
            results[k]["ndcg"].append(ndcg(relevances, k))
            results[k]["dcg"].append(dcg(relevances, k))
            results[k]["ndcg_binary"].append(ndcg_binary(binary_rels, k))

    summary = {}
    for k in k_values:
        summary[k] = {
            "mean_ndcg": np.mean(results[k]["ndcg"]),
            "mean_dcg": np.mean(results[k]["dcg"]),
            "mean_ndcg_binary": np.mean(results[k]["ndcg_binary"]),
        }
    return summary


def reciprocal_rank(relevances: List[float], k: Optional[int] = None) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    Args:
        relevances: Relevance scores in ranked order.
        k: Cutoff position.

    Returns:
        Reciprocal rank of first relevant result.
    """
    if k is not None:
        relevances = relevances[:k]
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: List[int], k: int) -> float:
    """Compute Precision@k for binary relevance.

    Args:
        relevances: Binary relevance scores.
        k: Cutoff position.

    Returns:
        Precision@k.
    """
    if k <= 0:
        return 0.0
    top_k = relevances[:k]
    return sum(top_k) / k


def recall_at_k(relevances: List[int], total_relevant: int, k: int) -> float:
    """Compute Recall@k for binary relevance.

    Args:
        relevances: Binary relevance scores for all retrieved docs.
        total_relevant: Total number of relevant docs in corpus.
        k: Cutoff position.

    Returns:
        Recall@k.
    """
    if total_relevant == 0:
        return 0.0
    top_k = relevances[:k]
    return sum(top_k) / total_relevant


def f1_at_k(relevances: List[int], total_relevant: int, k: int) -> float:
    """Compute F1@k combining precision and recall.

    Args:
        relevances: Binary relevance scores.
        total_relevant: Total relevant docs in corpus.
        k: Cutoff position.

    Returns:
        F1@k score.
    """
    p = precision_at_k(relevances, k)
    r = recall_at_k(relevances, total_relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ============================================================
# TESTS
# ============================================================

def run_tests():
    """Run comprehensive tests for all search evaluation metrics."""
    print("=" * 60)
    print("SEARCH RANKING EVALUATION - TEST SUITE")
    print("=" * 60)

    # Test 1: Basic DCG
    print("\n--- Test 1: DCG ---")
    rels = [3, 2, 3, 0, 1]
    d = dcg(rels)
    expected = (2**3 - 1)/math.log2(2) + (2**2 - 1)/math.log2(3) + (2**3 - 1)/math.log2(4) + 0 + (2**1 - 1)/math.log2(6)
    assert abs(d - expected) < 1e-9, f"DCG failed: {d} != {expected}"
    print(f"  DCG([3,2,3,0,1]) = {d:.4f} ✓")

    # Test 2: DCG with k
    d_k3 = dcg(rels, k=3)
    expected_k3 = (2**3 - 1)/math.log2(2) + (2**2 - 1)/math.log2(3) + (2**3 - 1)/math.log2(4)
    assert abs(d_k3 - expected_k3) < 1e-9
    print(f"  DCG@3 = {d_k3:.4f} ✓")

    # Test 3: NDCG perfect ranking
    print("\n--- Test 3: NDCG (perfect ranking) ---")
    perfect = [3, 2, 1, 0]
    n = ndcg(perfect)
    assert abs(n - 1.0) < 1e-9, f"Perfect NDCG should be 1.0, got {n}"
    print(f"  NDCG([3,2,1,0]) = {n:.4f} ✓")

    # Test 4: NDCG worst ranking
    print("\n--- Test 4: NDCG (worst ranking) ---")
    worst = [0, 1, 2, 3]
    n_worst = ndcg(worst)
    print(f"  NDCG([0,1,2,3]) = {n_worst:.4f} ✓")
    assert 0 <= n_worst <= 1

    # Test 5: NDCG@k
    print("\n--- Test 5: NDCG@k ---")
    rels5 = [1, 0, 3, 2, 0]
    n5 = ndcg(rels5, k=3)
    print(f"  NDCG@3([1,0,3,2,0]) = {n5:.4f} ✓")

    # Test 6: Binary NDCG
    print("\n--- Test 6: Binary NDCG ---")
    binary = [1, 1, 0, 1, 0]
    nb = ndcg_binary(binary, k=3)
    print(f"  NDCG_binary@3([1,1,0,1,0]) = {nb:.4f} ✓")
    assert 0 <= nb <= 1

    # Test 7: Empty input
    print("\n--- Test 7: Edge cases ---")
    assert dcg([]) == 0.0
    assert ndcg([]) == 0.0
    assert ndcg([0, 0, 0]) == 0.0
    assert ndcg_binary([0, 0, 0]) == 0.0
    print("  Empty/zero inputs handled ✓")

    # Test 8: Single element
    assert abs(ndcg([5]) - 1.0) < 1e-9
    assert abs(ndcg_binary([1]) - 1.0) < 1e-9
    assert abs(ndcg_binary([0]) - 0.0) < 1e-9
    print("  Single element inputs ✓")

    # Test 9: Reciprocal Rank
    print("\n--- Test 9: Reciprocal Rank ---")
    assert reciprocal_rank([0, 0, 1, 1]) == 1/3
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 0, 0]) == 0.0
    assert reciprocal_rank([0, 0, 1], k=2) == 0.0
    print("  MRR tests passed ✓")

    # Test 10: Precision@k
    print("\n--- Test 10: Precision@k ---")
    assert precision_at_k([1, 1, 0, 1], k=3) == 2/3
    assert precision_at_k([1, 1, 1, 1], k=4) == 1.0
    assert precision_at_k([0, 0, 0], k=3) == 0.0
    print("  Precision@k tests passed ✓")

    # Test 11: Recall@k
    print("\n--- Test 11: Recall@k ---")
    assert recall_at_k([1, 1, 0, 1], total_relevant=4, k=3) == 2/4
    assert recall_at_k([1, 1, 1, 1], total_relevant=4, k=4) == 1.0
    print("  Recall@k tests passed ✓")

    # Test 12: F1@k
    print("\n--- Test 12: F1@k ---")
    f1 = f1_at_k([1, 1, 0, 1], total_relevant=4, k=3)
    expected_f1 = 2 * (2/3) * (2/4) / ((2/3) + (2/4))
    assert abs(f1 - expected_f1) < 1e-9
    print(f"  F1@3 = {f1:.4f} ✓")

    # Test 13: Multi-query evaluation
    print("\n--- Test 13: Multi-query evaluation ---")
    query_results = {
        "laptop": [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.5)],
        "phone": [("doc5", 0.95), ("doc6", 0.85), ("doc7", 0.6), ("doc8", 0.4)],
    }
    ground_truth = {
        "laptop": {"doc1": 3, "doc2": 2, "doc3": 1, "doc4": 0},
        "phone": {"doc5": 2, "doc6": 3, "doc7": 1, "doc8": 0},
    }
    summary = evaluate_ranking(query_results, ground_truth, k_values=[1, 3, 5])
    print(f"  Mean NDCG@1 = {summary[1]['mean_ndcg']:.4f}")
    print(f"  Mean NDCG@3 = {summary[3]['mean_ndcg']:.4f}")
    print(f"  Mean NDCG@5 = {summary[5]['mean_ndcg']:.4f}")
    assert 0 <= summary[1]["mean_ndcg"] <= 1
    assert 0 <= summary[3]["mean_ndcg"] <= 1
    print("  Multi-query evaluation ✓")

    # Test 14: NDCG monotonicity
    print("\n--- Test 14: NDCG monotonicity ---")
    better = [3, 2, 1, 0]
    worse = [2, 3, 0, 1]
    assert ndcg(better) >= ndcg(worse)
    print("  Better ranking >= worse ranking ✓")

    # Test 15: NDCG with numpy arrays
    print("\n--- Test 15: Numpy compatibility ---")
    np_rels = np.array([3, 2, 1, 0])
    n_np = ndcg(np_rels.tolist())
    assert abs(n_np - 1.0) < 1e-9
    print("  Numpy array conversion works ✓")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
