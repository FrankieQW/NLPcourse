from collections import defaultdict


def reciprocal_rank_fusion(rankings: dict[str, list[tuple[str, float]]], k: int = 60):
    fused: dict[str, float] = defaultdict(float)
    branches: dict[str, dict[str, float]] = defaultdict(dict)
    for name, results in rankings.items():
        for rank, (image_id, original_score) in enumerate(results, start=1):
            fused[image_id] += 1.0 / (k + rank)
            branches[image_id][name] = original_score
    return [(image_id, fused[image_id], branches[image_id])
            for image_id in sorted(fused, key=lambda key: (-fused[key], key))]
