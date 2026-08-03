from __future__ import annotations

from pathlib import Path

from PIL import Image

from anima_search.annotation.validation import extract_json_object
from anima_search.schemas import SearchResult


class VisualReranker:
    def __init__(self, client: object, prompt: str, project_root: Path,
                 rrf_weight: float = 0.35, vlm_weight: float = 0.65) -> None:
        self.client = client
        self.prompt = prompt
        self.project_root = project_root
        self.rrf_weight = rrf_weight
        self.vlm_weight = vlm_weight

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        max_rrf = max(item.fused_score for item in candidates) or 1.0
        scored: list[tuple[float, SearchResult]] = []
        for item in candidates:
            try:
                with Image.open(self.project_root / item.relative_path) as image:
                    raw = self.client.generate(image.copy(),
                        f"{self.prompt}\n用户查询：{query}\n当前 image_id：{item.image_id}", max_new_tokens=384)
                payload = extract_json_object(raw)
                score = min(100.0, max(0.0, float(payload.get("score", 0.0))))
                item.rerank_score = score
                item.evidence = [str(value) for value in payload.get("evidence", [])]
                item.mismatch = [str(value) for value in payload.get("mismatch", [])]
            except Exception as exc:
                item.mismatch = [f"视觉重排不可用：{type(exc).__name__}"]
                score = 0.0
            combined = self.rrf_weight * (item.fused_score / max_rrf) + self.vlm_weight * (score / 100.0)
            scored.append((combined, item))
        return [item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].image_id))]
