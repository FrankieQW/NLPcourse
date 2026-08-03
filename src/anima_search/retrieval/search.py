from anima_search.indexing.documents import annotation_to_document
from anima_search.retrieval.fusion import reciprocal_rank_fusion
from anima_search.schemas import ImageAnnotation, SearchQuery, SearchResult


def query_to_document(query: SearchQuery) -> str:
    fields = [("主体", query.objects), ("动作", query.actions), ("场景", query.scene),
              ("情绪", query.mood), ("颜色", query.colors), ("风格", query.style),
              ("必须", query.required_terms)]
    structured = " ".join(f"{name}:{' '.join(values)}" for name, values in fields if values)
    return f"{query.raw_text} {structured}".strip()


class HybridSearcher:
    def __init__(self, annotations: dict[str, ImageAnnotation], bm25: object, vector: object, rrf_k: int = 60) -> None:
        self.annotations, self.bm25, self.vector, self.rrf_k = annotations, bm25, vector, rrf_k

    def search(self, query: SearchQuery, candidate_count: int = 50, result_count: int = 30) -> list[SearchResult]:
        retrieval_query = query_to_document(query)
        rankings = {"bm25": self.bm25.search(retrieval_query, candidate_count),
                    "vector": self.vector.search(retrieval_query, candidate_count)}
        results: list[SearchResult] = []
        for image_id, score, branch_scores in reciprocal_rank_fusion(rankings, self.rrf_k):
            annotation = self.annotations.get(image_id)
            if annotation is None:
                continue
            document = annotation_to_document(annotation).lower()
            if any(term.lower() in document for term in query.excluded_terms):
                continue
            if any(term.lower() not in document for term in query.required_terms):
                continue
            results.append(SearchResult(image_id=image_id, relative_path=annotation.relative_path,
                                        fused_score=score, branch_scores=branch_scores))
            if len(results) >= result_count:
                break
        return results
