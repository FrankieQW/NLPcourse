from anima_search.annotation.validation import extract_json_object
from anima_search.retrieval.fusion import reciprocal_rank_fusion
from anima_search.retrieval.search import query_to_document
from anima_search.schemas import SearchQuery


def test_extract_json_object_ignores_surrounding_text():
    assert extract_json_object('prefix {"summary": "ok"} suffix')["summary"] == "ok"


def test_rrf_rewards_multiple_branches():
    results = reciprocal_rank_fusion({"a": [("x", 1.0), ("y", 0.5)], "b": [("y", 1.0)]})
    assert results[0][0] == "y"


def test_search_query_defaults_to_empty_filters():
    assert SearchQuery(raw_text="雨夜城市").excluded_terms == []


def test_structured_query_fields_enter_retrieval_text():
    query = SearchQuery(raw_text="找照片", objects=["汽车"], scene=["城市"], colors=["冷色"])
    document = query_to_document(query)
    assert "主体:汽车" in document and "场景:城市" in document and "颜色:冷色" in document
