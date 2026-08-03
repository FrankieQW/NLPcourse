from __future__ import annotations

import json
from pathlib import Path

from anima_search.annotation.qwen_client import QwenVLClient
from anima_search.app.service import SearchService
from anima_search.config import load_config, resolve_path
from anima_search.generation.sd_generator import StableDiffusionGenerator
from anima_search.indexing.bm25_index import BM25Index
from anima_search.indexing.vector_index import VectorIndex
from anima_search.retrieval.query_parser import QueryParser
from anima_search.retrieval.search import HybridSearcher
from anima_search.runtime.model_manager import ModelManager
from anima_search.schemas import ImageAnnotation


def create_service(config_path: str = "configs/default.yaml", split: str = "val") -> SearchService:
    config = load_config(config_path); artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    index_dir = artifacts / "indexes" / split
    annotations_list = [ImageAnnotation.model_validate(item) for item in json.loads((index_dir / "annotations.json").read_text(encoding="utf-8"))]
    annotations = {item.image_id: item for item in annotations_list}
    bm25 = BM25Index.load(index_dir / "bm25.pkl"); vector = VectorIndex.load(index_dir / "vector")
    vector.device = config["runtime"]["device"]
    manager = ModelManager(
        lambda: QwenVLClient(resolve_path(config, config["models"]["qwen_vl"]),
            config["runtime"]["dtype"], config["runtime"]["device"]),
        lambda: StableDiffusionGenerator(resolve_path(config, config["models"]["stable_diffusion"]),
            config["runtime"]["dtype"], config["runtime"]["device"]))
    prompt_dir = Path(config["project_root"]) / "configs" / "prompts"
    parser = QueryParser(None, (prompt_dir / "query_parser.txt").read_text(encoding="utf-8"))
    searcher = HybridSearcher(annotations, bm25, vector, config["retrieval"]["rrf_k"])
    return SearchService(config, parser, searcher, manager, annotations,
        (prompt_dir / "reranker.txt").read_text(encoding="utf-8"),
        (prompt_dir / "content_writer.txt").read_text(encoding="utf-8"),
        (prompt_dir / "sd_prompt.txt").read_text(encoding="utf-8"))
