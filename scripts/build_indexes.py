from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.config import load_config, resolve_path
from anima_search.indexing.bm25_index import BM25Index
from anima_search.indexing.documents import annotation_to_document
from anima_search.indexing.vector_index import VectorIndex
from anima_search.schemas import ImageAnnotation


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["Train", "Val"], required=True); args = parser.parse_args()
    config = load_config(args.config); split = args.split.lower(); artifacts = resolve_path(config, config["data"]["artifacts_dir"])
    path = artifacts / "annotations" / f"{split}.{config['annotation']['prompt_version']}.jsonl"
    annotations = [ImageAnnotation.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [x.image_id for x in annotations]; documents = [annotation_to_document(x) for x in annotations]
    output = artifacts / "indexes" / split
    BM25Index(ids, documents, config["annotation"]["prompt_version"],
        {"tokenizer": "jieba", "candidate_count": config["retrieval"]["candidate_count"]}).save(output / "bm25.pkl")
    vector = VectorIndex(resolve_path(config, config["models"]["embedder"]), config["runtime"]["device"],
        config["annotation"]["prompt_version"], {"batch_size": 32, "normalize_embeddings": True,
        "similarity": "inner_product"}); vector.build(ids, documents); vector.save(output / "vector")
    (output / "annotations.json").write_text(json.dumps([x.model_dump() for x in annotations], ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(ids)} index records at {output}")


if __name__ == "__main__": main()
