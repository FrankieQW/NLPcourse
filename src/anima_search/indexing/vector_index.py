from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anima_search.provenance import model_directory_fingerprint


class VectorIndex:
    def __init__(self, model_path: str | Path, device: str | None = None,
                 annotation_version: str = "", build_parameters: dict | None = None) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.annotation_version = annotation_version
        self.build_parameters = build_parameters or {}
        self.model_digest = model_directory_fingerprint(self.model_path)
        self.model = None
        self.index = None
        self.image_ids: list[str] = []

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path, device=self.device)
        return self.model

    def build(self, image_ids: list[str], documents: list[str], batch_size: int = 32) -> None:
        import faiss
        vectors = self._load_model().encode(documents, batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.image_ids = image_ids

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        vector = self._load_model().encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(np.asarray(vector, dtype=np.float32), limit)
        return [(self.image_ids[int(i)], float(score)) for score, i in zip(scores[0], indices[0]) if i >= 0]

    def save(self, directory: Path) -> None:
        import faiss
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        (directory / "metadata.json").write_text(json.dumps(
            {"image_ids": self.image_ids, "model_path": self.model_path, "device": self.device,
             "annotation_version": self.annotation_version,
             "model_digest": self.model_digest,
             "build_parameters": self.build_parameters}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        import faiss
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(payload["model_path"], payload.get("device"), payload.get("annotation_version", ""),
                       payload.get("build_parameters", {}))
        instance.model_digest = payload.get("model_digest", instance.model_digest)
        instance.image_ids = payload["image_ids"]
        instance.index = faiss.read_index(str(directory / "vectors.faiss"))
        return instance
