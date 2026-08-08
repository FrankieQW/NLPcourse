from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


class M1ValidationError(ValueError):
    pass


def _format_schema_errors(errors: list[Any]) -> str:
    messages: list[str] = []
    for error in sorted(errors, key=lambda item: list(item.absolute_path)):
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        messages.append(f"{path or '/'}: {error.message}")
    return "; ".join(messages)


class M1Validators:
    def __init__(self, annotation_schema_path: str | Path, candidate_schema_path: str | Path) -> None:
        self.annotation_schema_path = Path(annotation_schema_path).resolve()
        self.candidate_schema_path = Path(candidate_schema_path).resolve()
        self.annotation_schema = json.loads(self.annotation_schema_path.read_text(encoding="utf-8"))
        self.candidate_schema = json.loads(self.candidate_schema_path.read_text(encoding="utf-8"))

        annotation_resource = Resource.from_contents(self.annotation_schema)
        resolved_ref = urljoin(self.candidate_schema["$id"], "./annotation_payload.schema.json")
        registry = Registry().with_resource(resolved_ref, annotation_resource)
        if "$id" in self.annotation_schema:
            registry = registry.with_resource(self.annotation_schema["$id"], annotation_resource)

        self.annotation_validator = Draft202012Validator(self.annotation_schema)
        self.candidate_validator = Draft202012Validator(self.candidate_schema, registry=registry)

    def validate_annotation(self, annotation: dict[str, Any]) -> None:
        schema_errors = list(self.annotation_validator.iter_errors(annotation))
        if schema_errors:
            raise M1ValidationError(_format_schema_errors(schema_errors))
        validate_annotation_semantics(annotation)

    def validate_candidate(self, candidate: dict[str, Any]) -> None:
        errors = list(self.candidate_validator.iter_errors(candidate))
        if errors:
            raise M1ValidationError(_format_schema_errors(errors))


def _validate_bbox(bbox: Any, path: str, errors: list[str]) -> None:
    if bbox is None:
        return
    if len(bbox) == 4 and not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        errors.append(f"{path} must satisfy x1 < x2 and y1 < y2")


def validate_annotation_semantics(annotation: dict[str, Any]) -> None:
    errors: list[str] = []
    scene = annotation["scene"]
    if scene["primary_type"] in scene["secondary_types"]:
        errors.append("/scene/secondary_types must not contain primary_type")

    entities = annotation["entities"]
    expected_entity_ids = [f"e{index}" for index in range(1, len(entities) + 1)]
    entity_ids = [entity["entity_id"] for entity in entities]
    if entity_ids != expected_entity_ids:
        errors.append(f"/entities entity_id values must be consecutive: {expected_entity_ids}")
    entity_id_set = set(entity_ids)

    for index, entity in enumerate(entities):
        prefix = f"/entities/{index}"
        _validate_bbox(entity["bbox_norm_1000"], f"{prefix}/bbox_norm_1000", errors)
        if entity["count"] is None and entity["count_exact"]:
            errors.append(f"{prefix}/count_exact must be false when count is null")
        if entity["entity_type"] != "person" and entity["attributes"]["attire_zh"]:
            errors.append(f"{prefix}/attributes/attire_zh must be empty for non-person entities")

    ocr = annotation["ocr"]
    expected_text_ids = [f"t{index}" for index in range(1, len(ocr) + 1)]
    text_ids = [item["text_id"] for item in ocr]
    if text_ids != expected_text_ids:
        errors.append(f"/ocr text_id values must be consecutive: {expected_text_ids}")
    for index, item in enumerate(ocr):
        _validate_bbox(item["bbox_norm_1000"], f"/ocr/{index}/bbox_norm_1000", errors)

    for index, relation in enumerate(annotation["relations"]):
        if relation["subject_id"] not in entity_id_set:
            errors.append(f"/relations/{index}/subject_id references a missing entity")
        if relation["object_id"] not in entity_id_set:
            errors.append(f"/relations/{index}/object_id references a missing entity")
        if relation["predicate"] == "other" and not relation["predicate_other_zh"]:
            errors.append(f"/relations/{index}/predicate_other_zh is required for predicate=other")
        if relation["predicate"] != "other" and relation["predicate_other_zh"] is not None:
            errors.append(f"/relations/{index}/predicate_other_zh must be null unless predicate=other")

    for index, entity_id in enumerate(annotation["event"]["evidence_entity_ids"]):
        if entity_id not in entity_id_set:
            errors.append(f"/event/evidence_entity_ids/{index} references a missing entity")

    if errors:
        raise M1ValidationError("; ".join(errors))
