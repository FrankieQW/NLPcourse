from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from PIL import Image

from anima_search.data.manifest import sha256_file
from anima_search.m1.prompt import M1Prompt
from anima_search.m1.records import M1ImageItem
from anima_search.m1.validation import M1Validators


ANNOTATION_SCHEMA_VERSION = "1.2.0"


def parse_strict_json_object(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("Model output must be one JSON object")
    return value


def _short_error(error: BaseException) -> str:
    value = f"{type(error).__name__}: {error}"
    return value[:1000]


def _load_existing_candidates(path: Path, validators: M1Validators) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        record = json.loads(line)
        validators.validate_candidate(record)
        image_id = record["image_id"]
        if image_id in records:
            raise ValueError(f"Duplicate image_id in candidates file at line {line_number}: {image_id}")
        records[image_id] = record
    return records


def _failed_candidate(
    item: M1ImageItem,
    model_id: str,
    prompt: M1Prompt,
    error: str,
    raw_response_path: str | None = None,
) -> dict[str, Any]:
    return {
        "image_id": item.image_id,
        "processed_sha256": item.processed_sha256,
        "source_kind": "local",
        "model_id": model_id,
        "prompt_version": prompt.version,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "status": "failed",
        "raw_response_path": raw_response_path,
        "annotation": None,
        "error": error[:1000],
    }


def _write_candidate(handle: Any, candidate: dict[str, Any], validators: M1Validators) -> None:
    validators.validate_candidate(candidate)
    handle.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def _generate_resilient(
    items: list[M1ImageItem],
    client: Any,
    project_root: Path,
    prompt: M1Prompt,
    max_new_tokens: int,
) -> list[tuple[M1ImageItem, str | None, str | None, float, int | None]]:
    images: list[Image.Image] = []
    try:
        for item in items:
            with Image.open(project_root / item.processed_path) as image:
                images.append(image.convert("RGB"))
        result = client.generate_batch(
            images,
            system_prompt=prompt.system,
            user_prompt=prompt.user,
            max_new_tokens=max_new_tokens,
        )
        if len(result.texts) != len(items):
            raise RuntimeError(f"Batch returned {len(result.texts)} outputs for {len(items)} inputs")
        elapsed_per_item = result.elapsed_seconds / len(items)
        return [
            (item, text, None, elapsed_per_item, result.peak_vram_bytes)
            for item, text in zip(items, result.texts)
        ]
    except Exception as error:
        client.clear_cuda_cache()
        if len(items) > 1:
            midpoint = len(items) // 2
            return (
                _generate_resilient(items[:midpoint], client, project_root, prompt, max_new_tokens)
                + _generate_resilient(items[midpoint:], client, project_root, prompt, max_new_tokens)
            )
        return [(items[0], None, _short_error(error), 0.0, None)]
    finally:
        for image in images:
            image.close()


def annotate_local_items(
    items: Iterable[M1ImageItem],
    client: Any,
    prompt: M1Prompt,
    validators: M1Validators,
    project_root: str | Path,
    output_path: str | Path,
    raw_dir: str | Path,
    run_summary_path: str | Path,
    batch_size: int,
    max_new_tokens: int,
    retry_failed: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    root = Path(project_root).resolve()
    output = Path(output_path)
    output = output if output.is_absolute() else root / output
    raw_root = Path(raw_dir)
    raw_root = raw_root if raw_root.is_absolute() else root / raw_root
    summary_path = Path(run_summary_path)
    summary_path = summary_path if summary_path.is_absolute() else root / summary_path
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    requested_items = list(items)
    if any(item.split != "train" for item in requested_items):
        raise ValueError("Current M1 scope only permits Train items")
    existing = _load_existing_candidates(output, validators)
    if retry_failed:
        retained = {
            image_id: record
            for image_id, record in existing.items()
            if record["status"] == "succeeded"
        }
        if len(retained) != len(existing):
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for record in retained.values()
                ),
                encoding="utf-8",
            )
            temporary.replace(output)
            existing = retained
    pending: list[M1ImageItem] = []
    for item in requested_items:
        record = existing.get(item.image_id)
        if record is not None:
            if record["processed_sha256"] != item.processed_sha256:
                raise ValueError(f"Existing candidate hash does not match manifest: {item.image_id}")
            continue
        pending.append(item)

    started = time.perf_counter()
    succeeded = failed = 0
    peak_vram_bytes = 0
    generation_seconds = 0.0
    with output.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            valid_for_generation: list[M1ImageItem] = []
            for item in batch:
                processed_path = root / item.processed_path
                if not processed_path.exists():
                    candidate = _failed_candidate(item, client.model_id, prompt, "processed image is missing")
                    _write_candidate(handle, candidate, validators)
                    failed += 1
                elif sha256_file(processed_path) != item.processed_sha256:
                    candidate = _failed_candidate(item, client.model_id, prompt, "processed_sha256 mismatch")
                    _write_candidate(handle, candidate, validators)
                    failed += 1
                else:
                    valid_for_generation.append(item)

            if not valid_for_generation:
                continue
            outcomes = _generate_resilient(
                valid_for_generation, client, root, prompt, max_new_tokens,
            )
            for item, raw, generation_error, elapsed, peak_vram in outcomes:
                generation_seconds += elapsed
                peak_vram_bytes = max(peak_vram_bytes, peak_vram or 0)
                if generation_error is not None or raw is None:
                    candidate = _failed_candidate(
                        item, client.model_id, prompt, generation_error or "empty model response",
                    )
                    _write_candidate(handle, candidate, validators)
                    failed += 1
                    continue

                raw_path = raw_root / f"{item.image_id}.attempt-01.txt"
                raw_path.write_text(raw, encoding="utf-8")
                raw_relative = raw_path.relative_to(root).as_posix()
                try:
                    annotation = parse_strict_json_object(raw)
                    validators.validate_annotation(annotation)
                    candidate = {
                        "image_id": item.image_id,
                        "processed_sha256": item.processed_sha256,
                        "source_kind": "local",
                        "model_id": client.model_id,
                        "prompt_version": prompt.version,
                        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
                        "status": "succeeded",
                        "raw_response_path": raw_relative,
                        "annotation": annotation,
                        "error": None,
                    }
                    validators.validate_candidate(candidate)
                    succeeded += 1
                except Exception as error:
                    candidate = _failed_candidate(
                        item, client.model_id, prompt, _short_error(error), raw_relative,
                    )
                    failed += 1
                _write_candidate(handle, candidate, validators)

    summary = {
        "source_kind": "local",
        "model_id": client.model_id,
        "model_path": str(client.model_path),
        "model_digest": client.model_digest,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_schema_sha256": hashlib.sha256(
            validators.annotation_schema_path.read_bytes()
        ).hexdigest(),
        "candidate_schema_sha256": hashlib.sha256(
            validators.candidate_schema_path.read_bytes()
        ).hexdigest(),
        "configured_batch_size": batch_size,
        "attempted_batch_sizes": client.attempted_batch_sizes,
        "max_new_tokens": max_new_tokens,
        "dtype": client.dtype_name,
        "requested_attention_implementation": client.requested_attention_implementation,
        "effective_attention_implementation": client.effective_attention_implementation,
        "requested": len(requested_items),
        "skipped_existing": len(requested_items) - len(pending),
        "retry_failed": retry_failed,
        "succeeded": succeeded,
        "failed": failed,
        "generation_seconds": generation_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": peak_vram_bytes or None,
        "output_path": output.relative_to(root).as_posix(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
