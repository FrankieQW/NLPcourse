from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Iterable

from PIL import Image, ImageOps

from anima_search.data.manifest import sha256_file
from anima_search.m1.records import M1ImageItem
from anima_search.schemas import ManifestItem


PREPROCESS_VERSION = "m1-img-v1"


def _safe_filename(image_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", image_id):
        raise ValueError(f"image_id cannot be used safely as a filename: {image_id}")
    return image_id


def _resize_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    resized_width = max(1, math.floor(width * scale))
    resized_height = max(1, math.floor(height * scale))
    while resized_width * resized_height > max_pixels:
        if resized_width >= resized_height:
            resized_width -= 1
        else:
            resized_height -= 1
    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def preprocess_image(source: Path, destination: Path, max_pixels: int) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = _resize_to_pixel_budget(image, max_pixels)
        width, height = image.size
        image.save(
            destination,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
    return width, height


def load_source_manifest(path: str | Path) -> dict[str, ManifestItem]:
    records: dict[str, ManifestItem] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        item = ManifestItem.model_validate_json(line)
        if item.image_id in records:
            raise ValueError(f"Duplicate image_id in source manifest at line {line_number}: {item.image_id}")
        records[item.image_id] = item
    return records


def prepare_smoke_images(
    source_manifest_path: str | Path,
    image_ids: Iterable[str],
    project_root: str | Path,
    output_images_dir: str | Path,
    output_manifest_path: str | Path,
    max_pixels: int,
) -> list[M1ImageItem]:
    root = Path(project_root).resolve()
    source_records = load_source_manifest(source_manifest_path)
    requested = list(image_ids)
    if len(requested) != 12 or len(set(requested)) != 12:
        raise ValueError("M1 smoke requires exactly 12 unique Train image IDs")

    output_dir = Path(output_images_dir)
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_manifest = Path(output_manifest_path)
    output_manifest = output_manifest if output_manifest.is_absolute() else root / output_manifest

    prepared: list[M1ImageItem] = []
    for image_id in requested:
        if image_id not in source_records:
            raise ValueError(f"image_id is missing from the source manifest: {image_id}")
        source = source_records[image_id]
        if source.split != "Train":
            raise ValueError(f"M1 smoke may only contain Train images: {image_id}")
        if not source.valid:
            raise ValueError(f"M1 smoke image is invalid: {image_id}")
        if source.duplicate_of is not None:
            raise ValueError(f"M1 smoke image is a duplicate: {image_id} -> {source.duplicate_of}")

        source_path = root / source.relative_path
        if sha256_file(source_path) != source.sha256:
            raise ValueError(f"Source image SHA-256 does not match the source manifest: {image_id}")
        destination = output_dir / f"{_safe_filename(image_id)}.jpg"
        width, height = preprocess_image(source_path, destination, max_pixels)
        prepared.append(M1ImageItem(
            image_id=image_id,
            split="train",
            source_path=source.relative_path,
            source_sha256=source.sha256,
            processed_path=destination.relative_to(root).as_posix(),
            processed_sha256=sha256_file(destination),
            width=width,
            height=height,
            preprocess_version=PREPROCESS_VERSION,
        ))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "\n".join(item.model_dump_json() for item in prepared) + "\n",
        encoding="utf-8",
    )
    return prepared


def load_m1_manifest(path: str | Path) -> list[M1ImageItem]:
    items: list[M1ImageItem] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        item = M1ImageItem.model_validate_json(line)
        if item.image_id in seen:
            raise ValueError(f"Duplicate image_id in M1 manifest at line {line_number}: {item.image_id}")
        seen.add(item.image_id)
        items.append(item)
    return items


def load_image_ids_file(path: str | Path) -> list[str]:
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values
