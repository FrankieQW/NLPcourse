from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class M1ImageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1, max_length=128)
    split: Literal["train", "val"]
    source_path: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    processed_path: str
    processed_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    preprocess_version: str


class BatchGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str]
    elapsed_seconds: float = Field(ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
