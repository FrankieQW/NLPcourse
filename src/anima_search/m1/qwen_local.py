from __future__ import annotations

import importlib.util
from pathlib import Path
import time
from typing import Any

from PIL import Image

from anima_search.m1.records import BatchGenerationResult
from anima_search.provenance import model_directory_fingerprint


class QwenLocalBatchClient:
    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attention_implementation: str = "flash_attention_2",
        local_files_only: bool = True,
        min_pixels: int = 256 * 32 * 32,
        max_pixels: int = 1280 * 32 * 32,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.dtype_name = dtype
        self.device_map = device_map
        self.requested_attention_implementation = attention_implementation
        self.effective_attention_implementation = attention_implementation
        self.local_files_only = local_files_only
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model_digest = model_directory_fingerprint(self.model_path)
        self.attempted_batch_sizes: list[int] = []
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if (
            self.requested_attention_implementation == "flash_attention_2"
            and importlib.util.find_spec("flash_attn") is None
        ):
            self.effective_attention_implementation = "sdpa"

        dtype = getattr(torch, self.dtype_name)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self.device_map,
            local_files_only=self.local_files_only,
            attn_implementation=self.effective_attention_implementation,
        ).eval()

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.clear_cuda_cache()

    @staticmethod
    def clear_cuda_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate_batch(
        self,
        images: list[Image.Image],
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> BatchGenerationResult:
        if not images:
            return BatchGenerationResult(texts=[], elapsed_seconds=0)
        self.attempted_batch_sizes.append(len(images))
        self.load()
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            for image in images
        ]
        rendered = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(
            text=rendered,
            images=rgb_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.perf_counter() - started
        input_length = inputs.input_ids.shape[1]
        trimmed = [tokens[input_length:] for tokens in generated]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        return BatchGenerationResult(
            texts=texts,
            elapsed_seconds=elapsed,
            peak_vram_bytes=peak_vram,
        )
