from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.m1.pipeline import annotate_local_items
from anima_search.m1.preprocess import load_m1_manifest
from anima_search.m1.prompt import load_m1_prompt
from anima_search.m1.qwen_local import QwenLocalBatchClient
from anima_search.m1.validation import M1Validators


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen M1 structured annotation.")
    parser.add_argument("--config", default="configs/m1_local.yaml")
    parser.add_argument("--model-path", help="Override the local Qwen checkpoint directory")
    parser.add_argument("--model-id", help="Override the exact model ID recorded in candidates")
    parser.add_argument("--batch-size", type=int, help="Override the configured physical batch size")
    parser.add_argument("--limit", type=int, choices=range(1, 13), metavar="1..12")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry failed records while preserving succeeded records",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["m1"]
    data, model_settings, output = settings["data"], settings["model"], settings["output"]
    prompt = load_m1_prompt(resolve_path(config, settings["prompt_file"]))
    validators = M1Validators(
        resolve_path(config, settings["annotation_schema"]),
        resolve_path(config, settings["candidate_schema"]),
    )
    items = load_m1_manifest(resolve_path(config, data["smoke_manifest"]))
    if len(items) != 12:
        raise ValueError(f"M1 smoke manifest must contain exactly 12 images, found {len(items)}")
    if args.limit:
        items = items[:args.limit]

    model_path = args.model_path or model_settings["model_path"]
    client = QwenLocalBatchClient(
        model_path=resolve_path(config, model_path),
        model_id=args.model_id or model_settings["model_id"],
        dtype=model_settings["dtype"],
        device_map=model_settings["device_map"],
        attention_implementation=model_settings["attention_implementation"],
        local_files_only=bool(model_settings["local_files_only"]),
        min_pixels=int(model_settings["min_pixels"]),
        max_pixels=int(model_settings["max_pixels"]),
    )
    try:
        summary = annotate_local_items(
            items=items,
            client=client,
            prompt=prompt,
            validators=validators,
            project_root=config["project_root"],
            output_path=output["candidates"],
            raw_dir=output["raw_dir"],
            run_summary_path=output["run_summary"],
            batch_size=(
                args.batch_size
                if args.batch_size is not None
                else int(model_settings["batch_size"])
            ),
            max_new_tokens=int(model_settings["max_new_tokens"]),
            retry_failed=args.retry_failed,
        )
    finally:
        client.unload()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
