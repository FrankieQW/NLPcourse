from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.m1.validation import M1Validators


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local M1 candidate JSONL.")
    parser.add_argument("--config", default="configs/m1_local.yaml")
    parser.add_argument("--input", help="Override the configured candidates file")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["m1"]
    validators = M1Validators(
        resolve_path(config, settings["annotation_schema"]),
        resolve_path(config, settings["candidate_schema"]),
    )
    candidate_path = resolve_path(
        config, args.input or settings["output"]["candidates"],
    )
    succeeded = failed = 0
    seen: set[str] = set()
    for line_number, line in enumerate(candidate_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        candidate = json.loads(line)
        validators.validate_candidate(candidate)
        if candidate["image_id"] in seen:
            raise ValueError(f"Duplicate image_id at line {line_number}: {candidate['image_id']}")
        seen.add(candidate["image_id"])
        if candidate["status"] == "succeeded":
            validators.validate_annotation(candidate["annotation"])
            succeeded += 1
        else:
            failed += 1
    print(json.dumps({
        "input": str(candidate_path),
        "records": len(seen),
        "succeeded": succeeded,
        "failed": failed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
