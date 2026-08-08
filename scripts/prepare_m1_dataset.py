from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.m1.preprocess import prepare_full_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a complete Train or Val split for AskAlbum M1.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="configs/m1_local.yaml")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["m1"]
    split_settings = settings["full"][args.split]
    items = prepare_full_split(
        source_manifest_path=resolve_path(config, split_settings["source_manifest"]),
        split=args.split,
        project_root=config["project_root"],
        output_images_dir=split_settings["processed_images_dir"],
        output_manifest_path=split_settings["manifest"],
        max_pixels=int(settings["data"]["max_pixels"]),
    )
    print(json.dumps({
        "split": args.split,
        "prepared": len(items),
        "manifest": str(resolve_path(config, split_settings["manifest"])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
