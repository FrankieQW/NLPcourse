from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.config import load_config, resolve_path
from anima_search.m1.preprocess import load_image_ids_file, prepare_smoke_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 12-image AskAlbum M1 Train smoke set.")
    parser.add_argument("--config", default="configs/m1_local.yaml")
    parser.add_argument("--ids-file", help="Override the configured smoke ID file")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["m1"]
    data = settings["data"]
    ids_path = resolve_path(config, args.ids_file or data["smoke_ids_file"])
    image_ids = load_image_ids_file(ids_path)
    items = prepare_smoke_images(
        source_manifest_path=resolve_path(config, data["source_manifest"]),
        image_ids=image_ids,
        project_root=config["project_root"],
        output_images_dir=data["processed_images_dir"],
        output_manifest_path=data["smoke_manifest"],
        max_pixels=int(data["max_pixels"]),
    )
    print(json.dumps({
        "prepared": len(items),
        "manifest": str(resolve_path(config, data["smoke_manifest"])),
        "image_ids": [item.image_id for item in items],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
