from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


EXPECTED_PROMPT_VERSION = "m1-shared-v1.0.0-draft"


@dataclass(frozen=True)
class M1Prompt:
    system: str
    user: str
    version: str
    sha256: str
    source_path: Path


def _extract_text_block(markdown: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*\n\s*```text\s*\n(.*?)\n```\s*$"
    matches = re.findall(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one text block under '## {heading}', found {len(matches)}")
    return matches[0]


def load_m1_prompt(path: str | Path) -> M1Prompt:
    source_path = Path(path)
    raw = source_path.read_bytes()
    markdown = raw.decode("utf-8-sig").replace("\r\n", "\n")
    version_match = re.search(r"^版本：`([^`]+)`\s*$", markdown, flags=re.MULTILINE)
    if not version_match:
        raise ValueError("Prompt version was not found")
    version = version_match.group(1)
    if version != EXPECTED_PROMPT_VERSION:
        raise ValueError(f"Unexpected prompt version: {version}")
    return M1Prompt(
        system=_extract_text_block(markdown, "System Prompt"),
        user=_extract_text_block(markdown, "User Prompt"),
        version=version,
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=source_path.resolve(),
    )
