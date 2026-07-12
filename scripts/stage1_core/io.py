from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_frame_jsons(directory: Path) -> Iterator[Path]:
    yield from sorted(directory.glob("frame_*.json"))


def validate_video_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        print(f"[WARN] Uncommon video extension: {path.suffix}")
