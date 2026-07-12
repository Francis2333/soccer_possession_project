from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io import read_json


def parse_half(game_time: Any) -> Optional[int]:
    if not isinstance(game_time, str) or "-" not in game_time:
        return None
    try:
        return int(game_time.split("-", 1)[0].strip())
    except ValueError:
        return None


def infer_half(video_path: Path) -> Optional[int]:
    stem = video_path.stem.lower()
    if stem == "1" or stem.startswith("1_"):
        return 1
    if stem == "2" or stem.startswith("2_"):
        return 2
    return None


def annotation_time_sec(annotation: Dict[str, Any]) -> float:
    return float(annotation.get("position", 0)) / 1000.0


def is_active_shot(
    annotation: Dict[str, Any],
    keep_label_prefixes: Sequence[str],
) -> bool:
    replay = str(annotation.get("replay", "real-time")).strip().lower()
    if replay not in {"real-time", "live", "not replay", "none", ""}:
        return False
    label = str(annotation.get("camera_label", annotation.get("label", ""))).strip().lower()
    return any(label.startswith(prefix.lower()) for prefix in keep_label_prefixes)


def build_shots(
    annotations: Iterable[Dict[str, Any]],
    half: int,
    video_duration_sec: float,
    label_applies_to: str = "before",
) -> List[Dict[str, Any]]:
    selected = [a for a in annotations if parse_half(a.get("gameTime")) == half]
    selected.sort(key=annotation_time_sec)
    if not selected:
        raise ValueError(f"No camera annotations found for half {half}.")
    if label_applies_to not in {"before", "after"}:
        raise ValueError("label_applies_to must be 'before' or 'after'.")

    shots: List[Dict[str, Any]] = []
    for index, annotation in enumerate(selected):
        time_sec = annotation_time_sec(annotation)
        if label_applies_to == "before":
            start = annotation_time_sec(selected[index - 1]) if index > 0 else 0.0
            end = time_sec
        else:
            start = time_sec
            end = (
                annotation_time_sec(selected[index + 1])
                if index + 1 < len(selected)
                else video_duration_sec
            )
        if end <= start:
            continue
        shots.append({
            "start_sec": max(0.0, start),
            "end_sec": min(video_duration_sec, end) if video_duration_sec > 0 else end,
            "camera_label": annotation.get("label"),
            "replay": annotation.get("replay"),
            "change_type": annotation.get("change_type"),
            "visibility": annotation.get("visibility"),
            "annotation_position_ms": int(annotation.get("position", 0)),
            "annotation_index": index,
        })
    return shots


def merge_windows(
    windows: List[Dict[str, Any]],
    merge_gap_sec: float = 0.0,
    padding_sec: float = 0.0,
    video_duration_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    intervals: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    for item in sorted(windows, key=lambda x: float(x["start_sec"])):
        start = max(0.0, float(item["start_sec"]) - padding_sec)
        end = float(item["end_sec"]) + padding_sec
        if video_duration_sec > 0:
            end = min(video_duration_sec, end)
        if end > start:
            intervals.append((start, end, [item]))

    merged: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    for start, end, sources in intervals:
        if not merged or start > merged[-1][1] + merge_gap_sec:
            merged.append((start, end, sources))
        else:
            old_start, old_end, old_sources = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_sources + sources)

    return [{
        "window_id": index,
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "source_intervals": sources,
    } for index, (start, end, sources) in enumerate(merged)]


def active_windows_from_soccernet(
    labels: Dict[str, Any],
    half: int,
    video_duration_sec: float,
    keep_label_prefixes: Sequence[str],
    boundary_margin_sec: float,
    merge_gap_sec: float,
    label_applies_to: str = "before",
) -> List[Dict[str, Any]]:
    shots = build_shots(
        labels.get("annotations", []), half, video_duration_sec, label_applies_to
    )
    active: List[Dict[str, Any]] = []
    for shot in shots:
        if not is_active_shot(shot, keep_label_prefixes):
            continue
        start = float(shot["start_sec"]) + boundary_margin_sec
        end = float(shot["end_sec"]) - boundary_margin_sec
        if end > start:
            active.append({**shot, "start_sec": start, "end_sec": end})
    return merge_windows(active, merge_gap_sec, 0.0, video_duration_sec)


def load_windows_json(path: Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    raw = data if isinstance(data, list) else data.get("gameplay_windows")
    if not isinstance(raw, list):
        raise ValueError(f"No gameplay_windows list in: {path}")
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start, end = float(item["start_sec"]), float(item["end_sec"])
        if end > start:
            cleaned.append({**item, "start_sec": start, "end_sec": end})
    cleaned.sort(key=lambda item: float(item["start_sec"]))
    for window_id, item in enumerate(cleaned):
        item["window_id"] = window_id
    return cleaned
