from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .io import read_json


def indexed_records(path: Optional[Path], key: str) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        data = data["records"]
    if isinstance(data, list):
        output = {}
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                continue
            value = record.get(key, record.get("frame", index))
            output[int(value)] = record
        return output
    if isinstance(data, dict):
        output = {}
        for record_key, record in data.items():
            if isinstance(record, dict):
                try:
                    output[int(record.get(key, record_key))] = record
                except (TypeError, ValueError):
                    continue
        return output
    raise ValueError(f"Unsupported record JSON: {path}")


def nearest_record(records: Dict[int, Dict[str, Any]], index: int, max_gap: int) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    if index in records:
        return records[index]
    closest = min(records, key=lambda value: abs(value - index))
    return records[closest] if abs(closest - index) <= max_gap else None


def extract_homography(record: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for key, direction in (
        ("homography_image_to_pitch", "image_to_pitch"),
        ("image_to_pitch_homography", "image_to_pitch"),
        ("homography_pitch_to_image", "pitch_to_image"),
        ("pitch_to_image_homography", "pitch_to_image"),
    ):
        matrix = record.get(key)
        if matrix is None:
            continue
        array = np.asarray(matrix, dtype=float)
        if array.size == 9:
            return array.reshape(3, 3), direction
    return None, None


def project_point(point: List[float], matrix: np.ndarray, direction: str) -> Optional[List[float]]:
    if direction == "pitch_to_image":
        try:
            matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None
    vector = matrix @ np.array([float(point[0]), float(point[1]), 1.0])
    if abs(vector[2]) < 1e-9:
        return None
    return [float(vector[0] / vector[2]), float(vector[1] / vector[2])]


def enrich_frame(
    frame: Dict[str, Any],
    calibration: Optional[Dict[str, Any]],
    identities: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    if calibration is not None:
        frame["calibration"] = calibration
    matrix, direction = extract_homography(calibration or {})

    for player in frame.get("players") or []:
        track_id = player.get("track_id")
        if track_id is not None and int(track_id) in identities:
            player["identity"] = identities[int(track_id)]
        if matrix is not None and player.get("bottom_center"):
            player["pitch_position"] = project_point(player["bottom_center"], matrix, direction or "image_to_pitch")

    ball = frame.get("ball")
    if matrix is not None and ball and ball.get("bbox_center"):
        ball["pitch_position"] = project_point(ball["bbox_center"], matrix, direction or "image_to_pitch")
    return frame
