from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def parse_hsv(text: str) -> Tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in text.split(","))
    if len(values) != 3 or not (0 <= values[0] <= 179):
        raise ValueError(f"Invalid HSV triplet: {text}")
    return values


def green_features(
    frame: np.ndarray,
    lower: Tuple[int, int, int],
    upper: Tuple[int, int, int],
) -> Dict[str, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    height, width = frame.shape[:2]
    return {"green_ratio": float(np.count_nonzero(mask) / max(1, height * width))}


def bbox_iou(first: List[float], second: List[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def player_geometry(
    players: List[Dict[str, Any]],
    width: int,
    height: int,
    confident_threshold: float,
) -> Dict[str, Any]:
    if not players:
        return {
            "person_count": 0,
            "confident_person_count": 0,
            "mean_person_confidence": None,
            "largest_person_height_ratio": 0.0,
            "total_person_bbox_area_ratio": 0.0,
            "player_x_span_ratio": 0.0,
            "mean_pairwise_iou": None,
        }
    confidences = np.array([float(p["conf"]) for p in players])
    heights = np.array([float(p["bbox_size"]["h"]) for p in players])
    areas = np.array([float(p["bbox_size"]["area"]) for p in players])
    centers = np.array([p["bbox_center"] for p in players])
    pairwise = [
        bbox_iou(players[i]["bbox_xyxy"], players[j]["bbox_xyxy"])
        for i in range(len(players)) for j in range(i + 1, len(players))
    ]
    return {
        "person_count": len(players),
        "confident_person_count": int(np.count_nonzero(confidences >= confident_threshold)),
        "mean_person_confidence": float(np.mean(confidences)),
        "largest_person_height_ratio": float(np.max(heights) / max(1, height)),
        "total_person_bbox_area_ratio": float(np.sum(areas) / max(1, width * height)),
        "player_x_span_ratio": float((np.max(centers[:, 0]) - np.min(centers[:, 0])) / max(1, width)),
        "mean_pairwise_iou": float(np.mean(pairwise)) if pairwise else None,
    }


def temporal_features(
    players: List[Dict[str, Any]],
    balls: List[Dict[str, Any]],
    previous_players: Dict[int, Point],
    previous_ball: Optional[Point],
    frame_diagonal: float,
    source_frame_delta: int,
) -> Tuple[Dict[str, Any], Dict[int, Point], Optional[Point]]:
    current_players: Dict[int, Point] = {}
    movement: List[float] = []
    for player in players:
        track_id = player.get("track_id")
        if track_id is None:
            continue
        center = tuple(map(float, player["bbox_center"]))
        current_players[int(track_id)] = center
        if int(track_id) in previous_players:
            old = previous_players[int(track_id)]
            movement.append(math.hypot(center[0] - old[0], center[1] - old[1]) / max(1.0, frame_diagonal))

    current_ball = tuple(map(float, balls[0]["bbox_center"])) if balls else None
    ball_move = None
    if current_ball is not None and previous_ball is not None:
        ball_move = math.hypot(current_ball[0] - previous_ball[0], current_ball[1] - previous_ball[1]) / max(1.0, frame_diagonal)
    features = {
        "source_frame_delta": source_frame_delta,
        "matched_player_count_from_previous_sample": len(movement),
        "median_player_displacement_frame_diagonal_ratio": float(np.median(movement)) if movement else None,
        "ball_displacement_frame_diagonal_ratio": ball_move,
    }
    return features, current_players, current_ball
