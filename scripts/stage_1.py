"""
stage_1.py

Stage 1: frame observation and feature extraction for soccer broadcast video.

This script does not decide whether a frame is live gameplay. It extracts:
- tracked player detections;
- ball candidates;
- per-frame visual and geometric scene features;
- simple frame-to-frame motion observations.

The resulting JSON files are intended for a later Stage 2A that classifies
contiguous gameplay windows, followed by Stage 2B possession estimation.

Supported input containers depend on the OpenCV/FFmpeg installation. Typical
.mp4 and .mkv files are supported when OpenCV was built with FFmpeg.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

Point = Tuple[float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def xyxy_to_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def center_from_xyxy(xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = map(float, xyxy)
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def bottom_center_from_xyxy(xyxy: Sequence[float]) -> List[float]:
    x1, _, x2, y2 = map(float, xyxy)
    return [(x1 + x2) / 2.0, y2]


def bbox_size(xyxy: Sequence[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {"w": width, "h": height, "area": width * height}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def parse_hsv_triplet(text: str, argument_name: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{argument_name} must contain H,S,V, for example 30,35,25")
    values = tuple(int(part) for part in parts)
    if not (0 <= values[0] <= 179 and 0 <= values[1] <= 255 and 0 <= values[2] <= 255):
        raise ValueError(f"Invalid OpenCV HSV value for {argument_name}: {values}")
    return values


def parse_ratio_box(text: str, argument_name: str) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{argument_name} must contain x1,y1,x2,y2")
    values = tuple(float(part) for part in parts)
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"Invalid normalized box for {argument_name}: {values}")
    return values


def resolve_class_ids(
    model: YOLO,
    class_id: Optional[int],
    class_name: Optional[str],
    fallback_class_id: Optional[int] = None,
) -> Optional[List[int]]:
    if class_id is not None:
        return [int(class_id)]

    if class_name:
        wanted = class_name.strip().lower()
        names = getattr(model, "names", None) or {}
        found = [int(key) for key, value in names.items() if str(value).strip().lower() == wanted]
        if found:
            return found
        print(f"[WARN] class_name='{class_name}' not found in model names.")
        if names:
            print(f"[WARN] Available names: {names}")

    if fallback_class_id is not None:
        return [int(fallback_class_id)]

    return None


def extract_tracked_detections(
    result: Any,
    keep_class_ids: Optional[List[int]],
    min_conf: float,
    include_track_id: bool = True,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    boxes = result.boxes

    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy_array = boxes.xyxy.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy_array))
    class_array = (
        boxes.cls.cpu().numpy().astype(int)
        if boxes.cls is not None
        else np.full(len(xyxy_array), -1)
    )

    id_array = None
    if include_track_id and boxes.id is not None:
        id_array = boxes.id.cpu().numpy().astype(int)

    names = getattr(result, "names", {}) or {}

    for index, xyxy_values in enumerate(xyxy_array):
        class_id = int(class_array[index])
        confidence = float(conf_array[index])

        if confidence < min_conf:
            continue
        if keep_class_ids is not None and class_id not in keep_class_ids:
            continue

        xyxy = xyxy_to_list(xyxy_values)
        item: Dict[str, Any] = {
            "det_index": int(index),
            "class_id": class_id,
            "class_name": str(names.get(class_id, class_id)),
            "bbox_xyxy": xyxy,
            "bbox_center": center_from_xyxy(xyxy),
            "bottom_center": bottom_center_from_xyxy(xyxy),
            "bbox_size": bbox_size(xyxy),
            "conf": confidence,
        }

        if include_track_id:
            item["track_id"] = int(id_array[index]) if id_array is not None else None

        detections.append(item)

    detections.sort(key=lambda detection: (detection.get("track_id") is None, -detection["conf"]))
    return detections


def extract_ball_detections(
    result: Any,
    keep_class_ids: Optional[List[int]],
    min_conf: float,
) -> List[Dict[str, Any]]:
    balls = extract_tracked_detections(
        result=result,
        keep_class_ids=keep_class_ids,
        min_conf=min_conf,
        include_track_id=False,
    )
    balls.sort(key=lambda ball: ball["conf"], reverse=True)
    return balls


def calculate_green_features(
    frame: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
) -> Tuple[Dict[str, float], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))

    # Remove isolated compression noise but preserve broad pitch regions.
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = max(1, frame.shape[0] * frame.shape[1])
    green_ratio = float(np.count_nonzero(mask) / total_pixels)

    row_coverage = np.count_nonzero(mask, axis=1) / max(1, frame.shape[1])
    column_coverage = np.count_nonzero(mask, axis=0) / max(1, frame.shape[0])

    return {
        "green_ratio": green_ratio,
        "green_row_coverage_mean": float(np.mean(row_coverage)),
        "green_row_coverage_max": float(np.max(row_coverage)) if row_coverage.size else 0.0,
        "green_column_coverage_mean": float(np.mean(column_coverage)),
        "green_column_coverage_max": float(np.max(column_coverage)) if column_coverage.size else 0.0,
    }, mask


def calculate_player_geometry_features(
    players: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    confident_threshold: float,
) -> Dict[str, Any]:
    frame_area = max(1.0, float(frame_width * frame_height))
    frame_diagonal = max(1.0, math.hypot(frame_width, frame_height))

    if not players:
        return {
            "person_count": 0,
            "confident_person_count": 0,
            "mean_person_confidence": None,
            "largest_person_height_ratio": 0.0,
            "median_person_height_ratio": 0.0,
            "largest_person_area_ratio": 0.0,
            "total_person_bbox_area_ratio": 0.0,
            "player_x_span_ratio": 0.0,
            "player_y_span_ratio": 0.0,
            "mean_nearest_player_distance_ratio": None,
            "median_nearest_player_distance_ratio": None,
            "mean_pairwise_iou": None,
        }

    confidences = np.array([float(player.get("conf", 0.0)) for player in players], dtype=np.float64)
    heights = np.array([float(player["bbox_size"]["h"]) for player in players], dtype=np.float64)
    areas = np.array([float(player["bbox_size"]["area"]) for player in players], dtype=np.float64)
    centers = np.array([player["bbox_center"] for player in players], dtype=np.float64)

    x_span = float((np.max(centers[:, 0]) - np.min(centers[:, 0])) / max(1, frame_width))
    y_span = float((np.max(centers[:, 1]) - np.min(centers[:, 1])) / max(1, frame_height))

    nearest_distances: List[float] = []
    if len(centers) >= 2:
        for index, center in enumerate(centers):
            distances = np.linalg.norm(centers - center, axis=1)
            distances[index] = np.inf
            nearest_distances.append(float(np.min(distances) / frame_diagonal))

    pairwise_ious: List[float] = []
    for first in range(len(players)):
        ax1, ay1, ax2, ay2 = map(float, players[first]["bbox_xyxy"])
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for second in range(first + 1, len(players)):
            bx1, by1, bx2, by2 = map(float, players[second]["bbox_xyxy"])
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = area_a + area_b - intersection
            pairwise_ious.append(float(intersection / union) if union > 0 else 0.0)

    return {
        "person_count": int(len(players)),
        "confident_person_count": int(np.count_nonzero(confidences >= confident_threshold)),
        "mean_person_confidence": float(np.mean(confidences)),
        "largest_person_height_ratio": float(np.max(heights) / max(1, frame_height)),
        "median_person_height_ratio": float(np.median(heights) / max(1, frame_height)),
        "largest_person_area_ratio": float(np.max(areas) / frame_area),
        "total_person_bbox_area_ratio": float(np.sum(areas) / frame_area),
        "player_x_span_ratio": clamp(x_span, 0.0, 1.0),
        "player_y_span_ratio": clamp(y_span, 0.0, 1.0),
        "mean_nearest_player_distance_ratio": (
            float(np.mean(nearest_distances)) if nearest_distances else None
        ),
        "median_nearest_player_distance_ratio": (
            float(np.median(nearest_distances)) if nearest_distances else None
        ),
        "mean_pairwise_iou": float(np.mean(pairwise_ious)) if pairwise_ious else None,
    }


def torso_roi_from_bbox(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    torso_ratio: Tuple[float, float, float, float],
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = map(float, bbox)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width < 2 or box_height < 2:
        return None

    rx1, ry1, rx2, ry2 = torso_ratio
    tx1 = int(clamp(x1 + rx1 * box_width, 0, frame_width - 1))
    ty1 = int(clamp(y1 + ry1 * box_height, 0, frame_height - 1))
    tx2 = int(clamp(x1 + rx2 * box_width, 0, frame_width - 1))
    ty2 = int(clamp(y1 + ry2 * box_height, 0, frame_height - 1))

    if tx2 <= tx1 or ty2 <= ty1:
        return None
    return tx1, ty1, tx2, ty2


def extract_shirt_color_observations(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    torso_ratio: Tuple[float, float, float, float],
    minimum_pixels: int,
) -> List[Dict[str, Any]]:
    frame_height, frame_width = frame.shape[:2]
    observations: List[Dict[str, Any]] = []

    for player in players:
        roi = torso_roi_from_bbox(
            player["bbox_xyxy"], frame_width, frame_height, torso_ratio
        )
        if roi is None:
            continue

        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        saturation = pixels[:, 1]
        value = pixels[:, 2]

        valid = (saturation >= 30) & (value >= 35) & (value <= 245)
        if int(np.count_nonzero(valid)) < minimum_pixels:
            continue

        valid_pixels = pixels[valid].astype(np.float64)
        median_hsv = np.median(valid_pixels, axis=0)
        mean_hsv = np.mean(valid_pixels, axis=0)

        observation = {
            "track_id": player.get("track_id"),
            "roi_xyxy": [x1, y1, x2, y2],
            "median_hsv": [float(value) for value in median_hsv],
            "mean_hsv": [float(value) for value in mean_hsv],
            "valid_pixel_count": int(valid_pixels.shape[0]),
        }
        observations.append(observation)
        player["shirt_color_observation"] = observation

    return observations


def circular_hue_distance(first: float, second: float) -> float:
    difference = abs(first - second)
    return min(difference, 180.0 - difference)


def cluster_shirt_colors(
    observations: List[Dict[str, Any]],
    hue_threshold: float,
    saturation_threshold: float,
    value_threshold: float,
) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []

    # Greedy clustering is deliberately simple and transparent. Stage 2A can
    # later replace it with a temporal or learned color model.
    for observation in observations:
        color = np.array(observation["median_hsv"], dtype=np.float64)
        assigned = False

        for cluster in clusters:
            center = np.array(cluster["center_hsv"], dtype=np.float64)
            if (
                circular_hue_distance(color[0], center[0]) <= hue_threshold
                and abs(color[1] - center[1]) <= saturation_threshold
                and abs(color[2] - center[2]) <= value_threshold
            ):
                cluster["members"].append(observation)
                member_colors = np.array(
                    [member["median_hsv"] for member in cluster["members"]], dtype=np.float64
                )
                # Hue averaging is approximate here; sufficient for an observational feature.
                cluster["center_hsv"] = [float(value) for value in np.mean(member_colors, axis=0)]
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "center_hsv": [float(value) for value in color],
                    "members": [observation],
                }
            )

    counts = [len(cluster["members"]) for cluster in clusters]
    serializable_clusters = [
        {
            "cluster_id": index,
            "center_hsv": cluster["center_hsv"],
            "member_count": len(cluster["members"]),
            "track_ids": [member.get("track_id") for member in cluster["members"]],
        }
        for index, cluster in enumerate(clusters)
    ]

    return {
        "shirt_color_valid_player_count": len(observations),
        "shirt_color_cluster_count": len(clusters),
        "dominant_shirt_color_cluster_fraction": (
            float(max(counts) / len(observations)) if observations else None
        ),
        "shirt_color_clusters": serializable_clusters,
    }


def calculate_field_line_features(
    frame: np.ndarray,
    green_mask: np.ndarray,
    white_saturation_max: int,
    white_value_min: int,
    line_min_length_ratio: float,
) -> Dict[str, Any]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, white_value_min], dtype=np.uint8),
        np.array([179, white_saturation_max, 255], dtype=np.uint8),
    )

    # Keep white pixels close to green regions. This suppresses much of the
    # scoreboard, crowd, and advertising-board text.
    nearby_green = cv2.dilate(green_mask, np.ones((15, 15), dtype=np.uint8), iterations=1)
    candidate_mask = cv2.bitwise_and(white_mask, nearby_green)
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )

    edges = cv2.Canny(candidate_mask, 50, 150)
    min_length = max(12, int(frame.shape[1] * line_min_length_ratio))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(20, min_length // 2),
        minLineLength=min_length,
        maxLineGap=max(8, min_length // 3),
    )

    lengths: List[float] = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, line)
            lengths.append(math.hypot(x2 - x1, y2 - y1))

    frame_diagonal = max(1.0, math.hypot(frame.shape[1], frame.shape[0]))
    total_length_ratio = float(sum(lengths) / frame_diagonal)
    score = float(1.0 - math.exp(-total_length_ratio))

    return {
        "white_near_green_ratio": float(
            np.count_nonzero(candidate_mask) / max(1, frame.shape[0] * frame.shape[1])
        ),
        "field_line_count": int(len(lengths)),
        "field_line_total_length_ratio": total_length_ratio,
        "field_line_score": score,
    }


def calculate_temporal_observations(
    players: List[Dict[str, Any]],
    balls: List[Dict[str, Any]],
    previous_player_centers: Dict[int, Point],
    previous_ball_center: Optional[Point],
    frame_diagonal: float,
) -> Tuple[Dict[str, Any], Dict[int, Point], Optional[Point]]:
    player_displacements: List[float] = []
    player_displacements_by_height: List[float] = []
    current_player_centers: Dict[int, Point] = {}

    for player in players:
        track_id = player.get("track_id")
        if track_id is None:
            continue

        center = tuple(map(float, player["bbox_center"]))
        current_player_centers[int(track_id)] = center

        previous_center = previous_player_centers.get(int(track_id))
        if previous_center is None:
            continue

        displacement = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
        player_displacements.append(displacement / max(1.0, frame_diagonal))
        player_height = max(1.0, float(player["bbox_size"]["h"]))
        player_displacements_by_height.append(displacement / player_height)

    current_ball_center: Optional[Point] = None
    ball_displacement_ratio = None
    ball_displacement_by_player_height = None

    if balls:
        current_ball_center = tuple(map(float, balls[0]["bbox_center"]))
        if previous_ball_center is not None:
            displacement = math.hypot(
                current_ball_center[0] - previous_ball_center[0],
                current_ball_center[1] - previous_ball_center[1],
            )
            ball_displacement_ratio = float(displacement / max(1.0, frame_diagonal))

            player_heights = [float(player["bbox_size"]["h"]) for player in players]
            if player_heights:
                ball_displacement_by_player_height = float(
                    displacement / max(1.0, float(np.median(player_heights)))
                )

    features = {
        "matched_player_count_from_previous_saved_frame": len(player_displacements),
        "median_player_displacement_frame_diagonal_ratio": (
            float(np.median(player_displacements)) if player_displacements else None
        ),
        "median_player_displacement_by_player_height": (
            float(np.median(player_displacements_by_height))
            if player_displacements_by_height
            else None
        ),
        "ball_displacement_frame_diagonal_ratio": ball_displacement_ratio,
        "ball_displacement_by_median_player_height": ball_displacement_by_player_height,
    }

    return features, current_player_centers, current_ball_center


def draw_debug(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    balls: List[Dict[str, Any]],
    scene_features: Dict[str, Any],
) -> np.ndarray:
    visualization = frame.copy()

    for player in players:
        x1, y1, x2, y2 = map(int, player["bbox_xyxy"])
        track_id = player.get("track_id")
        label = (
            f"Player ID {track_id} {player['conf']:.2f}"
            if track_id is not None
            else f"Player ? {player['conf']:.2f}"
        )
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            visualization,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        bottom_x, bottom_y = map(int, player["bottom_center"])
        cv2.circle(visualization, (bottom_x, bottom_y), 4, (0, 255, 255), -1)

        shirt = player.get("shirt_color_observation")
        if shirt:
            sx1, sy1, sx2, sy2 = map(int, shirt["roi_xyxy"])
            cv2.rectangle(visualization, (sx1, sy1), (sx2, sy2), (255, 255, 0), 1)

    for index, ball in enumerate(balls):
        x1, y1, x2, y2 = map(int, ball["bbox_xyxy"])
        center_x, center_y = map(int, ball["bbox_center"])
        label = f"BALL {ball['conf']:.2f}" if index == 0 else f"ball? {ball['conf']:.2f}"
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(visualization, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(
            visualization,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    panel_lines = [
        f"green={scene_features.get('green_ratio', 0.0):.3f}",
        f"people={scene_features.get('person_count', 0)} "
        f"confident={scene_features.get('confident_person_count', 0)}",
        f"largest_h={scene_features.get('largest_person_height_ratio', 0.0):.3f} "
        f"box_area={scene_features.get('total_person_bbox_area_ratio', 0.0):.3f}",
        f"x_span={scene_features.get('player_x_span_ratio', 0.0):.3f} "
        f"shirt_clusters={scene_features.get('shirt_color_cluster_count', 0)}",
        f"line_score={scene_features.get('field_line_score', 0.0):.3f}",
    ]

    panel_height = 28 * len(panel_lines) + 12
    cv2.rectangle(visualization, (0, 0), (630, panel_height), (0, 0, 0), -1)
    for line_index, text in enumerate(panel_lines):
        cv2.putText(
            visualization,
            text,
            (10, 25 + 28 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return visualization


def validate_video_path(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        print(
            f"[WARN] Extension '{video_path.suffix}' is not in the common supported list. "
            "OpenCV will still attempt to open it."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: detect players/ball and extract per-frame scene features."
    )

    parser.add_argument("--video", required=True, help="Input video path, including .mkv or .mp4.")
    parser.add_argument("--player_model", default="yolo26x.pt")
    parser.add_argument("--ball_model", default="yolo26x.pt")

    parser.add_argument("--out_dir", default="outputs/json")
    parser.add_argument("--out_debug_video", default=None)
    parser.add_argument("--output_prefix", default=None)

    parser.add_argument("--player_conf", type=float, default=0.10)
    parser.add_argument(
        "--confident_player_conf",
        type=float,
        default=0.35,
        help="Second threshold used only for confident_person_count.",
    )
    parser.add_argument("--ball_conf", type=float, default=0.05)
    parser.add_argument("--player_imgsz", type=int, default=1920)
    parser.add_argument("--ball_imgsz", type=int, default=1280)

    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker", default="botsort.yaml")

    parser.add_argument("--person_class_id", type=int, default=COCO_PERSON_CLASS)
    parser.add_argument("--person_class_name", default=None)
    parser.add_argument("--ball_class_id", type=int, default=None)
    parser.add_argument("--ball_class_name", default="sports ball")

    parser.add_argument("--save_every_n_frames", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--start_frame_number_at_one", action="store_true")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="FPS for metadata/debug output. Use <=0 to preserve native video FPS.",
    )

    parser.add_argument("--green_lower_hsv", default="25,30,25")
    parser.add_argument("--green_upper_hsv", default="95,255,255")
    parser.add_argument("--shirt_roi", default="0.25,0.18,0.75,0.62")
    parser.add_argument("--shirt_min_pixels", type=int, default=12)
    parser.add_argument("--shirt_hue_threshold", type=float, default=14.0)
    parser.add_argument("--shirt_saturation_threshold", type=float, default=70.0)
    parser.add_argument("--shirt_value_threshold", type=float, default=80.0)

    parser.add_argument("--white_saturation_max", type=int, default=55)
    parser.add_argument("--white_value_min", type=int, default=165)
    parser.add_argument("--line_min_length_ratio", type=float, default=0.025)
    parser.add_argument(
        "--disable_field_line_features",
        action="store_true",
        help="Skip white-line feature extraction for faster processing.",
    )
    parser.add_argument(
        "--disable_shirt_color_features",
        action="store_true",
        help="Skip torso color observations and clustering.",
    )

    args = parser.parse_args()

    if args.save_every_n_frames < 1:
        raise ValueError("--save_every_n_frames must be at least 1")

    video_path = Path(args.video)
    validate_video_path(video_path)
    output_prefix = args.output_prefix or video_path.stem

    out_dir = Path(args.out_dir) / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    green_lower_hsv = parse_hsv_triplet(args.green_lower_hsv, "--green_lower_hsv")
    green_upper_hsv = parse_hsv_triplet(args.green_upper_hsv, "--green_upper_hsv")
    shirt_roi = parse_ratio_box(args.shirt_roi, "--shirt_roi")

    ball_class_name = args.ball_class_name
    if ball_class_name is not None and ball_class_name.strip() == "":
        ball_class_name = None

    print("[INFO] Loading models...")
    player_model = YOLO(args.player_model)
    ball_model = YOLO(args.ball_model)

    person_keep_ids = resolve_class_ids(
        player_model,
        class_id=args.person_class_id,
        class_name=args.person_class_name,
        fallback_class_id=COCO_PERSON_CLASS,
    )
    ball_keep_ids = resolve_class_ids(
        ball_model,
        class_id=args.ball_class_id,
        class_name=ball_class_name,
        fallback_class_id=(
            COCO_SPORTS_BALL_CLASS if ball_class_name == "sports ball" else None
        ),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open: {video_path}. For .mkv input, verify that your "
            "OpenCV installation includes FFmpeg support and that the codec is available."
        )

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = float(args.fps) if args.fps > 0 else (native_fps or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions reported for {video_path}: {width}x{height}")

    metadata = {
        "stage": "stage_1_frame_observation_and_feature_extraction",
        "video": str(video_path),
        "video_extension": video_path.suffix.lower(),
        "fps": fps,
        "native_fps": native_fps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "save_every_n_frames": args.save_every_n_frames,
        "player_model": args.player_model,
        "ball_model": args.ball_model,
        "player_imgsz": args.player_imgsz,
        "ball_imgsz": args.ball_imgsz,
        "player_conf": args.player_conf,
        "confident_player_conf": args.confident_player_conf,
        "ball_conf": args.ball_conf,
        "tracker": args.tracker,
        "coordinate_system": "original_video_pixels",
        "player_filter": {
            "person_class_id": args.person_class_id,
            "person_class_name": args.person_class_name,
            "resolved_class_ids": person_keep_ids,
        },
        "ball_filter": {
            "ball_class_id": args.ball_class_id,
            "ball_class_name": ball_class_name,
            "resolved_class_ids": ball_keep_ids,
        },
        "scene_feature_settings": {
            "green_lower_hsv": list(green_lower_hsv),
            "green_upper_hsv": list(green_upper_hsv),
            "shirt_roi": list(shirt_roi),
            "shirt_min_pixels": args.shirt_min_pixels,
            "shirt_hue_threshold": args.shirt_hue_threshold,
            "shirt_saturation_threshold": args.shirt_saturation_threshold,
            "shirt_value_threshold": args.shirt_value_threshold,
            "white_saturation_max": args.white_saturation_max,
            "white_value_min": args.white_value_min,
            "line_min_length_ratio": args.line_min_length_ratio,
            "field_line_features_enabled": not args.disable_field_line_features,
            "shirt_color_features_enabled": not args.disable_shirt_color_features,
        },
        "output_prefix": output_prefix,
        "notes": [
            "Stage 1 records observations and does not classify live gameplay.",
            "Stage 2A should smooth features and classify contiguous gameplay windows.",
            "Stage 2B should reset possession state at each accepted window boundary.",
            "Temporal displacement features compare consecutive saved frames, not necessarily consecutive source frames.",
        ],
    }
    write_json(out_dir / "_metadata.json", metadata)

    writer = None
    debug_path: Optional[Path] = None
    if args.out_debug_video is None:
        args.out_debug_video = str(
            Path("outputs")
            / "debug_vid"
            / "stage_1"
            / f"{output_prefix}_stage_1_debug.mp4"
        )

    if args.out_debug_video:
        debug_path = Path(args.out_debug_video)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create debug video: {debug_path}")

    frame_idx = 0
    saved_count = 0
    previous_player_centers: Dict[int, Point] = {}
    previous_ball_center: Optional[Point] = None
    frame_diagonal = math.hypot(width, height)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        if frame_idx % args.save_every_n_frames != 0:
            frame_idx += 1
            continue

        player_results = player_model.track(
            frame,
            persist=True,
            conf=args.player_conf,
            imgsz=args.player_imgsz,
            tracker=args.tracker,
            device=args.device,
            classes=person_keep_ids,
            verbose=False,
        )
        players = extract_tracked_detections(
            result=player_results[0],
            keep_class_ids=person_keep_ids,
            min_conf=args.player_conf,
            include_track_id=True,
        )

        ball_results = ball_model.predict(
            frame,
            conf=args.ball_conf,
            imgsz=args.ball_imgsz,
            device=args.device,
            classes=ball_keep_ids,
            verbose=False,
        )
        balls = extract_ball_detections(
            result=ball_results[0],
            keep_class_ids=ball_keep_ids,
            min_conf=args.ball_conf,
        )

        green_features, green_mask = calculate_green_features(
            frame, green_lower_hsv, green_upper_hsv
        )
        geometry_features = calculate_player_geometry_features(
            players,
            frame_width=width,
            frame_height=height,
            confident_threshold=args.confident_player_conf,
        )

        if args.disable_shirt_color_features:
            shirt_features = {
                "shirt_color_valid_player_count": 0,
                "shirt_color_cluster_count": 0,
                "dominant_shirt_color_cluster_fraction": None,
                "shirt_color_clusters": [],
            }
        else:
            observations = extract_shirt_color_observations(
                frame,
                players,
                torso_ratio=shirt_roi,
                minimum_pixels=args.shirt_min_pixels,
            )
            shirt_features = cluster_shirt_colors(
                observations,
                hue_threshold=args.shirt_hue_threshold,
                saturation_threshold=args.shirt_saturation_threshold,
                value_threshold=args.shirt_value_threshold,
            )

        if args.disable_field_line_features:
            field_line_features = {
                "white_near_green_ratio": None,
                "field_line_count": None,
                "field_line_total_length_ratio": None,
                "field_line_score": None,
            }
        else:
            field_line_features = calculate_field_line_features(
                frame,
                green_mask,
                white_saturation_max=args.white_saturation_max,
                white_value_min=args.white_value_min,
                line_min_length_ratio=args.line_min_length_ratio,
            )

        temporal_features, previous_player_centers, previous_ball_center = (
            calculate_temporal_observations(
                players,
                balls,
                previous_player_centers=previous_player_centers,
                previous_ball_center=previous_ball_center,
                frame_diagonal=frame_diagonal,
            )
        )

        scene_features: Dict[str, Any] = {
            **green_features,
            **geometry_features,
            **shirt_features,
            **field_line_features,
            **temporal_features,
        }

        file_number = frame_idx + 1 if args.start_frame_number_at_one else frame_idx
        frame_record = {
            "frame_idx": frame_idx,
            "frame_file_number": file_number,
            "time_sec": float(frame_idx / fps),
            "source_frame_step": args.save_every_n_frames,
            "players": players,
            "ball_candidates": balls,
            "ball": balls[0] if balls else None,
            "scene_features": scene_features,
        }
        write_json(out_dir / f"frame_{file_number:06d}.json", frame_record)

        if writer is not None:
            writer.write(draw_debug(frame, players, balls, scene_features))

        saved_count += 1
        if saved_count % 100 == 0:
            print(
                f"[INFO] saved {saved_count} JSON files; "
                f"current source frame_idx={frame_idx}"
            )

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    metadata["frames_read"] = frame_idx
    metadata["frames_saved"] = saved_count
    write_json(out_dir / "_metadata.json", metadata)

    print(f"[DONE] Stage 1 JSON directory: {out_dir}")
    print(f"[DONE] Source frames read: {frame_idx}")
    print(f"[DONE] Frame JSON files saved: {saved_count}")
    if debug_path is not None:
        print(f"[DONE] Debug video: {debug_path}")


if __name__ == "__main__":
    main()
