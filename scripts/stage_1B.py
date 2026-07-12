"""
stage_1B.py

High-rate refinement for candidate gameplay windows.

Accepted window-source formats
------------------------------
1. Stage 1A window JSON:
   {"gameplay_windows": [{"start_sec": 12.5, "end_sec": 28.0}, ...]}

2. SoccerNet Labels-cameras.json:
   {"annotations": [{"gameTime": "1 - 00:59", "position": "59752",
                      "label": "Main camera center", "replay": "real-time"}, ...]}

For SoccerNet, this script keeps real-time wide/main-camera intervals by default and
rejects replays and close-ups. The annotation "position" is interpreted as
milliseconds within the selected half-video.

Outputs
-------
<out_dir>/<output_prefix>/
    _metadata.json
    _refined_windows.json
    frame_000000.json, frame_000012.json, ...   # source-frame-indexed

The flat frame JSON layout is intentional: the existing possession stage can read
all *frame_*.json files directly from this directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
Point = Tuple[float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def validate_video_path(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        print(f"[WARN] Uncommon video extension '{video_path.suffix}'. OpenCV will still try it.")


def parse_hsv_triplet(text: str, argument_name: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{argument_name} must be H,S,V, for example 25,30,25")
    values = tuple(int(part) for part in parts)
    if not (0 <= values[0] <= 179 and 0 <= values[1] <= 255 and 0 <= values[2] <= 255):
        raise ValueError(f"Invalid HSV triplet for {argument_name}: {values}")
    return values


def resolve_class_ids(
    model: YOLO,
    class_id: Optional[int],
    class_name: Optional[str],
    fallback_class_id: Optional[int],
) -> Optional[List[int]]:
    if class_id is not None:
        return [int(class_id)]
    if class_name:
        wanted = class_name.strip().lower()
        names = getattr(model, "names", None) or {}
        found = [int(k) for k, v in names.items() if str(v).strip().lower() == wanted]
        if found:
            return found
        print(f"[WARN] class_name='{class_name}' not found in model names.")
    return [int(fallback_class_id)] if fallback_class_id is not None else None


def xyxy_to_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def bbox_size(xyxy: Sequence[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {"w": width, "h": height, "area": width * height}


def extract_detections(
    result: Any,
    keep_class_ids: Optional[List[int]],
    min_conf: float,
    include_track_id: bool,
) -> List[Dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []

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
    detections: List[Dict[str, Any]] = []
    for index, values in enumerate(xyxy_array):
        class_id = int(class_array[index])
        confidence = float(conf_array[index])
        if confidence < min_conf:
            continue
        if keep_class_ids is not None and class_id not in keep_class_ids:
            continue

        xyxy = xyxy_to_list(values)
        x1, y1, x2, y2 = xyxy
        item: Dict[str, Any] = {
            "det_index": int(index),
            "class_id": class_id,
            "class_name": str(names.get(class_id, class_id)),
            "bbox_xyxy": xyxy,
            "bbox_center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            "bottom_center": [(x1 + x2) / 2.0, y2],
            "bbox_size": bbox_size(xyxy),
            "conf": confidence,
        }
        if include_track_id:
            item["track_id"] = int(id_array[index]) if id_array is not None else None
        detections.append(item)

    detections.sort(key=lambda item: (item.get("track_id") is None, -float(item["conf"])))
    return detections


def calculate_green_features(
    frame: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
) -> Dict[str, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    height, width = frame.shape[:2]
    row_coverage = np.count_nonzero(mask, axis=1) / max(1, width)
    column_coverage = np.count_nonzero(mask, axis=0) / max(1, height)
    return {
        "green_ratio": float(np.count_nonzero(mask) / max(1, height * width)),
        "green_row_coverage_mean": float(np.mean(row_coverage)) if row_coverage.size else 0.0,
        "green_row_coverage_max": float(np.max(row_coverage)) if row_coverage.size else 0.0,
        "green_column_coverage_mean": float(np.mean(column_coverage)) if column_coverage.size else 0.0,
        "green_column_coverage_max": float(np.max(column_coverage)) if column_coverage.size else 0.0,
    }


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def calculate_player_geometry_features(
    players: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    confident_threshold: float,
) -> Dict[str, Any]:
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

    frame_area = max(1.0, float(frame_width * frame_height))
    frame_diagonal = max(1.0, math.hypot(frame_width, frame_height))
    confidences = np.array([float(p["conf"]) for p in players], dtype=np.float64)
    heights = np.array([float(p["bbox_size"]["h"]) for p in players], dtype=np.float64)
    areas = np.array([float(p["bbox_size"]["area"]) for p in players], dtype=np.float64)
    centers = np.array([p["bbox_center"] for p in players], dtype=np.float64)

    nearest: List[float] = []
    if len(centers) >= 2:
        for index, center in enumerate(centers):
            distances = np.linalg.norm(centers - center, axis=1)
            distances[index] = np.inf
            nearest.append(float(np.min(distances) / frame_diagonal))

    pairwise_ious = [
        bbox_iou(players[i]["bbox_xyxy"], players[j]["bbox_xyxy"])
        for i in range(len(players))
        for j in range(i + 1, len(players))
    ]

    return {
        "person_count": len(players),
        "confident_person_count": int(np.count_nonzero(confidences >= confident_threshold)),
        "mean_person_confidence": float(np.mean(confidences)),
        "largest_person_height_ratio": float(np.max(heights) / max(1, frame_height)),
        "median_person_height_ratio": float(np.median(heights) / max(1, frame_height)),
        "largest_person_area_ratio": float(np.max(areas) / frame_area),
        "total_person_bbox_area_ratio": float(np.sum(areas) / frame_area),
        "player_x_span_ratio": float((np.max(centers[:, 0]) - np.min(centers[:, 0])) / max(1, frame_width)),
        "player_y_span_ratio": float((np.max(centers[:, 1]) - np.min(centers[:, 1])) / max(1, frame_height)),
        "mean_nearest_player_distance_ratio": float(np.mean(nearest)) if nearest else None,
        "median_nearest_player_distance_ratio": float(np.median(nearest)) if nearest else None,
        "mean_pairwise_iou": float(np.mean(pairwise_ious)) if pairwise_ious else None,
    }


def calculate_temporal_features(
    players: List[Dict[str, Any]],
    balls: List[Dict[str, Any]],
    previous_player_centers: Dict[int, Point],
    previous_ball_center: Optional[Point],
    frame_diagonal: float,
    source_frame_delta: int,
) -> Tuple[Dict[str, Any], Dict[int, Point], Optional[Point]]:
    current_player_centers: Dict[int, Point] = {}
    player_motion: List[float] = []
    for player in players:
        track_id = player.get("track_id")
        if track_id is None:
            continue
        center = tuple(map(float, player["bbox_center"]))
        current_player_centers[int(track_id)] = center
        previous = previous_player_centers.get(int(track_id))
        if previous is not None:
            displacement = math.hypot(center[0] - previous[0], center[1] - previous[1])
            player_motion.append(displacement / max(1.0, frame_diagonal))

    current_ball_center: Optional[Point] = None
    ball_displacement_ratio = None
    if balls:
        current_ball_center = tuple(map(float, balls[0]["bbox_center"]))
        if previous_ball_center is not None:
            displacement = math.hypot(
                current_ball_center[0] - previous_ball_center[0],
                current_ball_center[1] - previous_ball_center[1],
            )
            ball_displacement_ratio = float(displacement / max(1.0, frame_diagonal))

    return (
        {
            "source_frame_delta": int(source_frame_delta),
            "matched_player_count_from_previous_sample": len(player_motion),
            "median_player_displacement_frame_diagonal_ratio": (
                float(np.median(player_motion)) if player_motion else None
            ),
            "ball_displacement_frame_diagonal_ratio": ball_displacement_ratio,
        },
        current_player_centers,
        current_ball_center,
    )


def draw_debug(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    balls: List[Dict[str, Any]],
    scene_features: Dict[str, Any],
    window_id: int,
    time_sec: float,
) -> np.ndarray:
    vis = frame.copy()
    for player in players:
        x1, y1, x2, y2 = map(int, player["bbox_xyxy"])
        track_id = player.get("track_id")
        label = f"ID {track_id}" if track_id is not None else "ID ?"
        label += f" {player['conf']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 0), 2)

    for index, ball in enumerate(balls):
        x1, y1, x2, y2 = map(int, ball["bbox_xyxy"])
        label = "BALL" if index == 0 else "ball?"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{label} {ball['conf']:.2f}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    panel = (
        f"window={window_id} t={time_sec:.2f}s people={scene_features.get('person_count', 0)} "
        f"green={scene_features.get('green_ratio', 0.0):.3f}"
    )
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 900), 38), (0, 0, 0), -1)
    cv2.putText(vis, panel, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Window input handling
# ---------------------------------------------------------------------------

def detect_windows_format(data: Any) -> str:
    if isinstance(data, list):
        if not data:
            return "stage1a"
        if isinstance(data[0], dict) and "start_sec" in data[0]:
            return "stage1a"
    if isinstance(data, dict):
        if isinstance(data.get("gameplay_windows"), list):
            return "stage1a"
        annotations = data.get("annotations")
        if isinstance(annotations, list) and annotations:
            sample = annotations[0]
            if isinstance(sample, dict) and {"position", "label"}.issubset(sample):
                return "soccernet"
    raise ValueError("Could not recognize windows JSON as Stage 1A or SoccerNet Labels-cameras format.")


def parse_game_time_half(game_time: Any) -> Optional[int]:
    if not isinstance(game_time, str) or "-" not in game_time:
        return None
    try:
        return int(game_time.split("-", 1)[0].strip())
    except ValueError:
        return None


def infer_half_from_video_name(video_path: Path) -> Optional[int]:
    stem = video_path.stem.lower()
    if stem.startswith("1_") or stem == "1":
        return 1
    if stem.startswith("2_") or stem == "2":
        return 2
    return None


def is_soccernet_keep_interval(annotation: Dict[str, Any], keep_label_prefixes: Sequence[str]) -> bool:
    replay = str(annotation.get("replay", "real-time")).strip().lower()
    if replay != "real-time":
        return False
    label = str(annotation.get("label", "")).strip().lower()
    return any(label.startswith(prefix.lower()) for prefix in keep_label_prefixes)


def load_stage1a_windows(data: Any) -> List[Dict[str, Any]]:
    raw_windows = data if isinstance(data, list) else data.get("gameplay_windows")
    if not isinstance(raw_windows, list):
        raise ValueError("Stage 1A JSON must be a list or contain gameplay_windows.")

    windows: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_windows):
        if not isinstance(item, dict) or "start_sec" not in item or "end_sec" not in item:
            continue
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])
        if end_sec <= start_sec:
            continue
        windows.append({
            **item,
            "source_format": "stage1a",
            "source_window_index": index,
            "start_sec": start_sec,
            "end_sec": end_sec,
        })
    return windows


def load_soccernet_windows(
    data: Dict[str, Any],
    half: int,
    video_duration_sec: float,
    keep_label_prefixes: Sequence[str],
    boundary_margin_sec: float,
) -> List[Dict[str, Any]]:
    annotations = [
        annotation
        for annotation in data.get("annotations", [])
        if isinstance(annotation, dict) and parse_game_time_half(annotation.get("gameTime")) == half
    ]
    annotations.sort(key=lambda item: int(item.get("position", 0)))
    if not annotations:
        raise ValueError(f"No SoccerNet camera annotations found for half {half}.")

    windows: List[Dict[str, Any]] = []
    for index, annotation in enumerate(annotations):
        if not is_soccernet_keep_interval(annotation, keep_label_prefixes):
            continue

        end_sec = float(annotation["position"]) / 1000.0 - boundary_margin_sec

        if index > 0:
            start_sec = (
                float(annotations[index - 1]["position"]) / 1000.0
                + boundary_margin_sec
            )
        else:
            start_sec = 0.0 + boundary_margin_sec

        if end_sec <= start_sec:
            continue
        windows.append({
            "source_format": "soccernet",
            "source_window_index": index,
            "start_sec": max(0.0, start_sec),
            "end_sec": max(0.0, end_sec),
            "camera_label": annotation.get("label"),
            "camera_replay": annotation.get("replay"),
            "camera_change_type": annotation.get("change_type"),
            "camera_annotation_position_ms": int(annotation["position"]),
        })
    return windows


def load_windows(
    path: Path,
    requested_format: str,
    video_path: Path,
    video_duration_sec: float,
    soccernet_half: Optional[int],
    keep_label_prefixes: Sequence[str],
    boundary_margin_sec: float,
) -> Tuple[List[Dict[str, Any]], str, Optional[int]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    detected_format = detect_windows_format(data) if requested_format == "auto" else requested_format
    if detected_format == "stage1a":
        return load_stage1a_windows(data), detected_format, None

    half = soccernet_half or infer_half_from_video_name(video_path)
    if half not in {1, 2}:
        raise ValueError(
            "SoccerNet Labels-cameras contains both halves. Pass --soccernet_half 1 or 2, "
            "or use a video filename beginning with 1_ or 2_."
        )
    return (
        load_soccernet_windows(
            data=data,
            half=half,
            video_duration_sec=video_duration_sec,
            keep_label_prefixes=keep_label_prefixes,
            boundary_margin_sec=boundary_margin_sec,
        ),
        detected_format,
        half,
    )


def merge_windows(
    windows: List[Dict[str, Any]],
    merge_gap_sec: float,
    padding_sec: float,
    video_duration_sec: float,
) -> List[Dict[str, Any]]:
    intervals: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    for window in sorted(windows, key=lambda item: float(item["start_sec"])):
        start = max(0.0, float(window["start_sec"]) - padding_sec)
        end = float(window["end_sec"]) + padding_sec
        if video_duration_sec > 0:
            end = min(video_duration_sec, end)
        if end > start:
            intervals.append((start, end, [window]))

    merged: List[Tuple[float, float, List[Dict[str, Any]]]] = []
    for start, end, sources in intervals:
        if not merged or start > merged[-1][1] + merge_gap_sec:
            merged.append((start, end, sources))
        else:
            old_start, old_end, old_sources = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_sources + sources)

    return [
        {
            "window_id": index,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "source_intervals": sources,
        }
        for index, (start, end, sources) in enumerate(merged)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1B: high-rate player tracking and ball detection inside candidate gameplay windows."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--windows_json",
        required=True,
        help="Stage 1A _gameplay_windows.json or SoccerNet Labels-cameras.json.",
    )
    parser.add_argument("--windows_format", choices=["auto", "stage1a", "soccernet"], default="auto")
    parser.add_argument("--soccernet_half", type=int, choices=[1, 2], default=None)
    parser.add_argument(
        "--soccernet_keep_label_prefixes",
        default="Main camera,Main behind the goal",
        help="Comma-separated camera-label prefixes retained for real-time SoccerNet intervals.",
    )
    parser.add_argument(
        "--soccernet_boundary_margin_sec",
        type=float,
        default=0.20,
        help="Trim this many seconds from both sides of each SoccerNet camera interval.",
    )

    parser.add_argument("--player_model", default="yolo26x.pt")
    parser.add_argument("--ball_model", default="yolo26x.pt")
    parser.add_argument("--out_dir", default="outputs/json_stage_1B")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--out_debug_video", default=None)
    parser.add_argument("--make_debug_video", action="store_true")

    parser.add_argument("--target_fps", type=float, default=10.0)
    parser.add_argument("--player_conf", type=float, default=0.10)
    parser.add_argument("--confident_player_conf", type=float, default=0.35)
    parser.add_argument("--ball_conf", type=float, default=0.05)
    parser.add_argument("--player_iou", type=float, default=0.50)
    parser.add_argument("--ball_iou", type=float, default=0.50)
    parser.add_argument("--player_imgsz", type=int, default=1280)
    parser.add_argument("--ball_imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--person_class_id", type=int, default=COCO_PERSON_CLASS)
    parser.add_argument("--person_class_name", default=None)
    parser.add_argument("--ball_class_id", type=int, default=None)
    parser.add_argument("--ball_class_name", default="sports ball")

    parser.add_argument("--window_padding_sec", type=float, default=0.0)
    parser.add_argument("--merge_gap_sec", type=float, default=0.25)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_output_frames", type=int, default=None)
    parser.add_argument("--green_lower_hsv", default="25,30,25")
    parser.add_argument("--green_upper_hsv", default="95,255,255")
    parser.add_argument("--start_frame_number_at_one", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    windows_path = Path(args.windows_json)
    validate_video_path(video_path)
    if not windows_path.exists():
        raise FileNotFoundError(f"Windows JSON does not exist: {windows_path}")
    if args.target_fps <= 0:
        raise ValueError("--target_fps must be positive")

    output_prefix = args.output_prefix or f"{video_path.stem}_stage_1B"
    out_dir = Path(args.out_dir) / output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_sec = total_frames / native_fps if total_frames > 0 else 0.0
    source_step = max(1, int(round(native_fps / args.target_fps)))
    effective_fps = native_fps / source_step
    frame_diagonal = math.hypot(width, height)

    keep_prefixes = [value.strip() for value in args.soccernet_keep_label_prefixes.split(",") if value.strip()]
    raw_windows, detected_format, resolved_half = load_windows(
        path=windows_path,
        requested_format=args.windows_format,
        video_path=video_path,
        video_duration_sec=video_duration_sec,
        soccernet_half=args.soccernet_half,
        keep_label_prefixes=keep_prefixes,
        boundary_margin_sec=args.soccernet_boundary_margin_sec,
    )
    windows = merge_windows(
        windows=raw_windows,
        merge_gap_sec=args.merge_gap_sec,
        padding_sec=args.window_padding_sec,
        video_duration_sec=video_duration_sec,
    )
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if not windows:
        cap.release()
        raise RuntimeError("No accepted gameplay windows were produced from the input JSON.")

    green_lower_hsv = parse_hsv_triplet(args.green_lower_hsv, "--green_lower_hsv")
    green_upper_hsv = parse_hsv_triplet(args.green_upper_hsv, "--green_upper_hsv")

    print(f"[INFO] Windows format: {detected_format}")
    if resolved_half is not None:
        print(f"[INFO] SoccerNet half: {resolved_half}")
    print(f"[INFO] Accepted/merged windows: {len(windows)}")

    print("[INFO] Loading ball model...")
    ball_model = YOLO(args.ball_model)
    ball_class_name = args.ball_class_name.strip() if args.ball_class_name else None
    ball_keep_ids = resolve_class_ids(
        ball_model,
        class_id=args.ball_class_id,
        class_name=ball_class_name,
        fallback_class_id=COCO_SPORTS_BALL_CLASS if ball_class_name == "sports ball" else None,
    )

    print("[INFO] Loading player model metadata...")
    player_template = YOLO(args.player_model)
    person_keep_ids = resolve_class_ids(
        player_template,
        class_id=args.person_class_id,
        class_name=args.person_class_name,
        fallback_class_id=COCO_PERSON_CLASS,
    )
    del player_template

    debug_path: Optional[Path] = None
    writer = None
    if args.make_debug_video:
        debug_path = (
            Path(args.out_debug_video)
            if args.out_debug_video
            else Path("outputs") / "debug_vid" / "stage_1B" / f"{output_prefix}_debug.mp4"
        )
        if debug_path.exists() and debug_path.is_dir():
            debug_path = debug_path / f"{output_prefix}_debug.mp4"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            effective_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create debug video: {debug_path}")

    refined_windows_path = out_dir / "_refined_windows.json"
    write_json(
        refined_windows_path,
        {
            "stage": "stage_1B_input_windows",
            "source_format": detected_format,
            "source_json": str(windows_path),
            "soccernet_half": resolved_half,
            "gameplay_windows": windows,
        },
    )

    metadata: Dict[str, Any] = {
        "stage": "stage_1B_high_rate_player_tracking_and_ball_detection",
        "video": str(video_path),
        "windows_json": str(windows_path),
        "windows_format": detected_format,
        "soccernet_half": resolved_half,
        "native_fps": native_fps,
        "requested_target_fps": args.target_fps,
        "effective_fps": effective_fps,
        "source_frame_step": source_step,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "player_model": args.player_model,
        "ball_model": args.ball_model,
        "player_conf": args.player_conf,
        "ball_conf": args.ball_conf,
        "player_iou": args.player_iou,
        "ball_iou": args.ball_iou,
        "tracker": args.tracker,
        "half_precision": bool(args.half),
        "window_count": len(windows),
        "refined_windows_file": str(refined_windows_path),
        "coordinate_system": "original_video_pixels",
        "notes": [
            "Player tracker state is reset at each merged gameplay window.",
            "Frame JSON files are flat in this directory for possession-stage compatibility.",
            "frame_idx always refers to the source/original video frame.",
            "The highest-confidence ball candidate is also stored under ball.",
        ],
    }
    write_json(out_dir / "_metadata.json", metadata)

    total_saved = 0
    processed_windows: List[Dict[str, Any]] = []
    stop_all = False

    for window in windows:
        if stop_all:
            break
        window_id = int(window["window_id"])
        start_frame = max(0, int(math.floor(float(window["start_sec"]) * native_fps)))
        end_frame = int(math.ceil(float(window["end_sec"]) * native_fps)) - 1
        if total_frames > 0:
            end_frame = min(total_frames - 1, end_frame)
        if end_frame < start_frame:
            continue

        print(
            f"[INFO] Window {window_id}: {window['start_sec']:.2f}-{window['end_sec']:.2f}s "
            f"| source frames {start_frame}-{end_frame}"
        )
        player_model = YOLO(args.player_model)
        previous_player_centers: Dict[int, Point] = {}
        previous_ball_center: Optional[Point] = None
        previous_sample_frame: Optional[int] = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        source_frame_idx = start_frame
        window_saved = 0

        while source_frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if (source_frame_idx - start_frame) % source_step != 0:
                source_frame_idx += 1
                continue

            player_results = player_model.track(
                frame,
                persist=True,
                conf=args.player_conf,
                iou=args.player_iou,
                imgsz=args.player_imgsz,
                tracker=args.tracker,
                device=args.device,
                classes=person_keep_ids,
                half=args.half,
                verbose=False,
            )
            players = extract_detections(
                player_results[0], person_keep_ids, args.player_conf, include_track_id=True
            )
            # Tracker IDs restart in every accepted window. Namespace them by
            # window so downstream possession logic cannot confuse two unrelated
            # local ID 1 tracks from different windows.
            for player in players:
                local_track_id = player.get("track_id")
                player["local_track_id"] = local_track_id
                if local_track_id is not None:
                    player["track_id"] = window_id * 1_000_000 + int(local_track_id)

            ball_results = ball_model.predict(
                frame,
                conf=args.ball_conf,
                iou=args.ball_iou,
                imgsz=args.ball_imgsz,
                device=args.device,
                classes=ball_keep_ids,
                half=args.half,
                verbose=False,
            )
            balls = extract_detections(
                ball_results[0], ball_keep_ids, args.ball_conf, include_track_id=False
            )
            balls.sort(key=lambda item: float(item["conf"]), reverse=True)

            green_features = calculate_green_features(frame, green_lower_hsv, green_upper_hsv)
            geometry_features = calculate_player_geometry_features(
                players, width, height, args.confident_player_conf
            )
            source_delta = (
                source_frame_idx - previous_sample_frame
                if previous_sample_frame is not None
                else source_step
            )
            temporal_features, previous_player_centers, previous_ball_center = calculate_temporal_features(
                players=players,
                balls=balls,
                previous_player_centers=previous_player_centers,
                previous_ball_center=previous_ball_center,
                frame_diagonal=frame_diagonal,
                source_frame_delta=source_delta,
            )
            previous_sample_frame = source_frame_idx
            scene_features = {**green_features, **geometry_features, **temporal_features}

            file_number = source_frame_idx + 1 if args.start_frame_number_at_one else source_frame_idx
            frame_record = {
                "window_id": window_id,
                "frame_idx": source_frame_idx,
                "frame_file_number": file_number,
                "time_sec": float(source_frame_idx / native_fps),
                "window_time_sec": float((source_frame_idx - start_frame) / native_fps),
                "source_frame_step": source_step,
                "analysis_fps": effective_fps,
                "players": players,
                "ball_candidates": balls,
                "ball": balls[0] if balls else None,
                "scene_features": scene_features,
            }
            write_json(out_dir / f"frame_{file_number:06d}.json", frame_record)

            if writer is not None:
                writer.write(
                    draw_debug(
                        frame, players, balls, scene_features, window_id, source_frame_idx / native_fps
                    )
                )

            total_saved += 1
            window_saved += 1
            if total_saved % 250 == 0:
                print(f"[INFO] Saved {total_saved} detailed frames")
            if args.max_output_frames is not None and total_saved >= args.max_output_frames:
                stop_all = True
                break
            source_frame_idx += 1

        processed_windows.append({
            **window,
            "start_source_frame": start_frame,
            "end_source_frame": end_frame,
            "frames_saved": window_saved,
        })
        del player_model

    cap.release()
    if writer is not None:
        writer.release()

    metadata["frames_saved"] = total_saved
    metadata["processed_windows"] = processed_windows
    write_json(out_dir / "_metadata.json", metadata)

    print(f"[DONE] Stage 1B output directory: {out_dir}")
    print(f"[DONE] Processed windows: {len(processed_windows)}")
    print(f"[DONE] Detailed frame JSON files: {total_saved}")
    print(f"[DONE] Refined windows: {refined_windows_path}")
    if debug_path is not None:
        print(f"[DONE] Debug video: {debug_path}")


if __name__ == "__main__":
    main()
