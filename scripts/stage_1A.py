"""
Stage 1A: lightweight coarse gameplay screening for full soccer broadcasts.

This stage intentionally does NOT perform ball detection, shirt-colour extraction,
field-line detection, pose estimation, or temporal tracking features. It samples
near a requested analysis FPS (2 FPS by default), runs player detection, extracts
coarse scene geometry, writes one JSONL observation stream, and proposes permissive
candidate gameplay windows for Stage 1B.

Primary outputs under <out_dir>/<output_prefix>/:
    _metadata.json
    <output_prefix>_stage_1A_observations.jsonl
    _gameplay_windows.json

Debug video is disabled by default and is created only with --make_debug_video.
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
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


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
        print(
            f"[WARN] Uncommon video extension '{video_path.suffix}'. "
            "OpenCV will still try to open it."
        )


def parse_hsv_triplet(text: str, argument_name: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{argument_name} must be H,S,V, for example 25,30,25")
    values = tuple(int(part) for part in parts)
    if not (0 <= values[0] <= 179 and 0 <= values[1] <= 255 and 0 <= values[2] <= 255):
        raise ValueError(f"Invalid OpenCV HSV triplet for {argument_name}: {values}")
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
        found = [
            int(key)
            for key, value in names.items()
            if str(value).strip().lower() == wanted
        ]
        if found:
            return found
        print(f"[WARN] class_name='{class_name}' was not found in model names.")

    if fallback_class_id is not None:
        return [int(fallback_class_id)]
    return None


def xyxy_to_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def bbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(first) + bbox_area(second) - intersection
    return float(intersection / union) if union > 0 else 0.0


def bbox_containment_ratio(smaller: Sequence[float], larger: Sequence[float]) -> float:
    sx1, sy1, sx2, sy2 = map(float, smaller)
    lx1, ly1, lx2, ly2 = map(float, larger)
    ix1, iy1 = max(sx1, lx1), max(sy1, ly1)
    ix2, iy2 = min(sx2, lx2), min(sy2, ly2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = bbox_area(smaller)
    return float(intersection / area) if area > 0 else 0.0


def remove_duplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float,
    containment_threshold: float,
) -> List[Dict[str, Any]]:
    """Greedy duplicate removal after YOLO NMS, highest confidence first."""
    ordered = sorted(detections, key=lambda item: float(item["conf"]), reverse=True)
    kept: List[Dict[str, Any]] = []

    for candidate in ordered:
        candidate_box = candidate["bbox_xyxy"]
        duplicate = False
        for accepted in kept:
            accepted_box = accepted["bbox_xyxy"]
            smaller, larger = (
                (candidate_box, accepted_box)
                if bbox_area(candidate_box) <= bbox_area(accepted_box)
                else (accepted_box, candidate_box)
            )
            if (
                bbox_iou(candidate_box, accepted_box) >= iou_threshold
                or bbox_containment_ratio(smaller, larger) >= containment_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda item: (item.get("track_id") is None, -float(item["conf"])))
    return kept


def extract_player_detections(
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

        box = xyxy_to_list(values)
        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        item: Dict[str, Any] = {
            "det_index": int(index),
            "class_id": class_id,
            "class_name": str(names.get(class_id, class_id)),
            "bbox_xyxy": box,
            "bbox_center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            "bottom_center": [(x1 + x2) / 2.0, y2],
            "bbox_size": {"w": width, "h": height, "area": width * height},
            "conf": confidence,
        }
        if include_track_id:
            item["track_id"] = int(id_array[index]) if id_array is not None else None
        detections.append(item)

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

    frame_height, frame_width = frame.shape[:2]
    total_pixels = max(1, frame_height * frame_width)
    row_coverage = np.count_nonzero(mask, axis=1) / max(1, frame_width)
    column_coverage = np.count_nonzero(mask, axis=0) / max(1, frame_height)

    return {
        "green_ratio": float(np.count_nonzero(mask) / total_pixels),
        "green_row_coverage_mean": float(np.mean(row_coverage)),
        "green_row_coverage_max": float(np.max(row_coverage)) if row_coverage.size else 0.0,
        "green_column_coverage_mean": float(np.mean(column_coverage)),
        "green_column_coverage_max": float(np.max(column_coverage)) if column_coverage.size else 0.0,
    }


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
            "max_pairwise_iou": None,
        }

    frame_area = max(1.0, float(frame_width * frame_height))
    frame_diagonal = max(1.0, math.hypot(frame_width, frame_height))
    confidences = np.array([float(player["conf"]) for player in players], dtype=np.float64)
    heights = np.array([float(player["bbox_size"]["h"]) for player in players], dtype=np.float64)
    areas = np.array([float(player["bbox_size"]["area"]) for player in players], dtype=np.float64)
    centers = np.array([player["bbox_center"] for player in players], dtype=np.float64)

    nearest_distances: List[float] = []
    if len(centers) >= 2:
        for index, center in enumerate(centers):
            distances = np.linalg.norm(centers - center, axis=1)
            distances[index] = np.inf
            nearest_distances.append(float(np.min(distances) / frame_diagonal))

    pairwise_ious = [
        bbox_iou(players[first]["bbox_xyxy"], players[second]["bbox_xyxy"])
        for first in range(len(players))
        for second in range(first + 1, len(players))
    ]

    return {
        "person_count": int(len(players)),
        "confident_person_count": int(np.count_nonzero(confidences >= confident_threshold)),
        "mean_person_confidence": float(np.mean(confidences)),
        "largest_person_height_ratio": float(np.max(heights) / max(1, frame_height)),
        "median_person_height_ratio": float(np.median(heights) / max(1, frame_height)),
        "largest_person_area_ratio": float(np.max(areas) / frame_area),
        "total_person_bbox_area_ratio": float(np.sum(areas) / frame_area),
        "player_x_span_ratio": clamp(
            float((np.max(centers[:, 0]) - np.min(centers[:, 0])) / max(1, frame_width)),
            0.0,
            1.0,
        ),
        "player_y_span_ratio": clamp(
            float((np.max(centers[:, 1]) - np.min(centers[:, 1])) / max(1, frame_height)),
            0.0,
            1.0,
        ),
        "mean_nearest_player_distance_ratio": (
            float(np.mean(nearest_distances)) if nearest_distances else None
        ),
        "median_nearest_player_distance_ratio": (
            float(np.median(nearest_distances)) if nearest_distances else None
        ),
        "mean_pairwise_iou": float(np.mean(pairwise_ious)) if pairwise_ious else None,
        "max_pairwise_iou": float(np.max(pairwise_ious)) if pairwise_ious else None,
    }


def calculate_gameplay_score(
    features: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[float, str, bool, Dict[str, float], List[str]]:
    """Return score, label, keep-for-Stage-1B flag, components, and reasons."""
    green = float(features.get("green_ratio") or 0.0)
    people = int(features.get("person_count") or 0)
    confident_people = int(features.get("confident_person_count") or 0)
    largest_height = float(features.get("largest_person_height_ratio") or 0.0)
    total_area = float(features.get("total_person_bbox_area_ratio") or 0.0)
    x_span = float(features.get("player_x_span_ratio") or 0.0)

    green_component = clamp(
        (green - args.gameplay_green_floor) /
        max(1e-6, args.gameplay_green_good - args.gameplay_green_floor),
        0.0,
        1.0,
    )

    if args.gameplay_min_people <= people <= args.gameplay_max_people:
        people_component = 1.0
    elif people < args.gameplay_min_people:
        people_component = clamp(people / max(1, args.gameplay_min_people), 0.0, 1.0)
    else:
        people_component = clamp(
            1.0 - (people - args.gameplay_max_people) / max(1, args.gameplay_max_people),
            0.0,
            1.0,
        )

    confident_component = clamp(
        confident_people / max(1, args.gameplay_confident_people_good), 0.0, 1.0
    )
    size_component = clamp(
        1.0 - largest_height / max(1e-6, args.gameplay_closeup_height), 0.0, 1.0
    )
    span_component = clamp(
        (x_span - args.gameplay_min_x_span) / max(1e-6, 0.65 - args.gameplay_min_x_span),
        0.0,
        1.0,
    )
    area_component = clamp(
        1.0 - total_area / max(1e-6, args.gameplay_large_total_area), 0.0, 1.0
    )

    components = {
        "green": green_component,
        "people": people_component,
        "confident_people": confident_component,
        "player_size": size_component,
        "horizontal_span": span_component,
        "box_area": area_component,
    }
    score = float(
        0.42 * green_component
        + 0.20 * people_component
        + 0.10 * confident_component
        + 0.12 * size_component
        + 0.11 * span_component
        + 0.05 * area_component
    )

    reasons: List[str] = []
    obvious_reject = False
    if green < args.reject_green_below:
        reasons.append("very_low_green")
        obvious_reject = True
    if largest_height >= args.reject_closeup_height:
        reasons.append("extreme_closeup")
        obvious_reject = True
    if total_area >= args.reject_total_area_above and people <= args.reject_area_max_people:
        reasons.append("large_foreground_people")
        obvious_reject = True

    if obvious_reject and score < args.keep_uncertain_score:
        label = "reject"
        keep = False
    elif score >= args.accept_gameplay_score:
        label = "gameplay"
        keep = True
    elif score >= args.keep_uncertain_score:
        label = "uncertain"
        keep = True
        reasons.append("permissive_uncertain_keep")
    else:
        label = "reject"
        keep = False
        reasons.append("low_gameplay_score")

    return score, label, keep, components, reasons


def bridge_boolean_gaps(values: List[bool], max_gap_samples: int) -> List[bool]:
    bridged = list(values)
    index = 0
    while index < len(bridged):
        if bridged[index]:
            index += 1
            continue
        start = index
        while index < len(bridged) and not bridged[index]:
            index += 1
        end = index
        if start > 0 and end < len(bridged) and end - start <= max_gap_samples:
            for gap_index in range(start, end):
                bridged[gap_index] = True
    return bridged


def build_gameplay_windows(
    samples: List[Dict[str, Any]],
    sample_period_sec: float,
    min_duration_sec: float,
    bridge_gap_sec: float,
    padding_sec: float,
    video_duration_sec: float,
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    keep_flags = bridge_boolean_gaps(
        [bool(sample["keep_for_stage_1B"]) for sample in samples],
        max_gap_samples=max(0, int(round(bridge_gap_sec / max(sample_period_sec, 1e-6)))),
    )

    windows: List[Dict[str, Any]] = []
    index = 0
    while index < len(keep_flags):
        if not keep_flags[index]:
            index += 1
            continue

        start = index
        while index + 1 < len(keep_flags) and keep_flags[index + 1]:
            index += 1
        end = index

        raw_start = float(samples[start]["time_sec"])
        raw_end = float(samples[end]["time_sec"] + sample_period_sec)
        if raw_end - raw_start >= min_duration_sec:
            start_sec = max(0.0, raw_start - padding_sec)
            end_sec = raw_end + padding_sec
            if video_duration_sec > 0:
                end_sec = min(video_duration_sec, end_sec)

            window_samples = samples[start : end + 1]
            labels = [sample["coarse_label"] for sample in window_samples]
            windows.append(
                {
                    "window_id": len(windows),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "raw_start_sec": raw_start,
                    "raw_end_sec": raw_end,
                    "duration_sec": max(0.0, end_sec - start_sec),
                    "start_source_frame": int(samples[start]["frame_idx"]),
                    "end_source_frame": int(samples[end]["frame_idx"]),
                    "sample_count": len(window_samples),
                    "mean_gameplay_score": float(
                        np.mean([sample["gameplay_score"] for sample in window_samples])
                    ),
                    "min_gameplay_score": float(
                        np.min([sample["gameplay_score"] for sample in window_samples])
                    ),
                    "gameplay_sample_count": int(labels.count("gameplay")),
                    "uncertain_sample_count": int(labels.count("uncertain")),
                    "source_observations_jsonl": None,
                }
            )
        index += 1

    # Padding can make neighboring windows overlap; merge those before Stage 1B.
    merged: List[Dict[str, Any]] = []
    for window in windows:
        if not merged or window["start_sec"] > merged[-1]["end_sec"]:
            merged.append(window)
            continue

        previous = merged[-1]
        total_samples = previous["sample_count"] + window["sample_count"]
        previous["end_sec"] = max(previous["end_sec"], window["end_sec"])
        previous["raw_end_sec"] = max(previous["raw_end_sec"], window["raw_end_sec"])
        previous["duration_sec"] = previous["end_sec"] - previous["start_sec"]
        previous["end_source_frame"] = max(
            previous["end_source_frame"], window["end_source_frame"]
        )
        previous["mean_gameplay_score"] = (
            previous["mean_gameplay_score"] * previous["sample_count"]
            + window["mean_gameplay_score"] * window["sample_count"]
        ) / max(1, total_samples)
        previous["min_gameplay_score"] = min(
            previous["min_gameplay_score"], window["min_gameplay_score"]
        )
        previous["sample_count"] = total_samples
        previous["gameplay_sample_count"] += window["gameplay_sample_count"]
        previous["uncertain_sample_count"] += window["uncertain_sample_count"]

    for window_id, window in enumerate(merged):
        window["window_id"] = window_id
    return merged


def draw_debug_frame(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    observation: Dict[str, Any],
) -> np.ndarray:
    visualization = frame.copy()
    for player in players:
        x1, y1, x2, y2 = map(int, player["bbox_xyxy"])
        track_id = player.get("track_id")
        label = f"Player {player['conf']:.2f}"
        if "track_id" in player:
            label = f"ID {track_id if track_id is not None else '?'} {player['conf']:.2f}"
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            visualization,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    features = observation["scene_features"]
    lines = [
        f"frame={observation['frame_idx']} time={observation['time_sec']:.2f}s",
        f"label={observation['coarse_label']} keep={observation['keep_for_stage_1B']}",
        f"score={observation['gameplay_score']:.3f} green={features['green_ratio']:.3f}",
        f"people={features['person_count']} confident={features['confident_person_count']}",
        f"largest_h={features['largest_person_height_ratio']:.3f} x_span={features['player_x_span_ratio']:.3f}",
    ]
    panel_height = 31 * len(lines) + 8
    cv2.rectangle(visualization, (0, 0), (760, panel_height), (0, 0, 0), -1)
    for line_index, text in enumerate(lines):
        cv2.putText(
            visualization,
            text,
            (10, 27 + line_index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return visualization


def read_frame_at_index(
    cap: cv2.VideoCapture,
    target_frame_idx: int,
    current_frame_idx: int,
) -> Tuple[bool, Optional[np.ndarray], int]:
    """Sequentially decode to target. This is reliable for long-GOP MKV files."""
    frame: Optional[np.ndarray] = None
    while current_frame_idx <= target_frame_idx:
        ok, decoded = cap.read()
        if not ok:
            return False, None, current_frame_idx
        if current_frame_idx == target_frame_idx:
            frame = decoded
        current_frame_idx += 1
    return frame is not None, frame, current_frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1A: lightweight low-FPS gameplay screening."
    )
    parser.add_argument("--video", required=True, help="Input .mkv, .mp4, or other OpenCV-readable video.")
    parser.add_argument("--player_model", default="yolo26m.pt")
    parser.add_argument("--out_dir", default="outputs/stage_1A")
    parser.add_argument("--output_prefix", default=None)

    parser.add_argument("--analysis_fps", type=float, default=2.0, help="Approximate sampling rate. Default: 2 FPS.")
    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=None,
        help="Optional explicit frame step. Overrides --analysis_fps.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Stop after this many sampled frames.")
    parser.add_argument("--max_source_frames", type=int, default=None, help="Do not read source frames beyond this index.")

    parser.add_argument("--player_conf", type=float, default=0.20)
    parser.add_argument("--confident_player_conf", type=float, default=0.35)
    parser.add_argument("--player_imgsz", type=int, default=960)
    parser.add_argument("--player_iou", type=float, default=0.50)
    parser.add_argument("--player_mode", choices=["predict", "track"], default="predict")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="0", help="Examples: 0, cuda:0, cpu.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 on CUDA when possible. Disable with --no-half.",
    )
    parser.add_argument("--person_class_id", type=int, default=COCO_PERSON_CLASS)
    parser.add_argument("--person_class_name", default=None)

    parser.add_argument(
        "--remove_duplicate_players",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--duplicate_iou", type=float, default=0.65)
    parser.add_argument("--duplicate_containment", type=float, default=0.90)

    parser.add_argument("--green_lower_hsv", default="25,30,25")
    parser.add_argument("--green_upper_hsv", default="95,255,255")

    # Deliberately permissive defaults. Uncertain samples are retained for Stage 1B.
    parser.add_argument("--accept_gameplay_score", type=float, default=0.52)
    parser.add_argument("--keep_uncertain_score", type=float, default=0.34)
    parser.add_argument("--gameplay_green_floor", type=float, default=0.16)
    parser.add_argument("--gameplay_green_good", type=float, default=0.48)
    parser.add_argument("--gameplay_min_people", type=int, default=3)
    parser.add_argument("--gameplay_max_people", type=int, default=26)
    parser.add_argument("--gameplay_confident_people_good", type=int, default=6)
    parser.add_argument("--gameplay_closeup_height", type=float, default=0.42)
    parser.add_argument("--gameplay_large_total_area", type=float, default=0.48)
    parser.add_argument("--gameplay_min_x_span", type=float, default=0.16)
    parser.add_argument("--reject_green_below", type=float, default=0.06)
    parser.add_argument("--reject_closeup_height", type=float, default=0.68)
    parser.add_argument("--reject_total_area_above", type=float, default=0.62)
    parser.add_argument("--reject_area_max_people", type=int, default=5)

    parser.add_argument("--gameplay_min_duration_sec", type=float, default=2.0)
    parser.add_argument("--gameplay_bridge_gap_sec", type=float, default=3.0)
    parser.add_argument("--gameplay_padding_sec", type=float, default=2.0)

    parser.add_argument("--make_debug_video", action="store_true")
    parser.add_argument("--debug_video_path", default=None)
    args = parser.parse_args()

    if args.analysis_fps <= 0:
        raise ValueError("--analysis_fps must be greater than 0")
    if args.save_every_n_frames is not None and args.save_every_n_frames < 1:
        raise ValueError("--save_every_n_frames must be at least 1")
    if args.keep_uncertain_score > args.accept_gameplay_score:
        raise ValueError("--keep_uncertain_score must not exceed --accept_gameplay_score")

    video_path = Path(args.video)
    validate_video_path(video_path)
    output_prefix = args.output_prefix or video_path.stem
    output_dir = Path(args.out_dir) / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    observations_path = output_dir / f"{output_prefix}_stage_1A_observations.jsonl"
    windows_path = output_dir / "_gameplay_windows.json"
    metadata_path = output_dir / "_metadata.json"

    green_lower = parse_hsv_triplet(args.green_lower_hsv, "--green_lower_hsv")
    green_upper = parse_hsv_triplet(args.green_upper_hsv, "--green_upper_hsv")

    print(f"[INFO] Loading player model: {args.player_model}")
    player_model = YOLO(args.player_model)
    person_keep_ids = resolve_class_ids(
        player_model,
        class_id=args.person_class_id,
        class_name=args.person_class_name,
        fallback_class_id=COCO_PERSON_CLASS,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open {video_path}. Verify FFmpeg/codec support for the file."
        )

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_for_timestamps = native_fps if native_fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

    frame_step = (
        int(args.save_every_n_frames)
        if args.save_every_n_frames is not None
        else max(1, int(round(fps_for_timestamps / args.analysis_fps)))
    )
    effective_analysis_fps = fps_for_timestamps / frame_step
    sample_period_sec = frame_step / fps_for_timestamps
    video_duration_sec = total_frames / fps_for_timestamps if total_frames > 0 else 0.0

    device_text = str(args.device).strip().lower()
    use_half = bool(args.half and device_text not in {"cpu", "mps"} and not device_text.startswith("cpu"))
    if args.half and not use_half:
        print("[INFO] FP16 disabled because the selected device is not CUDA-like.")

    debug_path: Optional[Path] = None
    writer: Optional[cv2.VideoWriter] = None
    if args.make_debug_video:
        debug_path = (
            Path(args.debug_video_path)
            if args.debug_video_path
            else Path("outputs") / "debug_vid" / "stage_1A" / f"{output_prefix}_stage_1A_debug.mp4"
        )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            effective_analysis_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create debug video: {debug_path}")

    metadata: Dict[str, Any] = {
        "stage": "stage_1A_lightweight_coarse_gameplay_screening",
        "schema_version": 2,
        "video": str(video_path),
        "output_prefix": output_prefix,
        "native_fps": native_fps,
        "fps": fps_for_timestamps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "video_duration_sec_reported": video_duration_sec,
        "analysis_fps_requested": args.analysis_fps,
        "analysis_fps_approx": effective_analysis_fps,
        "source_frame_step": frame_step,
        "player_model": args.player_model,
        "player_mode": args.player_mode,
        "player_conf": args.player_conf,
        "confident_player_conf": args.confident_player_conf,
        "player_imgsz": args.player_imgsz,
        "player_iou": args.player_iou,
        "tracker": args.tracker if args.player_mode == "track" else None,
        "device": args.device,
        "half_requested": bool(args.half),
        "half_used": use_half,
        "coordinate_system": "original_video_pixels",
        "person_class_ids": person_keep_ids,
        "outputs": {
            "observations_jsonl": str(observations_path),
            "gameplay_windows_json": str(windows_path),
            "debug_video": str(debug_path) if debug_path else None,
        },
        "excluded_from_stage_1A": [
            "ball_detection",
            "shirt_color",
            "field_line_detection",
            "pose_estimation",
            "temporal_displacement_features",
            "per_frame_json_files",
        ],
        "stage_1B_contract": {
            "read_video_from": "video",
            "read_candidate_intervals_from": "_gameplay_windows.json -> gameplay_windows",
            "window_time_fields": ["start_sec", "end_sec"],
            "window_frame_fields": ["start_source_frame", "end_source_frame"],
            "reset_tracker_at_each_window": True,
        },
    }
    write_json(metadata_path, metadata)

    samples_for_windows: List[Dict[str, Any]] = []
    sample_count = 0
    source_frames_decoded = 0
    next_sample_frame = 0

    try:
        with observations_path.open("w", encoding="utf-8") as observations_file:
            while True:
                if args.max_samples is not None and sample_count >= args.max_samples:
                    break
                if args.max_source_frames is not None and next_sample_frame >= args.max_source_frames:
                    break
                if total_frames > 0 and next_sample_frame >= total_frames:
                    break

                ok, frame, source_frames_decoded = read_frame_at_index(
                    cap,
                    target_frame_idx=next_sample_frame,
                    current_frame_idx=source_frames_decoded,
                )
                if not ok or frame is None:
                    break

                inference_kwargs = {
                    "source": frame,
                    "conf": args.player_conf,
                    "iou": args.player_iou,
                    "imgsz": args.player_imgsz,
                    "device": args.device,
                    "classes": person_keep_ids,
                    "verbose": False,
                    "half": use_half,
                }
                if args.player_mode == "track":
                    results = player_model.track(
                        **inference_kwargs,
                        persist=True,
                        tracker=args.tracker,
                    )
                    include_track_id = True
                else:
                    results = player_model.predict(**inference_kwargs)
                    include_track_id = False

                players = extract_player_detections(
                    result=results[0],
                    keep_class_ids=person_keep_ids,
                    min_conf=args.player_conf,
                    include_track_id=include_track_id,
                )
                raw_person_count = len(players)
                if args.remove_duplicate_players:
                    players = remove_duplicate_detections(
                        players,
                        iou_threshold=args.duplicate_iou,
                        containment_threshold=args.duplicate_containment,
                    )

                green_features = calculate_green_features(frame, green_lower, green_upper)
                geometry_features = calculate_player_geometry_features(
                    players,
                    frame_width=width,
                    frame_height=height,
                    confident_threshold=args.confident_player_conf,
                )
                scene_features: Dict[str, Any] = {
                    **green_features,
                    **geometry_features,
                    "raw_person_count_before_custom_dedup": raw_person_count,
                    "duplicate_player_count_removed": raw_person_count - len(players),
                }
                score, label, keep, components, reasons = calculate_gameplay_score(
                    scene_features, args
                )
                scene_features["gameplay_score_components"] = components

                time_sec = float(next_sample_frame / fps_for_timestamps)
                observation = {
                    "frame_idx": int(next_sample_frame),
                    "source_frame_idx": int(next_sample_frame),
                    "time_sec": time_sec,
                    "source_frame_step": frame_step,
                    "analysis_sample_index": sample_count,
                    "players": players,
                    "scene_features": scene_features,
                    "gameplay_score": score,
                    "coarse_label": label,
                    "is_gameplay_candidate": keep,
                    "keep_for_stage_1B": keep,
                    "decision_reasons": reasons,
                }
                observations_file.write(json.dumps(observation, separators=(",", ":")) + "\n")

                samples_for_windows.append(
                    {
                        "frame_idx": int(next_sample_frame),
                        "time_sec": time_sec,
                        "gameplay_score": score,
                        "coarse_label": label,
                        "keep_for_stage_1B": keep,
                    }
                )

                if writer is not None:
                    writer.write(draw_debug_frame(frame, players, observation))

                sample_count += 1
                if sample_count % 100 == 0:
                    print(
                        f"[INFO] samples={sample_count} source_frame={next_sample_frame} "
                        f"time={time_sec / 60.0:.1f} min"
                    )
                next_sample_frame += frame_step
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    gameplay_windows = build_gameplay_windows(
        samples=samples_for_windows,
        sample_period_sec=sample_period_sec,
        min_duration_sec=args.gameplay_min_duration_sec,
        bridge_gap_sec=args.gameplay_bridge_gap_sec,
        padding_sec=args.gameplay_padding_sec,
        video_duration_sec=video_duration_sec,
    )
    for window in gameplay_windows:
        window["source_observations_jsonl"] = str(observations_path)

    windows_record = {
        "stage": "stage_1A_provisional_gameplay_windows",
        "schema_version": 2,
        "video": str(video_path),
        "output_prefix": output_prefix,
        "native_fps": native_fps,
        "analysis_fps_requested": args.analysis_fps,
        "analysis_fps_approx": effective_analysis_fps,
        "source_frame_step": frame_step,
        "player_mode": args.player_mode,
        "observations_jsonl": str(observations_path),
        "window_semantics": (
            "Permissive candidate intervals for Stage 1B. Gameplay and uncertain samples "
            "are retained; only likely non-gameplay regions are excluded."
        ),
        "thresholds": {
            "accept_gameplay_score": args.accept_gameplay_score,
            "keep_uncertain_score": args.keep_uncertain_score,
            "gameplay_min_duration_sec": args.gameplay_min_duration_sec,
            "gameplay_bridge_gap_sec": args.gameplay_bridge_gap_sec,
            "gameplay_padding_sec": args.gameplay_padding_sec,
            "reject_green_below": args.reject_green_below,
            "reject_closeup_height": args.reject_closeup_height,
        },
        "sample_counts": {
            "total": sample_count,
            "gameplay": sum(s["coarse_label"] == "gameplay" for s in samples_for_windows),
            "uncertain": sum(s["coarse_label"] == "uncertain" for s in samples_for_windows),
            "reject": sum(s["coarse_label"] == "reject" for s in samples_for_windows),
        },
        "gameplay_windows": gameplay_windows,
    }
    write_json(windows_path, windows_record)

    metadata["source_frames_decoded"] = source_frames_decoded
    metadata["samples_written"] = sample_count
    metadata["provisional_gameplay_window_count"] = len(gameplay_windows)
    write_json(metadata_path, metadata)

    print(f"[DONE] Output directory: {output_dir}")
    print(f"[DONE] Observations JSONL: {observations_path}")
    print(f"[DONE] Gameplay windows: {windows_path} ({len(gameplay_windows)} windows)")
    print(f"[DONE] Samples written: {sample_count}")
    print(f"[DONE] Approximate analysis FPS: {effective_analysis_fps:.3f}")
    if debug_path is not None:
        print(f"[DONE] Debug video: {debug_path}")


if __name__ == "__main__":
    main()
