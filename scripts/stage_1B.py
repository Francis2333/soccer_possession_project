"""Stage 1B: player/ball extraction inside canonical gameplay windows."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from stage1_core.detection import (
    COCO_PERSON_CLASS,
    COCO_SPORTS_BALL_CLASS,
    DetectorConfig,
    YoloDetector,
    draw_detections,
)
from stage1_core.features import green_features, parse_hsv, player_geometry, temporal_features
from stage1_core.io import write_json
from stage1_core.video import full_video_window, frame_step, iter_window_frames, probe_video
from stage1_core.windows import load_windows_json, merge_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--windows_json", default=None, help="Stage 1A canonical windows; omit for full video.")
    parser.add_argument("--out_dir", default="outputs/stage_1B")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--player_model", default="yolo26x.pt")
    parser.add_argument("--ball_model", default="yolo26x.pt")
    parser.add_argument("--player_mode", choices=["predict", "track"], default="track")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--target_fps", type=float, default=0.0, help="0 means native FPS.")
    parser.add_argument("--player_conf", type=float, default=0.10)
    parser.add_argument("--ball_conf", type=float, default=0.05)
    parser.add_argument("--player_iou", type=float, default=0.70)
    parser.add_argument("--ball_iou", type=float, default=0.70)
    parser.add_argument("--player_imgsz", type=int, default=1920)
    parser.add_argument("--ball_imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--person_class_id", type=int, default=COCO_PERSON_CLASS)
    parser.add_argument("--ball_class_id", type=int, default=COCO_SPORTS_BALL_CLASS)
    parser.add_argument("--window_padding_sec", type=float, default=0.0)
    parser.add_argument("--merge_gap_sec", type=float, default=0.0)
    parser.add_argument("--include_scene_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confident_player_conf", type=float, default=0.35)
    parser.add_argument("--green_lower_hsv", default="25,30,25")
    parser.add_argument("--green_upper_hsv", default="95,255,255")
    parser.add_argument("--make_debug_video", action="store_true")
    parser.add_argument("--out_debug_video", default=None)
    parser.add_argument("--max_output_frames", type=int, default=None)
    parser.add_argument("--start_frame_number_at_one", action="store_true")
    return parser


def make_detector_configs(args: argparse.Namespace) -> Tuple[DetectorConfig, DetectorConfig]:
    player = DetectorConfig(
        model_path=args.player_model,
        confidence=args.player_conf,
        image_size=args.player_imgsz,
        iou=args.player_iou,
        device=args.device,
        class_id=args.person_class_id,
        fallback_class_id=COCO_PERSON_CLASS,
        half=args.half,
        tracker=args.tracker if args.player_mode == "track" else None,
    )
    ball = DetectorConfig(
        model_path=args.ball_model,
        confidence=args.ball_conf,
        image_size=args.ball_imgsz,
        iou=args.ball_iou,
        device=args.device,
        class_id=args.ball_class_id,
        fallback_class_id=COCO_SPORTS_BALL_CLASS,
        half=args.half,
    )
    return player, ball


def namespace_tracks(players: List[Dict[str, Any]], window_id: int) -> None:
    for player in players:
        local_id = player.get("track_id")
        player["local_track_id"] = local_id
        if local_id is not None:
            player["track_id"] = window_id * 1_000_000 + int(local_id)


def create_writer(args: argparse.Namespace, prefix: str, fps: float, size: Tuple[int, int]):
    if not args.make_debug_video:
        return None, None
    path = Path(args.out_debug_video) if args.out_debug_video else Path("outputs/debug_vid/stage_1B") / f"{prefix}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create debug video: {path}")
    return writer, path


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    info = probe_video(video_path)
    windows = load_windows_json(Path(args.windows_json)) if args.windows_json else [full_video_window(info)]
    windows = merge_windows(windows, args.merge_gap_sec, args.window_padding_sec, info.duration_sec)
    if not windows:
        raise RuntimeError("No gameplay windows available.")

    step = frame_step(info.fps, args.target_fps)
    effective_fps = info.fps / step
    prefix = args.output_prefix or f"{video_path.stem}_stage_1B"
    output_dir = Path(args.out_dir) / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    player_config, ball_config = make_detector_configs(args)
    player_detector, ball_detector = YoloDetector(player_config), YoloDetector(ball_config)
    lower, upper = parse_hsv(args.green_lower_hsv), parse_hsv(args.green_upper_hsv)
    writer, debug_path = create_writer(args, prefix, effective_fps, (info.width, info.height))

    metadata: Dict[str, Any] = {
        "stage": "stage_1B_object_extraction",
        "schema_version": 1,
        "video": str(video_path),
        "windows_json": args.windows_json,
        "native_fps": info.fps,
        "effective_fps": effective_fps,
        "source_frame_step": step,
        "player_mode": args.player_mode,
        "player_model": args.player_model,
        "ball_model": args.ball_model,
        "player_imgsz": args.player_imgsz,
        "ball_imgsz": args.ball_imgsz,
        "player_iou": args.player_iou,
        "ball_iou": args.ball_iou,
        "half_precision": args.half,
        "coordinate_system": "original_video_pixels",
        "gameplay_windows": windows,
    }
    write_json(output_dir / "_metadata.json", metadata)
    write_json(output_dir / "_refined_windows.json", {"gameplay_windows": windows})

    saved = 0
    current_window: Optional[int] = None
    previous_players: Dict[int, Tuple[float, float]] = {}
    previous_ball = None
    previous_frame: Optional[int] = None
    frame_diagonal = math.hypot(info.width, info.height)

    try:
        for window_id, frame_idx, window_start, frame in iter_window_frames(video_path, windows, step):
            if window_id != current_window:
                current_window = window_id
                previous_players, previous_ball, previous_frame = {}, None, None
                if args.player_mode == "track":
                    player_detector.reset()

            players = player_detector.infer(frame, args.player_mode)
            if args.player_mode == "track":
                namespace_tracks(players, window_id)
            balls = ball_detector.infer(frame, "predict")
            balls.sort(key=lambda item: float(item["conf"]), reverse=True)

            scene: Dict[str, Any] = {}
            if args.include_scene_features:
                scene.update(green_features(frame, lower, upper))
                scene.update(player_geometry(players, info.width, info.height, args.confident_player_conf))
                delta = frame_idx - previous_frame if previous_frame is not None else step
                temporal, previous_players, previous_ball = temporal_features(
                    players, balls, previous_players, previous_ball, frame_diagonal, delta
                )
                scene.update(temporal)
                previous_frame = frame_idx

            file_number = frame_idx + 1 if args.start_frame_number_at_one else frame_idx
            record = {
                "window_id": window_id,
                "frame_idx": frame_idx,
                "frame_file_number": file_number,
                "time_sec": frame_idx / info.fps,
                "window_time_sec": (frame_idx - window_start) / info.fps,
                "source_frame_step": step,
                "analysis_fps": effective_fps,
                "players": players,
                "ball_candidates": balls,
                "ball": balls[0] if balls else None,
                "scene_features": scene,
            }
            write_json(output_dir / f"frame_{file_number:06d}.json", record)
            if writer is not None:
                writer.write(draw_detections(frame, players, balls, f"window={window_id} frame={frame_idx}"))

            saved += 1
            if saved % 250 == 0:
                print(f"[INFO] Saved {saved} frame records")
            if args.max_output_frames is not None and saved >= args.max_output_frames:
                break
    finally:
        if writer is not None:
            writer.release()

    metadata["frames_saved"] = saved
    write_json(output_dir / "_metadata.json", metadata)
    print(f"[DONE] Frame JSON files: {saved}")
    print(f"[DONE] Output directory: {output_dir}")
    if debug_path is not None:
        print(f"[DONE] Debug video: {debug_path}")


if __name__ == "__main__":
    main()
