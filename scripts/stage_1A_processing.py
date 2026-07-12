"""Stage 1A (model-driven): coarse gameplay screening when labels are unavailable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from stage1_core.detection import COCO_PERSON_CLASS, DetectorConfig, YoloDetector
from stage1_core.features import green_features, parse_hsv, player_geometry
from stage1_core.gameplay import GameplayConfig, score_gameplay, windows_from_samples
from stage1_core.io import write_json
from stage1_core.video import full_video_window, frame_step, iter_window_frames, probe_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--player_model", default="yolo26m.pt")
    parser.add_argument("--out_dir", default="outputs/stage_1A_processing")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--analysis_fps", type=float, default=2.0)
    parser.add_argument("--player_conf", type=float, default=0.20)
    parser.add_argument("--confident_player_conf", type=float, default=0.35)
    parser.add_argument("--player_imgsz", type=int, default=960)
    parser.add_argument("--player_iou", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--green_lower_hsv", default="25,30,25")
    parser.add_argument("--green_upper_hsv", default="95,255,255")
    parser.add_argument("--min_duration_sec", type=float, default=2.0)
    parser.add_argument("--bridge_gap_sec", type=float, default=3.0)
    parser.add_argument("--padding_sec", type=float, default=2.0)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    info = probe_video(video_path)
    step = frame_step(info.fps, args.analysis_fps)
    effective_fps = info.fps / step
    lower, upper = parse_hsv(args.green_lower_hsv), parse_hsv(args.green_upper_hsv)

    detector = YoloDetector(DetectorConfig(
        model_path=args.player_model,
        confidence=args.player_conf,
        image_size=args.player_imgsz,
        iou=args.player_iou,
        device=args.device,
        class_id=COCO_PERSON_CLASS,
        fallback_class_id=COCO_PERSON_CLASS,
        half=args.half,
    ))
    prefix = args.output_prefix or f"{video_path.stem}_coarse"
    output_dir = Path(args.out_dir) / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / f"{prefix}_observations.jsonl"

    samples: List[Dict[str, Any]] = []
    count = 0
    with observations_path.open("w", encoding="utf-8") as handle:
        for _, frame_idx, _, frame in iter_window_frames(
            video_path, [full_video_window(info)], step
        ):
            players = detector.infer(frame, mode="predict")
            features = {
                **green_features(frame, lower, upper),
                **player_geometry(players, info.width, info.height, args.confident_player_conf),
            }
            score, label, keep, components = score_gameplay(features, GameplayConfig())
            observation = {
                "frame_idx": frame_idx,
                "time_sec": frame_idx / info.fps,
                "players": players,
                "scene_features": {**features, "gameplay_score_components": components},
                "gameplay_score": score,
                "coarse_label": label,
                "keep_for_stage_1B": keep,
            }
            handle.write(json.dumps(observation, separators=(",", ":")) + "\n")
            samples.append({
                "frame_idx": frame_idx,
                "time_sec": frame_idx / info.fps,
                "coarse_label": label,
                "keep_for_stage_1B": keep,
            })
            count += 1
            if args.max_samples is not None and count >= args.max_samples:
                break

    windows = windows_from_samples(
        samples=samples,
        sample_period_sec=1.0 / effective_fps,
        min_duration_sec=args.min_duration_sec,
        bridge_gap_sec=args.bridge_gap_sec,
        padding_sec=args.padding_sec,
        video_duration_sec=info.duration_sec,
    )
    windows_path = output_dir / "_gameplay_windows.json"
    write_json(windows_path, {
        "stage": "stage_1A_model_driven_windowing",
        "schema_version": 1,
        "video": str(video_path),
        "observations_jsonl": str(observations_path),
        "native_fps": info.fps,
        "analysis_fps": effective_fps,
        "source_frame_step": step,
        "gameplay_windows": windows,
    })
    print(f"[DONE] Samples: {count}")
    print(f"[DONE] Windows: {len(windows)}")
    print(f"[DONE] Windows JSON: {windows_path}")


if __name__ == "__main__":
    main()
