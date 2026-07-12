"""Stage 1A (label-driven): build active-gameplay windows from SoccerNet camera/replay labels."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from stage1_core.io import read_json, write_json
from stage1_core.video import export_window_clips, probe_video
from stage1_core.windows import active_windows_from_soccernet, infer_half


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--labels", required=True, help="SoccerNet Labels-cameras.json")
    parser.add_argument("--out_dir", default="outputs/stage_1A_replay")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--half", type=int, choices=[1, 2], default=None)
    parser.add_argument(
        "--keep_label_prefixes",
        default="Main camera,Main behind the goal",
        help="Comma-separated camera labels treated as active gameplay.",
    )
    parser.add_argument("--boundary_margin_sec", type=float, default=0.20)
    parser.add_argument("--merge_gap_sec", type=float, default=0.25)
    parser.add_argument(
        "--label_applies_to",
        choices=["before", "after"],
        default="before",
        help="SoccerNet boundary labels normally describe the shot ending at the timestamp.",
    )
    parser.add_argument("--write_clips", action="store_true")
    parser.add_argument("--clip_codec", default="mp4v")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path, labels_path = Path(args.video), Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file does not exist: {labels_path}")

    info = probe_video(video_path)
    half = args.half or infer_half(video_path)
    if half not in {1, 2}:
        raise ValueError("Pass --half 1 or 2 when the video name does not identify the half.")

    labels = read_json(labels_path)
    prefixes = [value.strip() for value in args.keep_label_prefixes.split(",") if value.strip()]
    windows = active_windows_from_soccernet(
        labels=labels,
        half=half,
        video_duration_sec=info.duration_sec,
        keep_label_prefixes=prefixes,
        boundary_margin_sec=args.boundary_margin_sec,
        merge_gap_sec=args.merge_gap_sec,
        label_applies_to=args.label_applies_to,
    )
    if not windows:
        raise RuntimeError("No active-gameplay windows were produced.")

    prefix = args.output_prefix or f"{video_path.stem}_active"
    output_dir = Path(args.out_dir) / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "_gameplay_windows.json"

    record: Dict[str, Any] = {
        "stage": "stage_1A_replay_label_windowing",
        "schema_version": 1,
        "video": str(video_path),
        "source_labels": str(labels_path),
        "half": half,
        "native_fps": info.fps,
        "video_duration_sec": info.duration_sec,
        "keep_label_prefixes": prefixes,
        "boundary_margin_sec": args.boundary_margin_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "label_applies_to": args.label_applies_to,
        "window_semantics": "Original-video time intervals; source frame indices remain unchanged.",
        "gameplay_windows": windows,
    }

    if args.write_clips:
        record["clips"] = export_window_clips(
            video_path, windows, output_dir / "clips", codec=args.clip_codec
        )
    write_json(windows_path, record)

    active_seconds = sum(float(window["duration_sec"]) for window in windows)
    print(f"[DONE] Active windows: {len(windows)}")
    print(f"[DONE] Active duration: {active_seconds / 60.0:.2f} min")
    print(f"[DONE] Windows JSON: {windows_path}")


if __name__ == "__main__":
    main()
