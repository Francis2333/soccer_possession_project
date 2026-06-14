import argparse
import subprocess
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def run_cmd(cmd):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def collect_videos(input_dir: Path, recursive: bool = False):
    if recursive:
        videos = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
    else:
        videos = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]

    return sorted(videos)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="Directory containing original videos.")
    parser.add_argument("--recursive", action="store_true", help="Search videos recursively.")

    parser.add_argument("--stage1_script", default="scripts/extract_player_ball_json.py")
    parser.add_argument("--stage2_script", default="scripts/possession_logic_from_frame_json.py")

    parser.add_argument("--json_root", default="outputs/json")
    parser.add_argument("--possessions_dir", default="outputs/possessions")
    parser.add_argument("--debug_vid_dir", default="outputs/debug_vid")

    parser.add_argument("--player_model", default="yolo26x.pt")
    parser.add_argument("--ball_model", default="yolo26x.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--fps", type=float, default=25.0, help="Pipeline FPS passed to Stage 1 and Stage 2. Default: 25.")

    parser.add_argument("--player_conf", default="0.10")
    parser.add_argument("--ball_conf", default="0.05")
    parser.add_argument("--player_imgsz", default="1920")
    parser.add_argument("--ball_imgsz", default="1280")

    parser.add_argument("--make_stage1_debug_video", action="store_true")
    parser.add_argument("--make_debug_videos", action="store_true")
    parser.add_argument("--make_full_debug_video", action="store_true")

    parser.add_argument("--use_shirt_color", action="store_true")

    parser.add_argument("--square_crop_scale", type=float, default=1.30, help="Stage 2 active 1:1 crop side = this * max(width,height) of tight player+ball box. Default: 1.30.")
    parser.add_argument("--draw_legacy_crop_box", action="store_true", help="Ask Stage 2 debug videos to also draw the old rectangular crop for comparison.")

    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    videos = collect_videos(input_dir, recursive=args.recursive)

    if not videos:
        raise FileNotFoundError(f"No video files found in {input_dir}")

    print(f"[INFO] Found {len(videos)} videos.")

    for video_path in videos:
        prefix = video_path.stem
        json_dir = Path(args.json_root) / prefix

        print("\n" + "=" * 80)
        print(f"[VIDEO] {video_path}")
        print(f"[PREFIX] {prefix}")
        print(f"[JSON DIR] {json_dir}")
        print("=" * 80)

        if not args.skip_stage1:
            stage1_cmd = [
                "python",
                args.stage1_script,
                "--video",
                str(video_path),
                "--player_model",
                args.player_model,
                "--ball_model",
                args.ball_model,
                "--out_dir",
                args.json_root,
                "--player_conf",
                args.player_conf,
                "--ball_conf",
                args.ball_conf,
                "--player_imgsz",
                args.player_imgsz,
                "--ball_imgsz",
                args.ball_imgsz,
                "--device",
                args.device,
                "--fps",
                str(args.fps),
                "--start_frame_number_at_one",
            ]

            if args.make_stage1_debug_video:
                stage1_cmd.extend([
                    "--out_debug_video",
                    str(Path(args.debug_vid_dir) / "player_ball" / f"{prefix}_debug_player_ball.mp4"),
                ])
            else:
                # Empty string disables Stage 1 debug video.
                stage1_cmd.extend(["--out_debug_video", ""])

            run_cmd(stage1_cmd)

        if not args.skip_stage2:
            video_possession_dir = Path(args.possessions_dir) / prefix
            summary_json = video_possession_dir / f"{prefix}_possessions_summary.json"
            debug_csv = video_possession_dir / f"{prefix}_possession_debug.csv"

            stage2_cmd = [
                "python",
                args.stage2_script,
                "--frames_dir",
                str(json_dir),
                "--out_dir",
                args.possessions_dir,
                "--summary_json",
                str(summary_json),
                "--debug_csv",
                str(debug_csv),
                "--debug_vid_dir",
                args.debug_vid_dir,
                "--output_prefix",
                prefix,
                "--fps",
                str(args.fps),
                "--square_crop_scale",
                str(args.square_crop_scale),
            ]

            # Usually not necessary if Stage 1 _metadata.json has the video path,
            # but passing it explicitly is safer.
            stage2_cmd.extend(["--video", str(video_path)])

            if args.make_debug_videos:
                stage2_cmd.append("--make_debug_videos")

            if args.make_full_debug_video:
                stage2_cmd.append("--make_full_debug_video")

            if args.use_shirt_color:
                stage2_cmd.append("--use_shirt_color")

            if args.draw_legacy_crop_box:
                stage2_cmd.append("--draw_legacy_crop_box")

            run_cmd(stage2_cmd)

    print("\n[DONE] Pipeline completed.")


if __name__ == "__main__":
    main()