"""
extract_original_video_player_ball_json.py

Stage 1 detector/tracker for the full original video.

Purpose:
- Use normal YOLO detection/tracking for players. This is more stable than pose for far broadcast players.
- Use YOLO detection for ball.
- Save EACH frame as an individual JSON file:
    outputs/json/frame_000001.json
    outputs/json/frame_000002.json
    ...
- Optionally save a debug video with player IDs and ball boxes.

Important:
- This script does NOT run pose yet.
- Later pipeline:
    1. this script: player boxes + IDs + ball position
    2. possession logic: decide possessor and possession segments
    3. crop original frames around possessing player
    4. run pose on cropped player/clip for better 17-keypoint data

Coordinates:
- All boxes and centers are in ORIGINAL video pixel coordinates.
- player_imgsz and ball_imgsz can be different. Ultralytics maps results back to original pixels.

Example:
python extract_original_video_player_ball_json.py ^
  --video "data/input.mp4" ^
  --player_model "yolo26x.pt" ^
  --ball_model "yolo26x.pt" ^
  --out_dir "outputs/json" ^
  --out_debug_video "outputs/debug_player_ball.mp4" ^
  --player_conf 0.10 ^
  --ball_conf 0.05 ^
  --player_imgsz 1920 ^
  --ball_imgsz 1280 ^
  --device 0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32


def xyxy_to_list(x) -> List[float]:
    return [float(v) for v in x]


def center_from_xyxy(xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]


def bottom_center_from_xyxy(xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [float((x1 + x2) / 2.0), float(y2)]


def bbox_size(xyxy: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = xyxy
    return {
        "w": float(max(0.0, x2 - x1)),
        "h": float(max(0.0, y2 - y1)),
        "area": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
    }


def resolve_class_ids(
    model: YOLO,
    class_id: Optional[int],
    class_name: Optional[str],
    fallback_class_id: Optional[int] = None,
) -> Optional[List[int]]:
    """
    Returns class IDs to keep.
    If class_id is given, use it.
    Else if class_name is found in model.names, use that.
    Else if fallback_class_id is given, use that.
    Else return None = keep all detections.
    """
    if class_id is not None:
        return [int(class_id)]

    if class_name is not None:
        wanted = class_name.strip().lower()
        names = getattr(model, "names", None)
        found = []
        if names:
            for k, v in names.items():
                if str(v).strip().lower() == wanted:
                    found.append(int(k))
        if found:
            return found
        print(f"[WARN] class_name='{class_name}' not found in model names.")
        if names:
            print(f"[WARN] Available names: {names}")

    if fallback_class_id is not None:
        return [int(fallback_class_id)]

    return None


def extract_tracked_detections(
    result,
    keep_class_ids: Optional[List[int]],
    min_conf: float,
    include_track_id: bool = True,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy_arr = boxes.xyxy.cpu().numpy()
    conf_arr = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy_arr))
    cls_arr = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy_arr), -1)

    ids_arr = None
    if include_track_id and boxes.id is not None:
        ids_arr = boxes.id.cpu().numpy().astype(int)

    names = getattr(result, "names", {}) or {}

    for i, xyxy_np in enumerate(xyxy_arr):
        cls_id = int(cls_arr[i])
        conf = float(conf_arr[i])

        if conf < min_conf:
            continue
        if keep_class_ids is not None and cls_id not in keep_class_ids:
            continue

        xyxy = xyxy_to_list(xyxy_np)
        item = {
            "det_index": int(i),
            "class_id": cls_id,
            "class_name": str(names.get(cls_id, cls_id)),
            "bbox_xyxy": xyxy,
            "bbox_center": center_from_xyxy(xyxy),
            "bottom_center": bottom_center_from_xyxy(xyxy),
            "bbox_size": bbox_size(xyxy),
            "conf": conf,
        }

        if include_track_id:
            item["track_id"] = int(ids_arr[i]) if ids_arr is not None else None

        detections.append(item)

    detections.sort(key=lambda d: (d.get("track_id") is None, -(d["conf"])))
    return detections


def extract_ball_detections(
    result,
    keep_class_ids: Optional[List[int]],
    min_conf: float,
) -> List[Dict[str, Any]]:
    balls = extract_tracked_detections(
        result=result,
        keep_class_ids=keep_class_ids,
        min_conf=min_conf,
        include_track_id=False,
    )
    balls.sort(key=lambda b: b["conf"], reverse=True)
    return balls


def draw_debug(frame, players: List[Dict[str, Any]], balls: List[Dict[str, Any]]):
    vis = frame.copy()

    for p in players:
        x1, y1, x2, y2 = map(int, p["bbox_xyxy"])
        tid = p.get("track_id")
        label = f"Player ID {tid} {p['conf']:.2f}" if tid is not None else f"Player ? {p['conf']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
        )

        bx, by = map(int, p["bottom_center"])
        cv2.circle(vis, (bx, by), 4, (0, 255, 255), -1)

    for j, b in enumerate(balls):
        x1, y1, x2, y2 = map(int, b["bbox_xyxy"])
        cx, cy = map(int, b["bbox_center"])
        label = f"BALL {b['conf']:.2f}" if j == 0 else f"ball? {b['conf']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    return vis


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True, help="Path to original video.")

    # Normal detector/tracker for players. This replaces pose as the primary player source.
    parser.add_argument("--player_model", default="yolo26x.pt", help="Normal YOLO detection model for players.")
    parser.add_argument("--ball_model", default="yolo26x.pt", help="YOLO detection model for ball.")

    parser.add_argument("--out_dir", default="outputs/json", help="Directory for per-frame JSON files.")
    parser.add_argument("--out_debug_video", default="outputs/debug_player_ball/debug_player_ball.mp4")

    parser.add_argument("--player_conf", type=float, default=0.10)
    parser.add_argument("--ball_conf", type=float, default=0.05)
    parser.add_argument("--player_imgsz", type=int, default=1920)
    parser.add_argument("--ball_imgsz", type=int, default=1280)

    parser.add_argument("--device", default="0", help="Example: 0, cuda:0, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")

    parser.add_argument("--person_class_id", type=int, default=COCO_PERSON_CLASS)
    parser.add_argument("--person_class_name", default=None, help="Optional class name if custom model uses a different person/player class name.")

    parser.add_argument("--ball_class_id", type=int, default=None)
    parser.add_argument("--ball_class_name", default="sports ball", help="COCO: sports ball. Custom model may use ball. Empty string keeps all classes.")

    parser.add_argument("--save_every_n_frames", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--start_frame_number_at_one", action="store_true", help="Use frame_000001.json for video frame_idx=0.")

    args = parser.parse_args()

    video_path = Path(args.video)
    video_stem = video_path.stem  # example: "match_clip_01"

    out_root = Path(args.out_dir)
    out_dir = out_root / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_class_name = args.ball_class_name
    if ball_class_name is not None and ball_class_name.strip() == "":
        ball_class_name = None

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
        fallback_class_id=COCO_SPORTS_BALL_CLASS if ball_class_name == "sports ball" else None,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "player_model": args.player_model,
        "ball_model": args.ball_model,
        "player_imgsz": args.player_imgsz,
        "ball_imgsz": args.ball_imgsz,
        "player_conf": args.player_conf,
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
        "notes": [
            "Players are from normal YOLO detection/tracking, not pose.",
            "Use bottom_center or bbox_center for initial possession logic.",
            "Run pose later on possession crops for better 17-keypoint quality.",
        ],
    }
    write_json(out_dir / "_metadata.json", metadata)

    writer = None
    if args.out_debug_video:
        debug_path = Path(args.out_debug_video)

        # If user left default debug path, rename it based on input video
        if str(debug_path) == "outputs/debug_player_ball/debug_player_ball.mp4":
            debug_path = Path("outputs/debug_player_ball") / f"{video_stem}_debug_player_ball.mp4"

        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (width, height))

    frame_idx = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        if frame_idx % args.save_every_n_frames != 0:
            frame_idx += 1
            continue

        # Normal YOLO player tracking.
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

        # Ball detection on original frame.
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

        file_number = frame_idx + 1 if args.start_frame_number_at_one else frame_idx
        out_path = out_dir / f"frame_{file_number:06d}.json"

        frame_record = {
            "frame_idx": frame_idx,
            "frame_file_number": file_number,
            "time_sec": float(frame_idx / fps),
            "players": players,
            "ball_candidates": balls,
            "ball": balls[0] if balls else None,
        }
        write_json(out_path, frame_record)

        if writer is not None:
            writer.write(draw_debug(frame, players, balls))

        saved_count += 1
        if saved_count % 100 == 0:
            print(f"[INFO] saved {saved_count} JSON files; current video frame_idx={frame_idx}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"[DONE] saved per-frame JSON files to: {out_dir}")
    print(f"[DONE] saved frames: {saved_count}")
    if args.out_debug_video:
        print(f"[DONE] saved debug video: {debug_path}")


if __name__ == "__main__":
    main()
