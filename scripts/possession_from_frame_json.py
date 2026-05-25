"""
possession_logic_from_frame_json.py

Stage 2 possession logic.

Input:
    soccer_possession_project/outputs/json/frame_000001.json ...
    or any directory containing per-frame JSON from the detector.

Output:
    soccer_possession_project/outputs/possessions/possession_000001.json ...
    soccer_possession_project/outputs/possessions_summary.json
    optional debug CSV

Recommended project layout:
    soccer_possession_project/
      data/
      scripts/
        possession_logic_from_frame_json.py
      outputs/
        json/
          frame_000001.json
          frame_000002.json
        possessions/

Example:
python scripts/possession_logic_from_frame_json.py ^
  --frames_dir outputs/json ^
  --out_dir outputs/possessions ^
  --summary_json outputs/possessions_summary.json ^
  --pre_frames 10 ^
  --post_frames 10

Core idea:
- Use raw detected ball if available.
- If ball missing for a short gap, estimate using previous ball velocity.
- For each frame, choose nearest player to ball using bottom_center / bbox_center.
- Add temporal momentum so owner does not switch too easily.
- Create possession segments when owner remains stable for enough frames.
"""

import argparse
import csv
import glob
import json
import math
import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[float, float]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_size(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def bbox_center(bbox: List[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_bottom_center(bbox: List[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_from_list(x: Any) -> Optional[Point]:
    if not x or len(x) < 2:
        return None
    if x[0] is None or x[1] is None:
        return None
    return float(x[0]), float(x[1])


def get_player_anchor(player: Dict[str, Any], anchor: str) -> Optional[Point]:
    """
    Anchor options:
    - bottom_center: best for ball near feet
    - bbox_center: more stable for far views
    - hip_midpoint: if available from pose
    """
    if anchor == "bottom_center":
        p = point_from_list(player.get("bottom_center"))
        if p is not None:
            return p
        if player.get("bbox_xyxy"):
            return bbox_bottom_center(player["bbox_xyxy"])

    if anchor == "hip_midpoint":
        p = point_from_list(player.get("hip_midpoint"))
        if p is not None:
            return p

    p = point_from_list(player.get("bbox_center"))
    if p is not None:
        return p

    if player.get("bbox_xyxy"):
        return bbox_center(player["bbox_xyxy"])

    return None


def get_ball_center(frame: Dict[str, Any]) -> Optional[Point]:
    ball = frame.get("ball")
    if ball:
        p = point_from_list(ball.get("center") or ball.get("bbox_center"))
        if p is not None:
            return p

    # Defensive fallback: use best candidate if detector file only has ball_candidates.
    cands = frame.get("ball_candidates") or []
    if cands:
        best = max(cands, key=lambda b: float(b.get("conf", 0.0)))
        p = point_from_list(best.get("center") or best.get("bbox_center"))
        if p is not None:
            return p

    return None


def get_track_id(player: Dict[str, Any]) -> Optional[int]:
    tid = player.get("track_id")
    if tid is None:
        return None
    try:
        return int(tid)
    except Exception:
        return None


def expand_bbox(
    bbox: List[float],
    pad_ratio: float,
    width: Optional[int],
    height: Optional[int],
) -> List[float]:
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = bbox_size([x1, y1, x2, y2])
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    nx1 = x1 - pad_x
    ny1 = y1 - pad_y
    nx2 = x2 + pad_x
    ny2 = y2 + pad_y

    if width is not None:
        nx1 = clamp(nx1, 0, width - 1)
        nx2 = clamp(nx2, 0, width - 1)
    if height is not None:
        ny1 = clamp(ny1, 0, height - 1)
        ny2 = clamp(ny2, 0, height - 1)

    return [float(nx1), float(ny1), float(nx2), float(ny2)]


def make_crop_bbox_for_player(
    player: Optional[Dict[str, Any]],
    ball_center: Optional[Point],
    pad_ratio: float,
    width: Optional[int],
    height: Optional[int],
) -> Optional[List[float]]:
    if player is None or not player.get("bbox_xyxy"):
        return None

    x1, y1, x2, y2 = map(float, player["bbox_xyxy"])

    # If ball is visible, include both player and ball before padding.
    if ball_center is not None:
        bx, by = ball_center
        x1 = min(x1, bx)
        y1 = min(y1, by)
        x2 = max(x2, bx)
        y2 = max(y2, by)

    return expand_bbox([x1, y1, x2, y2], pad_ratio, width, height)


def read_frames(frames_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata = {}
    meta_path = frames_dir / "_metadata.json"
    if meta_path.exists():
        metadata = load_json(meta_path)

    paths = sorted(Path(p) for p in glob.glob(str(frames_dir / "frame_*.json")))
    if not paths:
        raise FileNotFoundError(f"No frame_*.json files found in {frames_dir}")

    frames = []
    for p in paths:
        d = load_json(p)
        d["_source_json"] = str(p)
        frames.append(d)

    frames.sort(key=lambda f: int(f.get("frame_idx", 0)))
    return frames, metadata


def estimate_ball_positions(
    frames: List[Dict[str, Any]],
    max_missing: int,
    max_speed_px_per_frame: float,
) -> List[Dict[str, Any]]:
    """
    Adds:
      frame["ball_observed_center"]
      frame["ball_estimated_center"]
      frame["ball_is_estimated"]
      frame["ball_missing_gap"]
    """
    last_obs_idx: Optional[int] = None
    last_obs_pos: Optional[Point] = None
    prev_obs_idx: Optional[int] = None
    prev_obs_pos: Optional[Point] = None

    for i, frame in enumerate(frames):
        obs = get_ball_center(frame)
        frame["ball_observed_center"] = list(obs) if obs is not None else None
        frame["ball_estimated_center"] = None
        frame["ball_is_estimated"] = False
        frame["ball_missing_gap"] = 0

        if obs is not None:
            prev_obs_idx, prev_obs_pos = last_obs_idx, last_obs_pos
            last_obs_idx, last_obs_pos = i, obs
            frame["ball_estimated_center"] = list(obs)
            continue

        if last_obs_idx is None or last_obs_pos is None:
            continue

        gap = i - last_obs_idx
        frame["ball_missing_gap"] = int(gap)

        if gap > max_missing:
            continue

        # Constant velocity from last two observed ball points.
        if prev_obs_idx is not None and prev_obs_pos is not None and last_obs_idx != prev_obs_idx:
            dt = last_obs_idx - prev_obs_idx
            vx = (last_obs_pos[0] - prev_obs_pos[0]) / dt
            vy = (last_obs_pos[1] - prev_obs_pos[1]) / dt

            speed = math.hypot(vx, vy)
            if speed > max_speed_px_per_frame:
                scale = max_speed_px_per_frame / max(speed, 1e-6)
                vx *= scale
                vy *= scale

            est = (last_obs_pos[0] + vx * gap, last_obs_pos[1] + vy * gap)
        else:
            # No velocity yet; hold last known location.
            est = last_obs_pos

        frame["ball_estimated_center"] = [float(est[0]), float(est[1])]
        frame["ball_is_estimated"] = True

    return frames


def score_players_for_frame(
    frame: Dict[str, Any],
    previous_owner: Optional[int],
    anchor: str,
    max_possession_distance: float,
    switch_margin: float,
    momentum_bonus: float,
) -> Dict[str, Any]:
    """
    Returns candidate owner for one frame.
    Lower score is better.
    previous_owner gets a score bonus to reduce jitter.
    """
    ball = point_from_list(frame.get("ball_estimated_center"))
    players = frame.get("players") or []

    result = {
        "candidate_owner": None,
        "candidate_score": None,
        "candidate_distance": None,
        "nearest_track_id": None,
        "nearest_distance": None,
        "scores": [],
        "reason": "",
    }

    if ball is None:
        result["reason"] = "no_ball"
        return result

    scored = []

    for p in players:
        tid = get_track_id(p)
        if tid is None:
            continue

        anchor_pt = get_player_anchor(p, anchor)
        if anchor_pt is None:
            continue

        d = dist(ball, anchor_pt)

        # Normalize distance lightly by player height so far/near scale is not too unfair.
        # But keep raw distance as the main measurement.
        score = d

        if previous_owner is not None and tid == previous_owner:
            score -= momentum_bonus

        scored.append(
            {
                "track_id": tid,
                "distance": float(d),
                "score": float(score),
                "anchor": [float(anchor_pt[0]), float(anchor_pt[1])],
                "bbox_xyxy": p.get("bbox_xyxy"),
            }
        )

    if not scored:
        result["reason"] = "no_players"
        return result

    scored.sort(key=lambda x: x["score"])
    nearest_raw = min(scored, key=lambda x: x["distance"])

    result["scores"] = scored
    result["nearest_track_id"] = nearest_raw["track_id"]
    result["nearest_distance"] = nearest_raw["distance"]

    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    if best["distance"] > max_possession_distance:
        result["reason"] = "too_far"
        return result

    # If best and second are extremely close, keep previous owner if it is one of them.
    if second is not None:
        if abs(second["score"] - best["score"]) < switch_margin:
            if previous_owner in {best["track_id"], second["track_id"]}:
                result["candidate_owner"] = previous_owner
                result["candidate_score"] = float(
                    next(s["score"] for s in scored if s["track_id"] == previous_owner)
                )
                result["candidate_distance"] = float(
                    next(s["distance"] for s in scored if s["track_id"] == previous_owner)
                )
                result["reason"] = "ambiguous_keep_previous"
                return result

    result["candidate_owner"] = best["track_id"]
    result["candidate_score"] = best["score"]
    result["candidate_distance"] = best["distance"]
    result["reason"] = "nearest"
    return result


def assign_possession(
    frames: List[Dict[str, Any]],
    anchor: str,
    max_possession_distance: float,
    switch_margin: float,
    momentum_bonus: float,
    min_confirm_frames: int,
    max_empty_keep_frames: int,
    pass_speed_threshold: float,
    pass_keep_previous_frames: int,
    receive_distance: float,
) -> List[Dict[str, Any]]:
    """
    Adds possession fields per frame.
    Uses simple hysteresis:
    - Do not switch to a new owner until new owner appears for min_confirm_frames.
    - Keep old owner through short uncertain/missing spans.
    """
    stable_owner: Optional[int] = None
    pending_owner: Optional[int] = None
    pending_count = 0
    empty_count = 0
    pass_count = 0

    for frame in frames:
        scored = score_players_for_frame(
            frame=frame,
            previous_owner=stable_owner,
            anchor=anchor,
            max_possession_distance=max_possession_distance,
            switch_margin=switch_margin,
            momentum_bonus=momentum_bonus,
        )

        cand = scored["candidate_owner"]

        ball_speed = frame.get("ball_speed_px_per_frame")
        is_fast_ball = ball_speed is not None and float(ball_speed) >= pass_speed_threshold

        # During a fast ball movement, it is usually a pass/shot, not true control.
        # Avoid switching to a defender just because the ball passes near him for 1-2 frames.
        if stable_owner is not None and cand is not None and cand != stable_owner and is_fast_ball:
            cand_dist = scored.get("candidate_distance")
            if cand_dist is None or float(cand_dist) > receive_distance:
                pass_count += 1
                if pass_count <= pass_keep_previous_frames:
                    frame["possession"] = {
                        "owner_track_id": stable_owner,
                        "candidate_owner": cand,
                        "state": "kept_previous_fast_ball",
                        "candidate_distance": scored["candidate_distance"],
                        "nearest_track_id": scored["nearest_track_id"],
                        "nearest_distance": scored["nearest_distance"],
                        "reason": "fast_ball_pass_inertia",
                        "ball_used_center": frame.get("ball_estimated_center"),
                        "ball_is_estimated": frame.get("ball_is_estimated", False),
                        "ball_speed_px_per_frame": ball_speed,
                    }
                    continue
            else:
                pass_count = 0
        else:
            pass_count = 0

        if cand is None:
            empty_count += 1
            if stable_owner is not None and empty_count <= max_empty_keep_frames:
                owner = stable_owner
                state = "kept_previous_uncertain"
            else:
                owner = None
                stable_owner = None
                state = "no_possession"
            pending_owner = None
            pending_count = 0
        else:
            empty_count = 0

            if stable_owner is None:
                if pending_owner == cand:
                    pending_count += 1
                else:
                    pending_owner = cand
                    pending_count = 1

                if pending_count >= min_confirm_frames:
                    stable_owner = cand
                    owner = stable_owner
                    state = "confirmed_new"
                else:
                    owner = None
                    state = "pending_new"
            elif cand == stable_owner:
                pending_owner = None
                pending_count = 0
                owner = stable_owner
                state = "stable"
            else:
                if pending_owner == cand:
                    pending_count += 1
                else:
                    pending_owner = cand
                    pending_count = 1

                if pending_count >= min_confirm_frames:
                    stable_owner = cand
                    owner = stable_owner
                    pending_owner = None
                    pending_count = 0
                    state = "switched"
                else:
                    owner = stable_owner
                    state = "kept_previous_pending_switch"

        frame["possession"] = {
            "owner_track_id": owner,
            "candidate_owner": cand,
            "state": state,
            "candidate_distance": scored["candidate_distance"],
            "nearest_track_id": scored["nearest_track_id"],
            "nearest_distance": scored["nearest_distance"],
            "reason": scored["reason"],
            "ball_used_center": frame.get("ball_estimated_center"),
            "ball_is_estimated": frame.get("ball_is_estimated", False),
            "ball_speed_px_per_frame": frame.get("ball_speed_px_per_frame"),
        }

    return frames


def find_player_by_id(frame: Dict[str, Any], track_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if track_id is None:
        return None
    for p in frame.get("players") or []:
        if get_track_id(p) == track_id:
            return p
    return None


def build_segments(
    frames: List[Dict[str, Any]],
    min_segment_frames: int,
    pre_frames: int,
    post_frames: int,
    crop_pad_ratio: float,
    width: Optional[int],
    height: Optional[int],
) -> List[Dict[str, Any]]:
    segments = []
    current_owner = None
    start_i = None

    def close_segment(end_i: int):
        nonlocal start_i, current_owner
        if current_owner is None or start_i is None:
            return

        raw_len = end_i - start_i + 1
        if raw_len < min_segment_frames:
            return

        clip_start_i = max(0, start_i - pre_frames)
        clip_end_i = min(len(frames) - 1, end_i + post_frames)

        seg_id = len(segments) + 1
        seg_frames = []

        for i in range(clip_start_i, clip_end_i + 1):
            f = frames[i]
            frame_idx = int(f.get("frame_idx", i))
            owner_id = f.get("possession", {}).get("owner_track_id")

            phase = "possession"
            if i < start_i:
                phase = "pre"
            elif i > end_i:
                phase = "post"

            owner_player = find_player_by_id(f, current_owner)
            ball_center = point_from_list(f.get("ball_estimated_center"))

            crop_bbox = make_crop_bbox_for_player(
                owner_player,
                ball_center,
                pad_ratio=crop_pad_ratio,
                width=width,
                height=height,
            )

            seg_frames.append(
                {
                    "frame_idx": frame_idx,
                    "phase": phase,
                    "owner_track_id_for_segment": current_owner,
                    "frame_possession_owner": owner_id,
                    "player_bbox_xyxy": owner_player.get("bbox_xyxy") if owner_player else None,
                    "crop_bbox_xyxy": crop_bbox,
                    "ball_center": list(ball_center) if ball_center is not None else None,
                    "ball_is_estimated": bool(f.get("ball_is_estimated", False)),
                    "source_json": f.get("_source_json"),
                }
            )

        segments.append(
            {
                "possession_id": seg_id,
                "player_track_id": current_owner,
                "start_frame": int(frames[start_i].get("frame_idx", start_i)),
                "end_frame": int(frames[end_i].get("frame_idx", end_i)),
                "clip_start_frame": int(frames[clip_start_i].get("frame_idx", clip_start_i)),
                "clip_end_frame": int(frames[clip_end_i].get("frame_idx", clip_end_i)),
                "num_possession_frames": raw_len,
                "num_clip_frames": len(seg_frames),
                "pre_frames_requested": pre_frames,
                "post_frames_requested": post_frames,
                "crop_pad_ratio": crop_pad_ratio,
                "frames": seg_frames,
            }
        )

    for i, f in enumerate(frames):
        owner = f.get("possession", {}).get("owner_track_id")

        if owner != current_owner:
            if current_owner is not None:
                close_segment(i - 1)
            current_owner = owner
            start_i = i if owner is not None else None

    if current_owner is not None and start_i is not None:
        close_segment(len(frames) - 1)

    return segments


def save_debug_csv(path: Path, frames: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "owner_track_id",
                "candidate_owner",
                "state",
                "nearest_track_id",
                "nearest_distance",
                "candidate_distance",
                "ball_x",
                "ball_y",
                "ball_is_estimated",
                "reason",
            ]
        )
        for fr in frames:
            poss = fr.get("possession", {})
            ball = point_from_list(fr.get("ball_estimated_center"))
            writer.writerow(
                [
                    fr.get("frame_idx"),
                    poss.get("owner_track_id"),
                    poss.get("candidate_owner"),
                    poss.get("state"),
                    poss.get("nearest_track_id"),
                    poss.get("nearest_distance"),
                    poss.get("candidate_distance"),
                    ball[0] if ball else None,
                    ball[1] if ball else None,
                    fr.get("ball_is_estimated"),
                    poss.get("reason"),
                ]
            )



def draw_possession_debug_frame(
    frame_img,
    frame_record: Dict[str, Any],
    segment_owner_id: Optional[int],
    crop_bbox: Optional[List[float]],
) -> Any:
    """
    Draws players, owner, ball, estimated ball, and crop window on one frame.
    """
    vis = frame_img.copy()

    # Draw all players.
    for p in frame_record.get("players") or []:
        bbox = p.get("bbox_xyxy")
        if not bbox:
            continue

        tid = get_track_id(p)
        x1, y1, x2, y2 = map(int, bbox)

        is_owner = tid is not None and tid == segment_owner_id
        color = (0, 255, 255) if is_owner else (80, 220, 80)
        thickness = 3 if is_owner else 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {tid}" if tid is not None else "ID ?"
        if is_owner:
            label += " OWNER"

        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        anchor = get_player_anchor(p, "bottom_center")
        if anchor is not None:
            ax, ay = map(int, anchor)
            cv2.circle(vis, (ax, ay), 4, color, -1)

    # Draw ball.
    ball = point_from_list(frame_record.get("ball_estimated_center"))
    if ball is not None:
        bx, by = map(int, ball)
        is_est = bool(frame_record.get("ball_is_estimated", False))
        color = (0, 165, 255) if is_est else (0, 0, 255)
        label = "BALL EST" if is_est else "BALL"

        cv2.circle(vis, (bx, by), 7, color, -1)
        cv2.putText(
            vis,
            label,
            (bx + 8, max(20, by - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    # Draw crop box.
    if crop_bbox is not None:
        cx1, cy1, cx2, cy2 = map(int, crop_bbox)
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
        cv2.putText(
            vis,
            "CROP",
            (cx1, max(20, cy1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    poss = frame_record.get("possession", {})
    frame_idx = frame_record.get("frame_idx")
    owner = poss.get("owner_track_id")
    state = poss.get("state")
    reason = poss.get("reason")

    info = f"frame={frame_idx} owner={owner} seg_owner={segment_owner_id} state={state} reason={reason}"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 1000), 32), (0, 0, 0), -1)
    cv2.putText(
        vis,
        info,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return vis


def make_debug_videos(
    video_path: Path,
    frames: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    debug_vid_dir: Path,
    fps: float,
    max_debug_videos: int,
    make_full_debug_video: bool,
) -> None:
    """
    Creates:
    - one optional full_video_possession_debug.mp4
    - one debug clip per possession, up to max_debug_videos

    The clips use original video frames and draw:
    - all tracked players
    - owner player
    - ball / estimated ball
    - crop bbox
    """

    if not video_path.exists():
        print(f"[WARN] Cannot create debug videos; video does not exist: {video_path}")
        return

    debug_vid_dir.mkdir(parents=True, exist_ok=True)

    frame_by_idx = {int(f.get("frame_idx", i)): f for i, f in enumerate(frames)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for debug: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)

    # Optional full-video debug.
    if make_full_debug_video:
        full_path = debug_vid_dir / "full_video_possession_debug.mp4"
        writer = cv2.VideoWriter(
            str(full_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, img = cap.read()
            if not ret:
                break

            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            fr = frame_by_idx.get(idx)
            if fr is None:
                writer.write(img)
                continue

            owner = fr.get("possession", {}).get("owner_track_id")
            owner_player = find_player_by_id(fr, owner)
            ball_center = point_from_list(fr.get("ball_estimated_center"))
            crop_bbox = make_crop_bbox_for_player(
                owner_player,
                ball_center,
                pad_ratio=0.30,
                width=width,
                height=height,
            )

            writer.write(draw_possession_debug_frame(img, fr, owner, crop_bbox))

        writer.release()
        print(f"[DONE] full debug video: {full_path}")

    # Per-possession debug clips.
    made = 0
    for seg in segments:
        if max_debug_videos >= 0 and made >= max_debug_videos:
            break

        seg_id = int(seg["possession_id"])
        owner_id = int(seg["player_track_id"])
        clip_start = int(seg["clip_start_frame"])
        clip_end = int(seg["clip_end_frame"])

        out_path = debug_vid_dir / f"possession_{seg_id:06d}_debug.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

        # Map crop boxes from segment frame list.
        seg_frame_info = {
            int(x["frame_idx"]): x
            for x in seg.get("frames", [])
            if x.get("frame_idx") is not None
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

        for idx in range(clip_start, clip_end + 1):
            ret, img = cap.read()
            if not ret:
                break

            fr = frame_by_idx.get(idx)
            if fr is None:
                writer.write(img)
                continue

            info = seg_frame_info.get(idx, {})
            crop_bbox = info.get("crop_bbox_xyxy")

            writer.write(draw_possession_debug_frame(img, fr, owner_id, crop_bbox))

        writer.release()
        made += 1

    cap.release()
    print(f"[DONE] possession debug videos: {debug_vid_dir} ({made} clips)")


def parse_id_set(text: Optional[str]) -> set:
    """
    Parse comma-separated IDs, e.g. "3,7,12".
    Empty / None -> empty set.
    """
    if text is None or str(text).strip() == "":
        return set()
    ids = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            print(f"[WARN] Could not parse track id: {part}")
    return ids


def filter_players_by_track_id(
    frames: List[Dict[str, Any]],
    ignore_track_ids: set,
    keep_track_ids: set,
) -> List[Dict[str, Any]]:
    """
    Removes referee / unwanted tracks before possession logic.
    - ignore_track_ids: always remove these IDs.
    - keep_track_ids: if non-empty, keep only these IDs.
    """
    if not ignore_track_ids and not keep_track_ids:
        return frames

    for fr in frames:
        new_players = []
        for p in fr.get("players") or []:
            tid = get_track_id(p)
            if tid is None:
                continue
            if tid in ignore_track_ids:
                continue
            if keep_track_ids and tid not in keep_track_ids:
                continue
            new_players.append(p)
        fr["players"] = new_players

    return frames


def add_ball_speed(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds ball_speed_px_per_frame using estimated/observed ball centers.
    """
    prev_pos = None
    prev_idx = None

    for fr in frames:
        cur = point_from_list(fr.get("ball_estimated_center"))
        idx = int(fr.get("frame_idx", 0))

        fr["ball_speed_px_per_frame"] = None

        if cur is not None and prev_pos is not None and prev_idx is not None and idx != prev_idx:
            dt = max(1, idx - prev_idx)
            fr["ball_speed_px_per_frame"] = float(dist(cur, prev_pos) / dt)

        if cur is not None:
            prev_pos = cur
            prev_idx = idx

    return frames


def merge_same_owner_segments(
    segments: List[Dict[str, Any]],
    max_gap_frames: int,
) -> List[Dict[str, Any]]:
    """
    Merge neighboring possession segments if:
    - same player_track_id
    - gap between them <= max_gap_frames

    This helps when ball tracking is briefly lost and the same possession is split.
    """
    if not segments:
        return segments

    merged = []
    cur = segments[0]

    for nxt in segments[1:]:
        same_owner = cur["player_track_id"] == nxt["player_track_id"]
        gap = int(nxt["start_frame"]) - int(cur["end_frame"]) - 1

        if same_owner and gap <= max_gap_frames:
            cur["end_frame"] = nxt["end_frame"]
            cur["clip_end_frame"] = max(cur["clip_end_frame"], nxt["clip_end_frame"])
            cur["num_possession_frames"] += nxt["num_possession_frames"]
            cur["frames"].extend(nxt["frames"])
            cur["num_clip_frames"] = len(cur["frames"])
            cur["merged_from"] = cur.get("merged_from", [cur["possession_id"]]) + [nxt["possession_id"]]
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)

    # Re-number after merging.
    for i, seg in enumerate(merged, start=1):
        seg["possession_id"] = i

    return merged

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames_dir", default="outputs/json", help="Directory containing frame_*.json")
    parser.add_argument("--out_dir", default="outputs/possessions", help="Where to write possession_*.json")
    parser.add_argument("--summary_json", default="outputs/possessions_summary.json")
    parser.add_argument("--debug_csv", default="outputs/possession_debug.csv")
    parser.add_argument("--video", default=None, help="Original video path. Needed only for debug videos.")
    parser.add_argument("--debug_vid_dir", default="outputs/debug_vid", help="Directory for debug videos.")
    parser.add_argument("--make_debug_videos", action="store_true", help="Create per-possession debug videos.")
    parser.add_argument("--make_full_debug_video", action="store_true", help="Create one full-length debug video too.")
    parser.add_argument("--max_debug_videos", type=int, default=30, help="Max possession debug clips. Use -1 for all.")

    parser.add_argument("--anchor", default="bottom_center", choices=["bottom_center", "bbox_center", "hip_midpoint"])

    parser.add_argument("--ignore_track_ids", default="", help="Comma-separated track IDs to ignore, e.g. referee: 4,19")
    parser.add_argument("--keep_track_ids", default="", help="If non-empty, keep only these comma-separated player IDs.")

    parser.add_argument("--max_missing_ball_frames", type=int, default=12)
    parser.add_argument("--max_ball_speed_px_per_frame", type=float, default=120.0)

    parser.add_argument("--max_possession_distance", type=float, default=90.0)
    parser.add_argument("--switch_margin", type=float, default=18.0)
    parser.add_argument("--momentum_bonus", type=float, default=25.0)
    parser.add_argument("--pass_speed_threshold", type=float, default=35.0, help="If ball speed is above this, avoid switching owner too quickly.")
    parser.add_argument("--pass_keep_previous_frames", type=int, default=12, help="Keep previous owner during fast pass-like motion for this many frames.")
    parser.add_argument("--receive_distance", type=float, default=55.0, help="New owner must be this close to fast ball to count as receiver.")

    parser.add_argument("--min_confirm_frames", type=int, default=3)
    parser.add_argument("--max_empty_keep_frames", type=int, default=12)
    parser.add_argument("--min_segment_frames", type=int, default=5)

    parser.add_argument("--pre_frames", type=int, default=10)
    parser.add_argument("--post_frames", type=int, default=10)
    parser.add_argument("--crop_pad_ratio", type=float, default=0.30)
    parser.add_argument("--bridge_gap_frames", type=int, default=15, help="Merge same-owner segments separated by this many frames or fewer.")

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    summary_path = Path(args.summary_json)
    debug_csv_path = Path(args.debug_csv)

    frames, metadata = read_frames(frames_dir)

    width = metadata.get("width")
    height = metadata.get("height")
    width = int(width) if width is not None else None
    height = int(height) if height is not None else None

    ignore_track_ids = parse_id_set(args.ignore_track_ids)
    keep_track_ids = parse_id_set(args.keep_track_ids)
    frames = filter_players_by_track_id(frames, ignore_track_ids, keep_track_ids)

    frames = estimate_ball_positions(
        frames,
        max_missing=args.max_missing_ball_frames,
        max_speed_px_per_frame=args.max_ball_speed_px_per_frame,
    )
    frames = add_ball_speed(frames)

    frames = assign_possession(
        frames,
        anchor=args.anchor,
        max_possession_distance=args.max_possession_distance,
        switch_margin=args.switch_margin,
        momentum_bonus=args.momentum_bonus,
        min_confirm_frames=args.min_confirm_frames,
        max_empty_keep_frames=args.max_empty_keep_frames,
        pass_speed_threshold=args.pass_speed_threshold,
        pass_keep_previous_frames=args.pass_keep_previous_frames,
        receive_distance=args.receive_distance,
    )

    segments = build_segments(
        frames,
        min_segment_frames=args.min_segment_frames,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        crop_pad_ratio=args.crop_pad_ratio,
        width=width,
        height=height,
    )

    segments = merge_same_owner_segments(segments, max_gap_frames=args.bridge_gap_frames)

    out_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        out_path = out_dir / f"possession_{seg['possession_id']:06d}.json"
        write_json(out_path, seg)

    summary = {
        "source_frames_dir": str(frames_dir),
        "out_dir": str(out_dir),
        "num_input_frames": len(frames),
        "num_possessions": len(segments),
        "settings": {
            "anchor": args.anchor,
            "ignore_track_ids": sorted(list(ignore_track_ids)),
            "keep_track_ids": sorted(list(keep_track_ids)),
            "max_missing_ball_frames": args.max_missing_ball_frames,
            "max_ball_speed_px_per_frame": args.max_ball_speed_px_per_frame,
            "max_possession_distance": args.max_possession_distance,
            "switch_margin": args.switch_margin,
            "momentum_bonus": args.momentum_bonus,
            "pass_speed_threshold": args.pass_speed_threshold,
            "pass_keep_previous_frames": args.pass_keep_previous_frames,
            "receive_distance": args.receive_distance,
            "min_confirm_frames": args.min_confirm_frames,
            "max_empty_keep_frames": args.max_empty_keep_frames,
            "min_segment_frames": args.min_segment_frames,
            "pre_frames": args.pre_frames,
            "post_frames": args.post_frames,
            "crop_pad_ratio": args.crop_pad_ratio,
            "bridge_gap_frames": args.bridge_gap_frames,
        },
        "possessions": [
            {
                "possession_id": s["possession_id"],
                "player_track_id": s["player_track_id"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "clip_start_frame": s["clip_start_frame"],
                "clip_end_frame": s["clip_end_frame"],
                "num_possession_frames": s["num_possession_frames"],
                "num_clip_frames": s["num_clip_frames"],
                "file": str(out_dir / f"possession_{s['possession_id']:06d}.json"),
            }
            for s in segments
        ],
    }

    write_json(summary_path, summary)
    save_debug_csv(debug_csv_path, frames)

    if args.make_debug_videos or args.make_full_debug_video:
        video_path_arg = args.video or metadata.get("video")
        if video_path_arg is None:
            print("[WARN] No --video provided and metadata has no video path; skipping debug videos.")
        else:
            make_debug_videos(
                video_path=Path(video_path_arg),
                frames=frames,
                segments=segments,
                debug_vid_dir=Path(args.debug_vid_dir),
                fps=float(metadata.get("fps", 30.0)),
                max_debug_videos=args.max_debug_videos,
                make_full_debug_video=args.make_full_debug_video,
            )

    print(f"[DONE] input frames: {len(frames)}")
    print(f"[DONE] possessions: {len(segments)}")
    print(f"[DONE] possession files: {out_dir}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] debug csv: {debug_csv_path}")


if __name__ == "__main__":
    main()
