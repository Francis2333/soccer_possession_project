"""
possession_logic_from_frame_json.py

Stage 2 possession logic.

Input:
    soccer_possession_project/outputs/json/frame_000001.json ...
    or prefixed frame files such as:
    soccer_possession_project/outputs/json/sample_attack_1_frame_000001.json

Output:
    soccer_possession_project/outputs/possessions/<prefix>_possession_0001.json
    soccer_possession_project/outputs/<prefix>_possessions_summary.json
    soccer_possession_project/outputs/<prefix>_possession_debug.csv
    soccer_possession_project/outputs/debug_vid/<prefix>_possession_0001_debug.mp4

State meanings:
    controlled:
        Confirmed possession after enough frames and enough contact/control evidence.

    pass:
        Ball is in transit. A previous player may be shown as CTX in the debug video,
        but this is NOT counted as active possession.

    loose:
        Ball is visible/estimated, but no player has enough control evidence.
        This includes deflections, one-touch uncertainty, bouncing/contested balls,
        and cases where the nearest player is not close enough.

    uncertain:
        Not enough information, e.g. missing ball, long estimated ball gap,
        ambiguous players, or weak evidence.

Debug label meanings:
    CTRL:
        Confirmed controller.

    CTX:
        Contextual previous owner. This is the blue/orange-ish box in debug video.
        It means "previous owner remembered during pass/loose/uncertain," not true contact.

Main logic:
- Estimate short missing-ball gaps.
- Compute ball speed.
- Score player control using:
    distance to ball,
    normalized distance by player height,
    contact threshold around 1/3 player height,
    significant closeness versus second-nearest player,
    expanded player bbox,
    ball speed/pass state,
    temporal confirmation lower bound,
    optional shirt-color same-team momentum.
- Exclude one-touch/deflection by requiring enough confirmed frames.
"""

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


# -----------------------------
# Basic IO / geometry
# -----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_size(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def bbox_height(bbox: List[float]) -> float:
    return bbox_size(bbox)[1]


def bbox_center(bbox: List[float]) -> Point:
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_bottom_center(bbox: List[float]) -> Point:
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, y2)


def point_from_list(x: Any) -> Optional[Point]:
    if not x or len(x) < 2:
        return None
    if x[0] is None or x[1] is None:
        return None
    return float(x[0]), float(x[1])


def point_inside_bbox(p: Point, bbox: List[float]) -> bool:
    x, y = p
    x1, y1, x2, y2 = map(float, bbox)
    return x1 <= x <= x2 and y1 <= y <= y2


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


# -----------------------------
# Player / ball helpers
# -----------------------------

def get_track_id(player: Dict[str, Any]) -> Optional[int]:
    tid = player.get("track_id")
    if tid is None:
        return None
    try:
        return int(tid)
    except Exception:
        return None


def get_player_anchor(player: Dict[str, Any], anchor: str) -> Optional[Point]:
    """
    Anchor options:
    - bottom_center: best for ball near feet
    - bbox_center: more stable for far views
    - hip_midpoint: if pose was already available
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

    cands = frame.get("ball_candidates") or []
    if cands:
        best = max(cands, key=lambda b: float(b.get("conf", 0.0)))
        p = point_from_list(best.get("center") or best.get("bbox_center"))
        if p is not None:
            return p

    return None


def find_player_by_id(frame: Dict[str, Any], track_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if track_id is None:
        return None
    for p in frame.get("players") or []:
        if get_track_id(p) == track_id:
            return p
    return None


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


# -----------------------------
# Prefix / path helpers
# -----------------------------

def infer_output_prefix(args: argparse.Namespace, metadata: Dict[str, Any]) -> str:
    if args.output_prefix:
        return Path(args.output_prefix).stem

    # If you still use mixed JSON files like sample_attack_1_frame_*.json
    if hasattr(args, "input_prefix") and args.input_prefix:
        return Path(args.input_prefix).stem

    video_path = args.video or metadata.get("video")
    if video_path:
        return Path(video_path).stem

    # New behavior:
    # If frames_dir is outputs/json/sample_attack_1,
    # use sample_attack_1 as prefix.
    if args.frames_dir:
        return Path(args.frames_dir).name

    return "video"


def resolve_output_paths(args: argparse.Namespace, prefix: str) -> Tuple[Path, Path, Path, Path]:
    out_dir = Path(args.out_dir)
    debug_vid_dir = Path(args.debug_vid_dir)

    if args.summary_json == "outputs/possessions/possessions_summary.json":
        summary_path = Path("outputs") / "possessions" / prefix / f"{prefix}_possessions_summary.json"
    else:
        summary_path = Path(args.summary_json)

    if args.debug_csv == "outputs/possessions/possessions_debug.csv":
        debug_csv_path = Path("outputs") / "possessions" / prefix / f"{prefix}_possession_debug.csv"
    else:
        debug_csv_path = Path(args.debug_csv)

    return out_dir, summary_path, debug_csv_path, debug_vid_dir


def read_frames(frames_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata = {}
    meta_path = frames_dir / "_metadata.json"
    if meta_path.exists():
        metadata = load_json(meta_path)

    # Supports both:
    #   frame_000001.json
    #   sample_attack_1_frame_000001.json
    paths = sorted(Path(p) for p in glob.glob(str(frames_dir / "*frame_*.json")))
    paths = [p for p in paths if p.name != "_metadata.json"]

    if not paths:
        raise FileNotFoundError(f"No *frame_*.json files found in {frames_dir}")

    frames = []
    for p in paths:
        d = load_json(p)
        d["_source_json"] = str(p)
        frames.append(d)

    frames.sort(key=lambda f: int(f.get("frame_idx", 0)))
    return frames, metadata


# -----------------------------
# Shirt color / team-color momentum
# -----------------------------

def get_keypoint_xy(player: Dict[str, Any], named_key: str) -> Optional[Point]:
    kp = player.get(named_key)
    if isinstance(kp, dict):
        x, y = kp.get("x"), kp.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def get_keypoint_from_list(player: Dict[str, Any], idx: int) -> Optional[Point]:
    kpts = player.get("keypoints_17")
    if not isinstance(kpts, list) or idx >= len(kpts):
        return None

    kp = kpts[idx]
    if isinstance(kp, dict):
        x, y = kp.get("x"), kp.get("y")
        if x is not None and y is not None:
            return float(x), float(y)

    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
        if kp[0] is not None and kp[1] is not None:
            return float(kp[0]), float(kp[1])

    return None


def torso_roi_from_player(
    player: Dict[str, Any],
    width: int,
    height: int,
    fallback_bbox_ratio: Tuple[float, float, float, float],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Prefer shoulder/hip keypoints if available:
        left/right shoulders: 5, 6
        left/right hips: 11, 12

    If keypoints are not available, use a central upper-body region inside bbox.
    """
    bbox = player.get("bbox_xyxy")
    if not bbox:
        return None

    pts = []

    # Named pose fields.
    for key in ["left_shoulder_5", "right_shoulder_6", "left_hip_11", "right_hip_12"]:
        p = get_keypoint_xy(player, key)
        if p is not None:
            pts.append(p)

    # Generic keypoints_17 list.
    for idx in [5, 6, 11, 12]:
        p = get_keypoint_from_list(player, idx)
        if p is not None:
            pts.append(p)

    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        bw = max(8.0, x2 - x1)
        bh = max(8.0, y2 - y1)
        x1 -= 0.40 * bw
        x2 += 0.40 * bw
        y1 -= 0.25 * bh
        y2 += 0.25 * bh
    else:
        bx1, by1, bx2, by2 = map(float, bbox)
        bw, bh = bbox_size(bbox)
        rx1, ry1, rx2, ry2 = fallback_bbox_ratio
        x1 = bx1 + rx1 * bw
        x2 = bx1 + rx2 * bw
        y1 = by1 + ry1 * bh
        y2 = by1 + ry2 * bh

    x1 = int(clamp(x1, 0, width - 1))
    x2 = int(clamp(x2, 0, width - 1))
    y1 = int(clamp(y1, 0, height - 1))
    y2 = int(clamp(y2, 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def sample_shirt_color_from_roi(frame_img: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
    x1, y1, x2, y2 = roi
    crop = frame_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    flat_hsv = hsv.reshape(-1, 3)
    flat_bgr = crop.reshape(-1, 3)

    s = flat_hsv[:, 1]
    v = flat_hsv[:, 2]

    # Remove likely shadows/field lines/overexposure.
    mask = (v > 35) & (v < 245) & (s > 20)
    if int(mask.sum()) < 10:
        mask = np.ones(len(flat_hsv), dtype=bool)

    mean_hsv = flat_hsv[mask].mean(axis=0)
    mean_bgr = flat_bgr[mask].mean(axis=0)

    return {
        "mean_hsv": [float(x) for x in mean_hsv],
        "mean_bgr": [float(x) for x in mean_bgr],
        "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "num_pixels": int(mask.sum()),
    }


def color_distance_hsv(c1: Optional[Dict[str, Any]], c2: Optional[Dict[str, Any]]) -> Optional[float]:
    if not c1 or not c2:
        return None

    h1, s1, v1 = c1["mean_hsv"]
    h2, s2, v2 = c2["mean_hsv"]

    # OpenCV H is circular [0, 179].
    dh = abs(h1 - h2)
    dh = min(dh, 180.0 - dh) / 90.0
    ds = abs(s1 - s2) / 255.0
    dv = abs(v1 - v2) / 255.0

    return float(math.sqrt((2.0 * dh) ** 2 + ds ** 2 + 0.5 * dv ** 2))


def add_shirt_colors(
    frames: List[Dict[str, Any]],
    video_path: Optional[Path],
    enabled: bool,
    fallback_bbox_ratio: Tuple[float, float, float, float],
) -> List[Dict[str, Any]]:
    if not enabled:
        return frames

    if video_path is None or not video_path.exists():
        print("[WARN] --use_shirt_color was set, but no valid --video path was provided. Skipping shirt color.")
        return frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video for shirt color: {video_path}")
        return frames

    frame_by_idx = {int(fr.get("frame_idx", i)): fr for i, fr in enumerate(frames)}

    for idx in sorted(frame_by_idx.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
        if not ret:
            continue

        h, w = img.shape[:2]
        fr = frame_by_idx[idx]

        for p in fr.get("players") or []:
            roi = torso_roi_from_player(p, w, h, fallback_bbox_ratio)
            color = sample_shirt_color_from_roi(img, roi) if roi else None
            p["shirt_color"] = color

    cap.release()
    return frames


# -----------------------------
# Ball estimation / motion
# -----------------------------

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

        if prev_obs_idx is not None and prev_obs_pos is not None and last_obs_idx != prev_obs_idx:
            dt = max(1, last_obs_idx - prev_obs_idx)
            vx = (last_obs_pos[0] - prev_obs_pos[0]) / dt
            vy = (last_obs_pos[1] - prev_obs_pos[1]) / dt

            speed = math.hypot(vx, vy)
            if speed > max_speed_px_per_frame:
                scale = max_speed_px_per_frame / max(speed, 1e-6)
                vx *= scale
                vy *= scale

            est = (last_obs_pos[0] + vx * gap, last_obs_pos[1] + vy * gap)
        else:
            est = last_obs_pos

        frame["ball_estimated_center"] = [float(est[0]), float(est[1])]
        frame["ball_is_estimated"] = True

    return frames


def add_ball_motion(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds:
      ball_speed_px_per_frame
      ball_velocity
    """
    prev_pos = None
    prev_idx = None

    for fr in frames:
        cur = point_from_list(fr.get("ball_estimated_center"))
        idx = int(fr.get("frame_idx", 0))

        fr["ball_speed_px_per_frame"] = None
        fr["ball_velocity"] = None

        if cur is not None and prev_pos is not None and prev_idx is not None and idx != prev_idx:
            dt = max(1, idx - prev_idx)
            vx = (cur[0] - prev_pos[0]) / dt
            vy = (cur[1] - prev_pos[1]) / dt
            fr["ball_speed_px_per_frame"] = float(math.hypot(vx, vy))
            fr["ball_velocity"] = [float(vx), float(vy)]

        if cur is not None:
            prev_pos = cur
            prev_idx = idx

    return frames


# -----------------------------
# Possession scoring and assignment
# -----------------------------

def score_players_for_frame(
    frame: Dict[str, Any],
    previous_owner: Optional[int],
    previous_owner_color: Optional[Dict[str, Any]],
    anchor: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Higher score is better.
    Uses strict contact / significant-closeness requirements.
    """
    ball = point_from_list(frame.get("ball_estimated_center"))
    players = frame.get("players") or []

    result = {
        "candidate_owner": None,
        "candidate_score": None,
        "candidate_distance": None,
        "candidate_norm_distance": None,
        "nearest_track_id": None,
        "nearest_distance": None,
        "second_nearest_distance": None,
        "distance_ratio_to_second": None,
        "scores": [],
        "reason": "",
        "raw_state": "uncertain",
    }

    if ball is None:
        result["reason"] = "no_ball"
        return result

    scored = []

    for p in players:
        tid = get_track_id(p)
        if tid is None:
            continue

        bbox = p.get("bbox_xyxy")
        if not bbox:
            continue

        anchor_pt = get_player_anchor(p, anchor)
        if anchor_pt is None:
            continue

        h = bbox_height(bbox)
        d = dist(ball, anchor_pt)
        norm_d = d / max(h, 1.0)

        expanded = expand_bbox(bbox, args.expanded_box_ratio, None, None)
        inside_expanded_box = point_inside_bbox(ball, expanded)

        # Base score: close to ball is good.
        score = max(0.0, 1.20 - norm_d) * 100.0

        if norm_d <= args.contact_norm_distance:
            score += args.contact_bonus

        if inside_expanded_box:
            score += args.inside_box_bonus

        if previous_owner is not None and tid == previous_owner:
            score += args.momentum_bonus

        # Optional same-team color momentum.
        shirt_color = p.get("shirt_color")
        color_dist = color_distance_hsv(shirt_color, previous_owner_color)
        same_team_like = None
        if args.use_shirt_color and previous_owner_color is not None and shirt_color is not None:
            same_team_like = color_dist is not None and color_dist <= args.same_team_color_threshold

            if previous_owner is not None and tid != previous_owner:
                if same_team_like:
                    score += args.same_team_momentum_bonus
                else:
                    score -= args.opponent_color_penalty

        # Ball velocity direction: reward if moving toward player.
        velocity = point_from_list(frame.get("ball_velocity"))
        direction_score = 0.0
        if velocity is not None and norm(velocity) > 1e-6:
            ball_to_player = sub(anchor_pt, ball)
            denom = max(norm(velocity) * norm(ball_to_player), 1e-6)
            direction_score = max(0.0, dot(velocity, ball_to_player) / denom)
            score += direction_score * args.velocity_direction_bonus

        scored.append(
            {
                "track_id": tid,
                "score": float(score),
                "distance": float(d),
                "norm_distance": float(norm_d),
                "anchor": [float(anchor_pt[0]), float(anchor_pt[1])],
                "bbox_xyxy": bbox,
                "inside_expanded_box": bool(inside_expanded_box),
                "shirt_color": shirt_color,
                "color_distance_to_previous_owner": color_dist,
                "same_team_like_previous_owner": same_team_like,
                "direction_score": float(direction_score),
            }
        )

    if not scored:
        result["reason"] = "no_players"
        return result

    nearest_sorted = sorted(scored, key=lambda x: x["distance"])
    nearest = nearest_sorted[0]
    second_nearest = nearest_sorted[1] if len(nearest_sorted) > 1 else None

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]
    second_best = scored[1] if len(scored) > 1 else None

    ratio = None
    if second_nearest is not None:
        ratio = nearest["distance"] / max(second_nearest["distance"], 1e-6)

    result["scores"] = scored[:8]
    result["nearest_track_id"] = nearest["track_id"]
    result["nearest_distance"] = nearest["distance"]
    result["second_nearest_distance"] = second_nearest["distance"] if second_nearest else None
    result["distance_ratio_to_second"] = ratio
    result["candidate_owner"] = best["track_id"]
    result["candidate_score"] = best["score"]
    result["candidate_distance"] = best["distance"]
    result["candidate_norm_distance"] = best["norm_distance"]

    ball_speed = frame.get("ball_speed_px_per_frame")
    ball_is_estimated = bool(frame.get("ball_is_estimated", False))
    missing_gap = int(frame.get("ball_missing_gap", 0) or 0)

    close_contact = best["norm_distance"] <= args.contact_norm_distance
    significantly_closer = ratio is not None and ratio <= args.significant_closer_ratio

    if ball_is_estimated and missing_gap > args.max_estimated_control_gap:
        result["raw_state"] = "uncertain"
        result["reason"] = "estimated_ball_gap_too_long"
        return result

    # If not close enough and not much closer than others, call it loose/pass.
    if not close_contact and not significantly_closer and not best["inside_expanded_box"]:
        if ball_speed is not None and ball_speed >= args.pass_speed_threshold:
            result["raw_state"] = "pass"
            result["reason"] = "fast_ball_not_controlled"
        else:
            result["raw_state"] = "loose"
            result["reason"] = "not_contact_or_not_significantly_closer"
        return result

    if best["norm_distance"] > args.max_control_norm_distance and not best["inside_expanded_box"]:
        result["raw_state"] = "loose"
        result["reason"] = "too_far_from_players"
        return result

    # Fast ball: possible receiver only if quite close.
    if ball_speed is not None and ball_speed >= args.pass_speed_threshold:
        evidence_gap = best["score"] - (second_best["score"] if second_best else 0.0)
        receiver_close = (
            best["norm_distance"] <= args.receive_norm_distance
            or best["distance"] <= args.receive_distance
            or best["inside_expanded_box"]
        )
        if receiver_close and evidence_gap >= args.receiver_score_margin:
            result["raw_state"] = "controlled_candidate"
            result["reason"] = "fast_ball_possible_receiver"
        else:
            result["raw_state"] = "pass"
            result["reason"] = "fast_ball_in_transit"
        return result

    if best["score"] < args.min_control_score:
        result["raw_state"] = "loose"
        result["reason"] = "weak_control_score"
        return result

    # Ambiguity check: if two players score similarly, keep previous only if involved.
    if second_best is not None:
        score_gap = best["score"] - second_best["score"]
        if score_gap < args.switch_margin:
            if previous_owner in {best["track_id"], second_best["track_id"]}:
                prev = next(s for s in scored if s["track_id"] == previous_owner)
                result["candidate_owner"] = previous_owner
                result["candidate_score"] = prev["score"]
                result["candidate_distance"] = prev["distance"]
                result["candidate_norm_distance"] = prev["norm_distance"]
                result["raw_state"] = "controlled_candidate"
                result["reason"] = "ambiguous_keep_previous"
                return result

            result["raw_state"] = "uncertain"
            result["reason"] = "ambiguous_no_previous"
            return result

    result["raw_state"] = "controlled_candidate"
    result["reason"] = "controlled_candidate_best_score"
    return result


def assign_possession(
    frames: List[Dict[str, Any]],
    anchor: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Adds possession fields per frame.

    owner_track_id:
        Contextual owner. Can be shown as CTX during pass/loose/uncertain.

    controlled_owner_track_id:
        Confirmed possession owner. Only this is used to build possession segments.
    """
    stable_owner: Optional[int] = None
    stable_owner_color: Optional[Dict[str, Any]] = None

    pending_owner: Optional[int] = None
    pending_count = 0

    empty_count = 0
    pass_count = 0

    for frame in frames:
        scored = score_players_for_frame(
            frame=frame,
            previous_owner=stable_owner,
            previous_owner_color=stable_owner_color,
            anchor=anchor,
            args=args,
        )

        raw_state = scored["raw_state"]
        cand = scored["candidate_owner"]

        owner = stable_owner
        controlled_owner = None
        state = raw_state
        decision = ""

        if raw_state == "controlled_candidate" and cand is not None:
            empty_count = 0
            pass_count = 0

            if stable_owner is None:
                if pending_owner == cand:
                    pending_count += 1
                else:
                    pending_owner = cand
                    pending_count = 1

                if pending_count >= args.min_confirm_frames:
                    stable_owner = cand
                    owner = stable_owner
                    controlled_owner = stable_owner
                    state = "controlled"
                    decision = "confirmed_initial_owner"
                    p = find_player_by_id(frame, stable_owner)
                    stable_owner_color = p.get("shirt_color") if p and p.get("shirt_color") else stable_owner_color
                else:
                    owner = None
                    controlled_owner = None
                    state = "uncertain"
                    decision = "pending_initial_owner"

            elif cand == stable_owner:
                pending_owner = None
                pending_count = 0
                owner = stable_owner
                controlled_owner = stable_owner
                state = "controlled"
                decision = "stable_owner"
                p = find_player_by_id(frame, stable_owner)
                stable_owner_color = p.get("shirt_color") if p and p.get("shirt_color") else stable_owner_color

            else:
                if pending_owner == cand:
                    pending_count += 1
                else:
                    pending_owner = cand
                    pending_count = 1

                required = args.min_confirm_frames
                ball_speed = frame.get("ball_speed_px_per_frame")
                if ball_speed is not None and float(ball_speed) >= args.pass_speed_threshold:
                    required = max(required, args.receiver_confirm_frames)

                if pending_count >= required:
                    stable_owner = cand
                    owner = stable_owner
                    controlled_owner = stable_owner
                    state = "controlled"
                    decision = "confirmed_switch"
                    pending_owner = None
                    pending_count = 0
                    p = find_player_by_id(frame, stable_owner)
                    stable_owner_color = p.get("shirt_color") if p and p.get("shirt_color") else stable_owner_color
                else:
                    # Keep previous as context only, do not count as controlled.
                    owner = stable_owner
                    controlled_owner = None
                    state = "pass" if frame.get("ball_speed_px_per_frame", 0) >= args.pass_speed_threshold else "uncertain"
                    decision = "pending_switch_context_previous"

        elif raw_state == "pass":
            pass_count += 1
            pending_owner = None
            pending_count = 0

            if stable_owner is not None and pass_count <= args.pass_keep_previous_frames:
                owner = stable_owner
                controlled_owner = None
                state = "pass"
                decision = "pass_keep_previous_context"
            else:
                owner = None
                controlled_owner = None
                state = "pass"
                decision = "pass_no_context"
                if pass_count > args.pass_keep_previous_frames:
                    stable_owner = None

        elif raw_state in {"loose", "uncertain"}:
            empty_count += 1
            pending_owner = None
            pending_count = 0

            if stable_owner is not None and empty_count <= args.max_empty_keep_frames:
                owner = stable_owner
                controlled_owner = None
                state = raw_state
                decision = "short_gap_keep_previous_context"
            else:
                owner = None
                controlled_owner = None
                state = raw_state
                decision = "lost_owner"
                if empty_count > args.max_empty_keep_frames:
                    stable_owner = None

        frame["possession"] = {
            "owner_track_id": owner,
            "controlled_owner_track_id": controlled_owner,
            "candidate_owner": cand,
            "state": state,
            "raw_state": raw_state,
            "candidate_score": scored["candidate_score"],
            "candidate_distance": scored["candidate_distance"],
            "candidate_norm_distance": scored["candidate_norm_distance"],
            "nearest_track_id": scored["nearest_track_id"],
            "nearest_distance": scored["nearest_distance"],
            "second_nearest_distance": scored["second_nearest_distance"],
            "distance_ratio_to_second": scored["distance_ratio_to_second"],
            "reason": scored["reason"],
            "decision": decision,
            "ball_used_center": frame.get("ball_estimated_center"),
            "ball_is_estimated": frame.get("ball_is_estimated", False),
            "ball_missing_gap": frame.get("ball_missing_gap", 0),
            "ball_speed_px_per_frame": frame.get("ball_speed_px_per_frame"),
            "top_scores": scored["scores"],
        }

    return frames


# -----------------------------
# Segments
# -----------------------------

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

    if ball_center is not None:
        bx, by = ball_center
        x1 = min(x1, bx)
        y1 = min(y1, by)
        x2 = max(x2, bx)
        y2 = max(y2, by)

    return expand_bbox([x1, y1, x2, y2], pad_ratio, width, height)


def build_segments(
    frames: List[Dict[str, Any]],
    min_segment_frames: int,
    pre_frames: int,
    post_frames: int,
    crop_pad_ratio: float,
    width: Optional[int],
    height: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Segments are built only from controlled_owner_track_id.
    Therefore pass/loose/uncertain and one-touch events are excluded unless they
    become long enough confirmed controlled possessions.
    """
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
            poss = f.get("possession", {})
            context_owner = poss.get("owner_track_id")
            controlled_owner = poss.get("controlled_owner_track_id")

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
                    "state": poss.get("state"),
                    "owner_track_id_for_segment": current_owner,
                    "frame_context_owner": context_owner,
                    "frame_controlled_owner": controlled_owner,
                    "player_bbox_xyxy": owner_player.get("bbox_xyxy") if owner_player else None,
                    "crop_bbox_xyxy": crop_bbox,
                    "ball_center": list(ball_center) if ball_center is not None else None,
                    "ball_is_estimated": bool(f.get("ball_is_estimated", False)),
                    "ball_speed_px_per_frame": f.get("ball_speed_px_per_frame"),
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
        owner = f.get("possession", {}).get("controlled_owner_track_id")

        if owner != current_owner:
            if current_owner is not None:
                close_segment(i - 1)
            current_owner = owner
            start_i = i if owner is not None else None

    if current_owner is not None and start_i is not None:
        close_segment(len(frames) - 1)

    return segments


def merge_same_owner_segments(
    segments: List[Dict[str, Any]],
    frames: List[Dict[str, Any]],
    max_gap_frames: int,
    allowed_bridge_states: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Merge neighboring possession segments if:

    CTRL owner A
        ↓
    CTX owner A during pass/loose/uncertain
        ↓
    CTRL owner A again

    This prevents one real possession from being split just because the ball
    temporarily enters pass/loose/uncertain state.
    """

    if not segments:
        return segments

    if allowed_bridge_states is None:
        allowed_bridge_states = {"pass", "loose", "uncertain"}

    frame_by_idx = {
        int(fr.get("frame_idx", i)): fr
        for i, fr in enumerate(frames)
    }

    def can_bridge_same_owner(prev_seg, next_seg) -> bool:
        owner = prev_seg["player_track_id"]

        if owner != next_seg["player_track_id"]:
            return False

        gap_start = int(prev_seg["end_frame"]) + 1
        gap_end = int(next_seg["start_frame"]) - 1
        gap_len = gap_end - gap_start + 1

        if gap_len <= 0:
            return True

        if gap_len > max_gap_frames:
            return False

        for frame_idx in range(gap_start, gap_end + 1):
            fr = frame_by_idx.get(frame_idx)
            if fr is None:
                return False

            poss = fr.get("possession", {})
            state = poss.get("state")
            context_owner = poss.get("owner_track_id")
            controlled_owner = poss.get("controlled_owner_track_id")

            # During bridge frames, it should NOT be controlled by another player.
            if controlled_owner is not None and controlled_owner != owner:
                return False

            # It must be same-player CTX, not random loose state.
            if context_owner != owner:
                return False

            if state not in allowed_bridge_states:
                return False

        return True

    merged = []
    cur = segments[0]

    for nxt in segments[1:]:
        if can_bridge_same_owner(cur, nxt):
            cur["end_frame"] = nxt["end_frame"]
            cur["clip_end_frame"] = max(cur["clip_end_frame"], nxt["clip_end_frame"])
            cur["num_possession_frames"] += nxt["num_possession_frames"]
            cur["frames"].extend(nxt["frames"])
            cur["num_clip_frames"] = len(cur["frames"])
            cur["merged_from"] = cur.get("merged_from", [cur["possession_id"]]) + [nxt["possession_id"]]
            cur["merge_reason"] = "ctrl_ctx_ctrl_same_player"
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)

    for i, seg in enumerate(merged, start=1):
        seg["possession_id"] = i

    return merged


# -----------------------------
# Debug CSV / videos
# -----------------------------

def save_debug_csv(path: Path, frames: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "state",
                "context_owner_track_id",
                "controlled_owner_track_id",
                "candidate_owner",
                "candidate_score",
                "candidate_distance",
                "candidate_norm_distance",
                "nearest_track_id",
                "nearest_distance",
                "second_nearest_distance",
                "distance_ratio_to_second",
                "ball_x",
                "ball_y",
                "ball_speed",
                "ball_is_estimated",
                "ball_missing_gap",
                "reason",
                "decision",
            ]
        )
        for fr in frames:
            poss = fr.get("possession", {})
            ball = point_from_list(fr.get("ball_estimated_center"))
            writer.writerow(
                [
                    fr.get("frame_idx"),
                    poss.get("state"),
                    poss.get("owner_track_id"),
                    poss.get("controlled_owner_track_id"),
                    poss.get("candidate_owner"),
                    poss.get("candidate_score"),
                    poss.get("candidate_distance"),
                    poss.get("candidate_norm_distance"),
                    poss.get("nearest_track_id"),
                    poss.get("nearest_distance"),
                    poss.get("second_nearest_distance"),
                    poss.get("distance_ratio_to_second"),
                    ball[0] if ball else None,
                    ball[1] if ball else None,
                    fr.get("ball_speed_px_per_frame"),
                    fr.get("ball_is_estimated"),
                    fr.get("ball_missing_gap"),
                    poss.get("reason"),
                    poss.get("decision"),
                ]
            )


def draw_possession_debug_frame(
    frame_img,
    frame_record: Dict[str, Any],
    segment_owner_id: Optional[int],
    crop_bbox: Optional[List[float]],
) -> Any:
    """
    Draws players, contextual owner, controlled owner, ball, and crop window.
    """
    vis = frame_img.copy()

    poss = frame_record.get("possession", {})
    context_owner = poss.get("owner_track_id")
    controlled_owner = poss.get("controlled_owner_track_id")

    for p in frame_record.get("players") or []:
        bbox = p.get("bbox_xyxy")
        if not bbox:
            continue

        tid = get_track_id(p)
        x1, y1, x2, y2 = map(int, bbox)

        if tid is not None and tid == controlled_owner:
            color = (0, 255, 255)
            thickness = 3
            label_suffix = " CTRL"
        elif tid is not None and tid == context_owner:
            color = (255, 180, 0)
            thickness = 2
            label_suffix = " CTX"
        else:
            color = (80, 220, 80)
            thickness = 1
            label_suffix = ""

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {tid}" if tid is not None else "ID ?"
        label += label_suffix

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

        # Draw sampled shirt-color ROI if available.
        shirt = p.get("shirt_color")
        if shirt and shirt.get("roi_xyxy"):
            rx1, ry1, rx2, ry2 = map(int, shirt["roi_xyxy"])
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (180, 180, 180), 1)

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

    frame_idx = frame_record.get("frame_idx")
    state = poss.get("state")
    reason = poss.get("reason")
    decision = poss.get("decision")
    speed = frame_record.get("ball_speed_px_per_frame")
    speed_s = f"{speed:.1f}" if speed is not None else "None"

    info = (
        f"frame={frame_idx} state={state} ctx={context_owner} ctrl={controlled_owner} "
        f"seg={segment_owner_id} speed={speed_s} reason={reason} decision={decision}"
    )
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 1450), 34), (0, 0, 0), -1)
    cv2.putText(
        vis,
        info,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )

    return vis


def make_debug_videos(
    video_path: Path,
    frames: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    debug_vid_dir: Path,
    prefix: str,
    fps: float,
    max_debug_videos: int,
    make_full_debug_video: bool,
) -> None:
    if not video_path.exists():
        print(f"[WARN] Cannot create debug videos; video does not exist: {video_path}")
        return

    full_debug_dir = debug_vid_dir / "full_video_possession"
    possession_debug_dir = debug_vid_dir / "possession"
    full_debug_dir.mkdir(parents=True, exist_ok=True)
    possession_debug_dir.mkdir(parents=True, exist_ok=True)

    frame_by_idx = {int(f.get("frame_idx", i)): f for i, f in enumerate(frames)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for debug: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)

    if make_full_debug_video:
        full_path = full_debug_dir / f"{prefix}_full_video_possession_debug.mp4"
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

    made = 0
    for seg in segments:
        if max_debug_videos >= 0 and made >= max_debug_videos:
            break

        seg_id = int(seg["possession_id"])
        owner_id = int(seg["player_track_id"])
        clip_start = int(seg["clip_start_frame"])
        clip_end = int(seg["clip_end_frame"])

        out_path = possession_debug_dir / f"{prefix}_possession_{seg_id:04d}_debug.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

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
    print(f"[DONE] full-possession debug dir: {full_debug_dir}")
    print(f"[DONE] possession debug videos: {possession_debug_dir} ({made} clips)")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames_dir", default="outputs/json", help="Directory containing frame_*.json or *_frame_*.json")
    parser.add_argument("--out_dir", default="outputs/possessions", help="Where to write possession JSON")
    parser.add_argument("--summary_json", default="outputs/possessions/possessions_summary.json")
    parser.add_argument("--debug_csv", default="outputs/possessions/possessions_debug.csv")
    parser.add_argument("--video", default=None, help="Original video path. Needed for debug videos and shirt color.")
    parser.add_argument("--output_prefix", default=None, help="Prefix for generated files. Default = video filename stem.")
    parser.add_argument("--debug_vid_dir", default="outputs/debug_vid", help="Directory for debug videos.")
    parser.add_argument("--make_debug_videos", action="store_true", help="Create per-possession debug videos.")
    parser.add_argument("--make_full_debug_video", action="store_true", help="Create one full-length debug video too.")
    parser.add_argument("--max_debug_videos", type=int, default=30, help="Max possession debug clips. Use -1 for all.")

    parser.add_argument("--anchor", default="bottom_center", choices=["bottom_center", "bbox_center", "hip_midpoint"])

    parser.add_argument("--ignore_track_ids", default="", help="Comma-separated track IDs to ignore, e.g. referee: 4,19")
    parser.add_argument("--keep_track_ids", default="", help="If non-empty, keep only these comma-separated player IDs.")

    # Shirt color.
    parser.add_argument("--use_shirt_color", action="store_true", help="Sample shirt colors from original video and apply same-team momentum.")
    parser.add_argument("--shirt_roi", default="0.30,0.25,0.70,0.62", help="Fallback bbox ratios x1,y1,x2,y2 for shirt sampling.")
    parser.add_argument("--same_team_color_threshold", type=float, default=0.58)
    parser.add_argument("--same_team_momentum_bonus", type=float, default=18.0)
    parser.add_argument("--opponent_color_penalty", type=float, default=18.0)

    # Ball estimate/motion.
    parser.add_argument("--max_missing_ball_frames", type=int, default=12)
    parser.add_argument("--max_ball_speed_px_per_frame", type=float, default=120.0)
    parser.add_argument("--max_estimated_control_gap", type=int, default=8)

    # Possession evidence.
    parser.add_argument("--expanded_box_ratio", type=float, default=0.35)
    parser.add_argument("--contact_norm_distance", type=float, default=0.80, help="Contact threshold: ball within about 0.8 player height.")
    parser.add_argument("--significant_closer_ratio", type=float, default=0.50, help="Best distance <= this * second-nearest distance.")
    parser.add_argument("--max_control_norm_distance", type=float, default=1.05)
    parser.add_argument("--min_control_score", type=float, default=45.0)
    parser.add_argument("--switch_margin", type=float, default=20.0)
    parser.add_argument("--momentum_bonus", type=float, default=25.0)
    parser.add_argument("--inside_box_bonus", type=float, default=45.0)
    parser.add_argument("--contact_bonus", type=float, default=35.0)
    parser.add_argument("--velocity_direction_bonus", type=float, default=15.0)

    # Pass/receiver.
    parser.add_argument("--pass_speed_threshold", type=float, default=35.0, help="Fast ball is treated as pass/in transit.")
    parser.add_argument("--pass_keep_previous_frames", type=int, default=16, help="Keep previous owner as CTX during pass.")
    parser.add_argument("--receive_distance", type=float, default=50.0, help="New receiver must be close to fast ball.")
    parser.add_argument("--receive_norm_distance", type=float, default=0.55)
    parser.add_argument("--receiver_score_margin", type=float, default=18.0)
    parser.add_argument("--receiver_confirm_frames", type=int, default=15)

    # Lower bound / stability.
    parser.add_argument("--min_confirm_frames", type=int, default=10, help="Lower bound for possession confirmation, around 0.5s at 30fps.")
    parser.add_argument("--max_empty_keep_frames", type=int, default=12)
    parser.add_argument("--min_segment_frames", type=int, default=10, help="Exclude short one-touch/deflection segments.")

    # Segment/crop.
    parser.add_argument("--pre_frames", type=int, default=10)
    parser.add_argument("--post_frames", type=int, default=10)
    parser.add_argument("--crop_pad_ratio", type=float, default=0.30)
    parser.add_argument("--bridge_gap_frames", type=int, default=20, help="Merge same-owner controlled segments separated by small gaps.")

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    frames, metadata = read_frames(frames_dir)

    prefix = infer_output_prefix(args, metadata)
    out_dir, summary_path, debug_csv_path, debug_vid_dir = resolve_output_paths(args, prefix)

    video_path_arg = args.video or metadata.get("video")
    video_path = Path(video_path_arg) if video_path_arg else None

    width = metadata.get("width")
    height = metadata.get("height")
    width = int(width) if width is not None else None
    height = int(height) if height is not None else None

    roi_parts = [float(x.strip()) for x in args.shirt_roi.split(",")]
    if len(roi_parts) != 4:
        raise ValueError("--shirt_roi must have four comma-separated floats: x1,y1,x2,y2")
    shirt_roi = tuple(roi_parts)

    ignore_track_ids = parse_id_set(args.ignore_track_ids)
    keep_track_ids = parse_id_set(args.keep_track_ids)

    frames = filter_players_by_track_id(frames, ignore_track_ids, keep_track_ids)
    frames = estimate_ball_positions(
        frames,
        max_missing=args.max_missing_ball_frames,
        max_speed_px_per_frame=args.max_ball_speed_px_per_frame,
    )
    frames = add_ball_motion(frames)
    frames = add_shirt_colors(frames, video_path, args.use_shirt_color, shirt_roi)

    frames = assign_possession(
        frames,
        anchor=args.anchor,
        args=args,
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

    segments = merge_same_owner_segments(
    segments=segments,
    frames=frames,
    max_gap_frames=args.bridge_gap_frames,
    )

    video_possession_dir = out_dir / prefix
    video_possession_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        out_path = video_possession_dir / f"{prefix}_possession_{seg['possession_id']:04d}.json"
        write_json(out_path, seg)

    state_counts: Dict[str, int] = {}
    for fr in frames:
        state = fr.get("possession", {}).get("state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

    summary = {
        "video_prefix": prefix,
        "source_frames_dir": str(frames_dir),
        "out_dir": str(video_possession_dir),
        "num_input_frames": len(frames),
        "num_possessions": len(segments),
        "state_counts": state_counts,
        "debug_video_dirs": {
            "player_ball": str(Path("outputs") / "debug_vid" / "player_ball"),
            "full_video_possession": str(debug_vid_dir / "full_video_possession"),
            "possession": str(debug_vid_dir / "possession"),
        },
        "settings": {
            **vars(args),
            "ignore_track_ids": sorted(list(ignore_track_ids)),
            "keep_track_ids": sorted(list(keep_track_ids)),
        },
        "state_meanings": {
            "controlled": "Confirmed possession after enough frames and contact/control evidence.",
            "pass": "Ball in transit. CTX owner may be shown, but it is not active possession.",
            "loose": "Ball visible/estimated but no player has enough control evidence.",
            "uncertain": "Insufficient information or ambiguous evidence.",
        },
        "debug_label_meanings": {
            "CTRL": "Confirmed active controller.",
            "CTX": "Contextual previous owner, not confirmed active possession.",
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
                "file": str(out_dir / prefix / f"{prefix}_possession_{s['possession_id']:04d}.json"),
            }
            for s in segments
        ],
    }

    write_json(summary_path, summary)
    save_debug_csv(debug_csv_path, frames)

    if args.make_debug_videos or args.make_full_debug_video:
        if video_path is None:
            print("[WARN] No --video provided and metadata has no video path; skipping debug videos.")
        else:
            make_debug_videos(
                video_path=video_path,
                frames=frames,
                segments=segments,
                debug_vid_dir=debug_vid_dir,
                prefix=prefix,
                fps=float(metadata.get("fps", 30.0)),
                max_debug_videos=args.max_debug_videos,
                make_full_debug_video=args.make_full_debug_video,
            )

    print(f"[DONE] input frames: {len(frames)}")
    print(f"[DONE] possessions: {len(segments)}")
    print(f"[DONE] possession files: {out_dir}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] debug csv: {debug_csv_path}")
    print(f"[DONE] states: {state_counts}")


if __name__ == "__main__":
    main()
