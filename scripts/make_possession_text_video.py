import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def safe_val(v, default="None"):
    if v is None:
        return default
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return default
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def to_float(v):
    try:
        s = safe_val(v, default="")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def load_csv_rows(csv_path):
    rows_by_frame = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = row.get("frame_idx")
            if frame_idx is None:
                continue
            try:
                frame_idx = int(float(frame_idx))
            except Exception:
                continue
            rows_by_frame[frame_idx] = row

    return rows_by_frame


def load_possession_segments(possession_jsons):
    frame_to_segment = {}
    segments = []

    for p in possession_jsons:
        p = Path(p)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = {
            "possession_id": data.get("possession_id"),
            "player_track_id": data.get("player_track_id"),
            "start_frame": data.get("start_frame"),
            "end_frame": data.get("end_frame"),
            "clip_start_frame": data.get("clip_start_frame"),
            "clip_end_frame": data.get("clip_end_frame"),
            "file": str(p),
        }
        segments.append(meta)

        start = meta["start_frame"]
        end = meta["end_frame"]

        if start is not None and end is not None:
            for frame_idx in range(int(start), int(end) + 1):
                frame_to_segment[frame_idx] = meta

    return frame_to_segment, segments


def draw_text_panel(row, frame_idx, panel_w, panel_h, frame_to_segment, summary=None):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.55
    normal = 0.65
    line_h = 28
    x = 24
    y = 36

    white = (235, 235, 235)
    gray = (170, 170, 170)
    green = (120, 255, 120)
    yellow = (80, 220, 255)
    red = (80, 80, 255)
    cyan = (255, 220, 120)

    state = safe_val(row.get("state"))
    ctrl_owner = safe_val(row.get("controlled_owner_track_id"))
    ctx_owner = safe_val(row.get("context_owner_track_id"))
    cand_owner = safe_val(row.get("candidate_owner"))
    nearest = safe_val(row.get("nearest_track_id"))

    seg = frame_to_segment.get(int(frame_idx))

    if state == "controlled":
        state_color = green
    elif state == "pass":
        state_color = cyan
    elif state == "loose":
        state_color = yellow
    else:
        state_color = red

    def put(text="", color=white, scale=small, thickness=1):
        nonlocal y
        if y < panel_h - 20:
            cv2.putText(
                panel,
                str(text),
                (x, y),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        y += line_h

    put("POSSESSION DEBUG / PER-FRAME INFERENCE", green, normal, 2)
    put("=" * 48, gray)
    put(f"frame_idx: {frame_idx}", white, normal, 2)
    put(f"state: {state}", state_color, normal, 2)

    put()
    put("[OWNER]")
    put(f"controlled_owner_track_id: {ctrl_owner}", green if ctrl_owner != "None" else gray)
    put(f"context_owner_track_id:    {ctx_owner}", white)
    put(f"candidate_owner:          {cand_owner}", white)
    put(f"nearest_track_id:         {nearest}", white)

    put()
    put("[SCORES / DISTANCE]")
    put(f"candidate_score:          {safe_val(row.get('candidate_score'))}")
    put(f"candidate_distance:       {safe_val(row.get('candidate_distance'))}")
    put(f"candidate_norm_distance:  {safe_val(row.get('candidate_norm_distance'))}")
    put(f"nearest_distance:         {safe_val(row.get('nearest_distance'))}")
    put(f"distance_ratio_to_second: {safe_val(row.get('distance_ratio_to_second'))}")

    put()
    put("[BALL]")
    put(f"ball_x, ball_y:           {safe_val(row.get('ball_x'))}, {safe_val(row.get('ball_y'))}")
    put(f"ball_speed:               {safe_val(row.get('ball_speed'))}")
    put(f"ball_is_estimated:        {safe_val(row.get('ball_is_estimated'))}")
    put(f"ball_missing_gap:         {safe_val(row.get('ball_missing_gap'))}")

    put()
    put("[DECISION]")
    reason = safe_val(row.get("reason"))
    decision = safe_val(row.get("decision"))

    for label, text in [("reason", reason), ("decision", decision)]:
        put(f"{label}:")
        max_chars = 46
        text = str(text)
        while len(text) > max_chars:
            cut = text.rfind("_", 0, max_chars)
            if cut == -1:
                cut = max_chars
            put(f"  {text[:cut]}", gray)
            text = text[cut:].lstrip("_ ")
        put(f"  {text}", gray)

    put()
    put("[POSSESSION SEGMENT]")
    if seg is None:
        put("active_segment: None", gray)
    else:
        pid = seg["possession_id"]
        if pid is not None:
            put(f"active_segment: possession_{int(pid):04d}", green)
        else:
            put("active_segment: possession_unknown", green)
        put(f"segment_player_track_id: {seg['player_track_id']}")
        put(f"segment_range: {seg['start_frame']} -> {seg['end_frame']}")

    if summary:
        put()
        put("[SUMMARY]")
        put(f"num_input_frames: {summary.get('num_input_frames')}")
        put(f"num_possessions:  {summary.get('num_possessions')}")

    return panel


def make_video(
    video_path,
    csv_path,
    out_path,
    possession_jsons,
    summary_json=None,
    panel_width=760,
    draw_ball_on_original=True,
):
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    rows_by_frame = load_csv_rows(csv_path)
    frame_to_segment, segments = load_possession_segments(possession_jsons)

    summary = None
    if summary_json is not None:
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = original_w + panel_width
    out_h = original_h

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        row = rows_by_frame.get(frame_idx, {"frame_idx": frame_idx, "state": "missing_csv_row"})

        if draw_ball_on_original:
            bx = to_float(row.get("ball_x"))
            by = to_float(row.get("ball_y"))
            if bx is not None and by is not None:
                bx, by = int(round(bx)), int(round(by))
                cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)
                cv2.circle(frame, (bx, by), 12, (0, 0, 0), 2)

        state = safe_val(row.get("state"))
        ctrl_owner = safe_val(row.get("controlled_owner_track_id"))
        ctx_owner = safe_val(row.get("context_owner_track_id"))

        label = f"frame {frame_idx} | {state}"
        if ctrl_owner != "None":
            label += f" | CTRL {ctrl_owner}"
        elif ctx_owner != "None":
            label += f" | CTX {ctx_owner}"

        cv2.rectangle(frame, (20, 20), (900, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        panel = draw_text_panel(
            row=row,
            frame_idx=frame_idx,
            panel_w=panel_width,
            panel_h=original_h,
            frame_to_segment=frame_to_segment,
            summary=summary,
        )

        combined = np.concatenate([frame, panel], axis=1)
        writer.write(combined)

        frame_idx += 1

    cap.release()
    writer.release()

    print("[DONE]")
    print(f"Saved: {out_path}")
    print(f"Video frames read: {frame_idx}")
    print(f"CSV rows: {len(rows_by_frame)}")
    print(f"Original video reported frames: {total_video_frames}")
    print(f"FPS: {fps}")
    print(f"Output size: {out_w}x{out_h}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--possession_jsons",
        nargs="+",
        required=True,
    )

    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--panel_width", type=int, default=760)
    parser.add_argument("--no_ball_dot", action="store_true")

    args = parser.parse_args()

    make_video(
        video_path=args.video,
        csv_path=args.csv,
        out_path=args.out,
        possession_jsons=args.possession_jsons,
        summary_json=args.summary_json,
        panel_width=args.panel_width,
        draw_ball_on_original=not args.no_ball_dot,
    )


if __name__ == "__main__":
    main()