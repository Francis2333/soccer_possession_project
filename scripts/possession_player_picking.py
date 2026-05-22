from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import json
from pathlib import Path


# COCO class IDs
PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


def center(box):
    """Return center point of a bounding box."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)


def box_height(box):
    """Return height of a bounding box."""
    return max(1.0, float(box[3] - box[1]))


def box_area(box):
    """Return area of a bounding box."""
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def make_crop_box(cx, cy, crop_w, crop_h, frame_w, frame_h):
    """
    Create a fixed-aspect crop box centered at (cx, cy).
    Keeps the crop inside the image boundary.
    """
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = int(cx + crop_w / 2)
    y2 = int(cy + crop_h / 2)

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        x1 -= x2 - frame_w
        x2 = frame_w
    if y2 > frame_h:
        y1 -= y2 - frame_h
        y2 = frame_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    return x1, y1, x2, y2


def choose_ball(ball_boxes):
    """
    Choose one ball detection.
    For now, use highest confidence.
    Each ball item: {"box": ..., "conf": ...}
    """
    if len(ball_boxes) == 0:
        return None

    return max(ball_boxes, key=lambda b: b["conf"])


def draw_original_debug(frame, players, ball_box, current_owner):
    """Draw player boxes, IDs, keypoints, ball, and current possessor on original frame."""
    for p in players:
        x1, y1, x2, y2 = map(int, p["box"])
        pid = p["id"]

        if pid == current_owner:
            color = (0, 255, 255)
            label = f"POSSESSOR ID {pid}"
        else:
            color = (0, 255, 0)
            label = f"ID {pid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Draw keypoints if available
        if p["keypoints"] is not None:
            for kp in p["keypoints"]:
                x, y, conf = kp
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    if ball_box is not None:
        x1, y1, x2, y2 = map(int, ball_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        bc = center(ball_box)
        cv2.circle(frame, (int(bc[0]), int(bc[1])), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "BALL",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output_video", default="outputs/possession_crops.mp4")
    parser.add_argument("--debug_video", default="outputs/debug_possession.mp4")
    parser.add_argument("--crop_dir", default="outputs/crops")
    parser.add_argument("--json_dir", default="outputs/json")

    # Pose model gives player boxes + keypoints
    parser.add_argument("--pose_model", default="yolov8x-pose.pt")

    # Ball model gives ball detections
    parser.add_argument("--ball_model", default="yolov8x.pt")

    # Separate thresholds: player should be cleaner, ball should be more sensitive
    parser.add_argument("--player_conf", type=float, default=0.25)
    parser.add_argument("--ball_conf", type=float, default=0.10)

    # Crop aspect ratio, default 4:3
    parser.add_argument("--aspect_w", type=int, default=4)
    parser.add_argument("--aspect_h", type=int, default=3)

    # If ball is within this many player-heights, player can own possession
    parser.add_argument("--proximity_height", type=float, default=2.0)

    # New player must win possession for this many frames before official switch
    parser.add_argument("--switch_frames", type=int, default=6)

    # How long to keep using last known ball position when ball disappears
    parser.add_argument("--max_ball_missing", type=int, default=15)

    # Crop height = player height * crop_scale
    parser.add_argument("--crop_scale", type=float, default=2.2)

    args = parser.parse_args()

    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.debug_video).parent.mkdir(parents=True, exist_ok=True)
    Path(args.crop_dir).mkdir(parents=True, exist_ok=True)
    Path(args.json_dir).mkdir(parents=True, exist_ok=True)

    pose_model = YOLO(args.pose_model)
    ball_model = YOLO(args.ball_model)

    cap = cv2.VideoCapture(args.input)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h_out = 720
    crop_w_out = int(crop_h_out * args.aspect_w / args.aspect_h)

    crop_writer = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_w_out, crop_h_out)
    )

    debug_writer = cv2.VideoWriter(
        args.debug_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h)
    )

    # Possession memory
    current_owner = None
    candidate_owner = None
    candidate_count = 0

    # Ball memory
    last_ball_box = None
    frames_since_ball_seen = 999

    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ------------------------------------------------------------
        # 1. Player detection + tracking + pose keypoints
        # ------------------------------------------------------------
        pose_results = pose_model.track(
            frame,
            persist=True,
            conf=args.player_conf,
            tracker="bytetrack.yaml",
            verbose=False
        )

        pose_r = pose_results[0]

        players = []

        if pose_r.boxes is not None:
            player_boxes = pose_r.boxes.xyxy.cpu().numpy()
            player_classes = pose_r.boxes.cls.cpu().numpy().astype(int)

            if pose_r.boxes.id is not None:
                player_ids = pose_r.boxes.id.cpu().numpy().astype(int)
            else:
                player_ids = np.array([-1] * len(player_boxes))

            # Keypoints: [num_people, 17, 2]
            if pose_r.keypoints is not None:
                kxy = pose_r.keypoints.xy.cpu().numpy()
                kconf = pose_r.keypoints.conf.cpu().numpy()
            else:
                kxy = None
                kconf = None

            for idx, (box, cls, tid) in enumerate(zip(player_boxes, player_classes, player_ids)):
                if cls != PERSON_CLASS:
                    continue

                keypoints = None

                if kxy is not None and idx < len(kxy):
                    # Store keypoints as [x, y, confidence]
                    keypoints = []
                    for j in range(kxy.shape[1]):
                        x, y = kxy[idx, j]
                        c = kconf[idx, j] if kconf is not None else 1.0
                        keypoints.append([float(x), float(y), float(c)])

                players.append({
                    "id": int(tid),
                    "box": box,
                    "keypoints": keypoints
                })

        # ------------------------------------------------------------
        # 2. Ball detection with lower confidence
        # ------------------------------------------------------------
        ball_results = ball_model.predict(
            frame,
            conf=args.ball_conf,
            verbose=False
        )

        ball_r = ball_results[0]
        balls = []

        if ball_r.boxes is not None:
            boxes = ball_r.boxes.xyxy.cpu().numpy()
            classes = ball_r.boxes.cls.cpu().numpy().astype(int)
            confs = ball_r.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                if cls == SPORTS_BALL_CLASS:
                    balls.append({
                        "box": box,
                        "conf": float(conf)
                    })

        chosen_ball = choose_ball(balls)

        if chosen_ball is not None:
            # Ball detected this frame
            ball_box = chosen_ball["box"]
            last_ball_box = ball_box
            frames_since_ball_seen = 0
            ball_status = "detected"
        else:
            # Ball missed this frame
            frames_since_ball_seen += 1

            if last_ball_box is not None and frames_since_ball_seen <= args.max_ball_missing:
                # Temporarily reuse last known ball location
                ball_box = last_ball_box
                ball_status = "last_known"
            else:
                # Ball missing for too long
                ball_box = None
                ball_status = "missing"

        # ------------------------------------------------------------
        # 3. Instant possession by nearest player to ball
        # ------------------------------------------------------------
        instant_owner = None

        if ball_box is not None and len(players) > 0:
            ball_c = center(ball_box)

            best_player = None
            best_dist = float("inf")

            for p in players:
                p_box = p["box"]
                p_c = center(p_box)

                dist = np.linalg.norm(ball_c - p_c)

                # Dynamic threshold: taller/larger player allows larger distance
                allowed_dist = args.proximity_height * box_height(p_box)

                if dist < best_dist and dist <= allowed_dist:
                    best_dist = dist
                    best_player = p

            if best_player is not None:
                instant_owner = best_player["id"]

        # ------------------------------------------------------------
        # 4. Smooth possession over time
        # ------------------------------------------------------------
        if current_owner is None:
            current_owner = instant_owner
            candidate_owner = None
            candidate_count = 0

        elif instant_owner is None:
            # If ball is missing, do NOT immediately erase possession.
            # Keep previous owner unless ball missing for too long.
            if ball_status == "missing":
                pass

        elif instant_owner == current_owner:
            candidate_owner = None
            candidate_count = 0

        else:
            # Another player appears to own possession.
            # Require several consecutive frames before switching.
            if candidate_owner == instant_owner:
                candidate_count += 1
            else:
                candidate_owner = instant_owner
                candidate_count = 1

            if candidate_count >= args.switch_frames:
                current_owner = candidate_owner
                candidate_owner = None
                candidate_count = 0

        # ------------------------------------------------------------
        # 5. Choose crop target
        # ------------------------------------------------------------
        target_player = None

        if current_owner is not None:
            for p in players:
                if p["id"] == current_owner:
                    target_player = p
                    break

        if target_player is not None:
            target_box = target_player["box"]
            cx, cy = center(target_box)

            crop_h = box_height(target_box) * args.crop_scale
            crop_w = crop_h * args.aspect_w / args.aspect_h

        elif ball_box is not None:
            # Fallback: no current player owner, crop around ball
            target_box = ball_box
            cx, cy = center(target_box)

            crop_h = frame_h * 0.30
            crop_w = crop_h * args.aspect_w / args.aspect_h

        else:
            target_box = None

        # ------------------------------------------------------------
        # 6. Save crop video + crop images + JSON metadata
        # ------------------------------------------------------------
        if target_box is not None:
            x1, y1, x2, y2 = make_crop_box(
                cx, cy, crop_w, crop_h, frame_w, frame_h
            )

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                crop = cv2.resize(crop, (crop_w_out, crop_h_out))

                cv2.putText(
                    crop,
                    f"Owner: {current_owner} | Ball: {ball_status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                crop_writer.write(crop)

                crop_path = f"{args.crop_dir}/frame_{frame_i:06d}_owner_{current_owner}.jpg"
                cv2.imwrite(crop_path, crop)

        metadata = {
            "frame": frame_i,
            "current_owner": current_owner,
            "instant_owner": instant_owner,
            "candidate_owner": candidate_owner,
            "candidate_count": candidate_count,
            "ball_status": ball_status,
            "ball_box": ball_box.tolist() if ball_box is not None else None,
            "players": []
        }

        for p in players:
            metadata["players"].append({
                "id": p["id"],
                "box": p["box"].tolist(),
                "keypoints": p["keypoints"]
            })

        json_path = f"{args.json_dir}/frame_{frame_i:06d}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # ------------------------------------------------------------
        # 7. Save debug full-frame video
        # ------------------------------------------------------------
        debug_frame = frame.copy()
        draw_original_debug(debug_frame, players, ball_box, current_owner)

        cv2.putText(
            debug_frame,
            f"Frame {frame_i} | Ball: {ball_status} | Owner: {current_owner}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        debug_writer.write(debug_frame)

        frame_i += 1

        if frame_i % 100 == 0:
            print(f"Processed {frame_i} frames")

    cap.release()
    crop_writer.release()
    debug_writer.release()

    print(f"Saved crop video to: {args.output_video}")
    print(f"Saved debug video to: {args.debug_video}")
    print(f"Saved crops to: {args.crop_dir}")
    print(f"Saved JSON metadata to: {args.json_dir}")


if __name__ == "__main__":
    main()