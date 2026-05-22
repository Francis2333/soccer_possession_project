from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/labeled_with_ids.mp4")
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            tracker="bytetrack.yaml",
            verbose=False
        )

        r = results[0]

        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            if r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
            else:
                ids = [-1] * len(boxes)

            for box, cls, conf, track_id in zip(boxes, classes, confs, ids):
                x1, y1, x2, y2 = map(int, box)

                if cls == PERSON_CLASS:
                    label = f"Player ID {track_id} {conf:.2f}"
                    color = (0, 255, 0)
                elif cls == SPORTS_BALL_CLASS:
                    label = f"Ball {conf:.2f}"
                    color = (0, 0, 255)
                else:
                    continue

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

        writer.write(frame)

        frame_i += 1
        if frame_i % 100 == 0:
            print(f"Processed {frame_i} frames")

    cap.release()
    writer.release()
    print(f"Saved labeled video with IDs to {args.output}")


if __name__ == "__main__":
    main()