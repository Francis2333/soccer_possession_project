from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str
    confidence: float
    image_size: int
    iou: float
    device: str
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    fallback_class_id: Optional[int] = None
    half: bool = False
    tracker: Optional[str] = None


def bbox_size(box: Sequence[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = map(float, box)
    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return {"w": width, "h": height, "area": width * height}


def resolve_class_ids(model: Any, config: DetectorConfig) -> Optional[List[int]]:
    if config.class_id is not None:
        return [int(config.class_id)]
    if config.class_name:
        wanted = config.class_name.strip().lower()
        names = getattr(model, "names", None) or {}
        found = [int(k) for k, value in names.items() if str(value).lower() == wanted]
        if found:
            return found
        print(f"[WARN] Class name not found: {config.class_name}")
    if config.fallback_class_id is not None:
        return [int(config.fallback_class_id)]
    return None


def extract_detections(
    result: Any,
    class_ids: Optional[List[int]],
    min_conf: float,
    include_track_id: bool,
) -> List[Dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    classes = (
        boxes.cls.cpu().numpy().astype(int)
        if boxes.cls is not None else np.full(len(xyxy), -1)
    )
    ids = None
    if include_track_id and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)

    names = getattr(result, "names", {}) or {}
    output: List[Dict[str, Any]] = []
    for index, values in enumerate(xyxy):
        class_id, confidence = int(classes[index]), float(conf[index])
        if confidence < min_conf or (class_ids is not None and class_id not in class_ids):
            continue
        box = [float(value) for value in values]
        x1, y1, x2, y2 = box
        item: Dict[str, Any] = {
            "det_index": index,
            "class_id": class_id,
            "class_name": str(names.get(class_id, class_id)),
            "bbox_xyxy": box,
            "bbox_center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            "bottom_center": [(x1 + x2) / 2.0, y2],
            "bbox_size": bbox_size(box),
            "conf": confidence,
        }
        if include_track_id:
            item["track_id"] = int(ids[index]) if ids is not None else None
        output.append(item)
    output.sort(key=lambda item: (item.get("track_id") is None, -float(item["conf"])))
    return output


class YoloDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = self._load_model()
        self.class_ids = resolve_class_ids(self.model, config)

    def _load_model(self) -> Any:
        from ultralytics import YOLO
        return YOLO(self.config.model_path)

    def reset(self) -> None:
        self.model = self._load_model()
        self.class_ids = resolve_class_ids(self.model, self.config)

    def infer(self, frame: np.ndarray, mode: str = "predict") -> List[Dict[str, Any]]:
        kwargs = {
            "source": frame,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.image_size,
            "device": self.config.device,
            "classes": self.class_ids,
            "half": self.config.half,
            "verbose": False,
        }
        if mode == "track":
            if not self.config.tracker:
                raise ValueError("Tracking mode requires a tracker config.")
            results = self.model.track(
                **kwargs, persist=True, tracker=self.config.tracker
            )
            return extract_detections(results[0], self.class_ids, self.config.confidence, True)
        if mode != "predict":
            raise ValueError("mode must be 'predict' or 'track'.")
        results = self.model.predict(**kwargs)
        return extract_detections(results[0], self.class_ids, self.config.confidence, False)


def draw_detections(
    frame: np.ndarray,
    players: List[Dict[str, Any]],
    balls: List[Dict[str, Any]],
    label: str,
) -> np.ndarray:
    visual = frame.copy()
    for player in players:
        x1, y1, x2, y2 = map(int, player["bbox_xyxy"])
        track_id = player.get("track_id")
        text = f"P {player['conf']:.2f}" if track_id is None else f"ID {track_id} {player['conf']:.2f}"
        cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(visual, text, (x1, max(20, y1 - 6)), 0, 0.5, (0, 220, 0), 2)
    for index, ball in enumerate(balls):
        x1, y1, x2, y2 = map(int, ball["bbox_xyxy"])
        text = "BALL" if index == 0 else "ball?"
        cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(visual, f"{text} {ball['conf']:.2f}", (x1, max(20, y1 - 6)), 0, 0.5, (0, 0, 255), 2)
    cv2.rectangle(visual, (0, 0), (min(visual.shape[1], 1000), 38), (0, 0, 0), -1)
    cv2.putText(visual, label, (10, 27), 0, 0.62, (255, 255, 255), 2)
    return visual
