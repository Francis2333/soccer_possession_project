from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .io import validate_video_path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps if self.total_frames > 0 else 0.0


def probe_video(path: Path) -> VideoInfo:
    validate_video_path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")
    info = VideoInfo(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    cap.release()
    return info


def frame_step(native_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        return 1
    return max(1, int(round(native_fps / target_fps)))


def full_video_window(info: VideoInfo) -> Dict[str, object]:
    return {
        "window_id": 0,
        "start_sec": 0.0,
        "end_sec": info.duration_sec,
        "duration_sec": info.duration_sec,
        "source_intervals": [{"source_format": "full_video"}],
    }


def iter_window_frames(
    video_path: Path,
    windows: List[Dict[str, object]],
    step: int,
) -> Iterator[Tuple[int, int, int, np.ndarray]]:
    """Yield window_id, frame_idx, window_start_frame, frame.

    OpenCV seeks once at each window boundary, then decodes sequentially inside
    the window. Canonical windows should be sorted and non-overlapping.
    """
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    try:
        for window in windows:
            window_id = int(window["window_id"])
            start = max(0, int(float(window["start_sec"]) * info.fps))
            end = int(np.ceil(float(window["end_sec"]) * info.fps)) - 1
            if info.total_frames > 0:
                end = min(end, info.total_frames - 1)
            if end < start:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frame_idx = start
            while frame_idx <= end:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame_idx - start) % step == 0:
                    yield window_id, frame_idx, start, frame
                frame_idx += 1
    finally:
        cap.release()


def export_window_clips(
    video_path: Path,
    windows: List[Dict[str, object]],
    out_dir: Path,
    codec: str = "mp4v",
) -> List[Dict[str, object]]:
    info = probe_video(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Dict[str, object]] = []

    for window in windows:
        window_id = int(window["window_id"])
        out_path = out_dir / f"active_{window_id:04d}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*codec), info.fps, (info.width, info.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create clip: {out_path}")

        frames_written = 0
        try:
            for _, _, _, frame in iter_window_frames(video_path, [window], step=1):
                writer.write(frame)
                frames_written += 1
        finally:
            writer.release()

        outputs.append({
            "window_id": window_id,
            "path": str(out_path),
            "frames_written": frames_written,
        })
    return outputs
