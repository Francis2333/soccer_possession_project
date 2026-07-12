"""Stage 1C: attach calibration, pitch coordinates, and track identity metadata."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from stage1_core.enrichment import enrich_frame, indexed_records, nearest_record
from stage1_core.io import iter_frame_jsons, read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", required=True, help="Stage 1B output directory.")
    parser.add_argument("--out_dir", default="outputs/stage_1C")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument(
        "--calibration_json",
        default=None,
        help="Frame-indexed records. Homography keys are projected when present.",
    )
    parser.add_argument(
        "--identity_json",
        default=None,
        help="Track-indexed identity metadata such as team, role, jersey, or reID labels.",
    )
    parser.add_argument("--calibration_max_gap_frames", type=int, default=15)
    parser.add_argument("--copy_metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

    prefix = args.output_prefix or f"{frames_dir.name}_stage_1C"
    output_dir = Path(args.out_dir) / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(args.calibration_json) if args.calibration_json else None
    identity_path = Path(args.identity_json) if args.identity_json else None
    calibrations = indexed_records(calibration_path, "frame_idx")
    identities = indexed_records(identity_path, "track_id")

    processed, projected = 0, 0
    for path in iter_frame_jsons(frames_dir):
        frame = read_json(path)
        frame_idx = int(frame.get("frame_idx", 0))
        calibration = nearest_record(
            calibrations, frame_idx, args.calibration_max_gap_frames
        )
        enrich_frame(frame, calibration, identities)
        if any(player.get("pitch_position") for player in frame.get("players") or []):
            projected += 1
        write_json(output_dir / path.name, frame)
        processed += 1

    source_metadata = frames_dir / "_metadata.json"
    metadata: Dict[str, Any] = {
        "stage": "stage_1C_coordinate_and_identity_enrichment",
        "schema_version": 1,
        "source_frames_dir": str(frames_dir),
        "calibration_json": str(calibration_path) if calibration_path else None,
        "identity_json": str(identity_path) if identity_path else None,
        "calibration_records": len(calibrations),
        "identity_records": len(identities),
        "frames_processed": processed,
        "frames_with_projected_players": projected,
        "notes": [
            "Raw line-segment labels are attached but are not treated as a homography.",
            "Pitch projection requires a 3x3 image-to-pitch or pitch-to-image homography.",
        ],
    }
    if args.copy_metadata and source_metadata.exists():
        shutil.copy2(source_metadata, output_dir / "_stage_1B_metadata.json")
    write_json(output_dir / "_metadata.json", metadata)
    print(f"[DONE] Enriched frames: {processed}")
    print(f"[DONE] Frames with pitch projection: {projected}")
    print(f"[DONE] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
