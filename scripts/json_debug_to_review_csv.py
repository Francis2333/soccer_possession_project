"""
Convert Stage 1A / Stage 1B JSON debug outputs into a CSV template for manual review.

Examples
--------
One Stage 1B run:
python scripts/json_debug_to_review_csv.py ^
  --input "video_1=outputs/json_stage_1B/chelsea_burnley_1B" ^
  --output "outputs/review/video_1_review.csv"

Compare several parameter settings side by side:
python scripts/json_debug_to_review_csv.py ^
  --input "m960=outputs/json_stage_1B/run_m_960" ^
  --input "l960=outputs/json_stage_1B/run_l_960" ^
  --input "m1280=outputs/json_stage_1B/run_m_1280" ^
  --input "x1920=outputs/json_stage_1B/run_x_1920" ^
  --output "outputs/review/player_detection_comparison.csv" ^
  --review_fps 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path}, line {line_number}: {error}"
                ) from error
            if isinstance(value, dict):
                records.append(value)
    return records


def find_records(input_path: Path) -> List[Dict[str, Any]]:
    """Accept a Stage 1A JSONL file, one Stage 1B JSON file, or a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            return load_jsonl(input_path)
        if input_path.suffix.lower() == ".json":
            return [read_json(input_path)]
        raise ValueError(f"Unsupported file type: {input_path}")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    # Stage 1A output directory.
    jsonl_files = sorted(input_path.glob("*_stage_1A_observations.jsonl"))
    if jsonl_files:
        if len(jsonl_files) > 1:
            raise ValueError(
                f"Multiple Stage 1A observation files found in {input_path}; "
                "pass the desired JSONL file directly."
            )
        return load_jsonl(jsonl_files[0])

    # Revised Stage 1B uses flat frame_*.json files. rglob also supports older
    # versions that stored frames inside window_* subdirectories.
    frame_files = sorted(
        input_path.rglob("frame_*.json"),
        key=lambda path: (
            int(read_json(path).get("frame_idx", 0)),
            str(path),
        ),
    )
    return [read_json(path) for path in frame_files]


def infer_fps(records: List[Dict[str, Any]]) -> Optional[float]:
    for record in records:
        value = record.get("analysis_fps")
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)

    times = [
        float(record["time_sec"])
        for record in records
        if isinstance(record.get("time_sec"), (int, float))
    ]
    positive_deltas = [
        later - earlier
        for earlier, later in zip(times, times[1:])
        if later > earlier and later - earlier < 2.0
    ]
    if positive_deltas:
        positive_deltas.sort()
        median_delta = positive_deltas[len(positive_deltas) // 2]
        if median_delta > 0:
            return 1.0 / median_delta
    return None


def automatic_values(record: Dict[str, Any]) -> Tuple[int, Optional[bool]]:
    players = record.get("players")
    players_identified = len(players) if isinstance(players, list) else 0

    # Stage 1B has ball / ball_candidates. Stage 1A intentionally has no ball model.
    if "ball" in record:
        ball_detected: Optional[bool] = record.get("ball") is not None
    elif "ball_candidates" in record:
        candidates = record.get("ball_candidates")
        ball_detected = isinstance(candidates, list) and len(candidates) > 0
    else:
        ball_detected = None

    return players_identified, ball_detected


def parse_named_input(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.stem, path

    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing name before '=' in --input {text!r}")
    return name, Path(raw_path.strip())


def bool_for_csv(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "TRUE" if value else "FALSE"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Excel-friendly CSV for manually reviewing Stage 1A/1B "
            "player and ball detections."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Input Stage 1A JSONL/file directory or Stage 1B directory. "
            "Repeat this argument to compare several runs side by side."
        ),
    )
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    parser.add_argument(
        "--review_fps",
        type=float,
        default=None,
        help=(
            "FPS of the debug video used for manual review. For example, use 5 "
            "to produce 0.0, 0.2, 0.4... If omitted, the script infers it."
        ),
    )
    args = parser.parse_args()

    datasets: List[Tuple[str, List[Dict[str, Any]], Optional[float]]] = []
    used_names = set()

    for raw_input in args.input:
        name, path = parse_named_input(raw_input)
        if name in used_names:
            raise ValueError(f"Duplicate input name: {name}")
        used_names.add(name)

        records = find_records(path)
        if not records:
            raise ValueError(f"No frame records found in {path}")

        records.sort(
            key=lambda record: (
                int(record.get("frame_idx", record.get("source_frame_idx", 0))),
                float(record.get("time_sec", 0.0)),
            )
        )
        datasets.append((name, records, infer_fps(records)))

    review_fps = args.review_fps
    if review_fps is None:
        inferred = [fps for _, _, fps in datasets if fps is not None]
        review_fps = inferred[0] if inferred else None

    if review_fps is None or review_fps <= 0:
        raise ValueError(
            "Could not infer the review FPS. Supply it explicitly, e.g. --review_fps 5"
        )

    max_rows = max(len(records) for _, records, _ in datasets)

    # The manually filled fields are intentionally written as empty strings:
    # actual_players, false_players, and ball_visible.
    headers = ["review_time_sec"]
    for name, _, _ in datasets:
        headers.extend(
            [
                f"{name}_actual_players",
                f"{name}_players_identified",
                f"{name}_false_players",
                f"{name}_ball_visible",
                f"{name}_ball_detected",
                f"{name}_source_time_sec",
                f"{name}_source_frame_idx",
            ]
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row_index in range(max_rows):
            row: Dict[str, Any] = {
                "review_time_sec": round(row_index / review_fps, 6)
            }

            for name, records, _ in datasets:
                if row_index >= len(records):
                    continue

                record = records[row_index]
                identified, ball_detected = automatic_values(record)

                row[f"{name}_actual_players"] = ""
                row[f"{name}_players_identified"] = identified
                row[f"{name}_false_players"] = ""
                row[f"{name}_ball_visible"] = ""
                row[f"{name}_ball_detected"] = bool_for_csv(ball_detected)
                row[f"{name}_source_time_sec"] = record.get("time_sec", "")
                row[f"{name}_source_frame_idx"] = record.get(
                    "frame_idx", record.get("source_frame_idx", "")
                )

            writer.writerow(row)

    print(f"[DONE] CSV written to: {output_path}")
    print(f"[DONE] Review rows: {max_rows}")
    print(f"[DONE] Review FPS: {review_fps:g}")
    print(
        "[INFO] Fill actual_players, false_players, and ball_visible manually. "
        "players_identified and ball_detected come from the JSON."
    )


if __name__ == "__main__":
    main()
