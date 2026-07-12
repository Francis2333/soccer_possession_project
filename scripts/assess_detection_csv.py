"""Assess player and ball detection performance from a manually annotated CSV.

Expected columns:
  time
  actual player
  <clip>_player_identified
  <clip>_0_player
  <clip>_ball_visible
  <clip>_ball_detected

Rows with blank actual-player values are ignored. Other blank cells are treated as 0.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PLAYER_IDENTIFIED_SUFFIXES = ("_player_identified", "_players_identified")
FALSE_PLAYER_SUFFIXES = ("_0_player", "_false_player", "_false_players", "_player_misidentified")
BALL_VISIBLE_SUFFIXES = ("_ball_visible", "_actual_ball")
BALL_DETECTED_SUFFIXES = ("_ball_detected", "_ball_identified")


def normalize_header(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def parse_number(value: object, default: float = 0.0) -> float:
    if is_blank(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return 1.0
    if text in {"false", "no", "n"}:
        return 0.0
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value: {value!r}")
    return number


def parse_count(value: object, default: int = 0) -> int:
    return max(0, int(round(parse_number(value, float(default)))))


def parse_bool01(value: object, default: int = 0) -> int:
    return 1 if parse_number(value, float(default)) != 0 else 0


def safe_div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def find_actual_player_column(fieldnames: Iterable[str]) -> str:
    lookup = {normalize_header(name): name for name in fieldnames}
    for candidate in ("actual_player", "actual_players", "actual_number_of_players", "actual_people"):
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError("Could not find 'actual player' or 'actual_players' column.")


def match_suffix(name: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return None


def discover_clips(fieldnames: Iterable[str]) -> Dict[str, Dict[str, str]]:
    clips: Dict[str, Dict[str, str]] = {}
    for original in fieldnames:
        normalized = normalize_header(original)
        for metric, suffixes in (
            ("player_identified", PLAYER_IDENTIFIED_SUFFIXES),
            ("false_player", FALSE_PLAYER_SUFFIXES),
            ("ball_visible", BALL_VISIBLE_SUFFIXES),
            ("ball_detected", BALL_DETECTED_SUFFIXES),
        ):
            prefix = match_suffix(normalized, suffixes)
            if prefix:
                clips.setdefault(prefix, {})[metric] = original
                break
    clips = {name: cols for name, cols in clips.items() if "player_identified" in cols}
    if not clips:
        raise ValueError("No clip groups found from headers.")
    return clips


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        actual_col = find_actual_player_column(reader.fieldnames)
        clips = discover_clips(reader.fieldnames)
        rows = list(reader)

    summaries: List[Dict[str, object]] = []

    for clip_name, cols in clips.items():
        evaluated = total_actual = total_boxes = total_false = 0
        total_correct = total_missed = total_abs_error = 0
        exact = under = over = 0
        tp = fp = tn = fn = 0

        for row in rows:
            if is_blank(row.get(actual_col)):
                continue
            evaluated += 1
            actual = parse_count(row.get(actual_col))
            identified = parse_count(row.get(cols["player_identified"], 0))
            false_players = min(parse_count(row.get(cols.get("false_player", ""), 0)), identified)
            raw_correct = max(identified - false_players, 0)
            matched_correct = min(raw_correct, actual)
            missed = max(actual - matched_correct, 0)

            total_actual += actual
            total_boxes += identified
            total_false += false_players
            total_correct += matched_correct
            total_missed += missed

            error = raw_correct - actual
            total_abs_error += abs(error)
            if error == 0:
                exact += 1
            elif error < 0:
                under += 1
            else:
                over += 1

            visible = parse_bool01(row.get(cols.get("ball_visible", ""), 0))
            detected = parse_bool01(row.get(cols.get("ball_detected", ""), 0))
            if visible and detected:
                tp += 1
            elif not visible and detected:
                fp += 1
            elif not visible and not detected:
                tn += 1
            else:
                fn += 1

        player_precision = safe_div(total_correct, total_correct + total_false)
        player_recall = safe_div(total_correct, total_actual)
        player_f1 = None
        if player_precision is not None and player_recall is not None and player_precision + player_recall:
            player_f1 = 2 * player_precision * player_recall / (player_precision + player_recall)

        ball_precision = safe_div(tp, tp + fp)
        ball_recall = safe_div(tp, tp + fn)
        ball_f1 = None
        if ball_precision is not None and ball_recall is not None and ball_precision + ball_recall:
            ball_f1 = 2 * ball_precision * ball_recall / (ball_precision + ball_recall)

        summaries.append({
            "clip_name": clip_name,
            "evaluated_rows": evaluated,
            "total_actual_players": total_actual,
            "total_detected_boxes": total_boxes,
            "total_correct_players": total_correct,
            "total_false_players": total_false,
            "total_missed_players": total_missed,
            "player_precision": fmt(player_precision),
            "player_recall": fmt(player_recall),
            "player_f1": fmt(player_f1),
            "mean_absolute_player_count_error": fmt(safe_div(total_abs_error, evaluated)),
            "exact_player_count_rate": fmt(safe_div(exact, evaluated)),
            "undercount_frame_rate": fmt(safe_div(under, evaluated)),
            "overcount_frame_rate": fmt(safe_div(over, evaluated)),
            "ball_true_positive": tp,
            "ball_false_positive": fp,
            "ball_true_negative": tn,
            "ball_false_negative": fn,
            "ball_precision": fmt(ball_precision),
            "ball_recall": fmt(ball_recall),
            "ball_f1": fmt(ball_f1),
            "ball_accuracy": fmt(safe_div(tp + tn, tp + fp + tn + fn)),
        })

    def rank_key(s: Dict[str, object]):
        f1 = float(s["player_f1"]) if s["player_f1"] != "" else -1.0
        recall = float(s["player_recall"]) if s["player_recall"] != "" else -1.0
        mae = float(s["mean_absolute_player_count_error"]) if s["mean_absolute_player_count_error"] != "" else float("inf")
        return (-f1, -recall, mae)

    summaries.sort(key=rank_key)
    for rank, summary in enumerate(summaries, start=1):
        summary["rank_by_player_f1"] = rank

    fieldnames = [
        "rank_by_player_f1", "clip_name", "evaluated_rows",
        "total_actual_players", "total_detected_boxes", "total_correct_players",
        "total_false_players", "total_missed_players", "player_precision",
        "player_recall", "player_f1", "mean_absolute_player_count_error",
        "exact_player_count_rate", "undercount_frame_rate", "overcount_frame_rate",
        "ball_true_positive", "ball_false_positive", "ball_true_negative",
        "ball_false_negative", "ball_precision", "ball_recall", "ball_f1",
        "ball_accuracy",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"[DONE] Assessment written to: {output_path}")
    print(f"[DONE] Configurations evaluated: {len(summaries)}")
    print(f"[INFO] Rows with blank '{actual_col}' were ignored.")
    print("[INFO] All other blank cells were treated as 0.")


if __name__ == "__main__":
    main()
