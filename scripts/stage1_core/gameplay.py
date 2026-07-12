from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .windows import merge_windows


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class GameplayConfig:
    accept_score: float = 0.52
    uncertain_score: float = 0.34
    green_floor: float = 0.16
    green_good: float = 0.48
    min_people: int = 3
    max_people: int = 26
    confident_people_good: int = 6
    closeup_height: float = 0.42
    large_total_area: float = 0.48
    min_x_span: float = 0.16
    reject_green_below: float = 0.06
    reject_closeup_height: float = 0.68


def score_gameplay(features: Dict[str, Any], config: GameplayConfig) -> Tuple[float, str, bool, Dict[str, float]]:
    green = float(features.get("green_ratio") or 0.0)
    people = int(features.get("person_count") or 0)
    confident = int(features.get("confident_person_count") or 0)
    height = float(features.get("largest_person_height_ratio") or 0.0)
    area = float(features.get("total_person_bbox_area_ratio") or 0.0)
    span = float(features.get("player_x_span_ratio") or 0.0)

    green_score = clamp((green - config.green_floor) / max(1e-6, config.green_good - config.green_floor), 0, 1)
    if config.min_people <= people <= config.max_people:
        people_score = 1.0
    elif people < config.min_people:
        people_score = clamp(people / max(1, config.min_people), 0, 1)
    else:
        people_score = clamp(1 - (people - config.max_people) / max(1, config.max_people), 0, 1)
    components = {
        "green": green_score,
        "people": people_score,
        "confident_people": clamp(confident / max(1, config.confident_people_good), 0, 1),
        "player_size": clamp(1 - height / max(1e-6, config.closeup_height), 0, 1),
        "horizontal_span": clamp((span - config.min_x_span) / max(1e-6, 0.65 - config.min_x_span), 0, 1),
        "box_area": clamp(1 - area / max(1e-6, config.large_total_area), 0, 1),
    }
    score = (
        0.42 * components["green"] + 0.20 * components["people"]
        + 0.10 * components["confident_people"] + 0.12 * components["player_size"]
        + 0.11 * components["horizontal_span"] + 0.05 * components["box_area"]
    )
    obvious_reject = green < config.reject_green_below or height >= config.reject_closeup_height
    if obvious_reject and score < config.uncertain_score:
        return score, "reject", False, components
    if score >= config.accept_score:
        return score, "gameplay", True, components
    if score >= config.uncertain_score:
        return score, "uncertain", True, components
    return score, "reject", False, components


def bridge_gaps(flags: List[bool], max_gap: int) -> List[bool]:
    result = list(flags)
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= max_gap:
            result[start:index] = [True] * (index - start)
    return result


def windows_from_samples(
    samples: List[Dict[str, Any]],
    sample_period_sec: float,
    min_duration_sec: float,
    bridge_gap_sec: float,
    padding_sec: float,
    video_duration_sec: float,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    flags = bridge_gaps(
        [bool(sample["keep_for_stage_1B"]) for sample in samples],
        max(0, int(round(bridge_gap_sec / max(sample_period_sec, 1e-6)))),
    )
    raw = []
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(flags) and flags[index + 1]:
            index += 1
        end = index
        start_sec = float(samples[start]["time_sec"])
        end_sec = float(samples[end]["time_sec"] + sample_period_sec)
        if end_sec - start_sec >= min_duration_sec:
            raw.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_source_frame": int(samples[start]["frame_idx"]),
                "end_source_frame": int(samples[end]["frame_idx"]),
                "sample_count": end - start + 1,
            })
        index += 1
    return merge_windows(raw, 0.0, padding_sec, video_duration_sec)
