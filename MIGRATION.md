# Migration from the current Stage 1 scripts

## Old responsibility map

- Old `stage_1A.py`: detection, scene features, gameplay scoring, window building,
  debug rendering, video reading, and CLI configuration in one file.
- Old `stage_1B.py`: SoccerNet parsing, window merging, YOLO extraction, tracking,
  scene features, temporal features, debug rendering, video reading, and output.

## New responsibility map

| New component | Responsibility |
|---|---|
| `stage_1A_replay.py` | Trust SoccerNet camera/replay annotations and output active intervals |
| `stage_1A_processing.py` | Coarse model-driven fallback when labels do not exist |
| `stage_1B.py` | Player and ball extraction; optional tracking |
| `stage_1C.py` | Calibration/identity attachment and optional pitch projection |
| `stage1_core/windows.py` | All interval parsing and merging |
| `stage1_core/detection.py` | One shared YOLO wrapper and output schema |
| `stage1_core/video.py` | Video probing, frame iteration, and optional clip export |
| `stage1_core/features.py` | Scene and temporal features |
| `stage1_core/enrichment.py` | Calibration and identity enrichment |

## Recommended migration sequence

1. Run `stage_1A_replay.py` on one SoccerNet half and inspect the windows JSON.
2. Run `stage_1B.py --player_mode predict` on a short window and compare raw
   boxes with the old extractor.
3. Repeat with `--player_mode track` and compare ID continuity separately.
4. Point the existing possession script at the new Stage 1B output folder.
5. Add Stage 1C only after the Stage 1B output is validated.

## Deliberate behavior changes

- Stage 1B defaults to native FPS, 1920 player inference size, explicit IoU 0.70,
  and full precision. These defaults favor reproducible detection validation.
- Detection-only validation is now explicit through `--player_mode predict`.
- SoccerNet label parsing is no longer duplicated inside Stage 1B.
- Stage 1C never assumes line annotations are camera homographies.
