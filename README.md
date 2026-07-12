# Reconstructed Stage 1 Pipeline

The new pipeline separates **window selection**, **object extraction**, and
**optional enrichment**. Every Python file is kept below 500 lines.

## Layout

```text
stage_1A_replay.py       SoccerNet Labels-cameras.json -> active gameplay windows
stage_1A_processing.py   Model-driven fallback when labels are unavailable
stage_1B.py              Player/ball detection or tracking inside windows
stage_1C.py              Calibration, pitch-coordinate, and identity enrichment
stage1_core/             Shared short modules
```

No Stage 1D is needed yet. Stage 1C remains small because calibration parsing and
projection live in `stage1_core/enrichment.py`.

## Canonical windows

Both Stage 1A variants produce:

```json
{
  "gameplay_windows": [
    {"window_id": 0, "start_sec": 12.0, "end_sec": 28.0}
  ]
}
```

Times always refer to the original video. Optional exported clips do not replace
these source timestamps.

## 1A: SoccerNet labels

```bash
python stage_1A_replay.py \
  --video "/path/to/1_720p.mkv" \
  --labels "/path/to/Labels-cameras.json" \
  --out_dir outputs/stage_1A_replay
```

Add `--write_clips` to export each active interval as a separate clip. The default
keeps real-time `Main camera*` and `Main behind the goal*` shots. The default
`--label_applies_to before` matches the original pipeline's interpretation of
SoccerNet camera-boundary labels; change it only after inspecting your labels.

## 1A: model-driven fallback

```bash
python stage_1A_processing.py \
  --video "/path/to/video.mkv" \
  --analysis_fps 2
```

This is intentionally a fallback. When SoccerNet labels exist, use the label-driven
version so detection quality is not mixed with gameplay-window classification.

## 1B: extraction

```bash
python stage_1B.py \
  --video "/path/to/1_720p.mkv" \
  --windows_json "outputs/stage_1A_replay/1_720p_active/_gameplay_windows.json" \
  --player_mode track \
  --device 0
```

Important defaults:

- native FPS (`--target_fps 0`)
- player `imgsz=1920`
- ball `imgsz=1280`
- explicit IoU `0.70`
- FP16 disabled unless `--half` is passed

Use `--player_mode predict` for detector-only validation. Use `track` when the
possession stage requires IDs.

## 1C: enrichment

```bash
python stage_1C.py \
  --frames_dir "outputs/stage_1B/1_720p_stage_1B" \
  --calibration_json "/path/to/calibration.json" \
  --identity_json "/path/to/track_identity.json"
```

Stage 1C accepts generic frame-indexed calibration records and track-indexed identity
records. It projects points only when a 3x3 homography is explicitly provided under
one of these keys:

- `homography_image_to_pitch`
- `image_to_pitch_homography`
- `homography_pitch_to_image`
- `pitch_to_image_homography`

Raw SoccerNet line-segment labels are preserved but are not falsely interpreted as a
homography. A calibration model or official camera-parameter converter can be added
later without changing Stage 1B.

## Tests

```bash
python -m pytest tests
```
