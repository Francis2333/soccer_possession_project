# Project Log

Formal project updates start from **June 14, 2026**.

## 2026-06-14 — Current Progress

### Completed

* Built a YOLO-based pipeline for detecting players and the soccer ball from broadcast video.
* Saved per-frame detection results as JSON.
* Implemented initial possession estimation using player-ball distance, temporal continuity, and possession-state rules.
* Generated possession segments, summary JSON files, debug CSV files, player-ball crops, and debug videos.
* Added `run_pipeline.py` to automate the main processing stages.

### New Work

* Started integrating **BoT-SORT** for persistent player tracking.
* Updated the main pipeline and possession scripts to support tracking data.
* Added research papers related to:

  * DeepSORT and BoT-SORT;
  * player re-identification;
  * SoccerNet tracking;
  * ST-GCN and skeleton-based action recognition.

### Current Pipeline

```text
Broadcast video
→ player and ball detection
→ player tracking
→ possession estimation
→ possession segments
→ pose extraction
→ event summary or commentary
```

Detection and possession estimation are implemented. Tracking integration is in progress. Pose estimation and commentary are planned next.

### Current Problems

* Player identities are not yet fully stable across frames.
* Ball detection can fail during fast motion or occlusion.
* There is not yet a formally labeled evaluation set.
* Pose extraction and temporal models are not yet connected to the pipeline.

### Next Steps

* Finish BoT-SORT integration.
* Save and visualize persistent player track IDs.
* Bind possession decisions to player tracks.
* Create a small manually labeled evaluation set.
* Connect pose estimation to possession clips.
* Generate simple event summaries from possession results.
* Try out mass video
* Implement Pose-Estimation pipeline

## 2026-06-21 — Stage 1 Full-Match Feature Extraction

### Completed

* Reworked the data-extraction pipeline into a dedicated **Stage 1 frame observation and feature extraction script**.
* Added support for full-match `.mkv` and `.mp4` broadcast videos.
* Added configurable frame sampling using `--save_every_n_frames`, allowing low-rate full-match analysis such as approximately 2 FPS.
* Integrated YOLO26x player detection, ball detection, and optional BoT-SORT player tracking.
* Saved detailed per-frame JSON containing:

  * player and ball bounding boxes;
  * detection confidence;
  * player track IDs when available;
  * bounding-box centers, bottom centers, sizes, and areas;
  * primary and alternative ball candidates.

### Scene Features Added

* Green-pitch percentage and row/column coverage.
* Player count and high-confidence player count.
* Largest and median player-size ratios.
* Total player bounding-box area.
* Horizontal and vertical player distribution.
* Nearest-player distance and pairwise overlap.
* Torso-region shirt-color observations and simple color clustering.
* White field-line detection near green pitch regions.
* Player and ball displacement between sampled frames.

### Outputs

* One JSON file for every analyzed frame.
* A metadata JSON recording video properties, models, thresholds, sampling rate, and feature settings.
* A debug video showing:

  * player and ball detections;
  * player IDs when tracking succeeds;
  * torso color regions;
  * major scene-feature values.

### Pipeline Design Update

The pipeline is now being separated into multiple temporal resolutions:

```text
Full broadcast video
→ low-FPS scene observation
→ gameplay-window detection
→ higher-FPS player and ball tracking
→ possession estimation
→ native-FPS pose and motion extraction
→ event understanding or commentary
```

Stage 1 currently records observations but does not yet make the final live-gameplay decision. Its JSON features are intended to be interpreted and temporally smoothed by the next stage.

### Findings and Current Problems

* Running BoT-SORT only on approximately 2 FPS sampled frames causes unstable or missing player IDs because the tracker does not receive the intermediate frames.
* Some detections therefore appear as `Player ?`, even though the person was detected.
* The low player confidence threshold can preserve distant players but may also create duplicate detections in close-up scenes.
* Raw prediction is more suitable than tracking for low-FPS gameplay filtering.
* Tracking should instead be used during the higher-FPS possession-analysis stage.
* The gameplay features and thresholds still need evaluation on a complete match half.

### Next Steps

* Test Stage 1 on one 45-minute SoccerNet video at approximately 2 FPS.
* Compare raw YOLO prediction with BoT-SORT tracking.
* Inspect gameplay, replay, crowd, close-up, and stoppage frames.
* Build a rule-based baseline for converting frame features into continuous gameplay windows.
* Reprocess accepted gameplay windows at a higher frame rate for tracking and possession inference.
* Extract consecutive native-video frames for pose-estimation training.
