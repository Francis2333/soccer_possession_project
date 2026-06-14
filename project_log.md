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
