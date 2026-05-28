# Soccer Possession Pose Estimation

This project explores how to extract soccer ball-possession segments from broadcast video and convert them into pose/keypoint sequences for later temporal modeling.

The long-term goal is to study whether player body pose, ball position, and short-term movement patterns can be used to model possession-related actions such as carrying, receiving, passing preparation, and changes of possession.

At the current stage, this is a prototype data-extraction pipeline. The focus is on building a workflow that can turn raw broadcast video into structured frame-level data and player-centered clips that can later be used for pose-based learning.

Data-extraction pipeline example: https://youtu.be/E4iRdTvv-d8 

---

## Motivation

Broadcast soccer video contains rich information about player movement and ball interaction, but it is difficult to use directly for machine learning because:

- the camera moves constantly;
- the ball is small and often blurred;
- players are frequently occluded;
- possession is not explicitly labeled;
- broadcast clips have changing zoom levels and viewing angles.

Instead of starting with a fully labeled dataset, this project attempts to build a weak-label extraction pipeline from raw video. The pipeline estimates which player is likely in possession of the ball and generates candidate possession segments for later pose extraction and temporal modeling.

---

## Current Pipeline

The current pipeline has three main stages:

### 1. Player and ball detection

The input is an original soccer broadcast video clip.

An object detection model is used to detect:

- players;
- the ball;
- candidate player bounding boxes;
- frame-level player and ball positions.

The output of this stage is a frame-level JSON file containing detected objects and their coordinates.

### 2. Possession estimation

Using the frame-level detection JSON, the possession logic estimates which player is most likely controlling the ball.

The current logic mainly uses:

- distance between the ball and detected players;
- temporal continuity of possession;
- loose-ball states;
- context states when possession is uncertain;
- short-term smoothing to avoid excessive switching.

The output is a possession JSON file that records the estimated possessor, possession state, and relevant bounding boxes for each frame.

### 3. Possession crop generation

Based on the possession JSON, the pipeline generates player-centered cropped clips or frames. These crops are intended to isolate the player who is likely in possession of the ball.

The current outputs include:

- cropped possession frames;
- debug videos;
- frame-level JSON files;
- possession segment information.

---

## Planned Pose Extraction Stage

The next stage is to run YOLO-Pose on the original video frames using the generated possession JSON.

Instead of running pose estimation blindly on the whole frame, the possession JSON provides the approximate region where the possessor should be located. The intended process is:

1. read the original video frame;
2. read the corresponding possession JSON entry;
3. locate the estimated possessor bounding box;
4. crop or expand the player-centered region;
5. run YOLO-Pose on that region or on the original frame;
6. extract body keypoints and their x-y coordinates;
7. save the keypoints as a temporal sequence.

The planned output format is a keypoint sequence such as:

```text
frame_id, player_id, keypoint_name, x, y, confidence