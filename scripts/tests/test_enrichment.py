import numpy as np

from stage1_core.enrichment import enrich_frame, project_point


def test_image_to_pitch_projection():
    matrix = np.eye(3)
    assert project_point([4, 7], matrix, "image_to_pitch") == [4.0, 7.0]


def test_identity_and_pitch_enrichment():
    frame = {
        "players": [{"track_id": 5, "bottom_center": [10, 20]}],
        "ball": {"bbox_center": [3, 4]},
    }
    calibration = {"homography_image_to_pitch": np.eye(3).tolist()}
    output = enrich_frame(frame, calibration, {5: {"team": "left", "jersey": 8}})
    assert output["players"][0]["identity"]["jersey"] == 8
    assert output["players"][0]["pitch_position"] == [10.0, 20.0]
    assert output["ball"]["pitch_position"] == [3.0, 4.0]
