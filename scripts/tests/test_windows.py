from stage1_core.windows import active_windows_from_soccernet, build_shots


def sample_labels():
    return {
        "annotations": [
            {"gameTime": "1 - 00:10", "position": "10000", "label": "Main camera center", "replay": "real-time"},
            {"gameTime": "1 - 00:20", "position": "20000", "label": "Close-up player", "replay": "real-time"},
            {"gameTime": "1 - 00:30", "position": "30000", "label": "Main camera left", "replay": "replay"},
            {"gameTime": "1 - 00:40", "position": "40000", "label": "Main camera right", "replay": "real-time"},
        ]
    }


def test_before_semantics():
    shots = build_shots(sample_labels()["annotations"], 1, 50.0, "before")
    assert shots[0]["start_sec"] == 0.0
    assert shots[0]["end_sec"] == 10.0
    assert shots[1]["start_sec"] == 10.0


def test_active_windows_filter_replay_and_closeup():
    windows = active_windows_from_soccernet(
        sample_labels(), 1, 50.0, ["Main camera"], 0.0, 0.0, "before"
    )
    assert [(w["start_sec"], w["end_sec"]) for w in windows] == [(0.0, 10.0), (30.0, 40.0)]
