import math

from vision.analysis import calculate_angle, evaluate_squat_pose, evaluate_pushup_pose


def test_calculate_angle_returns_expected_value():
    a = (0.0, 0.0)
    b = (1.0, 0.0)
    c = (1.0, 1.0)

    assert math.isclose(calculate_angle(a, b, c), 90.0, abs_tol=1e-6)


def test_squat_pose_detects_good_form():
    landmarks = {
        "left_hip": (0.0, 1.0),
        "left_knee": (0.0, 0.5),
        "left_ankle": (0.0, 0.1),
        "right_hip": (1.0, 1.0),
        "right_knee": (1.0, 0.5),
        "right_ankle": (1.0, 0.1),
        "left_shoulder": (-0.2, 1.5),
        "right_shoulder": (1.2, 1.5),
    }

    result = evaluate_squat_pose(landmarks)
    assert result["status"] in {"good", "warning"}
    assert result["score"] >= 0
    assert "advice" in result
    assert isinstance(result["advice"], list)


def test_pushup_pose_detects_alignment_issue():
    landmarks = {
        "left_shoulder": (0.0, 1.0),
        "left_elbow": (0.2, 0.6),
        "left_wrist": (0.1, 0.2),
        "right_shoulder": (1.0, 1.0),
        "right_elbow": (0.8, 0.6),
        "right_wrist": (0.9, 0.2),
        "left_hip": (0.2, 0.8),
        "right_hip": (0.8, 0.8),
    }

    result = evaluate_pushup_pose(landmarks)
    assert result["status"] in {"good", "warning", "bad"}
    assert "score" in result
    assert "advice" in result
    assert result["advice"]
