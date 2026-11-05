import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import mediapipe as mp


SUPPORTED_EXERCISES = ("squat", "push-up", "pushup")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker.task"


def calculate_angle(point_a: Tuple[float, float], point_b: Tuple[float, float], point_c: Tuple[float, float]) -> float:
    """Вычисляет угол ABC в градусах."""
    vector_ab = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_cb = (point_c[0] - point_b[0], point_c[1] - point_b[1])

    dot = vector_ab[0] * vector_cb[0] + vector_ab[1] * vector_cb[1]
    magnitude_ab = math.hypot(*vector_ab)
    magnitude_cb = math.hypot(*vector_cb)

    if magnitude_ab == 0 or magnitude_cb == 0:
        return 0.0

    cosine = max(-1.0, min(1.0, dot / (magnitude_ab * magnitude_cb)))
    return math.degrees(math.acos(cosine))


def _resolve_landmark(landmarks: Dict[str, Any], *keys: str):
    for key in keys:
        if key in landmarks:
            return landmarks[key]
    return None


def _average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _detect_pose_landmarks(image: Any) -> Dict[str, Tuple[float, float]]:
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {}

    pose = result.pose_landmarks[0]
    landmark_map = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    mapped: Dict[str, Tuple[float, float]] = {}
    for name, idx in landmark_map.items():
        if idx >= len(pose):
            continue
        point = pose[idx]
        mapped[name] = (float(point.x), float(point.y))
    return mapped


def _finalize_result(
    score: float,
    details: Dict[str, Any],
    default_message: str,
    issues: List[str] | None = None,
    advice: List[str] | None = None,
) -> Dict[str, Any]:
    if score >= 85:
        status = "good"
    elif score >= 60:
        status = "warning"
    else:
        status = "bad"

    return {
        "status": status,
        "score": round(score, 1),
        "message": default_message,
        "issues": issues or [],
        "advice": advice or [],
        "details": details,
    }


def _squat_advice(details: Dict[str, Any]) -> List[str]:
    advice: List[str] = []
    average_knee_angle = details.get("average_knee_angle", 0)
    alignment_offset = details.get("alignment_offset", 0)

    if average_knee_angle < 85:
        advice.append("Не ускоряйте движение: опускайтесь глубже, сохраняя ровную спину и контролируемые колени.")
    elif average_knee_angle > 170:
        advice.append("Контролируйте глубину приседа: не уходите слишком низко, чтобы не терять устойчивость и не перегружать колени.")

    if alignment_offset > 0.15:
        advice.append("Выровняйте таз и плечи: держите грудь вверх и не наклоняйте корпус в сторону.")

    if not advice:
        advice.append("Техника приседа выглядит устойчиво: сохраняйте ровный корпус и равномерное движение по всей амплитуде.")

    return advice


def _pushup_advice(details: Dict[str, Any]) -> List[str]:
    advice: List[str] = []
    average_elbow_angle = details.get("average_elbow_angle", 0)
    alignment_offset = details.get("alignment_offset", 0)

    if average_elbow_angle < 90:
        advice.append("Локти слишком сильно согнуты: опускайтесь плавнее и держите корпус прямой линией.")
    elif average_elbow_angle > 160:
        advice.append("Локти почти полностью выпрямлены — контролируйте глубину отжимания и не теряйте стабильность корпуса.")

    if alignment_offset > 0.12:
        advice.append("Проверьте выравнивание тела: плечи, таз и пятки должны находиться на одной линии.")

    if not advice:
        advice.append("Техника отжиманий в норме: сохраняйте прямую линию корпуса и контролируемое движение локтей.")

    return advice


def evaluate_squat_pose(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    left_hip = _resolve_landmark(landmarks, "left_hip", "hip_left")
    left_knee = _resolve_landmark(landmarks, "left_knee", "knee_left")
    left_ankle = _resolve_landmark(landmarks, "left_ankle", "ankle_left")
    right_hip = _resolve_landmark(landmarks, "right_hip", "hip_right")
    right_knee = _resolve_landmark(landmarks, "right_knee", "knee_right")
    right_ankle = _resolve_landmark(landmarks, "right_ankle", "ankle_right")
    left_shoulder = _resolve_landmark(landmarks, "left_shoulder", "shoulder_left")
    right_shoulder = _resolve_landmark(landmarks, "right_shoulder", "shoulder_right")

    if not all([left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
        return {"status": "bad", "score": 0, "message": "Недостаточно ключевых точек для оценки приседа", "details": {}}

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    knee_angle = _average([left_angle, right_angle])

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    shoulder_center_x = 0.0
    if left_shoulder and right_shoulder:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    alignment_offset = abs(hip_center_x - shoulder_center_x)

    score = 100.0
    if knee_angle < 60 or knee_angle > 170:
        score -= 35
    if knee_angle < 85:
        score -= 15
    if alignment_offset > 0.15:
        score -= 20

    details = {
        "left_knee_angle": round(left_angle, 1),
        "right_knee_angle": round(right_angle, 1),
        "average_knee_angle": round(knee_angle, 1),
        "alignment_offset": round(alignment_offset, 3),
    }

    issues: List[str] = []
    if knee_angle < 85:
        issues.append("Недостаточная глубина приседа")
    if knee_angle > 170:
        issues.append("Потеря контроля в нижней точке")
    if alignment_offset > 0.15:
        issues.append("Корпус отклоняется в сторону")

    message = "Техника приседа в норме" if score >= 80 else "Есть отклонения в глубине или выравнивании корпуса"
    return _finalize_result(max(0.0, score), details, message, issues=issues, advice=_squat_advice(details))


def evaluate_pushup_pose(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    left_shoulder = _resolve_landmark(landmarks, "left_shoulder", "shoulder_left")
    left_elbow = _resolve_landmark(landmarks, "left_elbow", "elbow_left")
    left_wrist = _resolve_landmark(landmarks, "left_wrist", "wrist_left")
    right_shoulder = _resolve_landmark(landmarks, "right_shoulder", "shoulder_right")
    right_elbow = _resolve_landmark(landmarks, "right_elbow", "elbow_right")
    right_wrist = _resolve_landmark(landmarks, "right_wrist", "wrist_right")
    left_hip = _resolve_landmark(landmarks, "left_hip", "hip_left")
    right_hip = _resolve_landmark(landmarks, "right_hip", "hip_right")

    if not all([left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist]):
        return {"status": "bad", "score": 0, "message": "Недостаточно точек для оценки отжиманий", "details": {}}

    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    elbow_angle = _average([left_angle, right_angle])

    if left_hip and right_hip:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        alignment_offset = abs(hip_center_x - shoulder_center_x)
    else:
        alignment_offset = 0.0

    score = 100.0
    if elbow_angle < 70 or elbow_angle > 160:
        score -= 30
    if elbow_angle < 90:
        score -= 10
    if alignment_offset > 0.12:
        score -= 20

    details = {
        "left_elbow_angle": round(left_angle, 1),
        "right_elbow_angle": round(right_angle, 1),
        "average_elbow_angle": round(elbow_angle, 1),
        "alignment_offset": round(alignment_offset, 3),
    }

    issues: List[str] = []
    if elbow_angle < 90:
        issues.append("Локти слишком сильно согнуты")
    if elbow_angle > 160:
        issues.append("Неполная амплитуда отжимания")
    if alignment_offset > 0.12:
        issues.append("Тело перекошено")

    message = "Техника отжиманий нормальная" if score >= 80 else "Проверьте положение корпуса и локтей"
    return _finalize_result(max(0.0, score), details, message, issues=issues, advice=_pushup_advice(details))


def analyze_pose_landmarks(landmarks: Dict[str, Tuple[float, float]], exercise: str) -> Dict[str, Any]:
    exercise_key = exercise.lower().replace("_", "-")
    if exercise_key == "squat":
        return evaluate_squat_pose(landmarks)
    if exercise_key in {"push-up", "pushup"}:
        return evaluate_pushup_pose(landmarks)
    raise ValueError(f"Не поддерживаемое упражнение: {exercise}")


def analyze_image_file(file_path: str | Path, exercise: str = "squat") -> Dict[str, Any]:
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {image_path}")

    landmarks = _detect_pose_landmarks(image)
    if not landmarks:
        return {"status": "bad", "score": 0, "message": "На изображении не найден человек", "details": {}}

    return {**analyze_pose_landmarks(landmarks, exercise), "exercise": exercise}


def analyze_video_file(file_path: str | Path, exercise: str = "squat") -> Dict[str, Any]:
    video_path = Path(file_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    samples: List[Dict[str, Any]] = []
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if frames % 8 != 0:
            continue

        landmarks = _detect_pose_landmarks(frame)
        if landmarks:
            samples.append(analyze_pose_landmarks(landmarks, exercise))

        if len(samples) >= 5:
            break

    cap.release()

    if not samples:
        return {"status": "bad", "score": 0, "message": "В видео не удалось обнаружить человека", "details": {"frames_processed": frames}, "exercise": exercise}

    score = _average(item["score"] for item in samples)
    statuses = {item["status"] for item in samples}
    status = "good" if "bad" not in statuses and score >= 75 else "warning" if score >= 55 else "bad"

    return {
        "status": status,
        "score": round(score, 1),
        "message": "Видео соответствует технике упражнения" if status == "good" else "Проверяйте положение корпуса и суставов",
        "details": {"frames_processed": frames, "sample_count": len(samples)},
        "exercise": exercise,
    }
