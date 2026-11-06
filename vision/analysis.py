import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import mediapipe as mp


SUPPORTED_EXERCISES = ("squat", "push-up", "pushup", "plank", "lunge", "lunges", "deadlift")
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


def _normalize_exercise_key(exercise: str) -> str:
    return (exercise or "auto").lower().replace("_", "-")


def _exercise_label(exercise: str) -> str:
    labels = {
        "squat": "присед",
        "push-up": "отжимания",
        "pushup": "отжимания",
        "plank": "планка",
        "lunge": "выпад",
        "lunges": "выпад",
        "deadlift": "тяга",
    }
    return labels.get(_normalize_exercise_key(exercise), exercise or "упражнение")


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


def _plank_advice(details: Dict[str, Any]) -> List[str]:
    advice: List[str] = []
    body_angle = details.get("body_angle", 0)
    alignment_offset = details.get("alignment_offset", 0)

    if body_angle < 160:
        advice.append("Корпус слишком согнут: держите линию тела прямой от плеч до пяток.")
    if alignment_offset > 0.14:
        advice.append("Проверьте, чтобы таз и плечи не уходили в сторону — тело должно быть ровным.")

    if not advice:
        advice.append("Планка выполнена ровно: сохраняйте нейтральное положение спины и напряжение корпуса.")

    return advice


def _lunge_advice(details: Dict[str, Any]) -> List[str]:
    advice: List[str] = []
    average_knee_angle = details.get("average_knee_angle", 0)
    alignment_offset = details.get("alignment_offset", 0)

    if average_knee_angle < 70:
        advice.append("Глубина выпада недостаточная: опускайтесь устойчиво и не теряйте контроль над коленом.")
    elif average_knee_angle > 160:
        advice.append("Колено слишком выпрямлено: контролируйте движение и немного увеличивайте глубину, если сохраняется баланс.")

    if alignment_offset > 0.12:
        advice.append("Старайтесь держать корпус вертикально и не заваливаться в сторону.")

    if not advice:
        advice.append("Выпад выполнен стабильно: сохраняйте одинаковую глубину и ровную спину.")

    return advice


def _deadlift_advice(details: Dict[str, Any]) -> List[str]:
    advice: List[str] = []
    body_angle = details.get("body_angle", 0)
    knee_angle = details.get("average_knee_angle", 0)

    if body_angle < 120:
        advice.append("Спина слишком округлена: держите грудь поднятой и таз под контролем.")
    if knee_angle > 160:
        advice.append("Колени почти полностью выпрямлены — поддерживайте небольшое сгибание и плавно поднимайте корпус.")

    if not advice:
        advice.append("Тяга выглядит безопасно: держите нейтральную спину, ровный корпус и контролируемый подъём.")

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


def evaluate_plank_pose(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    left_shoulder = _resolve_landmark(landmarks, "left_shoulder", "shoulder_left")
    left_hip = _resolve_landmark(landmarks, "left_hip", "hip_left")
    left_ankle = _resolve_landmark(landmarks, "left_ankle", "ankle_left")
    right_shoulder = _resolve_landmark(landmarks, "right_shoulder", "shoulder_right")
    right_hip = _resolve_landmark(landmarks, "right_hip", "hip_right")
    right_ankle = _resolve_landmark(landmarks, "right_ankle", "ankle_right")

    if not all([left_shoulder, left_hip, left_ankle, right_shoulder, right_hip, right_ankle]):
        return {"status": "bad", "score": 0, "message": "Недостаточно точек для оценки планки", "details": {}}

    left_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
    right_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
    body_angle = _average([left_angle, right_angle])

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
    alignment_offset = abs(shoulder_center_x - hip_center_x) + abs(hip_center_x - ankle_center_x)

    score = 100.0
    if body_angle < 160:
        score -= 30
    if alignment_offset > 0.18:
        score -= 25

    details = {
        "left_body_angle": round(left_angle, 1),
        "right_body_angle": round(right_angle, 1),
        "body_angle": round(body_angle, 1),
        "alignment_offset": round(alignment_offset, 3),
    }

    issues: List[str] = []
    if body_angle < 160:
        issues.append("Корпус прогнут или потерял прямую линию")
    if alignment_offset > 0.18:
        issues.append("Тело перекошено относительно таза и ног")

    message = "Планка выполнена ровно" if score >= 80 else "Проверьте прямую линию корпуса"
    return _finalize_result(max(0.0, score), details, message, issues=issues, advice=_plank_advice(details))


def evaluate_lunge_pose(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    left_hip = _resolve_landmark(landmarks, "left_hip", "hip_left")
    left_knee = _resolve_landmark(landmarks, "left_knee", "knee_left")
    left_ankle = _resolve_landmark(landmarks, "left_ankle", "ankle_left")
    right_hip = _resolve_landmark(landmarks, "right_hip", "hip_right")
    right_knee = _resolve_landmark(landmarks, "right_knee", "knee_right")
    right_ankle = _resolve_landmark(landmarks, "right_ankle", "ankle_right")
    left_shoulder = _resolve_landmark(landmarks, "left_shoulder", "shoulder_left")
    right_shoulder = _resolve_landmark(landmarks, "right_shoulder", "shoulder_right")

    if not all([left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
        return {"status": "bad", "score": 0, "message": "Недостаточно точек для оценки выпада", "details": {}}

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    knee_angle = _average([left_angle, right_angle])

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    shoulder_center_x = 0.0
    if left_shoulder and right_shoulder:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    alignment_offset = abs(hip_center_x - shoulder_center_x)

    score = 100.0
    if knee_angle < 70 or knee_angle > 170:
        score -= 30
    if knee_angle < 90:
        score -= 15
    if alignment_offset > 0.12:
        score -= 20

    details = {
        "left_knee_angle": round(left_angle, 1),
        "right_knee_angle": round(right_angle, 1),
        "average_knee_angle": round(knee_angle, 1),
        "alignment_offset": round(alignment_offset, 3),
    }

    issues: List[str] = []
    if knee_angle < 90:
        issues.append("Недостаточная глубина выпада")
    if knee_angle > 170:
        issues.append("Колени чрезмерно выпрямлены")
    if alignment_offset > 0.12:
        issues.append("Корпус смещён в сторону")

    message = "Выпад выполнен уверенно" if score >= 80 else "Проверьте глубину и симметрию выпада"
    return _finalize_result(max(0.0, score), details, message, issues=issues, advice=_lunge_advice(details))


def evaluate_deadlift_pose(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    left_shoulder = _resolve_landmark(landmarks, "left_shoulder", "shoulder_left")
    left_hip = _resolve_landmark(landmarks, "left_hip", "hip_left")
    left_knee = _resolve_landmark(landmarks, "left_knee", "knee_left")
    left_ankle = _resolve_landmark(landmarks, "left_ankle", "ankle_left")
    right_shoulder = _resolve_landmark(landmarks, "right_shoulder", "shoulder_right")
    right_hip = _resolve_landmark(landmarks, "right_hip", "hip_right")
    right_knee = _resolve_landmark(landmarks, "right_knee", "knee_right")
    right_ankle = _resolve_landmark(landmarks, "right_ankle", "ankle_right")

    if not all([left_shoulder, left_hip, left_knee, left_ankle, right_shoulder, right_hip, right_knee, right_ankle]):
        return {"status": "bad", "score": 0, "message": "Недостаточно точек для оценки мёртвой тяги", "details": {}}

    left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    body_angle = _average([left_back_angle, right_back_angle])
    knee_angle = _average([
        calculate_angle(left_hip, left_knee, left_ankle),
        calculate_angle(right_hip, right_knee, right_ankle),
    ])

    score = 100.0
    if body_angle < 120:
        score -= 30
    if body_angle > 170:
        score -= 20
    if knee_angle > 160:
        score -= 20

    details = {
        "left_back_angle": round(left_back_angle, 1),
        "right_back_angle": round(right_back_angle, 1),
        "body_angle": round(body_angle, 1),
        "average_knee_angle": round(knee_angle, 1),
    }

    issues: List[str] = []
    if body_angle < 120:
        issues.append("Спина слишком округлена")
    if body_angle > 170:
        issues.append("Тело слишком вертикально, похожее на присед")
    if knee_angle > 160:
        issues.append("Колени слишком выпрямлены")

    message = "Техника тяги в норме" if score >= 80 else "Поддерживайте нейтральную спину и контролируемое движение"
    return _finalize_result(max(0.0, score), details, message, issues=issues, advice=_deadlift_advice(details))


def _evaluate_exercise_by_key(landmarks: Dict[str, Tuple[float, float]], exercise_key: str) -> Dict[str, Any]:
    if exercise_key == "squat":
        return evaluate_squat_pose(landmarks)
    if exercise_key in {"push-up", "pushup"}:
        return evaluate_pushup_pose(landmarks)
    if exercise_key == "plank":
        return evaluate_plank_pose(landmarks)
    if exercise_key in {"lunge", "lunges"}:
        return evaluate_lunge_pose(landmarks)
    if exercise_key == "deadlift":
        return evaluate_deadlift_pose(landmarks)
    raise ValueError(f"Не поддерживаемое упражнение: {exercise_key}")


def detect_best_exercise(landmarks: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    left_shoulder = _resolve_landmark(landmarks, "left_shoulder", "shoulder_left")
    right_shoulder = _resolve_landmark(landmarks, "right_shoulder", "shoulder_right")
    left_hip = _resolve_landmark(landmarks, "left_hip", "hip_left")
    right_hip = _resolve_landmark(landmarks, "right_hip", "hip_right")
    left_knee = _resolve_landmark(landmarks, "left_knee", "knee_left")
    right_knee = _resolve_landmark(landmarks, "right_knee", "knee_right")
    left_ankle = _resolve_landmark(landmarks, "left_ankle", "ankle_left")
    right_ankle = _resolve_landmark(landmarks, "right_ankle", "ankle_right")
    left_elbow = _resolve_landmark(landmarks, "left_elbow", "elbow_left")
    right_elbow = _resolve_landmark(landmarks, "right_elbow", "elbow_right")
    left_wrist = _resolve_landmark(landmarks, "left_wrist", "wrist_left")
    right_wrist = _resolve_landmark(landmarks, "right_wrist", "wrist_right")

    scores: Dict[str, float] = {}

    if left_knee and right_knee and left_ankle and right_ankle:
        squat_score = 0.0
        knee_angle = _average([
            calculate_angle(left_hip, left_knee, left_ankle) if left_hip else 0,
            calculate_angle(right_hip, right_knee, right_ankle) if right_hip else 0,
        ]) if left_hip and right_hip else 0.0
        squat_score += max(0.0, 60.0 - abs(knee_angle - 110.0) * 0.6)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            squat_score += max(0.0, 25.0 - abs(hip_center_x - shoulder_center_x) * 100)
        if not (left_elbow and right_elbow and left_wrist and right_wrist):
            squat_score = max(squat_score, 90.0)
        scores["squat"] = squat_score

        if not (left_elbow and right_elbow and left_wrist and right_wrist):
            scores["lunge"] = max(scores.get("lunge", 0.0), 80.0)
            if "plank" in scores:
                scores["plank"] = min(scores["plank"], 55.0)

    if left_hip and right_hip and left_knee and right_knee and left_ankle and right_ankle:
        lunge_score = 0.0
        knee_angle = _average([
            calculate_angle(left_hip, left_knee, left_ankle) if left_hip else 0,
            calculate_angle(right_hip, right_knee, right_ankle) if right_hip else 0,
        ]) if left_hip and right_hip else 0.0
        lunge_score += max(0.0, 55.0 - abs(knee_angle - 100.0) * 0.7)
        if left_shoulder and right_shoulder:
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            lunge_score += max(0.0, 20.0 - abs(hip_center_x - shoulder_center_x) * 80)
        scores["lunge"] = lunge_score

    if left_shoulder and right_shoulder and left_hip and right_hip and left_ankle and right_ankle:
        body_angle = _average([
            calculate_angle(left_shoulder, left_hip, left_ankle) if left_shoulder and left_hip and left_ankle else 0,
            calculate_angle(right_shoulder, right_hip, right_ankle) if right_shoulder and right_hip and right_ankle else 0,
        ])
        alignment_offset = abs(((left_shoulder[0] + right_shoulder[0]) / 2) - ((left_hip[0] + right_hip[0]) / 2)) + abs(((left_hip[0] + right_hip[0]) / 2) - ((left_ankle[0] + right_ankle[0]) / 2))
        plank_score = max(0.0, 60.0 - abs(body_angle - 180.0) * 0.45) + max(0.0, 25.0 - alignment_offset * 100)
        scores["plank"] = plank_score

    if left_shoulder and right_shoulder and left_elbow and right_elbow and left_wrist and right_wrist:
        elbow_angle = _average([
            calculate_angle(left_shoulder, left_elbow, left_wrist) if left_shoulder and left_elbow and left_wrist else 0,
            calculate_angle(right_shoulder, right_elbow, right_wrist) if right_shoulder and right_elbow and right_wrist else 0,
        ])
        pushup_score = max(0.0, 65.0 - abs(elbow_angle - 95.0) * 0.75)
        if left_hip and right_hip:
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            pushup_score += max(0.0, 20.0 - abs(hip_center_x - shoulder_center_x) * 120)
        scores["pushup"] = pushup_score

    if left_shoulder and right_shoulder and left_hip and right_hip and left_knee and right_knee and left_ankle and right_ankle:
        body_angle = _average([
            calculate_angle(left_shoulder, left_hip, left_knee) if left_shoulder and left_hip and left_knee else 0,
            calculate_angle(right_shoulder, right_hip, right_knee) if right_shoulder and right_hip and right_knee else 0,
        ])
        knee_angle = _average([
            calculate_angle(left_hip, left_knee, left_ankle) if left_hip and left_knee and left_ankle else 0,
            calculate_angle(right_hip, right_knee, right_ankle) if right_hip and right_knee and right_ankle else 0,
        ])
        deadlift_score = 60.0 - abs(body_angle - 150.0) * 0.4 - max(0.0, knee_angle - 120.0) * 0.25
        if knee_angle <= 120:
            deadlift_score += 10.0
        if knee_angle > 160:
            deadlift_score -= 25.0
        scores["deadlift"] = max(0.0, deadlift_score)

    if not scores:
        return {"status": "bad", "score": 0, "message": "Не удалось определить упражнение по ключевым точкам", "details": {}, "exercise": "auto"}

    best_exercise = max(scores, key=scores.get)
    best_result = _evaluate_exercise_by_key(landmarks, best_exercise)
    best_result["exercise"] = best_exercise
    return best_result


def _validate_selected_exercise(landmarks: Dict[str, Tuple[float, float]], exercise: str) -> Dict[str, Any] | None:
    selected_key = _normalize_exercise_key(exercise)
    if selected_key == "auto":
        return None

    auto_result = detect_best_exercise(landmarks)
    detected_key = _normalize_exercise_key(auto_result.get("exercise", "auto"))

    if detected_key in {"auto", selected_key}:
        return None

    selected_label = _exercise_label(selected_key)
    detected_label = _exercise_label(detected_key)
    detected_score = float(auto_result.get("score", 0.0) or 0.0)
    return {
        "status": "warning" if detected_score >= 60 else "bad",
        "score": round(max(0.0, min(100.0, detected_score)), 1),
        "message": f"На фото больше похоже на {detected_label}, а не на {selected_label}",
        "issues": [f"Вы выбрали {selected_label}, но на фото больше похоже на {detected_label}."] ,
        "advice": ["Выберите соответствующее упражнение или включите режим auto."],
        "details": {
            "selected_exercise": selected_key,
            "detected_exercise": detected_key,
            "detected_score": round(detected_score, 1),
        },
        "exercise": detected_key,
    }


def analyze_pose_landmarks(landmarks: Dict[str, Tuple[float, float]], exercise: str) -> Dict[str, Any]:
    exercise_key = _normalize_exercise_key(exercise)
    if exercise_key == "auto":
        return detect_best_exercise(landmarks)

    mismatch_result = _validate_selected_exercise(landmarks, exercise_key)
    if mismatch_result is not None:
        return mismatch_result

    return _evaluate_exercise_by_key(landmarks, exercise_key)


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

    result = analyze_pose_landmarks(landmarks, exercise)
    return {**result, "exercise": result.get("exercise", exercise)}


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
            frame_result = detect_best_exercise(landmarks) if exercise.lower().replace("_", "-") == "auto" else analyze_pose_landmarks(landmarks, exercise)
            samples.append(frame_result)

        if len(samples) >= 5:
            break

    cap.release()

    if not samples:
        return {
            "status": "bad",
            "score": 0,
            "message": "В видео не удалось обнаружить человека",
            "details": {"frames_processed": frames},
            "exercise": exercise,
            "issues": ["На видео не найден человек"],
            "advice": ["Попробуйте другое видео с более широким ракурсом и хорошим освещением."],
        }

    exercise_key = exercise.lower().replace("_", "-")
    if exercise_key == "auto":
        detected_exercise = max(
            ((sample.get("exercise", "squat"), sample["score"]) for sample in samples),
            key=lambda item: item[1],
        )[0]
        samples = [sample for sample in samples if sample.get("exercise", "squat") == detected_exercise]

    score = _average(item["score"] for item in samples)
    statuses = {item["status"] for item in samples}
    status = "good" if "bad" not in statuses and score >= 75 else "warning" if score >= 55 else "bad"

    unique_issues: List[str] = []
    advice: List[str] = []
    for item in samples:
        for issue in item.get("issues", []) or []:
            if issue not in unique_issues:
                unique_issues.append(issue)
        for tip in item.get("advice", []) or []:
            if tip not in advice:
                advice.append(tip)

    return {
        "status": status,
        "score": round(score, 1),
        "message": "Видео соответствует технике упражнения" if status == "good" else "Проверяйте положение корпуса и суставов",
        "details": {"frames_processed": frames, "sample_count": len(samples)},
        "exercise": detected_exercise if exercise_key == "auto" else exercise,
        "issues": unique_issues,
        "advice": advice or ["Проверяйте ровность корпуса и стабильность движения по всей амплитуде."],
    }
