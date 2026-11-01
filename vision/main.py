import cv2
import mediapipe as mp


POSE_CONNECTIONS = [
    # Руки
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    # Корпус
    (11, 23),
    (12, 24),
    (23, 24),

    # Левая нога
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),

    # Правая нога
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),

    # Лицо
    (0, 1), (1, 2), (2, 3),
    (3, 7), (0, 4),
    (4, 5), (5, 6),
    (6, 8),
]

import cv2
import tkinter as tk

from tkinter import filedialog


root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Выберите фотографию",
    filetypes=[
        ("Images", "*.jpg *.jpeg *.png")
    ]
)

if not image_path:
    print("Фотография не выбрана")
    exit()

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Не удалось загрузить фотографию")

cv2.imshow("Selected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/pose_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1
)



if image is None:
    raise ValueError("Не удалось загрузить фотографию")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)


with PoseLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)

if result.pose_landmarks:

    for pose in result.pose_landmarks:

        # Рисуем точки
        for point in pose:

            if point.visibility is not None and point.visibility < 0.5:
                continue

            x = int(point.x * image.shape[1])
            y = int(point.y * image.shape[0])

            cv2.circle(
                image,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

        # Рисуем соединения
        for start, end in POSE_CONNECTIONS:

            p1 = pose[start]
            p2 = pose[end]

            x1 = int(p1.x * image.shape[1])
            y1 = int(p1.y * image.shape[0])

            x2 = int(p2.x * image.shape[1])
            y2 = int(p2.y * image.shape[0])

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

else:
    print("Человек не обнаружен")



cv2.imshow("Pose Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()