import cv2
import mediapipe as mp
from gestures.ok import is_ok_qesture


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)



with HandLandmarker.create_from_options(options) as landmarker:
    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = frame_id * 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                if is_ok_qesture(hand):
                    cap.replease()
                    cv2.destroyAllWindows()
                    exit()
                for i, point in enumerate(hand):
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                    3,
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        str(i),
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    

        cv2.imshow("Hand Tracker", frame)

        if cv2.waitKey(1) == 27:
            break

        frame_id += 1

cap.release()
cv2.destroyAllWindows()