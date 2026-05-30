import cv2
import av
import numpy as np
import mediapipe as mp
import threading
from streamlit_webrtc import VideoProcessorBase
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def _draw_skeleton(self, img, landmarks_list):
        h, w = img.shape[:2]
        landmarks = landmarks_list.landmark

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]
            if p1.visibility > 0.7 and p2.visibility > 0.7:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0), 8
                )

        for lm in landmarks:
            if lm.visibility > 0.7:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 8, (255, 0, 0), -1)

    def _draw_no_pose_warnings(self, img):
        cv2.putText(img, "NO POSE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "PLEASE FACE THE CAMERA", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_overlays(self, img, metrics, ex_type):
        h, _ = img.shape[:2]
        if ex_type == "Squats":
            text = f"DEPTH: {metrics.get('depth_status', '')}"
        elif ex_type == "Push-ups":
            text = f"BODY: {metrics.get('body_alignment', '')} | HIP: {metrics.get('hip_status', '')}"
        elif ex_type == "Biceps Curls (Dumbbell)":
            text = f"SWING: {metrics.get('swing_status', '')}"
        elif ex_type == "Shoulder Press":
            text = f"EXT: {metrics.get('extension_status', '')} | BACK: {metrics.get('back_arch_status', '')}"
        elif ex_type == "Lunges":
            text = f"BALANCE: {metrics.get('balance_status', '')}"
        else:
            text = ""
        if text:
            cv2.putText(img, text, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def recv(self, frame):
        image = np.asarray(cv2.flip(frame.to_ndarray(format="bgr24"), 1), dtype=np.uint8)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if result.pose_landmarks:
            self._draw_skeleton(image, result.pose_landmarks)
            ex_type = self.get_exercise()
            detector = self._detectors.get(ex_type)

            if detector:
                # Convert to same format detectors expect
                landmarks = result.pose_landmarks.landmark
                metrics = detector.process(landmarks)
                metrics["pose_detected"] = True
                self._draw_overlays(image, metrics, ex_type)
                self.set_latest_metrics(metrics)
        else:
            self._draw_no_pose_warnings(image)
            with self._lock:
                if self._latest_metrics is not None:
                    self._latest_metrics["pose_detected"] = False
                else:
                    self._latest_metrics = {"pose_detected": False}

        return av.VideoFrame.from_ndarray(image, format="bgr24")
