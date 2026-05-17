"""Webcam hand gesture detection using OpenCV and MediaPipe Hands."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Deque, Optional, Tuple
import urllib.request

from config import (
    ACTION_JUMP,
    ACTION_LEFT_LANE,
    ACTION_MIDDLE_LANE,
    ACTION_NONE,
    ACTION_RIGHT_LANE,
    ACTION_SLIDE,
    GESTURE_CHANGE_COOLDOWN_SECONDS,
    GESTURE_HISTORY_SIZE,
    GESTURE_STABLE_THRESHOLD,
    HAND_LANDMARKER_MODEL_URL,
)
from camera_selector import open_camera


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

FINGER_CHAINS = {
    "I": (5, 6, 7, 8),
    "M": (9, 10, 11, 12),
    "R": (13, 14, 15, 16),
    "P": (17, 18, 19, 20),
}


@dataclass
class GestureResult:
    """Result object returned once per frame to the game loop."""

    gesture: str = "No hand"
    action: str = ACTION_NONE
    finger_count: Optional[int] = None
    hand_detected: bool = False
    is_stable: bool = False
    message: str = ""
    raw_gesture: str = "No hand"
    raw_action: str = ACTION_NONE
    raw_finger_count: Optional[int] = None
    backend: str = "unavailable"


class GestureDetector:
    """Detects simple hand poses and smooths predictions over a short buffer."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.history: Deque[Tuple[str, str, int]] = deque(maxlen=GESTURE_HISTORY_SIZE)
        self.last_stable_action = ACTION_NONE
        self.last_stable_gesture = "No hand"
        self.last_stable_finger_count: Optional[int] = None
        self.last_change_time = 0.0
        self.available = False
        self.error_message = ""

        self.cv2 = None
        self.mp_hands = None
        self.mp_image = None
        self.mp_image_format = None
        self.vision = None
        self.backend = "unavailable"
        self.hands = None
        self.cap = None
        self.last_video_timestamp_ms = 0
        self.latest_rgb_frame = None

        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore

            self.cv2 = cv2
            self._initialize_mediapipe_backend(mp)
            self.cap = open_camera(camera_index, cv2)
            self.available = bool(self.cap.isOpened())
            if not self.available:
                self.error_message = "Webcam unavailable"
        except ImportError as exc:
            self.error_message = (
                "Missing webcam dependencies. Run: pip install -r requirements.txt"
            )
            print(f"[GestureDetector] {self.error_message} ({exc})")
        except Exception as exc:
            self.error_message = f"Gesture detector unavailable: {exc}"
            print(f"[GestureDetector] {self.error_message}")

    def _initialize_mediapipe_backend(self, mp) -> None:
        """Initialize either the legacy Solutions API or the newer Tasks API."""

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.62,
                min_tracking_confidence=0.55,
            )
            print("[GestureDetector] Using MediaPipe Solutions Hands backend")
            return

        from mediapipe.tasks.python import BaseOptions  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        model_path = self._ensure_hand_landmarker_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.62,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.backend = "tasks"
        self.mp_image = mp.Image
        self.mp_image_format = mp.ImageFormat
        self.vision = vision
        self.hands = vision.HandLandmarker.create_from_options(options)
        print("[GestureDetector] Using MediaPipe Tasks HandLandmarker backend")

    def _ensure_hand_landmarker_model(self) -> Path:
        """Download the MediaPipe Tasks hand model once if it is missing."""

        model_dir = Path(__file__).resolve().parent / "models"
        model_path = model_dir / "hand_landmarker.task"
        if model_path.exists() and model_path.stat().st_size > 0:
            return model_path

        model_dir.mkdir(parents=True, exist_ok=True)
        temp_path = model_path.with_suffix(".download")
        print("[GestureDetector] Downloading MediaPipe hand landmarker model...")
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, temp_path)
        temp_path.replace(model_path)
        print(f"[GestureDetector] Model saved to {model_path}")
        return model_path

    def detect(self) -> GestureResult:
        """Read a webcam frame and return the current smoothed gesture."""

        if not self.available or self.cap is None or self.cv2 is None or self.hands is None:
            return GestureResult(
                gesture=self.last_stable_gesture,
                action=self.last_stable_action,
                finger_count=self.last_stable_finger_count,
                hand_detected=False,
                is_stable=False,
                message=self.error_message or "Gesture detector unavailable",
                raw_gesture=self.last_stable_gesture,
                raw_action=self.last_stable_action,
                raw_finger_count=self.last_stable_finger_count,
                backend=self.backend,
            )

        ok, frame = self.cap.read()
        if not ok:
            return GestureResult(
                gesture=self.last_stable_gesture,
                action=self.last_stable_action,
                finger_count=self.last_stable_finger_count,
                hand_detected=False,
                is_stable=False,
                message="Could not read webcam frame",
                raw_gesture=self.last_stable_gesture,
                raw_action=self.last_stable_action,
                raw_finger_count=self.last_stable_finger_count,
                backend=self.backend,
            )

        # Mirror the frame so gestures feel natural to the player.
        frame = self.cv2.flip(frame, 1)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        self.latest_rgb_frame = rgb.copy()
        results = self._process_rgb_frame(rgb)
        hand_landmarks, handedness = self._extract_hand_data(results)

        if hand_landmarks is None:
            self._draw_no_hand_overlay()
            return GestureResult(
                gesture=f"No hand (last: {self.last_stable_gesture})",
                action=self.last_stable_action,
                finger_count=self.last_stable_finger_count,
                hand_detected=False,
                is_stable=False,
                message="Keyboard fallback active",
                raw_gesture="No hand",
                raw_action=ACTION_NONE,
                raw_finger_count=None,
                backend=self.backend,
            )

        finger_count = self._count_raised_fingers(hand_landmarks, handedness)
        raw_gesture, raw_action = self._map_fingers_to_action(finger_count)
        stable_gesture, stable_action, is_stable = self._smooth(
            raw_gesture, raw_action, finger_count
        )
        self._draw_hand_overlay(hand_landmarks, raw_gesture, stable_action, finger_count)

        return GestureResult(
            gesture=stable_gesture,
            action=stable_action,
            finger_count=finger_count,
            hand_detected=True,
            is_stable=is_stable,
            message="",
            raw_gesture=raw_gesture,
            raw_action=raw_action,
            raw_finger_count=finger_count,
            backend=self.backend,
        )

    def get_latest_rgb_frame(self):
        """Return the latest mirrored RGB camera frame for debugging previews."""

        return self.latest_rgb_frame

    def _process_rgb_frame(self, rgb_frame):
        """Run the active MediaPipe backend on one RGB frame."""

        if self.backend == "solutions":
            rgb_frame.flags.writeable = False
            return self.hands.process(rgb_frame)

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_video_timestamp_ms:
            timestamp_ms = self.last_video_timestamp_ms + 1
        self.last_video_timestamp_ms = timestamp_ms

        mp_image = self.mp_image(
            image_format=self.mp_image_format.SRGB,
            data=rgb_frame.copy(),
        )
        return self.hands.detect_for_video(mp_image, timestamp_ms)

    def _extract_hand_data(self, results):
        """Return the first hand's landmarks and handedness for either backend."""

        if self.backend == "solutions":
            if not results.multi_hand_landmarks:
                return None, None
            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[0]
            return results.multi_hand_landmarks[0], handedness

        if not results.hand_landmarks:
            return None, None
        handedness = None
        if results.handedness:
            handedness = results.handedness[0]
        return results.hand_landmarks[0], handedness

    def _count_raised_fingers(self, hand_landmarks, handedness=None) -> int:
        """Count raised control fingers from MediaPipe's 21 hand landmarks.

        The game uses index/middle/ring/pinky counts. This avoids the common
        webcam issue where a folded thumb is mistaken for one raised finger.
        """

        landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
        return sum(self._finger_states(landmarks))

    def _finger_states(self, landmarks) -> Tuple[bool, bool, bool, bool]:
        """Return raised states for index, middle, ring, and pinky fingers."""

        return tuple(
            landmarks[tip_id].y < landmarks[pip_id].y - 0.018
            for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18))
        )

    def _draw_no_hand_overlay(self) -> None:
        """Label the camera preview when no hand is visible."""

        if self.latest_rgb_frame is None or self.cv2 is None:
            return

        self.cv2.putText(
            self.latest_rgb_frame,
            "No hand detected",
            (16, 34),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 90, 90),
            2,
        )

    def _draw_hand_overlay(
        self,
        hand_landmarks,
        raw_gesture: str,
        stable_action: str,
        finger_count: int,
    ) -> None:
        """Draw green landmark lines, finger metrics, and action on the preview."""

        if self.latest_rgb_frame is None or self.cv2 is None:
            return

        landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
        height, width = self.latest_rgb_frame.shape[:2]
        points = [
            (
                int(max(0, min(1, landmark.x)) * width),
                int(max(0, min(1, landmark.y)) * height),
            )
            for landmark in landmarks
        ]
        raised_states = self._finger_states(landmarks)

        for start, end in HAND_CONNECTIONS:
            self.cv2.line(self.latest_rgb_frame, points[start], points[end], (0, 190, 0), 2)

        for point in points:
            self.cv2.circle(self.latest_rgb_frame, point, 4, (0, 255, 0), -1)

        for is_raised, (label, chain) in zip(raised_states, FINGER_CHAINS.items()):
            color = (40, 255, 40) if is_raised else (80, 120, 80)
            thickness = 4 if is_raised else 2
            for start, end in zip(chain, chain[1:]):
                self.cv2.line(self.latest_rgb_frame, points[start], points[end], color, thickness)
            tip = points[chain[-1]]
            self.cv2.circle(self.latest_rgb_frame, tip, 8, color, 2)
            self.cv2.putText(
                self.latest_rgb_frame,
                label,
                (tip[0] + 8, tip[1] - 8),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        metric_text = f"Fingers: {finger_count} | {raw_gesture}"
        action_text = f"Action: {stable_action}"
        self.cv2.putText(
            self.latest_rgb_frame,
            metric_text,
            (16, 34),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
        )
        self.cv2.putText(
            self.latest_rgb_frame,
            action_text,
            (16, 66),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

    def _map_fingers_to_action(self, finger_count: int) -> Tuple[str, str]:
        """Translate finger counts into game actions."""

        if finger_count == 0:
            return "Closed fist", ACTION_SLIDE
        if finger_count == 1:
            return "1 finger: left lane", ACTION_LEFT_LANE
        if finger_count == 2:
            return "2 fingers: middle lane", ACTION_MIDDLE_LANE
        if finger_count == 3:
            return "3 fingers: right lane", ACTION_RIGHT_LANE
        return "Open palm", ACTION_JUMP

    def _smooth(
        self, raw_gesture: str, raw_action: str, finger_count: int
    ) -> Tuple[str, str, bool]:
        """Return a stable action only after repeated matching predictions."""

        self.history.append((raw_action, raw_gesture, finger_count))
        action_counts = Counter(item[0] for item in self.history)
        best_action, best_count = action_counts.most_common(1)[0]
        now = time.monotonic()

        if best_count >= GESTURE_STABLE_THRESHOLD:
            matching = [item for item in self.history if item[0] == best_action]
            best_gesture = matching[-1][1]
            best_fingers = matching[-1][2]

            cooldown_ready = (
                now - self.last_change_time >= GESTURE_CHANGE_COOLDOWN_SECONDS
            )
            if best_action == self.last_stable_action or cooldown_ready:
                self.last_stable_action = best_action
                self.last_stable_gesture = best_gesture
                self.last_stable_finger_count = best_fingers
                self.last_change_time = now
                return best_gesture, best_action, True

        return (
            self.last_stable_gesture,
            self.last_stable_action,
            self.last_stable_action != ACTION_NONE,
        )

    def close(self) -> None:
        """Release camera and MediaPipe resources."""

        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()
