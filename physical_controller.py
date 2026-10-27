"""Physical Body Movement Controller for Subway Surfers.

Physical Gesture Controls:
1. Jump (Up Arrow)       : Raise both hands above the head.
2. Move Left (Left Arrow): Raise only the left hand above the shoulder.
3. Move Right (Right Arrow): Raise only the right hand above the shoulder.
4. Slide (Down Arrow)    : Cross both arms in front of the chest to form an X shape.

Important Gesture Priorities & Robustness:
- Both hands raised above head has highest priority (triggers Jump, not Left/Right/Slide).
- Crossed arms (wrists crossed in front of torso) triggers Slide (not Left or Right).
- Single hand raised triggers Left or Right only if not crossed and not both hands raised.
- Single-shot trigger: Triggers each action only once per gesture.
- Neutral reset: Requires returning to neutral / uncrossed position before triggering the same action again.
- Consecutive-frame smoothing prevents accidental spikes or false detections.
- Emergency exit (Q / ESC) safely releases all keys.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Deque, List, Optional, Tuple
import urllib.request

import cv2  # type: ignore
import numpy as np  # type: ignore

from camera_selector import open_camera, select_camera_interactively
from keyboard_controller import press_key, release_key, tap_key

# ---------------------------------------------------------------------------
# Constants and MediaPipe Pose Landmark Indices
# ---------------------------------------------------------------------------
# 0: nose, 7: left_ear, 8: right_ear
# 11: left_shoulder, 12: right_shoulder
# 13: left_elbow,    14: right_elbow
# 15: left_wrist,    16: right_wrist
# 23: left_hip,      24: right_hip
# 25: left_knee,     26: right_knee
# 27: left_ankle,    28: right_ankle

POSE_CONNECTIONS = (
    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Arms
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    # Legs
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


# ---------------------------------------------------------------------------
# Configuration Parameters
# ---------------------------------------------------------------------------
@dataclass
class PoseConfig:
    """Configurable thresholds and timing parameters for body gesture detection."""

    # Hand raise vertical margins (relative to head / shoulder level, normalized)
    # Note: in camera coordinates, smaller y is higher up.
    head_margin: float = 0.02          # Wrist must be above head reference by this margin
    shoulder_margin: float = 0.04      # Wrist must be above shoulder reference by this margin

    # Crossed arms parameters
    wrist_cross_margin: float = 0.03   # Wrist overlap threshold in horizontal direction

    # Smoothing & Confirmation
    confirm_frames: int = 3            # Consecutive matching frames to confirm action
    cooldown_s: float = 0.35           # Minimum cooldown between distinct action triggers

    # Startup Countdown & Calibration
    countdown_seconds: int = 5
    calibration_samples: int = 30


# ---------------------------------------------------------------------------
# PoseLandmarker Wrapper
# ---------------------------------------------------------------------------
class PoseDetector:
    """Handles MediaPipe Pose Landmarker initialization and frame inference."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or self._ensure_model()
        self.landmarker = self._init_landmarker(self.model_path)
        self.last_timestamp_ms = 0

    @staticmethod
    def _ensure_model() -> Path:
        """Download pose_landmarker_lite.task if missing."""
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "pose_landmarker_lite.task"
        if not model_path.exists() or model_path.stat().st_size == 0:
            print("[PoseDetector] Downloading MediaPipe Pose Landmarker model...")
            temp_path = model_path.with_suffix(".download")
            urllib.request.urlretrieve(MODEL_URL, temp_path)
            temp_path.replace(model_path)
            print(f"[PoseDetector] Saved model to {model_path}")
        return model_path

    @staticmethod
    def _init_landmarker(model_path: Path):
        """Create MediaPipe PoseLandmarker in VIDEO running mode."""
        from mediapipe.tasks.python import BaseOptions, vision  # type: ignore

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray) -> Optional[any]:
        """Process RGB frame and return pose landmarks if detected."""
        import mediapipe as mp  # type: ignore

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    def close(self) -> None:
        """Release landmarker resources."""
        if self.landmarker is not None:
            self.landmarker.close()


# ---------------------------------------------------------------------------
# Gesture Evaluation & State Management
# ---------------------------------------------------------------------------
class PhysicalController:
    """Evaluates physical body movements and emits Subway Surfers keystrokes."""

    def __init__(
        self,
        camera_index: int = 0,
        config: Optional[PoseConfig] = None,
    ) -> None:
        self.camera_index = camera_index
        self.config = config or PoseConfig()
        self.cap = None
        self.detector: Optional[PoseDetector] = None

        # Calibration baseline values
        self.calibrated = False
        self.baseline_shoulder_y: float = 0.35
        self.baseline_head_y: float = 0.20
        self.calibration_buffer: List[Tuple[float, float]] = []

        # Multi-frame smoothing buffer
        self.raw_gesture_history: Deque[str] = deque(maxlen=self.config.confirm_frames)

        # Anti-Repetition State Tracking
        # Only triggers once per gesture. Must return to NEUTRAL to trigger the same action again.
        self.active_gesture_state: str = "NEUTRAL"
        self.last_fired_action: str = "NONE"
        self.last_action_time: float = 0.0

        # UI display status
        self.current_action: str = "NEUTRAL"
        self.last_trigger_label: str = "None"
        self.last_trigger_time: float = 0.0

    def _check_visibility(self, landmarks) -> Tuple[bool, str]:
        """Ensure upper body landmarks (shoulders, elbows, wrists, nose) are tracked."""
        key_indices = [0, 11, 12, 13, 14, 15, 16]
        for idx in key_indices:
            lm = landmarks[idx]
            vis = getattr(lm, "visibility", 1.0)
            if vis is not None and vis < 0.35:
                return False, f"Low visibility on landmark {idx}"
        return True, "Upper body visible"

    def record_calibration_sample(self, landmarks) -> bool:
        """Collect neutral baseline head and shoulder heights."""
        nose = landmarks[0]
        l_sh = landmarks[11]
        r_sh = landmarks[12]

        sh_y = (l_sh.y + r_sh.y) / 2.0
        head_y = nose.y

        self.calibration_buffer.append((sh_y, head_y))
        if len(self.calibration_buffer) == self.config.calibration_samples:
            arr = np.array(self.calibration_buffer)
            self.baseline_shoulder_y = float(np.median(arr[:, 0]))
            self.baseline_head_y = float(np.median(arr[:, 1]))
            self.calibrated = True
            print(f"\n[Calibration Complete] Shoulder Y: {self.baseline_shoulder_y:.3f}, "
                  f"Head Y: {self.baseline_head_y:.3f}")
            return True
        return False

    def trigger_action(self, key_name: str, label: str) -> None:
        """Send hardware key press and update UI feedback."""
        safe_label = label.encode("ascii", "ignore").decode("ascii").strip()
        print(f"[PHYSICAL ACTION] ==> {safe_label.upper()} (Key: {key_name})")
        tap_key(key_name, duration=0.06)
        self.last_trigger_label = label
        self.last_trigger_time = time.monotonic()

    def _detect_raw_gesture(self, landmarks) -> Tuple[str, dict]:
        """Classify gesture in a single frame based on strict priority.
        
        Note: The camera frame is horizontally mirrored (flipped).
        In the mirrored frame:
        - Player's physical LEFT hand is shown on the LEFT side of the screen.
        - Player's physical RIGHT hand is shown on the RIGHT side of the screen.
        - landmarks[15] is left_wrist, landmarks[11] is left_shoulder.
        - landmarks[16] is right_wrist, landmarks[12] is right_shoulder.
        - MediaPipe landmark coordinate: smaller Y is HIGHER on screen.
        """
        metrics = {}

        nose = landmarks[0]
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        l_elb = landmarks[13]
        r_elb = landmarks[14]
        l_wri = landmarks[15]
        r_wri = landmarks[16]

        # Use highest point of head (nose or top of ears if available) as head reference
        head_y = min(nose.y, landmarks[7].y, landmarks[8].y)
        shoulder_y = (l_sh.y + r_sh.y) / 2.0

        # Elevation checks (smaller Y means higher up)
        l_above_head = l_wri.y < (head_y + self.config.head_margin)
        r_above_head = r_wri.y < (head_y + self.config.head_margin)

        l_above_shoulder = l_wri.y < (l_sh.y - self.config.shoulder_margin)
        r_above_shoulder = r_wri.y < (r_sh.y - self.config.shoulder_margin)

        metrics["head_y"] = head_y
        metrics["shoulder_y"] = shoulder_y
        metrics["l_wri_y"] = l_wri.y
        metrics["r_wri_y"] = r_wri.y

        # Crossed arms (X-shape) detection:
        # Both wrists crossed in front of torso / chest:
        # In a natural standing position, left wrist is to the left of right wrist (l_wri.x < r_wri.x in mirrored frame).
        # When arms are crossed, left wrist moves to the right and right wrist moves to the left (l_wri.x > r_wri.x).
        # Also check that wrists are near chest height (below shoulders, above hips).
        wrist_dist_x = l_wri.x - r_wri.x  # In mirrored frame: negative when uncrossed, positive when crossed
        hips_y = (landmarks[23].y + landmarks[24].y) / 2.0 if hasattr(landmarks[23], 'y') else shoulder_y + 0.35
        wrists_at_chest = (shoulder_y - 0.05) < ((l_wri.y + r_wri.y) / 2.0) < (hips_y + 0.10)

        # Arms crossed condition:
        # Wrists crossed (l_wri.x > r_wri.x) or wrists overlapping tightly between elbows at chest level
        elbows_min_x = min(l_elb.x, r_elb.x)
        elbows_max_x = max(l_elb.x, r_elb.x)
        wrists_between_elbows = (
            elbows_min_x <= l_wri.x <= elbows_max_x and
            elbows_min_x <= r_wri.x <= elbows_max_x
        )
        is_crossed = (
            wrists_at_chest and
            (wrist_dist_x > -self.config.wrist_cross_margin) and
            wrists_between_elbows and
            not l_above_head and
            not r_above_head
        )

        metrics["wrist_dist_x"] = wrist_dist_x
        metrics["is_crossed"] = is_crossed

        # -------------------------------------------------------------------
        # PRIORITY 1: JUMP (Both hands raised above head)
        # -------------------------------------------------------------------
        if l_above_head and r_above_head:
            return "JUMP", metrics

        # -------------------------------------------------------------------
        # PRIORITY 2: SLIDE (Crossed arms in front of chest - X shape)
        # -------------------------------------------------------------------
        if is_crossed:
            return "SLIDE", metrics

        # -------------------------------------------------------------------
        # PRIORITY 3: MOVE RIGHT (Raise hand on player's right side)
        # -------------------------------------------------------------------
        if l_above_shoulder and not r_above_shoulder:
            return "RIGHT", metrics

        # -------------------------------------------------------------------
        # PRIORITY 4: MOVE LEFT (Raise hand on player's left side)
        # -------------------------------------------------------------------
        if r_above_shoulder and not l_above_shoulder:
            return "LEFT", metrics

        return "NEUTRAL", metrics

    def process_body_movement(self, landmarks) -> Tuple[str, dict]:
        """Multi-frame confirmation and single-shot edge trigger."""
        raw_gesture, metrics = self._detect_raw_gesture(landmarks)
        self.raw_gesture_history.append(raw_gesture)

        # Determine if we have consecutive-frame consensus
        confirmed_gesture = "NEUTRAL"
        if len(self.raw_gesture_history) >= self.config.confirm_frames:
            counts = Counter(self.raw_gesture_history)
            best_g, count = counts.most_common(1)[0]
            if count >= self.config.confirm_frames:
                confirmed_gesture = best_g

        now = time.monotonic()

        # Check for returning to NEUTRAL to reset state
        if confirmed_gesture == "NEUTRAL":
            self.active_gesture_state = "NEUTRAL"
            self.current_action = "NEUTRAL"
            return "NEUTRAL", metrics

        # Single-shot trigger: Trigger once per gesture change
        # Requires returning to NEUTRAL or a different gesture before re-triggering
        if confirmed_gesture != self.active_gesture_state:
            # Check cooldown
            if (now - self.last_action_time) >= self.config.cooldown_s:
                self.active_gesture_state = confirmed_gesture
                self.last_action_time = now

                if confirmed_gesture == "JUMP":
                    self.trigger_action("up", "JUMP ⬆ (Both Hands Up)")
                    self.current_action = "JUMP"
                elif confirmed_gesture == "SLIDE":
                    self.trigger_action("down", "SLIDE ⬇ (Crossed Arms)")
                    self.current_action = "SLIDE"
                elif confirmed_gesture == "LEFT":
                    self.trigger_action("left", "MOVE LEFT ⬅ (Left Hand Up)")
                    self.current_action = "LEFT"
                elif confirmed_gesture == "RIGHT":
                    self.trigger_action("right", "MOVE RIGHT ➡ (Right Hand Up)")
                    self.current_action = "RIGHT"
        else:
            # Gesture is being held -> Show as holding, do NOT re-fire key
            self.current_action = f"{confirmed_gesture} (Holding)"

        return self.current_action, metrics

    def run(self) -> None:
        """Main camera acquisition, calibration countdown, and real-time control loop."""
        window_name = "Subway Surfers Physical Controller (Body Motion)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 70)
        print("    SUBWAY SURFERS PHYSICAL BODY MOTION CONTROLLER")
        print("=" * 70)
        print("Physical Controls:")
        print("  - JUMP       : Both hands raised above head     ==> [ Up Arrow ⬆ ]")
        print("  - MOVE LEFT  : Left hand raised above shoulder  ==> [ Left Arrow ⬅ ]")
        print("  - MOVE RIGHT : Right hand raised above shoulder ==> [ Right Arrow ➡ ]")
        print("  - SLIDE      : Cross both arms (X at chest)     ==> [ Down Arrow ⬇ ]")
        print("-" * 70)
        print("Safety rules:")
        print("  - Both hands up has priority over single-hand gestures.")
        print("  - Crossed arms triggers slide (not left/right).")
        print("  - Each action triggers once; return to neutral to trigger again.")
        print("  - Press 'Q' or 'ESC' to exit cleanly with all keys released.")
        print("=" * 70 + "\n")

        self.cap = open_camera(self.camera_index, cv2)
        if not self.cap.isOpened():
            print(f"[Error] Failed to open camera index {self.camera_index}")
            return

        self.detector = PoseDetector()

        fps_timer = time.monotonic()
        frame_count = 0
        current_fps = 0.0

        countdown_start_time = time.monotonic()
        calibrating_phase = True

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                # Mirror frame horizontally so your left hand is on the left side
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                # Calculate FPS
                frame_count += 1
                now = time.monotonic()
                if now - fps_timer >= 1.0:
                    current_fps = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer = now

                landmarks = self.detector.detect(rgb_frame)

                # -----------------------------------------------------------
                # Phase 1: Startup Countdown & Neutral Stance Calibration
                # -----------------------------------------------------------
                if calibrating_phase:
                    elapsed = now - countdown_start_time
                    remaining = int(np.ceil(self.config.countdown_seconds - elapsed))

                    if landmarks is not None:
                        self._draw_pose_skeleton(frame, landmarks)
                        if remaining <= 2:
                            self.record_calibration_sample(landmarks)

                    # Overlay countdown UI
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w // 2 - 250, h // 2 - 100), (w // 2 + 250, h // 2 + 100), (20, 24, 34), -1)
                    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                    if remaining > 0:
                        cv2.putText(
                            frame,
                            f"STAND READY: {remaining}s",
                            (w // 2 - 200, h // 2 - 15),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.2,
                            (0, 255, 255),
                            3,
                        )
                        cv2.putText(
                            frame,
                            "Stand in neutral position facing camera",
                            (w // 2 - 200, h // 2 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (220, 220, 220),
                            1,
                        )
                    else:
                        if not self.calibrated and landmarks is not None:
                            self.record_calibration_sample(landmarks)
                        if self.calibrated or elapsed > self.config.countdown_seconds + 2.0:
                            calibrating_phase = False
                            print("[Ready] Game active! Switch focus to Subway Surfers.")

                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    continue

                # -----------------------------------------------------------
                # Phase 2: Active Real-Time Game Control
                # -----------------------------------------------------------
                visibility_ok = False
                vis_msg = "No player detected"
                metrics = {}

                if landmarks is not None:
                    visibility_ok, vis_msg = self._check_visibility(landmarks)
                    self._draw_pose_skeleton(frame, landmarks)

                    if visibility_ok:
                        action, metrics = self.process_body_movement(landmarks)
                    else:
                        self.current_action = "ALIGN BODY"

                # Render HUD
                self._render_hud(frame, current_fps, visibility_ok, vis_msg, metrics)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("[Exit] User requested emergency exit.")
                    break

        finally:
            self._cleanup()

    def _draw_pose_skeleton(self, frame: np.ndarray, landmarks) -> None:
        """Draw pose joints and bone lines."""
        h, w = frame.shape[:2]
        pts = {}
        for idx, lm in enumerate(landmarks):
            vis = getattr(lm, "visibility", 1.0)
            if vis is None or vis >= 0.35:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pts[idx] = (px, py)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in pts and p2 in pts:
                cv2.line(frame, pts[p1], pts[p2], (0, 230, 255), 2, cv2.LINE_AA)

        for idx, pt in pts.items():
            # Highlight hands and shoulders
            if idx in (15, 16):  # Wrists
                cv2.circle(frame, pt, 8, (0, 255, 0), -1, cv2.LINE_AA)
            elif idx in (11, 12):  # Shoulders
                cv2.circle(frame, pt, 6, (255, 180, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (0, 200, 255), -1, cv2.LINE_AA)

    def _render_hud(
        self,
        frame: np.ndarray,
        fps: float,
        visibility_ok: bool,
        vis_msg: str,
        metrics: dict,
    ) -> None:
        """Draw sleek top and bottom telemetry overlays."""
        h, w = frame.shape[:2]

        # Top Banner
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 24, 34), -1)
        cv2.putText(
            frame,
            "SUBWAY SURFERS PHYSICAL CONTROLLER",
            (16, 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        status_col = (0, 255, 0) if visibility_ok else (0, 140, 255)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Tracking: {'OK' if visibility_ok else 'ALIGN'}",
            (w - 240, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_col,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Calibrated: {'YES' if self.calibrated else 'READY'}",
            (w - 240, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Bottom Panel
        panel_h = 100
        cv2.rectangle(frame, (0, h - panel_h), (w, h), (18, 20, 28), -1)

        # Action Badge (Green highlight when triggered recently)
        time_since_trigger = time.monotonic() - self.last_trigger_time
        recent_trigger = time_since_trigger < 0.65

        badge_col = (0, 255, 0) if recent_trigger else (120, 130, 140)
        cv2.rectangle(frame, (16, h - panel_h + 16), (220, h - 20), (32, 36, 48), -1)
        cv2.rectangle(frame, (16, h - panel_h + 16), (220, h - 20), badge_col, 2)

        cv2.putText(
            frame,
            "CURRENT ACTION:",
            (26, h - panel_h + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            self.current_action,
            (26, h - 34),
            cv2.FONT_HERSHEY_DUPLEX,
            0.72,
            badge_col,
            2,
            cv2.LINE_AA,
        )

        # Center Gesture Guides
        guide_x = w // 2 - 120
        cv2.putText(
            frame,
            "GESTURE GUIDE:",
            (guide_x, h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Both Hands Up: Jump | X-Arms: Slide",
            (guide_x, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 170, 185),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Left Hand Up: Left | Right Hand Up: Right",
            (guide_x, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 170, 185),
            1,
            cv2.LINE_AA,
        )

        # Last Sent Command Indicator
        cv2.putText(
            frame,
            f"Last Key Sent: {self.last_trigger_label}",
            (w - 260, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 200) if recent_trigger else (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press 'Q' or ESC to Quit",
            (w - 260, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (130, 130, 140),
            1,
            cv2.LINE_AA,
        )

    def _cleanup(self) -> None:
        """Release camera, detector, and ensure all game keys are released."""
        print("\n[Cleanup] Releasing game keyboard keys for safety...")
        for key in ("left", "right", "up", "down"):
            try:
                release_key(key)
            except Exception:
                pass

        if self.cap is not None:
            self.cap.release()
        if self.detector is not None:
            self.detector.close()
        cv2.destroyAllWindows()
        print("[Cleanup] Physical controller closed successfully.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subway Surfers Full-Body Physical Movement Controller"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index. If omitted, interactive selector opens.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=5,
        help="Maximum camera index to scan.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Startup calibration countdown duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.35,
        help="Cooldown between distinct action triggers in seconds (default: 0.35)",
    )
    args = parser.parse_args()

    camera_index = args.camera
    if camera_index is None:
        camera_index = select_camera_interactively(
            max_index=args.scan_limit, show_preview=True
        )

    if camera_index is None:
        print("[Aborted] No camera selected. Exiting.")
        sys.exit(0)

    cfg = PoseConfig(
        countdown_seconds=args.countdown,
        cooldown_s=args.cooldown,
    )

    controller = PhysicalController(camera_index=camera_index, config=cfg)
    controller.run()


if __name__ == "__main__":
    main()
