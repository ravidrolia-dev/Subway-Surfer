"""Temple Run Real-Game Gesture Controller.

Maps hand gestures to Temple Run PC keyboard controls:
- 'W' : Jump
- 'S' : Slide / Duck
- 'A' : Turn 90° Left
- 'D' : Turn 90° Right
- 'Left Arrow'  : Move Left Lane
- 'Right Arrow' : Move Right Lane

HOW TURNS WORK:
There are two ways you can trigger 90-degree turns:
1. HAND TILT / CHOP (Super intuitive & reliable):
   - Tilt your hand sideways to the LEFT (like pointing or tilting your palm left) -> Triggers 'A' (Turn Left).
   - Tilt your hand sideways to the RIGHT (like pointing or tilting your palm right) -> Triggers 'D' (Turn Right).
2. SWIPE MOTION:
   - Quickly wave/swipe your hand across the camera left or right -> Triggers 'A' or 'D'.
3. THUMB POINT (Optional sign):
   - Point thumb out to the left or right.

LANE MOVEMENTS:
- 1 Finger : Move to Left Lane (taps Left Arrow)
- 2 Fingers: Center Lane
- 3 Fingers: Move to Right Lane (taps Right Arrow)

JUMP & SLIDE:
- ✋ Open Palm (4-5 fingers) : Jump (taps 'W')
- ✊ Closed Fist (0 fingers) : Slide / Duck (taps 'S')
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import sys
import time
from typing import Deque, Optional, Tuple

import cv2  # type: ignore

from camera_selector import select_camera_interactively
from config import (
    ACTION_JUMP,
    ACTION_LEFT_LANE,
    ACTION_MIDDLE_LANE,
    ACTION_NONE,
    ACTION_RIGHT_LANE,
    ACTION_SLIDE,
)
from gesture_detector import GestureDetector
from keyboard_controller import tap_key


class TempleRunController:
    """Controls Temple Run using MediaPipe hand tracking and Windows hardware scan-code inputs."""

    def __init__(
        self,
        camera_index: int = 0,
        lane_mode: str = "relative",
        cooldown_ms: int = 280,
        turn_cooldown_ms: int = 500,
    ) -> None:
        self.camera_index = camera_index
        self.lane_mode = lane_mode
        self.cooldown_ms = cooldown_ms
        self.turn_cooldown_ms = turn_cooldown_ms

        self.last_action_time = 0.0
        self.last_turn_time = 0.0
        self.current_lane = 1  # 0: Left, 1: Middle, 2: Right

        # Motion tracking for swipes (x, y, timestamp)
        self.motion_history: Deque[Tuple[float, float, float]] = deque(maxlen=8)
        self.tilt_history: Deque[str] = deque(maxlen=4)

        print(f"[TempleRun] Initializing detector on camera {camera_index}...")
        self.detector = GestureDetector(camera_index=camera_index)

    def trigger_key(self, key_name: str, label: str, is_turn: bool = False) -> None:
        """Send key press event with cooldown check."""
        now = time.monotonic()
        cooldown = self.turn_cooldown_ms if is_turn else self.cooldown_ms
        last_time = self.last_turn_time if is_turn else self.last_action_time

        if (now - last_time) * 1000.0 < cooldown:
            return

        if is_turn:
            self.last_turn_time = now
        else:
            self.last_action_time = now

        print(f"🏃 [TEMPLE RUN INPUT] ==> {label.upper()} (Key: '{key_name}')")
        tap_key(key_name, duration=0.06)

    def detect_turn(self, landmarks) -> Tuple[Optional[str], float, str]:
        """Detect 90-degree turns purely using horizontal SWIPE gesture.

        Returns:
            (turn_action, velocity_x, turn_status_label)
        """
        now = time.monotonic()
        if (now - self.last_turn_time) * 1000.0 < self.turn_cooldown_ms:
            return None, 0.0, "Cooldown"

        # Wrist (0) and Middle MCP (9) - average their position for robust hand center tracking
        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        hand_x = (wrist.x + mid_mcp.x) / 2.0
        hand_y = (wrist.y + mid_mcp.y) / 2.0

        self.motion_history.append((hand_x, hand_y, now))

        # Check velocity across recent motion history (looking back over 0.05s to 0.35s)
        vel_x = 0.0
        if len(self.motion_history) >= 3:
            oldest_x, _, oldest_t = self.motion_history[0]
            dt = now - oldest_t
            if 0.05 < dt < 0.40:
                vel_x = (hand_x - oldest_x) / dt

                # Swipe Left -> 'A'
                if vel_x < -1.1:
                    self.trigger_key("a", "SWIPE TURN LEFT ⬅", is_turn=True)
                    self.motion_history.clear()
                    return "left", vel_x, "SWIPE LEFT ('A')"

                # Swipe Right -> 'D'
                elif vel_x > 1.1:
                    self.trigger_key("d", "SWIPE TURN RIGHT ➡", is_turn=True)
                    self.motion_history.clear()
                    return "right", vel_x, "SWIPE RIGHT ('D')"

        status = f"Vel: {vel_x:+.1f}" if abs(vel_x) > 0.3 else "Neutral"
        return None, vel_x, status

    def process_frame(self, gesture_result) -> Tuple[Optional[str], float, str]:
        """Process gestures for turns, jump, slide, and lane shifts."""
        tilt_ratio = 0.0
        tilt_status = "No Hand"

        if gesture_result.landmarks is not None:
            lm = getattr(gesture_result.landmarks, "landmark", gesture_result.landmarks)
            turn_dir, tilt_ratio, turn_status = self.detect_turn(lm)
            if turn_dir:
                return turn_status, tilt_ratio, turn_status
            tilt_status = turn_status

        if not gesture_result.hand_detected or not gesture_result.is_stable:
            return None, tilt_ratio, tilt_status

        action = gesture_result.action

        # Temple Run jump: 'W'
        if action == ACTION_JUMP:
            self.trigger_key("w", "JUMP ⬆")
            return "JUMP ('W')", tilt_ratio, tilt_status

        # Temple Run slide: 'S'
        if action == ACTION_SLIDE:
            self.trigger_key("s", "SLIDE / DUCK ⬇")
            return "SLIDE ('S')", tilt_ratio, tilt_status

        # Lane side movements: Left Arrow and Right Arrow
        if self.lane_mode == "relative":
            lane_map = {
                ACTION_LEFT_LANE: 0,
                ACTION_MIDDLE_LANE: 1,
                ACTION_RIGHT_LANE: 2,
            }
            if action in lane_map:
                target_lane = lane_map[action]
                if target_lane != self.current_lane:
                    if target_lane > self.current_lane:
                        steps = target_lane - self.current_lane
                        for _ in range(steps):
                            self.trigger_key("right", "MOVE RIGHT LANE ➡")
                            time.sleep(0.04)
                    else:
                        steps = self.current_lane - target_lane
                        for _ in range(steps):
                            self.trigger_key("left", "MOVE LEFT LANE ⬅")
                            time.sleep(0.04)
                    self.current_lane = target_lane
                    lane_name = "Left" if self.current_lane == 0 else ("Middle" if self.current_lane == 1 else "Right")
                    return f"LANE: {lane_name}", tilt_ratio, tilt_status
        else:
            if action == ACTION_LEFT_LANE:
                self.trigger_key("left", "MOVE LEFT LANE ⬅")
                return "MOVE LEFT (Left Arrow)", tilt_ratio, tilt_status
            elif action == ACTION_RIGHT_LANE:
                self.trigger_key("right", "MOVE RIGHT LANE ➡")
                return "MOVE RIGHT (Right Arrow)", tilt_ratio, tilt_status

        return None, tilt_ratio, tilt_status

    def run(self) -> None:
        """Main loop with real-time visual HUD showing tilt meter and active action."""
        window_name = "Temple Run Gesture Controller HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 68)
        print("           TEMPLE RUN GESTURE CONTROLLER READY")
        print("=" * 68)
        print("How to control:")
        print("  - 90° Turn Left  : TILT HAND LEFT or SWIPE LEFT  ==> [ A ]")
        print("  - 90° Turn Right : TILT HAND RIGHT or SWIPE RIGHT ==> [ D ]")
        print("  - Jump           : OPEN PALM (4-5 fingers)        ==> [ W ]")
        print("  - Slide / Duck   : CLOSED FIST (0 fingers)        ==> [ S ]")
        print("  - Move Left Lane : 1 FINGER                       ==> [ Left Arrow ]")
        print("  - Move Mid Lane  : 2 FINGERS")
        print("  - Move Right Lane: 3 FINGERS                      ==> [ Right Arrow ]")
        print("-" * 68)
        print("Switch focus to your Temple Run game window!")
        print("Press 'Q' or 'ESC' in the camera window to stop.")
        print("=" * 68 + "\n")

        last_display = "Waiting for gestures..."
        action_time = time.monotonic()

        try:
            while True:
                gesture_result = self.detector.detect()
                event_desc, tilt_val, tilt_status = self.process_frame(gesture_result)
                if event_desc:
                    last_display = event_desc
                    action_time = time.monotonic()

                frame = self.detector.get_latest_rgb_frame()
                if frame is not None:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]

                    # Status overlay panel
                    cv2.rectangle(bgr, (0, h - 90), (w, h), (18, 20, 28), -1)

                    time_diff = time.monotonic() - action_time
                    col = (0, 255, 180) if time_diff < 0.8 else (210, 210, 210)

                    # Display active action
                    cv2.putText(
                        bgr,
                        f"Active: {last_display}",
                        (15, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        col,
                        2,
                    )

                    # Swipe velocity meter indicator bar
                    bar_center_x = w - 120
                    bar_y = h - 55
                    bar_w = 80
                    cv2.rectangle(
                        bgr,
                        (bar_center_x - bar_w, bar_y - 6),
                        (bar_center_x + bar_w, bar_y + 6),
                        (50, 50, 60),
                        -1,
                    )
                    cv2.line(bgr, (bar_center_x, bar_y - 10), (bar_center_x, bar_y + 10), (120, 120, 120), 1)

                    # Marker for current swipe speed
                    # tilt_val now holds vel_x
                    marker_offset = int(max(-bar_w, min(bar_w, tilt_val * 45)))
                    marker_color = (0, 255, 255) if abs(tilt_val) > 1.1 else (0, 200, 0)
                    cv2.circle(bgr, (bar_center_x + marker_offset, bar_y), 7, marker_color, -1)
                    cv2.putText(
                        bgr,
                        f"Swipe: {tilt_status}",
                        (bar_center_x - bar_w, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (180, 180, 180),
                        1,
                    )

                    # Guide text
                    cv2.putText(
                        bgr,
                        "Swipe L/R: A/D (Turn) | Palm: W (Jump) | Fist: S (Slide) | 1-3 Fing: Lanes",
                        (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.44,
                        (140, 150, 165),
                        1,
                    )

                    cv2.imshow(window_name, bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("Stopping Temple Run Controller...")
                    break
                elif key == ord('a'):
                    self.trigger_key("a", "MANUAL TURN LEFT ('A')", is_turn=True)
                elif key == ord('d'):
                    self.trigger_key("d", "MANUAL TURN RIGHT ('D')", is_turn=True)

        finally:
            self.detector.close()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Temple Run Real-Game Gesture Controller")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--scan-limit", type=int, default=5, help="Camera scan limit")
    parser.add_argument(
        "--mode",
        choices=["relative", "direct"],
        default="relative",
        help="relative: tracks lane 0,1,2 smoothly; direct: triggers on gesture",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=280,
        help="Cooldown between lane moves in ms",
    )
    parser.add_argument(
        "--turn-cooldown",
        type=int,
        default=500,
        help="Cooldown between turns in ms",
    )
    args = parser.parse_args()

    camera_index = args.camera
    if camera_index is None:
        camera_index = select_camera_interactively(max_index=args.scan_limit, show_preview=True)

    if camera_index is None:
        print("No camera selected. Exiting.")
        sys.exit(0)

    controller = TempleRunController(
        camera_index=camera_index,
        lane_mode=args.mode,
        cooldown_ms=args.cooldown,
        turn_cooldown_ms=args.turn_cooldown,
    )
    controller.run()


if __name__ == "__main__":
    main()
