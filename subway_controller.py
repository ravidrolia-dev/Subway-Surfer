"""Subway Surfers Real-Game Gesture Controller.

Exact user controls requested:
- Jump        : Up Arrow   (Open Palm / 4-5 fingers)
- Slide       : Down Arrow (Closed Fist / 0 fingers)
- Move Left   : Left Arrow (1 Finger)
- Move Right  : Right Arrow (2 Fingers)

All other gestures removed. Trigger on each gesture change.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import sys
import time
from typing import Deque, Optional, Tuple

import cv2  # type: ignore

from camera_selector import select_camera_interactively
from gesture_detector import GestureDetector
from keyboard_controller import tap_key


class SubwaySurfersController:
    """Detects clean hand gestures and taps real Arrow keys for Subway Surfers."""

    def __init__(
        self,
        camera_index: int = 0,
        cooldown_ms: int = 320,
    ) -> None:
        self.camera_index = camera_index
        self.cooldown_ms = cooldown_ms
        self.last_input_time = 0.0

        # State tracking to avoid repetitive spamming when holding a gesture
        self.last_triggered_gesture: Optional[str] = None
        self.history: Deque[Tuple[str, int]] = deque(maxlen=5)

        print(f"[Controller] Initializing GestureDetector on camera {camera_index}...")
        self.detector = GestureDetector(camera_index=camera_index)

    def trigger_action(self, key_name: str, label: str) -> None:
        """Send key press to active game window."""
        now = time.monotonic()
        if (now - self.last_input_time) * 1000.0 < self.cooldown_ms:
            return

        self.last_input_time = now
        print(f"🎮 [SUBWAY SURFERS] ==> {label.upper()} (Key: {key_name})")
        tap_key(key_name, duration=0.06)

    def process_hand(self, gesture_result) -> Optional[str]:
        """Map only the 4 requested gestures:
        - 1 Finger  -> Left Arrow
        - 2 Fingers -> Right Arrow
        - 0 Fingers (Fist) -> Down Arrow (Slide)
        - 4-5 Fingers (Open Palm) -> Up Arrow (Jump)
        - 3 Fingers -> Ignored
        """
        if not gesture_result.hand_detected:
            self.history.clear()
            self.last_triggered_gesture = None
            return None

        raw_fingers = gesture_result.raw_finger_count
        if raw_fingers is None:
            raw_fingers = gesture_result.finger_count

        if raw_fingers is None:
            return None

        # Identify requested action
        target_action = None
        target_label = None
        target_key = None

        if raw_fingers == 1:
            target_action = "LEFT"
            target_label = "MOVE LEFT ⬅ (1 Finger)"
            target_key = "left"
        elif raw_fingers == 2:
            target_action = "RIGHT"
            target_label = "MOVE RIGHT ➡ (2 Fingers)"
            target_key = "right"
        elif raw_fingers == 0:
            target_action = "SLIDE"
            target_label = "SLIDE / ROLL ⬇ (Fist)"
            target_key = "down"
        elif raw_fingers >= 4:
            target_action = "JUMP"
            target_label = "JUMP ⬆ (Open Palm)"
            target_key = "up"
        else:
            # 3 fingers or unmapped -> ignored
            target_action = "NONE"

        # Smooth prediction over 3 frames
        self.history.append((target_action, raw_fingers))
        if len(self.history) >= 3:
            counts = Counter(item[0] for item in self.history)
            best_action, count = counts.most_common(1)[0]
            if count >= 3 and best_action != "NONE":
                # STRICT ONCE-PER-GESTURE:
                # Only fire when the user switches to a NEW gesture!
                # Holding 2 fingers will send RIGHT exactly ONCE, never spamming.
                if best_action != self.last_triggered_gesture:
                    self.last_triggered_gesture = best_action
                    if target_key:
                        self.trigger_action(target_key, target_label)
                        return target_label

        return None

    def run(self) -> None:
        """Main camera HUD loop."""
        window_name = "Subway Surfers Gesture Controller HUD"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 65)
        print("      SUBWAY SURFERS EXACT GESTURE CONTROLLER READY")
        print("=" * 65)
        print("Mapped Controls:")
        print("  - 1 Finger            : Move Left  ==> [ Left Arrow ⬅ ]")
        print("  - 2 Fingers           : Move Right ==> [ Right Arrow ➡ ]")
        print("  - Open Palm (4-5 fing): Jump       ==> [ Up Arrow ⬆ ]")
        print("  - Closed Fist (0 fing): Slide      ==> [ Down Arrow ⬇ ]")
        print("  - (All other gestures removed)")
        print("-" * 65)
        print("Click into your Subway Surfers window to play!")
        print("Press 'Q' or 'ESC' in the camera window to stop.")
        print("=" * 65 + "\n")

        last_display = "Show your hand to start..."
        action_time = time.monotonic()

        try:
            while True:
                gesture_result = self.detector.detect()
                event_desc = self.process_hand(gesture_result)
                if event_desc:
                    last_display = event_desc
                    action_time = time.monotonic()

                frame = self.detector.get_latest_rgb_frame()
                if frame is not None:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]

                    # Status overlay panel
                    cv2.rectangle(bgr, (0, h - 80), (w, h), (18, 20, 28), -1)

                    time_diff = time.monotonic() - action_time
                    col = (0, 255, 180) if time_diff < 0.8 else (200, 200, 200)

                    cv2.putText(
                        bgr,
                        f"Active: {last_display}",
                        (15, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        col,
                        2,
                    )
                    cv2.putText(
                        bgr,
                        "1 Finger: Left | 2 Fingers: Right | Palm: Jump | Fist: Slide",
                        (15, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (140, 150, 165),
                        1,
                    )

                    cv2.imshow(window_name, bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("Stopping Subway Surfers Controller...")
                    break

        finally:
            self.detector.close()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Subway Surfers Exact Gesture Controller")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--scan-limit", type=int, default=5, help="Camera scan limit")
    parser.add_argument(
        "--cooldown",
        type=int,
        default=300,
        help="Cooldown between repeated keystrokes in ms",
    )
    args = parser.parse_args()

    camera_index = args.camera
    if camera_index is None:
        camera_index = select_camera_interactively(max_index=args.scan_limit, show_preview=True)

    if camera_index is None:
        print("No camera selected. Exiting.")
        sys.exit(0)

    controller = SubwaySurfersController(
        camera_index=camera_index,
        cooldown_ms=args.cooldown,
    )
    controller.run()


if __name__ == "__main__":
    main()
