"""Small webcam gesture test utility for Gesture Runner."""

from __future__ import annotations

import time
import argparse

from gesture_detector import GestureDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gesture Runner webcam gestures.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to use. Try 1 or 2 if camera 0 is wrong.",
    )
    args = parser.parse_args()

    detector = GestureDetector(camera_index=args.camera)
    if not detector.available:
        print(f"Camera unavailable: {detector.error_message}")
        detector.close()
        return

    print("Gesture test running. Press Q or Esc in the preview window to quit.")
    print("Hold your hand upright and fully inside the camera frame.")
    last_log = 0.0

    try:
        while True:
            result = detector.detect()
            now = time.monotonic()
            if now - last_log >= 0.25:
                last_log = now
                print(
                    "raw="
                    f"{result.raw_gesture} stable={result.gesture} "
                    f"hand={result.hand_detected} action={result.action}"
                )

            frame = detector.get_latest_rgb_frame()
            if frame is not None and detector.cv2 is not None:
                preview = detector.cv2.cvtColor(frame, detector.cv2.COLOR_RGB2BGR)
                detector.cv2.putText(
                    preview,
                    f"Raw: {result.raw_gesture}",
                    (16, 32),
                    detector.cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                detector.cv2.putText(
                    preview,
                    f"Stable: {result.gesture}",
                    (16, 66),
                    detector.cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                detector.cv2.imshow("Gesture Runner camera test", preview)
                key = detector.cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        detector.close()
        if detector.cv2 is not None:
            detector.cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
