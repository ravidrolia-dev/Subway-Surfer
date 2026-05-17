"""Camera probing and selection helpers for Gesture Runner."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import time
from typing import List, Optional

from config import CAMERA_HEIGHT, CAMERA_WIDTH


@dataclass
class CameraInfo:
    """Basic information about a camera index."""

    index: int
    available: bool
    width: int = 0
    height: int = 0


def open_camera(index: int, cv2_module=None):
    """Open a camera using a Windows-friendly backend when available."""

    cv2 = cv2_module
    if cv2 is None:
        import cv2  # type: ignore

    backend = None
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        backend = cv2.CAP_DSHOW

    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def probe_cameras(max_index: int = 5, warmup_reads: int = 4) -> List[CameraInfo]:
    """Return available cameras from index 0 through max_index."""

    import cv2  # type: ignore

    cameras: List[CameraInfo] = []
    for index in range(max_index + 1):
        cap = open_camera(index, cv2)
        available = bool(cap.isOpened())
        ok = False
        if available:
            for _ in range(warmup_reads):
                ok, _ = cap.read()
                if ok:
                    break

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if available else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if available else 0
        cap.release()
        if available and ok:
            cameras.append(CameraInfo(index=index, available=True, width=width, height=height))
        time.sleep(0.08)

    return cameras


def preview_camera(index: int) -> str:
    """Show a camera preview and return start, choose, or quit."""

    import cv2  # type: ignore

    cap = open_camera(index, cv2)
    if not cap.isOpened():
        print(f"Camera {index} could not be opened.")
        return "choose"

    window_name = f"Gesture Runner - Camera {index} preview"
    print("")
    print(f"Previewing camera {index}.")
    print("Press S or Enter in the preview window to start the game.")
    print("Press C to choose another camera.")
    print("Press Q or Esc to quit.")

    decision = "choose"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"Camera {index} stopped returning frames.")
                break

            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                "S/Enter: start   C: choose again   Q/Esc: quit",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S"), 13):
                decision = "start"
                break
            if key in (ord("c"), ord("C")):
                decision = "choose"
                break
            if key in (ord("q"), ord("Q"), 27):
                decision = "quit"
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        time.sleep(0.25)

    return decision


def select_camera_interactively(max_index: int = 5, show_preview: bool = True) -> Optional[int]:
    """Ask the user to select a camera before the game starts."""

    print("")
    print("=== Gesture Runner Camera Setup ===")
    print(f"Scanning camera indexes 0 to {max_index}...")

    try:
        cameras = probe_cameras(max_index=max_index)
    except ImportError:
        print("OpenCV is not installed. Install dependencies with: pip install -r requirements.txt")
        return None

    if cameras:
        print("")
        print("Detected cameras:")
        for camera in cameras:
            print(f"  [{camera.index}] Camera {camera.index} ({camera.width}x{camera.height})")
        default_index = cameras[0].index
    else:
        print("")
        print("No camera returned a valid frame during scanning.")
        print("You can still type an index manually, for example 1 for a USB camera.")
        default_index = 0

    while True:
        print("")
        choice = input(f"Select camera index [default {default_index}, Q to quit]: ").strip()
        if choice.lower() == "q":
            return None

        if not choice:
            selected_index = default_index
        else:
            try:
                selected_index = int(choice)
            except ValueError:
                print("Please enter a number like 0, 1, or 2.")
                continue

        if not show_preview:
            return selected_index

        decision = preview_camera(selected_index)
        if decision == "start":
            return selected_index
        if decision == "quit":
            return None
