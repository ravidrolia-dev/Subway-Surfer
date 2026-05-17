"""Entry point for Gesture Runner."""

import argparse

from camera_selector import select_camera_interactively


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gesture Runner.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam index to use. If omitted, a camera chooser opens first.",
    )
    parser.add_argument(
        "--camera-scan-limit",
        type=int,
        default=5,
        help="Highest camera index to scan before startup.",
    )
    parser.add_argument(
        "--no-camera-preview",
        action="store_true",
        help="Choose from the terminal without opening the preview window.",
    )
    args = parser.parse_args()

    camera_index = args.camera
    if camera_index is None:
        camera_index = select_camera_interactively(
            max_index=args.camera_scan_limit,
            show_preview=not args.no_camera_preview,
        )

    if camera_index is None:
        print("No camera selected. Exiting before game start.")
        return

    print(f"Starting Gesture Runner with camera {camera_index}...")
    from game import GestureRunnerGame

    game = GestureRunnerGame(camera_index=camera_index)
    try:
        game.run()
    finally:
        game.close()


if __name__ == "__main__":
    main()
