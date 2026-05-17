# Gesture Runner

Gesture Runner is a lightweight Python endless runner inspired by Subway Surfers. The player stays on one of three lanes while obstacles move toward them. The main control method is real-time hand gesture recognition through a webcam using OpenCV and MediaPipe Hands, with keyboard controls as a fallback.

## Features

- Pygame 2D endless runner loop
- Three lanes/tracks
- Random obstacles with increasing speed and spawn frequency
- Score, FPS display, game-over screen, and restart option
- Webcam gesture controls with smoothing to reduce flickering
- Keyboard fallback controls
- Simple colored shapes instead of heavy assets
- Console logging for detected gesture, action, FPS, and score
- Compatibility with both older MediaPipe Solutions installs and newer MediaPipe Tasks installs
- Live camera preview with green hand landmark markup and detected action text

## Project Structure

```text
Gesture Runner/
|-- main.py
|-- game.py
|-- camera_selector.py
|-- gesture_detector.py
|-- gesture_test.py
|-- config.py
|-- requirements.txt
|-- models/
|   `-- hand_landmarker.task
`-- README.md
```

## Installation

Use Python 3.10 or newer. If MediaPipe installation fails on your machine, try Python 3.10 or 3.11 because webcam/ML packages can be version-sensitive.

```bash
cd "Gesture Runner"
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

You can also run directly with:

```bash
pip install -r requirements.txt
python main.py
```

`python main.py` starts with a camera setup step before the game opens. It scans available camera indexes, lets you choose the USB camera, and shows a preview. Press `S` or `Enter` inside the preview window to start the game.

If your USB camera is not listed, scan more indexes:

```bash
python main.py --camera-scan-limit 10
```

If the game opens but the hand gestures do not respond, first run:

```bash
python gesture_test.py
```

The test window should show your webcam and print raw/stable gestures in the terminal. If the wrong camera opens, try:

```bash
python main.py --camera 1
```

or:

```bash
python gesture_test.py --camera 1
```

You can skip the preview but still choose from the terminal:

```bash
python main.py --no-camera-preview
```

## Controls

### Gesture Controls

| Gesture | Action |
| --- | --- |
| 1 raised finger | Move to left lane / first track |
| 2 raised fingers | Move to middle lane / second track |
| 3 raised fingers | Move to right lane / third track |
| Open palm, 4 or 5 fingers | Jump |
| Closed fist, 0 fingers | Slide |

### Keyboard Fallback

| Key | Action |
| --- | --- |
| A or Left Arrow | Move to left lane |
| S or Down Arrow | Move to middle lane |
| D or Right Arrow | Move to right lane |
| Space | Jump |
| Ctrl or Shift | Slide |
| R or Enter | Restart after game over |
| Esc | Quit |

If no hand is detected, the game keeps the last stable gesture shown on screen and keyboard controls remain active.

On newer MediaPipe versions, the first run downloads the official `hand_landmarker.task` model into the local `models/` folder. This is needed because recent MediaPipe releases use the Tasks API instead of the older `mp.solutions.hands` API.

For best gesture results:

- Keep your hand upright, open toward the camera, and fully inside the preview.
- Use a bright room and a plain background.
- Hold each gesture for a short moment instead of switching instantly.
- Watch the `Raw:` line first. If `Raw:` changes but `Gesture:` does not, smoothing is waiting for a stable prediction.
- If the preview is dark, frozen, or showing the wrong camera, run with `--camera 1` or `--camera 2`.

## Gesture Recognition Logic

`gesture_detector.py` uses MediaPipe Hands to estimate 21 hand landmarks from each webcam frame. It counts the four main raised fingers by comparing fingertip positions with finger joint landmarks:

- Index, middle, ring, and pinky are raised when the fingertip is above the middle joint.
- The thumb is intentionally ignored for game control counts because webcam angle often makes a folded thumb look raised.
- 0 fingers maps to closed fist / slide.
- 1, 2, and 3 fingers map to the three lanes.
- 4 raised main fingers maps to open palm / jump.

The game preview draws green hand connections and labels raised fingers so you can see what MediaPipe is tracking in real time.

To avoid rapid flickering, raw gesture predictions are stored in a short history buffer. A gesture becomes stable only after repeated matching predictions and a small cooldown.

## Obstacles

The game uses three obstacle types:

- Red block: avoid by changing lanes.
- Orange low barrier: jump over it or change lanes.
- Purple high barrier: slide under it or change lanes.

Difficulty increases as the score grows: obstacles move faster and spawn more frequently.

## Console Evaluation Output

The game prints a status line about once per second:

```text
[Gesture Runner] gesture=Open palm action=jump source=gesture fps=60.0 score=120
```

This is useful during ICT demonstrations because it exposes:

- Detected gesture
- Action taken
- Control source
- FPS
- Score

## ICT Presentation Relevance

This project demonstrates a complete human-computer interaction pipeline:

- Real-time computer vision input through a webcam
- Hand landmark detection with MediaPipe
- Gesture-to-action mapping
- Smoothing for more stable interaction
- Game feedback through visuals, score, FPS, and console logs
- Accessibility through keyboard fallback controls

It is suitable for discussing gesture-based interfaces, interactive systems, real-time constraints, usability, and lightweight AI-assisted game control.

## Suggested Evaluation Metrics

- Average FPS during play
- Gesture recognition accuracy per gesture type
- Action latency from gesture to player movement
- False positive rate when no gesture is intended
- Robustness under different lighting conditions
- Score progression across repeated trials
- Number of collisions caused by gesture misclassification

## Limitations

- Gesture recognition depends on lighting, camera quality, and hand visibility.
- The thumb can be harder to classify when the hand is rotated.
- Only one hand is tracked.
- The game uses simple collision rules and placeholder graphics.
- MediaPipe package support can vary by Python version and operating system.

## Future Improvements

- Add a calibration screen for each user.
- Add gesture confidence indicators.
- Record gameplay and gesture metrics to a CSV file.
- Add sound effects and simple animations.
- Improve obstacle patterns so difficulty feels more balanced.
- Add a webcam preview panel for debugging.
- Train a custom gesture classifier on top of MediaPipe landmarks.
