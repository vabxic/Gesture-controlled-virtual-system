# Gesture-Controlled Virtual Apple System

A real-time computer vision application that allows you to control a virtual object (an apple) using hand gestures caught by your webcam.

## 🌟 Key Features
- **Dual-Window Architecture**: 
  - **Camera Window**: Clean webcam feed dedicated to tracking and gesture feedback.
  - **Apple Window**: A focused blank canvas where the virtual apple exists.
- **Natural Interaction**: Use spatial gestures (PULL, GRAB) to interact with the object.
- **Finite State Machine**: Robust logic to prevent jitter and ensure smooth transitions between interaction modes.
- **Smooth Animation**: LERP-based movement and scaling for a premium feel.

---

## 🖐️ Gestures

| Gesture | Movement | Action |
| :--- | :--- | :--- |
| **Open Hand** | Fingers extended, palm facing camera | **Activate**: Apple enters `RESPONDING` state. |
| **Pull** | Move open hand forward toward camera | **Attract**: Apple scales up and moves "closer". |
| **Grab / Pinch** | Touch thumb and index finger tips | **Attach**: Apple follows your hand's position. |
| **Release** | Open hand or remove hand from view | **Reset**: Apple returns to its default position. |

---

## 🛠️ Installation

1. **Clone or download** this project.
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```
3. **Activate the environment**:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. **Install dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

---

## 🚀 How to Run

1. Ensure your webcam is connected.
2. Run the main application:
   ```bash
   python main.py
   ```
3. **Usage Tips**:
   - Keep your hand roughly centered in the camera feed.
   - For the **Pull** gesture, move your hand steadily toward the lens.
   - For the **Grab** gesture, pinch with your thumb and index finger.
   - Press **ESC** on your keyboard to exit.

---

## 📂 Project Structure

- `main.py`: Entry point and window management loop.
- `hand_tracker.py`: MediaPipe wrapper for landmark extraction.
- `gesture_detector.py`: Temporal logic for classifying poses into gestures.
- `apple_controller.py`: Logic layer managing the apple's state and properties.
- `renderer.py`: OpenCV-based alpha-blending and sprite rendering.
- `state_machine.py`: FSM enforcing valid interaction transitions.
- `utils.py`: Shared constants and math helpers.
- `assets/`: Contains the apple sprite.

## 🔗 Share on LinkedIn

If you'd like to share this project on LinkedIn, here's a ready-to-use post and some short variants you can copy/paste.

Full post:

> Built a real-time gesture-controlled demo that lets you interact with a virtual object using your webcam. I call it the Gesture-Controlled Virtual Apple — two windows (camera + apple canvas), smooth LERP animations, and a finite-state machine to keep interactions stable and jitter-free.
>
> How it works: open-palm, pull, and pinch gestures (detected via MediaPipe) let you attract, grab, drag, and transfer the apple between windows. The app supports two-hand workflows: drag the Apple Window with a right-hand grab, and transfer the apple with a left-hand open palm. It also snaps windows and renders the apple on your hand for a polished AR-like feel.
>
> Tech: Python, OpenCV, MediaPipe, NumPy. Core files: `main.py`, `hand_tracker.py`, `gesture_detector.py`, `apple_controller.py`, and the sprite in `assets/apple.png`. Run locally with a webcam: create a venv, install `opencv-python mediapipe numpy`, then `python main.py`.
>
> Why I built it: exploring more intuitive, spatial UIs for desktop apps — hand gestures can make interactions feel tactile without extra hardware. If you’re into CV, HCI, or playful UI prototypes, I’d love feedback or collabs.

Short caption (image-ready):

> Gesture-controlled desktop demo — drag, pull, and transfer a virtual apple using just your webcam. Built with Python + MediaPipe.

Suggested hashtags:

`#ComputerVision #HumanComputerInteraction #MediaPipe #OpenCV #Python #GestureControl`

Call to action ideas:

- Try it locally and share which gestures felt most natural.
- Suggestions welcome: multi-object interactions, gesture recording, keyboard/shortcut export.

Image suggestion: a screenshot of the Camera Window with the apple rendered on the palm (transfer state) or a side-by-side of both windows.
