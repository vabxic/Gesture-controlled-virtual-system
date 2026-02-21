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
