# Reachy Tracking & Control

An autonomous tracking and control system for the Reachy robot, designed to detect and follow objects (faces, people, cats) using a hybrid tracking approach (YOLOv8 + Object Tracking + Sound). It provides a web-based dashboard for monitoring and manual control.

## Features

-   **Autonomous Tracking**: Detects and centers on targets using the robot's head.
-   **Hybrid Tracking Engine**:
    -   **YOLOv8**: For object detection (Person, Cat).
    -   **Haar Cascades**: For fast face detection.
    -   **Visual Tracking (KCF/CSRT)**: Locks onto targets for smooth following.
    -   **Ghosting Prevention**: Automatically resets tracking memory on robot movement.
-   **Audio Interaction**:
    -   **Greeting**: Plays a greeting sound ("What can I do for you?") upon initial detection. Reset logic ensures it only plays once per engagement session.
-   **Web Dashboard**:
    -   **Live Video Feed**: Annotated with detection boxes and status.
    -   **Manual Control**: Full control over Head (Pitch/Roll/Yaw), Body (Yaw), and Antennas via sliders.
    -   **Live Telemetry**: Sliders sync in real-time with the robot's physical position (even when paused).
    -   **Motor Modes**: Toggle between Stiff, Soft, and Limp modes.
    -   **Wiggle Mode**: Toggle "happy" antenna wiggles.
-   **State Management**:
    -   **Pause/Resume**: Stop tracking to take manual control.
    -   **Wiggle Toggle**: Enable/disable automatic idle animations.

## Prerequisites

-   **Python 3.10+** (Python 3.11 recommended).
-   Access to a Reachy robot (real or simulated) running the Reachy SDK server.
-   `yolov8n.onnx` model (automatically downloaded if using Ultralytics).
-   **Note for macOS Users**: System audio (CoreAudio) is used for playback.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/retsak/Reachy-Tracking.git
    cd Reachy-Tracking
    ```

2.  **Install Python 3.11 (if needed) & Create Virtual Environment:**
    
    *Using Homebrew (macOS)*:
    ```bash
    brew install python@3.11
    /opt/homebrew/bin/python3.11 -m venv .venv
    ```
    
    *Standard*:
    ```bash
    python3.11 -m venv .venv
    ```

3.  **Activate and Install Dependencies:**
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

1.  **Ensure the Robot is On:**
    Make sure the Reachy robot (or simulation) is running and its API is accessible (default: `localhost:8000`).

2.  **Run the Tracking Server:**
    **Important:** Ensure you are using the Virtual Environment created above.
    ```bash
    .venv/bin/python main.py
    ```

3.  **Access the Dashboard:**
    Open your browser and navigate to:
    [http://localhost:8082](http://localhost:8082)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.
**Note**: This project utilizes YOLOv8, which is AGPL-3.0 licensed.
