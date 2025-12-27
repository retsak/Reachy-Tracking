# Reachy Tracking & Control

An autonomous tracking and control system for the Reachy robot, designed to detect and follow objects (faces, people, cats) using a hybrid tracking approach (YOLOv8 + Object Tracking). It provides a web-based dashboard for monitoring and manual control.

## Features

-   **Autonomous Tracking**: Detects and centers on targets using the robot's head.
-   **Hybrid Tracking Engine**:
    -   **YOLOv8**: For object detection (Person, Cat).
    -   **Haar Cascades**: For fast face detection.
    -   **Visual Tracking (KCF/CSRT)**: Locks onto targets for smooth following.
    -   **Ghosting Prevention**: Automatically resets tracking memory on robot movement.
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

-   Python 3.9+
-   Access to a Reachy robot (real or simulated) running the Reachy SDK server.
-   `yolov8n.onnx` model (automatically downloaded if using Ultralytics, or placed in root).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Reachy-Tracking.git
    cd Reachy-Tracking
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Ensure the Robot is On:**
    Make sure the Reachy robot (or simulation) is running and its API is accessible (default: `localhost:8000`).

2.  **Run the Tracking Server:**
    ```bash
    python3 main.py
    ```

3.  **Access the Dashboard:**
    Open your browser and navigate to:
    [http://localhost:8082](http://localhost:8082)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.
**Note**: This project utilizes YOLOv8, which is AGPL-3.0 licensed.
