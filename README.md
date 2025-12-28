# Reachy Tracking & Control

An autonomous tracking and control system for the Reachy robot, designed to detect and follow objects (faces, people, cats) using a hybrid tracking approach (YOLOv8 + Object Tracking + Sound). It provides a web-based dashboard for monitoring and manual control.

## Features

- **Autonomous Tracking**: Detects and centers on targets using the robot's head.
- **Hybrid Tracking Engine**:
    - **YOLOv8**: For object detection (Person, Cat). Model stored in `models/yolov8n.onnx`.
    - **Haar Cascades**: For fast face detection.
    - **Visual Tracking (KCF/CSRT)**: Locks onto targets for smooth following.
    - **Ghosting Prevention**: Automatically resets tracking memory on robot movement.
- **Audio Interaction**:
    - **Greeting**: Plays a greeting sound ("What can I do for you?") upon initial detection. Reset logic ensures it only plays once per engagement session.
- **Web Dashboard**:
    - **Live Video Feed**: Annotated with detection boxes and status.
    - **Manual Control**: Full control over Head (Pitch/Roll/Yaw), Body (Yaw), and Antennas via sliders.
    - **Live Telemetry**: Sliders sync in real-time with the robot's physical position (even when paused).
    - **Motor Modes**: Toggle between Stiff, Soft, and Limp modes.
    - **Wiggle Mode**: Toggle "happy" antenna wiggles.
- **State Management**:
    - **Pause/Resume**: Stop tracking to take manual control.
    - **Wiggle Toggle**: Enable/disable automatic idle animations.

## Prerequisites

- **Python 3.10+** (Python 3.11+ recommended, Python 3.12 tested on Windows).
- **Reachy Robot Daemon**: A running Reachy robot (real or simulated) with the SDK server accessible on `localhost:8000` (default) or configured host.
- **Camera**: USB camera (index 0 or 1) for video capture. DirectShow (Windows) or AVFoundation (macOS) backends supported.
- **AI Models**: YOLOv8, Whisper, LLM, and Piper (download via `setup_model_assistant.py`).
- **Note for macOS Users**: System audio (CoreAudio) is used for playback; ensure audio device permissions are granted.

## Quick Start

### 1) Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Pre-cache AI models (recommended)
Use the setup helper to download YOLOv8, Whisper, LLMs, and a Piper voice into the local `models/` folder.

```powershell
python .\setup_model_assistant.py
```

What it does:
- Caches models under [models/](models/) so first-run is faster.
- Tries preferred LLMs (Gemma 3 4B → Gemma 2 2B → Phi-3 Mini 4k → Qwen2.5 0.5B → Phi-3.5 Mini → TinyLlama).
- If Gemma access is gated, it prompts for Hugging Face login and retries once.

### 4) Start the app
```powershell
python .\main.py
```
Open the dashboard at http://localhost:8082.

### Windows

1. **Clone the repository:**
    ```powershell
    git clone https://github.com/retsak/Reachy-Tracking.git
    cd "Reachy Tracking"
    ```

2. **Create Virtual Environment:**
    ```powershell
    python -m venv .venv
    ```

3. **Activate Virtual Environment:**
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
    
    *If PowerShell blocks script execution, run:*
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1
    ```

4. **Install Dependencies:**
    ```powershell
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

### macOS / Linux

1. **Clone the repository:**
    ```bash
    git clone https://github.com/retsak/Reachy-Tracking.git
    cd Reachy-Tracking
    ```

2. **Install Python 3.11+ (if needed) & Create Virtual Environment:**
    
    *Using Homebrew (macOS)*:
    ```bash
    brew install python@3.11
    /opt/homebrew/bin/python3.11 -m venv .venv
    ```
    
    *Standard*:
    ```bash
    python3.11 -m venv .venv
    ```

3. **Activate and Install Dependencies:**
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Reachy Robot Daemon Setup

This project requires a Reachy robot running the SDK daemon (HTTP API server) on port `8000`.

### Required Ports
- **Robot Daemon API**: `http://localhost:8000` (or configured host)
- **Tracking Dashboard**: `http://localhost:8082` (this server)

### Verifying Robot Connection
Check if the robot daemon is running:
```bash
curl http://localhost:8000/api/daemon/status
```

Expected response: `{"status": "ok"}` or similar.

If the robot daemon is on a different machine or port, update the `host` parameter in `robot_controller.py` (line ~15) or set via environment variable in future releases.

## Usage

1. **Ensure the Robot Daemon is Running:**
    Verify the Reachy SDK daemon is accessible at `http://localhost:8000` (see above).

2. **Run the Tracking Server:**
    
    **Windows (PowerShell with venv activated):**
    ```powershell
    python main.py
    ```
    
    **macOS/Linux (with venv activated):**
    ```bash
    .venv/bin/python main.py
    ```

3. **Access the Dashboard:**
    Open your browser and navigate to:
    [http://localhost:8082](http://localhost:8082)
    
    The dashboard opens automatically on startup.

## Models Directory

- **Location**: All AI models and caches live in [models/](models/).
- **Whisper**: Cache at [models/whisper](models/whisper) (faster-whisper base).
- **LLM**: Cache at [models/llm](models/llm) (TinyLlama 1.1B Chat by default).
- **Piper**: Voices at [models/piper](models/piper) (`.onnx` and `.json`).
- **Containment**: No environment variables; everything is stored within the project.

## Voice Assistant

- **Stack**: STT (Faster-Whisper), LLM (Gemma 3 4B preferred with fallbacks), TTS (Piper).
- **Caching**: Whisper uses local `download_root`; LLM uses local `cache_dir`; Piper reads voices from `models/piper`.
- **Enable**: In the dashboard, click “🎤 Voice: Off” to start listening.
- **Flow**: Listens → transcribes → generates → speaks (if a Piper voice is present).
- **Status**: `/api/voice/status` returns voice state, `models` info, and any `error`.
- **Commands**: POST `/api/voice/text_command` (optionally `speak: true`) or `/api/voice/speak`.

### Volume Control
- **UI sliders**: Voice panel and top bar include volume sliders (default 25%).
- **API**: `POST /api/voice/volume` to set; `GET /api/voice/volume` to read; `/status` includes current volume.

### Model Device Selection
- **Auto-detect**: Uses CUDA on Windows or MPS on macOS when available; otherwise CPU.
- **CPU optimization**: Applies dynamic quantization on Linear layers for smaller/faster inference.

### Generation Stability
- **Cache disabled**: `use_cache=False` to avoid `DynamicCache.seen_tokens` errors.
- **Prompt fallback**: If a tokenizer lacks `chat_template`, a simple prompt format is used.

## Setup Script (Recommended)

- Use [setup_model_assistant.py](setup_model_assistant.py) to download all required models into `models/`.

```powershell
& ".\.venv\Scripts\Activate.ps1"
python .\setup_model_assistant.py
```

- Downloads YOLOv8n ONNX into [models/yolov8n.onnx](models/yolov8n.onnx), Whisper base into [models/whisper](models/whisper), and TinyLlama 1.1B Chat into [models/llm](models/llm).
- Place a Piper voice into [models/piper](models/piper) (e.g., `en_US-amy-medium.onnx` and its `.json`). Piper CLI is used if available in PATH.

## Piper Voices

- **Getting Voices**: Download a voice from rhasspy/piper releases and place `.onnx` + `.json` into [models/piper](models/piper).
- **CLI**: Install Piper CLI (optional; required for TTS playback). Windows: follow rhasspy/piper instructions. macOS/Linux: `pip install piper-tts` or use packaged binaries.
- **Runtime**: If no voice is present, TTS is skipped with a log message. Camera reads pause during audio playback to avoid resource conflicts.

## Hugging Face Auth (Gated Models)

Some models (e.g., Gemma 3 4B) require accepting a license and logging in.

### Login
```powershell
huggingface-cli login
# or the new CLI
hf auth login
```

### Tokens
- Saved CLI tokens are used automatically by setup and runtime.
- You can also set an environment variable for non-interactive runs:
```powershell
$env:HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# Persist for future shells
setx HF_TOKEN "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

## Dependencies

Key packages in [requirements.txt](requirements.txt):
- fastapi, uvicorn, opencv-python, numpy, scipy
- transformers, torch, accelerate
- faster-whisper, piper-tts, webrtcvad, silero-vad
- reachy_mini
- huggingface-hub (CLI + login APIs)
- hf_xet (accelerated Hub downloads when Xet Storage is enabled)

## Troubleshooting


### Camera Issues

- **Camera not detected**: Ensure USB camera is connected and accessible (index 0 or 1). Check `robot_controller.py` lines ~106-130 for camera initialization.
- **Low FPS or lag**: Try lowering resolution or disabling auto-exposure in `robot_controller.py` (`_enforce_camera_config` method, line ~172).
- **Camera conflict with audio**: The system pauses camera reads during audio playback to prevent OpenCV/SDK resource conflicts. If audio fails, check logs for OpenCV errors.


### Robot Connection Issues

- **"Connecting to Robot..."**: Verify the daemon is running on `localhost:8000` or the configured host.
- **Moves not executing**: Ensure motors are enabled ("Stiff" mode) and system is not paused.
- **Manual control not available**: Manual control requires the system to be paused. Click "Pause" button in the dashboard.


### Detection/Tracking Issues

- **No detections**: Check lighting and camera angle. Adjust `min_score_threshold` in tuning panel (default: 250).
- **Jittery tracking**: Increase `command_interval` (default: 1.2s) to reduce move frequency.
- **Target switching too often**: Lower `min_score_threshold` or adjust detection class priorities in `detection_engine.py`.


### Performance Issues

- **High CPU usage**: YOLOv8 runs on CPU by default. Lower `detection_interval` (default: 0.2s = 5 Hz) or reduce camera resolution.
- **Slow dashboard**: Lower `stream_fps_cap` in tuning panel (default: 60 FPS).


### Voice/Model Issues

- **First-run delays**: Models load on first use. Use the setup script to pre-download.
- **Model selection**: Defaults to TinyLlama (CPU-friendly). Larger models are optional and may be slow/gated.
- **Local storage**: All caches are confined to [models/](models/) inside the project—no environment variables or external paths needed.


### Logs

- Structured logging is enabled by default. Check console output for `[INFO]`, `[WARNING]`, and `[ERROR]` messages.
- Increase log verbosity by editing `logging.basicConfig(level=logging.DEBUG)` in `main.py` (line ~17).

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.
**Note**: This project utilizes YOLOv8, which is AGPL-3.0 licensed.
