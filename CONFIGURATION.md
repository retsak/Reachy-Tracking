# Configuration Guide

Complete configuration reference for the Reachy Tracking & Control system.

## Table of Contents

- [Tuning Configuration](#tuning-configuration)
- [LLM Configuration](#llm-configuration)
- [Camera Settings](#camera-settings)
- [Theme Customization](#theme-customization)
- [File Locations](#file-locations)

## Tuning Configuration

Tuning parameters are stored in `.tuning_config.json` and can be adjusted via the dashboard or API.

### File: `.tuning_config.json`

```json
{
  "detection_interval": 0.2,
  "command_interval": 1.2,
  "min_score_threshold": 250,
  "stream_fps_cap": 60,
  "detection_classes": ["face", "person"]
}
```

### Parameters

#### detection_interval

- **Type**: Float (seconds)
- **Default**: `0.2`
- **Range**: `0.05` to `5.0`
- **Description**: Time between object detection runs. Lower = more frequent detection but higher CPU usage.
- **Recommendations**:
  - Fast tracking: `0.1` - `0.2`
  - Balanced: `0.2` - `0.3`
  - Power saving: `0.5` - `1.0`

#### command_interval

- **Type**: Float (seconds)
- **Default**: `1.2`
- **Range**: `0.1` to `5.0`
- **Description**: Minimum time between robot movement commands. Prevents jerky motion.
- **Recommendations**:
  - Smooth motion: `1.2` - `1.5`
  - Responsive: `0.8` - `1.0`
  - Slow/gentle: `2.0` - `3.0`

#### min_score_threshold

- **Type**: Integer
- **Default**: `250`
- **Range**: `0` to `500`
- **Description**: Minimum tracking score for target selection. Higher = more selective.
- **Score Components**:
  - Base priority (face: 120, person: 150, cat: 100)
  - Context bonus (+50 if currently tracked)
  - Confidence (0-100 from detector)
  - Detection count (persistent tracks score higher)
  - Size penalty (larger objects preferred)
- **Recommendations**:
  - Aggressive tracking: `150` - `200`
  - Balanced: `250` - `300`
  - Very selective: `350` - `400`

#### stream_fps_cap

- **Type**: Integer (FPS)
- **Default**: `60`
- **Range**: `1` to `120`
- **Description**: Maximum FPS for MJPEG video stream to browser.
- **Recommendations**:
  - Smooth viewing: `60`
  - Balanced: `30`
  - Low bandwidth: `15` - `20`
  - Debug only: `5` - `10`

#### detection_classes

- **Type**: Array of strings
- **Default**: `["face", "person"]`
- **Options**: `"face"`, `"person"`, `"cat"`
- **Description**: Which object types to detect and track.
- **Examples**:
  - Face only: `["face"]`
  - People only: `["person"]`
  - All: `["face", "person", "cat"]`

### Editing

**Via Dashboard**:

1. Click "⚙️ Tuning" button
2. Adjust sliders
3. Changes save automatically

**Via API**:

```bash
curl -X POST http://localhost:8082/api/tuning \
  -H "Content-Type: application/json" \
  -d '{"detection_interval": 0.3, "min_score_threshold": 300}'
```

**Direct File Edit**:

Edit `.tuning_config.json` and restart the application.

## LLM Configuration

LLM settings are stored in `llm_config.json` (not committed to git).

### File: `llm_config.json`

#### OpenAI Configuration

```json
{
  "provider": "openai",
  "openai": {
    "model": "gpt-5-nano",
    "api_key": "sk-...",
    "temperature": 0.7
  }
}
```

**Models**:

- `gpt-3.5-turbo`: Fast, economical
- `gpt-4`: High quality, slower
- `gpt-5-nano`: Latest reasoning model (recommended)
- `o1`, `o3`: Advanced reasoning (no temperature control)

**Temperature**:

- `0.0` - `1.0`: Controls randomness
- Lower = more deterministic
- Higher = more creative
- Not available for o-series and gpt-5 models

#### Ollama Configuration

```json
{
  "provider": "ollama",
  "ollama": {
    "model": "llama2",
    "endpoint": "http://localhost:11434"
  }
}
```

**Setup**:

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Start Ollama server: `ollama serve`
4. Configure endpoint and model name

**Recommended Models**:

- `llama2`: Balanced performance
- `mistral`: Fast and capable
- `codellama`: Code understanding
- `phi`: Microsoft's small model

#### Local Configuration

```json
{
  "provider": "local",
  "local": {
    "model": "google/gemma-2-2b-it"
  }
}
```

**Supported Models** (auto-fallback):

1. `google/gemma-2-2b-it` (5GB RAM, requires HF login)
2. `microsoft/Phi-3-mini-4k-instruct` (7.5GB RAM)
3. `Qwen/Qwen2.5-0.5B-Instruct` (1.5GB RAM)
4. `microsoft/Phi-3.5-mini-instruct` (8GB RAM)

**Memory Requirements**:

- System automatically selects models that fit available RAM
- Reserves 40% of RAM for OS and other processes
- Uses 60% budget for model selection

**Hugging Face Authentication**:

For gated models (Gemma), login required:

```bash
huggingface-cli login
# or
hf auth login
```

Or set environment variable:

```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_..."
setx HF_TOKEN "hf_..."

# Linux/macOS
export HF_TOKEN="hf_..."
```

### Editing

**Via Dashboard**:

1. Click "🔧 LLM Config" button
2. Select provider
3. Enter credentials
4. Save configuration

**Via API**:

```bash
# Set OpenAI
curl -X POST http://localhost:8082/api/llm/config \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-5-nano",
    "api_key": "sk-..."
  }'

# Set Ollama
curl -X POST http://localhost:8082/api/llm/config \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "model": "llama2",
    "endpoint": "http://localhost:11434"
  }'

# Set Local
curl -X POST http://localhost:8082/api/llm/config \
  -H "Content-Type: application/json" \
  -d '{"provider": "local"}'
```

**Test Configuration**:

```bash
curl -X POST http://localhost:8082/api/llm/test
```

## Camera Settings

Camera configuration is in `robot_controller.py`.

### Windows DirectShow Settings

```python
# Line ~20-25
self.win_exposure_mode = 0.75   # 0.75=manual, 0.25=auto
self.win_exposure_value = -4    # less negative = brighter
self.win_gain = 12.0            # ISO/gain boost
```

**Exposure Mode**:

- `0.25`: Auto exposure (camera adjusts)
- `0.75`: Manual exposure (fixed)

**Exposure Value**:

- `-7` to `0`: Darker to brighter
- Recommended: `-4` to `-2`

**Gain**:

- `0` to `30`: ISO equivalent
- Higher = brighter in low light but more noise
- Recommended: `10` - `15`

### Camera Index

```python
# Line ~110
camera_index = 1  # Try 0, 1, or 2
```

Change if your camera is not at index 1.

### Resolution & Format

```python
# Line ~175-185
self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
self.camera.set(cv2.CAP_PROP_FPS, 15)
```

**Recommendations**:

- 640x480 @ 15 FPS: Balanced
- 320x240 @ 30 FPS: Performance
- 1280x720 @ 10 FPS: Quality

## Theme Customization

Themes are defined in `static/style.css`.

### Available Themes

1. `dark` - Default dark mode
2. `light` - Clean light mode
3. `ocean` - Blue ocean tones
4. `forest` - Green forest tones
5. `hacker` - Matrix green
6. `halloween` - Spooky orange/purple
7. `christmas` - Festive red/green
8. `newyear` - Gold celebration
9. `sunset` - Warm orange gradient
10. `cyberpunk` - Neon pink/cyan
11. `midnight` - Deep blue night
12. `lavender` - Soft purple
13. `retro` - 80s brown/beige
14. `arctic` - Ice blue
15. `volcano` - Molten red/orange
16. `synthwave` - Purple/pink gradient
17. `coffee` - Warm brown
18. `cherry` - Cherry blossom pink
19. `unicorn` - Rainbow gradient
20. `nyan` - Nyan Cat animated

### Creating Custom Themes

Add to `static/style.css`:

```css
[data-theme="mytheme"] {
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface-hover: #0f3460;
    --text: #eaeaea;
    --text-muted: #a8a8a8;
    --accent: #e94560;
    --accent-hover: #d63651;
    --accent-light: rgba(233, 69, 96, 0.1);
    --border: #2a2a3e;
    --success: #00d9ff;
    --warning: #ffa41b;
    --danger: #f85e4e;
    --shadow: rgba(0, 0, 0, 0.3);
}
```

Add to theme selector in `static/index.html`:

```html
<option value="mytheme">My Theme</option>
```

### Theme Persistence

Selected theme is saved in browser `localStorage` and restored on page load.

## File Locations

### Configuration Files

- `.tuning_config.json` - Tracking parameters
- `llm_config.json` - LLM provider settings (git-ignored)

### Model Directories

- `models/` - Root model directory
  - `models/yolov8n.onnx` - Object detection model
  - `models/whisper/` - Whisper STT cache
  - `models/llm/` - Local LLM cache
  - `models/piper/` - TTS voice models

### Static Assets

- `static/index.html` - Dashboard UI
- `static/style.css` - Themes and styles
- `Audio/Default Voice/` - Audio samples

### Logs

- Console output (stdout/stderr)
- `voice_error.log` - Detailed LLM errors

### Temporary Files

- `C:\Users\<user>\AppData\Local\Temp\tmp*.wav` - TTS audio chunks
- Automatically cleaned up after playback

## Environment Variables

### Optional

```bash
# Hugging Face token for gated models
HF_TOKEN=hf_...

# Hugging Face alternative token name
HUGGINGFACE_TOKEN=hf_...

# OpenAI API key (can also set via dashboard)
OPENAI_API_KEY=sk-...
```

### Not Used

This project does NOT use:

- `HF_HOME` (uses local `models/` dir)
- `TRANSFORMERS_CACHE` (uses local `models/` dir)
- `XDG_CACHE_HOME` (Windows/macOS specific)

All caches are self-contained in the project directory.

## Port Configuration

Default ports (currently hardcoded):

- Dashboard: `8082`
- Robot API: `8000`

To change dashboard port, edit `main.py`:

```python
# Line ~1060
uvicorn.run(app, host="0.0.0.0", port=8082)
```

## Performance Tuning

### For Low-End Systems

```json
{
  "detection_interval": 0.5,
  "command_interval": 2.0,
  "min_score_threshold": 300,
  "stream_fps_cap": 15,
  "detection_classes": ["face"]
}
```

Camera: 320x240 @ 15 FPS

### For High-End Systems

```json
{
  "detection_interval": 0.1,
  "command_interval": 0.8,
  "min_score_threshold": 200,
  "stream_fps_cap": 60,
  "detection_classes": ["face", "person", "cat"]
}
```

Camera: 1280x720 @ 30 FPS

### GPU Acceleration

For YOLOv8 GPU acceleration, install ONNX Runtime GPU:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Requires CUDA toolkit and compatible GPU.
