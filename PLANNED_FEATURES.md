# Planned Features

Future enhancements and improvements for the Reachy Tracking & Control system.

## Planned Features

### High Priority (Now - 0-2 days)

#### GPU Acceleration
- **Status**: Planned
- **Description**: Enable CUDA/DirectML for Whisper/LLM inference on supported hardware
- **Impact**: 40-60% latency reduction for voice processing
- **Files**: `voice_assistant.py`, device detection logic
- **Success Criteria**: Significant latency reduction on supported hardware with graceful CPU fallback

#### Process Isolation for Voice Pipeline
- **Status**: Planned
- **Description**: Run STT/LLM/TTS in separate process to avoid blocking tracking
- **Impact**: Eliminates video frame drops during voice processing
- **Files**: `voice_assistant.py`, new `voice_worker.py`
- **Approach**: Use multiprocessing with IPC queues
- **Success Criteria**: No missed video frames during long voice generations

#### WebSocket Telemetry
- **Status**: Planned
- **Description**: Stream status/candidates/voice events via WebSocket instead of polling
- **Impact**: Reduced network overhead, lower UI latency
- **Files**: `main.py`, `static/index.html`
- **Success Criteria**: <200ms UI update latency, reduced server load

#### Command Grammar & Voice Control
- **Status**: Planned
- **Description**: Map voice intents to robot actions (look left/right, recenter, toggle wiggle, set motor mode)
- **Impact**: Full hands-free robot control
- **Files**: `main.py` (intent router), `robot_controller.py`
- **Commands**: 10+ natural language commands with safety clamps
- **Success Criteria**: Reliable command recognition with confirmation feedback

### Medium Priority (Next - 1-2 weeks)

#### Target Selection UI
- **Status**: Planned
- **Description**: Click-to-track feature allowing manual target selection from detection list
- **Impact**: Better control over which object to track
- **Files**: `main.py`, `static/index.html`
- **UI**: Interactive candidate list with click handlers

#### Patrol/Auto-Scan Mode
- **Status**: Planned
- **Description**: Automatic scanning behavior when no target detected
- **Impact**: More engaging idle behavior
- **Files**: `main.py`, tracking logic
- **Behavior**: Smooth panning with configurable pattern

#### Wake Word Detection
- **Status**: Planned
- **Description**: Integrate Porcupine/KWS for "Hey Reachy" wake word
- **Impact**: More natural voice activation (no button press needed)
- **Dependencies**: Porcupine SDK or similar KWS library
- **Success Criteria**: <1% false accept rate, <5% false reject rate

#### Conversation Memory Enhancement
- **Status**: Planned
- **Description**: Long-term memory using SQLite for facts and user preferences
- **Impact**: Persistent context across sessions
- **Files**: New `memory.py`, database schema
- **Features**: User profiles, remembered facts, preferences

### Low Priority (Later - 2-4+ weeks)

#### Multilingual Support
- **Status**: Planned
- **Description**: Multiple language support for voice assistant
- **Components**:
  - Whisper multilingual models
  - Multiple Piper voice packs
  - Language auto-detection
- **Success Criteria**: English + one additional language end-to-end

#### Skills & Plugins System
- **Status**: Planned
- **Description**: Extensible skill system for custom behaviors
- **Examples**: Reminders, jokes, facts, weather, timers
- **Architecture**: Plugin-based with registration system
- **Success Criteria**: 3+ working skills with persistence

#### Advanced Camera Features
- **Status**: Planned
- **Description**: Multi-camera support and advanced camera control
- **Features**:
  - Multiple camera inputs
  - Camera switching
  - PTZ camera support
- **Files**: `robot_controller.py`, camera initialization

#### Authentication System
- **Status**: Planned (if LAN deployment needed)
- **Description**: Simple auth for remote dashboard access
- **Scope**: Token-based authentication, user sessions
- **Files**: `main.py`, new `auth.py`

#### Full Test Suite & CI
- **Status**: Planned
- **Description**: Comprehensive testing and continuous integration
- **Components**:
  - Unit tests for VAD/STT/intent mapping
  - Mocked SDK for robot controller tests
  - Integration tests for tracking pipeline
  - GitHub Actions CI pipeline
- **Success Criteria**: Green CI on PRs, 80% coverage on core modules

### Performance Optimizations

#### Adaptive Rate/Resolution
- **Status**: Planned
- **Description**: Dynamically adjust detection rate and camera resolution based on CPU load
- **Impact**: Better performance on low-end hardware
- **Metrics**: Monitor CPU usage, adjust intervals automatically

#### ONNX Runtime Execution Providers
- **Status**: Planned
- **Description**: Leverage ONNX RT EPs for accelerated inference
- **Options**: DirectML (Windows), CoreML (macOS), TensorRT (Linux/NVIDIA)
- **Success Criteria**: Faster YOLOv8 inference without accuracy loss

### Robustness Improvements

#### Thread Health Monitoring
- **Status**: Planned
- **Description**: Monitor and auto-restart failed worker threads
- **Files**: `main.py`, worker threads
- **Features**: Health checks, automatic recovery, alerts

#### Camera Reconnect Logic
- **Status**: Planned
- **Description**: Handle camera disconnects gracefully with exponential backoff
- **Impact**: Better reliability with USB cameras
- **Files**: `robot_controller.py`, camera thread

#### Input Validation Enhancement
- **Status**: Planned
- **Description**: Stricter validation on all API endpoints
- **Scope**: Tuning parameters, control commands, configuration
- **Files**: `main.py`, all POST endpoints

### UX Enhancements

#### Live Logs Panel
- **Status**: Planned
- **Description**: Real-time log viewer in dashboard
- **Impact**: Easier debugging and monitoring
- **Files**: `static/index.html`, WebSocket streaming

#### Manual Control Sensitivity
- **Status**: Planned
- **Description**: Adjustable sensitivity/speed for manual movements
- **UI**: Slider or preset (coarse/fine) control
- **Files**: `static/index.html`, `main.py`

#### Status Enrichment
- **Status**: Planned
- **Description**: More detailed status information in UI
- **Info**: Target age, tracking confidence, FPS breakdown, model latency
- **Files**: `main.py`, status endpoint

### Documentation Improvements

#### Video Tutorials
- **Status**: Planned
- **Description**: Walkthrough videos for setup and features
- **Topics**: Installation, first run, voice setup, customization

#### Architecture Diagram
- **Status**: Planned
- **Description**: Visual system architecture documentation
- **Content**: Component interactions, data flow, threading model

---

## Completed Features

### ✅ Voice Assistant Core (Completed)
- **Completed**: December 2025
- **Description**: Full voice interaction pipeline integrated
- **Components**:
  - STT: Faster-Whisper for speech recognition
  - LLM: Multi-provider support (OpenAI, Ollama, Local)
  - TTS: Piper with natural voice synthesis
  - Audio pipeline via MediaManager
- **Features**:
  - Conversation memory (8 messages)
  - Smart text processing
  - Async speech (non-blocking)
  - Volume control

### ✅ Multi-LLM Provider Support (Completed)
- **Completed**: December 2025
- **Description**: Support for three LLM providers with dynamic switching
- **Providers**:
  - **OpenAI**: GPT-3.5, GPT-4, GPT-5, o1/o3 models
  - **Ollama**: Local models via Ollama server
  - **Local**: Gemma 2B, Phi-3, Qwen 2.5 with auto-fallback
- **Features**:
  - Memory-aware model selection
  - Configuration UI and API
  - HuggingFace authentication
  - Temperature control (where supported)

### ✅ Emotion System (Completed)
- **Completed**: December 2025
- **Description**: 25 pre-programmed emotions with choreographed animations
- **Categories**:
  - Basic: Happy, Sad, Surprised, Angry, Confused, Scared, Excited, Bored, Shy
  - Social: Greeting, Waving, Nodding, Shaking Head, Shrugging
  - Playful: Silly, Curious, Thinking, Dancing, Wiggle
  - Advanced: Love, Sleepy, Proud, Disappointed, Mischievous, Focused, Yawn
- **Implementation**: Coordinated head and antenna movements
- **UI**: One-click emotion triggers in dashboard

### ✅ Visual Themes (Completed)
- **Completed**: December 2025
- **Description**: 18 customizable visual themes for dashboard
- **Themes**: Dark, Light, Ocean, Forest, Hacker, Halloween, Christmas, New Year, Sunset, Cyberpunk, Midnight, Lavender, Retro, Arctic, Volcano, Synthwave, Coffee, Cherry Blossom, Unicorn, Nyan Cat
- **Special Features**:
  - Nyan Cat theme with animated flying cat and rainbow trail
  - Unicorn theme with rainbow gradient effects
- **Persistence**: Theme selection saved in localStorage

### ✅ TTS Audio Synchronization (Completed)
- **Completed**: December 2025
- **Description**: Proper audio chunk sequencing preventing overlaps
- **Features**:
  - Sentence-based text chunking (~500 chars)
  - Sequential playback with completion tracking
  - Threading events for synchronization
  - Automatic temporary file cleanup
- **Result**: No audio overlap, complete playback of long responses

### ✅ Speech Animation System (Completed)
- **Completed**: December 2025
- **Description**: Natural head movements synchronized with speech
- **Features**:
  - Smooth tilting and antenna motion during speech
  - Tracking pauses automatically during speech
  - Gentle reset after speaking
  - No conflicts with tracking system
- **Timing**: 1.5s duration with 15 iterations per cycle

### ✅ Dynamic Tuning System (Completed)
- **Completed**: December 2025
- **Description**: Real-time parameter adjustment via dashboard
- **Parameters**:
  - Detection interval (0.05-5.0s)
  - Command interval (0.1-5.0s)
  - Score threshold (0-500)
  - Stream FPS cap (1-120)
  - Detection classes (face, person, cat)
- **Persistence**: Settings saved to `.tuning_config.json`
- **UI**: Live sliders with instant feedback

### ✅ Hybrid Tracking Engine (Completed)
- **Completed**: Initial release + enhancements December 2025
- **Description**: Multi-stage object tracking system
- **Components**:
  - YOLOv8 for person/cat detection (ONNX CPU)
  - Haar Cascades for fast face detection
  - KCF visual tracker for smooth following
  - SimpleTracker for object persistence
- **Features**:
  - Smart target selection with scoring
  - Ghost prevention (automatic reset on movement)
  - Context-aware targeting (prefers current track)

### ✅ Web Dashboard (Completed)
- **Completed**: Initial release + major enhancements December 2025
- **Description**: Full-featured web interface for robot control
- **Features**:
  - Live MJPEG video stream with annotations
  - Manual control (pause/resume tracking)
  - Position sliders with live telemetry
  - Motor mode switching (stiff/soft/limp)
  - Voice controls and status
  - Emotion trigger buttons
  - Theme selector
  - Tuning panel
  - LLM configuration interface

### ✅ Configuration Persistence (Completed)
- **Completed**: December 2025
- **Description**: All settings persist between sessions
- **Files**:
  - `.tuning_config.json` - Tracking parameters
  - `llm_config.json` - LLM provider settings
  - Browser localStorage - UI preferences (theme)

### ✅ Camera Optimization (Completed)
- **Completed**: Initial release
- **Description**: Windows DirectShow camera configuration
- **Features**:
  - Manual exposure control
  - Gain adjustment
  - MJPEG format selection
  - Configurable resolution and FPS
  - Settings persistence during SDK operations

### ✅ Comprehensive Documentation (Completed)
- **Completed**: December 2025
- **Files**:
  - `README.md` - Main documentation with features
  - `API.md` - Complete REST API reference
  - `CONFIGURATION.md` - Detailed configuration guide
  - `TROUBLESHOOTING.md` - Common issues and solutions
- **Coverage**: Installation, usage, API, configuration, troubleshooting

### ✅ Auto-Reset Position After Speech (Completed)
- **Completed**: December 2025
- **Description**: Robot automatically returns to center position after speaking
- **Implementation**: Calls `recenter_head()` after TTS completion
- **Result**: Clean transitions between speech and tracking

### ✅ Tracking Pause During Speech (Completed)
- **Completed**: December 2025
- **Description**: Tracking automatically pauses during speech to prevent conflicts
- **Implementation**: `_pause_tracking` flag managed by voice assistant
- **Result**: Smooth speech animations without tracking interference

---

## Known Issues & Investigations

### Long Audio Playback Stability (Resolved)
- **Status**: ✅ Resolved
- **Issue**: Audio clips >30 seconds had truncation issues
- **Root Cause**: Text truncation at 240 characters in `_postprocess_response()`
- **Solution**: Removed character limit, implemented proper sentence-based chunking (500 char chunks)
- **Verification**: Full responses now play completely without cutoff

### Camera Conflicts with Audio (Resolved)
- **Status**: ✅ Resolved
- **Issue**: OpenCV errors during TTS playback due to SDK camera initialization
- **Solution**: Camera reads pause during audio playback via `_pause_camera_reads` flag
- **Verification**: No more camera conflicts during audio operations

---

## Design Decisions

### Why Sentence-Based Chunking?
- Prevents single-file length issues with TTS
- Enables streaming-like experience for long responses
- Better error recovery (chunk failure doesn't lose entire response)
- Current limit: 500 characters per chunk for efficiency

### Why Multi-Provider LLM?
- Flexibility: Use best model for the task (OpenAI for quality, Local for offline)
- Cost control: Switch to local models when appropriate
- Redundancy: Fallback options if one provider fails
- Development: Test with free/local models before production deployment

### Why Threading Events for Audio Sync?
- Prevents polling overhead
- Precise synchronization between chunks
- Clean timeout handling (60s per chunk)
- Minimal CPU usage while waiting

### Why Pause Tracking During Speech?
- Prevents conflicting head movements
- Allows dedicated animation for speaking
- Cleaner user experience (robot "focuses" on conversation)
- Automatic resume ensures seamless return to tracking

---

## Project Constraints

### Hardware
- **CPU**: Primary target (ONNX/PyTorch CPU inference)
- **GPU**: Optional CUDA/DirectML acceleration planned
- **RAM**: 8GB minimum, 16GB recommended for local LLMs
- **Camera**: USB webcam (DirectShow on Windows)

### Network
- **Current**: Localhost only (dashboard + robot daemon)
- **Future**: Optional LAN access with authentication

### OS Support
- **Primary**: Windows (tested on Windows 11)
- **Secondary**: macOS, Linux (community tested)

### Robot Requirements
- **SDK**: Reachy Mini SDK daemon on port 8000
- **Connection**: HTTP API (localhost or LAN)
- **Audio**: Robot speakers via MediaManager

---

## Success Metrics

### Performance
- Detection latency: <200ms per frame
- Tracking smoothness: <1.5s command interval
- Voice end-to-end: <4s (STT + LLM + TTS)
- Dashboard FPS: 15-60 stable

### Reliability
- Uptime: >4 hours continuous operation
- Camera recovery: Automatic reconnect
- Thread stability: Auto-restart on failure
- Audio quality: No dropouts or overlaps

### User Experience
- Setup time: <15 minutes for new users
- Configuration: All settings via UI
- Response time: <500ms for manual controls
- Error clarity: Actionable error messages

---

## Personal Development Notes

Areas for potential focus (priority order):

### Quick Wins

- Manual control sensitivity adjustment (UI + backend)
- Live logs panel in dashboard (WebSocket streaming)
- Status enrichment (confidence, FPS, model latency)

### Core Improvements

- GPU acceleration (CUDA/DirectML) - Would significantly improve voice latency
- Process isolation - Would eliminate video frame drops during voice processing
- WebSocket telemetry - Cleaner architecture than polling

### Advanced Features

- Voice command grammar - Full hands-free control
- Wake word detection - More natural interaction
- Conversation memory system - Persistent context
- Patrol/auto-scan - More engaging idle behavior

### Long-term

- Multi-language support
- Skills/plugins system
- Advanced camera features (multi-camera, PTZ)
- Comprehensive test suite and CI
