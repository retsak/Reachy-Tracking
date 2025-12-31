# Voice Assistant Quick Start

## Overview
The Reachy Tracking system now includes an interactive voice assistant that makes the robot conversational like a Google Home or Alexa.

## Features
- 🎤 **Microphone Input**: Real-time audio capture
- 🗣️ **Speech Recognition**: Local Whisper for accurate STT  
- 🤖 **Conversational AI**: Qwen3-1.7B for natural responses
- 🔊 **Text-to-Speech**: Piper TTS with natural Amy voice
- 💬 **Context Awareness**: Remembers conversation history
- ⌨️ **Text Commands**: Alternative to voice input

## Quick Setup

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Download Models (Optional)
```powershell
python setup_voice_assistant.py
```

This downloads:
- Whisper Base model (~150MB) 
- Optionally Qwen3-1.7B (~3GB)
- Checks for Piper TTS

**Note**: Models auto-download on first use if you skip this step.

### 3. Install Piper TTS

**Windows**:
- Download from: https://github.com/rhasspy/piper/releases
- Extract and add to PATH, or place in project directory

**Linux/Mac**:
```bash
pip install piper-tts
```

## Usage

### Enable Voice Assistant
1. Start the server: `python main.py`
2. Open dashboard: http://localhost:8082
3. Click **"🎤 Voice: Off"** button
4. Wait for models to load (30-60 seconds first time)
5. Start speaking when button shows **"🎤 Voice: On"**

### Text Commands
Alternative to voice:
1. Type in "Text Command" input box
2. Click **"Send"** (text only) or **"Send & Speak"** (with TTS)

### Example Conversations
- "Hello, what can you see?"
- "Tell me about yourself"
- "What are you tracking right now?"
- "Tell me a joke"

## System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **First Load**: 30-60 seconds (models loading)
- **Response Time**: 2-5 seconds on CPU, 1-2 seconds with GPU

## Models Used
- **STT**: faster-whisper (Whisper Base) - 150MB
- **LLM**: Qwen3-1.7B-Instruct - 3GB
- **TTS**: Piper (Amy Medium voice)

## Troubleshooting

### Voice not working
- Check microphone permissions
- Ensure Reachy SDK MediaManager is accessible
- Check logs for model loading errors

### Models not downloading
- Run `python setup_voice_assistant.py` manually
- Check internet connection
- Verify disk space (~4GB needed)

### Slow responses
- First run is slower (model loading)
- CPU mode: 2-5 seconds per response
- Consider GPU acceleration for faster inference

### TTS not playing
- Verify Piper is installed and in PATH
- Check `piper --version` works
- Audio conflicts resolved (camera pauses during playback)

## Advanced Configuration

### Change System Prompt
Edit `voice_assistant.py` line ~51:
```python
self.system_prompt = """Your custom prompt here..."""
```

### Use Different Model
Edit `voice_assistant.py` `_load_llm()` method:
```python
model_name = "your-preferred-model"
```

### Adjust VAD Sensitivity
Edit `voice_assistant.py` line ~158:
```python
is_speech = energy > 0.01  # Lower = more sensitive
```

## Wake Word Detection

### Overview
Wake word detection allows hands-free voice activation - just say the wake word (like "Hey Jarvis" or "Alexa") and the robot will start listening for your command. No button press needed!

### Features
- 🎤 **Hands-Free Activation**: Say the wake word to activate
- 🔄 **Multiple Wake Words**: Choose from Alexa, Hey Jarvis, Hey Mycroft, Hey Rhasspy
- ⚙️ **Adjustable Sensitivity**: Tune detection threshold to your environment
- ⏱️ **Timeout Control**: Configure how long to listen after wake word
- 💻 **CPU-Friendly**: ONNX-based models run efficiently on CPU

### Setup

Wake word detection is enabled by default. The required library (`openwakeword`) is included in `requirements.txt`.

On first use, wake word models (~10MB) will auto-download from the openwakeword repository.

### Configuration

#### Via Dashboard (Web UI)
1. Open dashboard: http://localhost:8082
2. Click **"⚙️ Settings"** button
3. Select **"Voice Settings"** tab
4. Configure wake word options:
   - **Enable/Disable**: Toggle wake word detection on/off
   - **Wake Word**: Select from available models
   - **Sensitivity**: Adjust detection threshold (0.1-0.9)
   - **Timeout**: Set listening duration after wake word (3-15 seconds)
5. Click **"💾 Save Wake Word Settings"**
6. Restart voice assistant for changes to take effect

#### Via API
**Get Current Settings:**
```bash
curl http://localhost:8082/api/voice/wake-word/status
```

**Configure Settings:**
```bash
curl -X POST http://localhost:8082/api/voice/wake-word/configure \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "wake_word": "hey_jarvis",
    "threshold": 0.5,
    "timeout": 5.0
  }'
```

### Available Wake Words

**Custom Models** (trained specifically for Reachy):
- **hay_ree_chee** - "Hay Reachy" (optimized pronunciation)
- **hey_ree_shee** - "Hey Reachy" (alternative pronunciation)
- **oh_kay_computer** - "Okay Computer" (pop culture reference)

**Built-in Models** (openw akeword defaults):
- **hey_jarvis** - "Hey Jarvis" (Iron Man reference)
- **alexa** - "Alexa" (Amazon)
- **hey_mycroft** - "Hey Mycroft" (Mycroft AI)
- **hey_rhasspy** - "Hey Rhasspy" (Rhasspy voice assistant)

**Custom Model Location**: Place `.onnx` files in `models/openwakeword/` directory for automatic detection.

### Usage Flow

1. **Enable Voice Assistant**: Click "🎤 Voice: Off" button
2. **Say Wake Word**: Speak "Hey Jarvis" (or your chosen wake word)
3. **Robot Acknowledges**: Robot will show "excited" emotion when wake word detected
4. **Speak Command**: You have 5 seconds (configurable) to speak your command
5. **Robot Responds**: Robot processes your speech and responds
6. **Returns to Listening**: After response, robot returns to wake word detection mode

### Tuning Sensitivity

**Threshold Values:**
- **0.001-0.3**: Very sensitive (more false triggers in noisy environments)
- **0.4-0.6**: Balanced (recommended for most use cases)
- **0.7-0.9**: Very selective (may miss wake word if not spoken clearly)

**Default**: 0.5

### Audio Device Selection

Choose which microphone to use for wake word detection:

**Via Dashboard:**
1. Settings → Voice Settings tab
2. Select audio input device from dropdown
3. Devices are auto-detected with Reachy Mini Audio prioritized
4. Save settings to persist across restarts

**Via API:**
```bash
curl http://localhost:8082/api/voice/audio-devices
```

**Device Priority:**
1. User-selected device (if configured)
2. Reachy device (auto-detected: "Reachy", "Echo", "Speakerphone")
3. System default device
4. First available input device

**Supported Platforms:**
- Windows: DirectSound, WASAPI, MME
- macOS: Core Audio
- Linux: ALSA, PulseAudio, JACK

**Tips:**
- Start with default (0.5) and adjust based on your environment
- Noisy environment → Increase threshold (0.6-0.7)
- Quiet environment → Decrease threshold (0.3-0.4)
- Test in actual usage conditions

### Timeout Control

**Timeout** determines how long the robot listens for commands after the wake word is detected.

- **3-5 seconds**: Quick commands only
- **5-10 seconds**: Standard conversations (recommended: 5s)
- **10-15 seconds**: Longer, complex commands

**Default**: 5 seconds

### Disabling Wake Word

If you prefer manual activation (button press), you can disable wake word detection:

1. Dashboard → Settings → Voice Settings
2. Uncheck "Enable Wake Word Detection"
3. Save settings
4. Restart voice assistant

With wake word disabled, the robot will continuously listen for speech when voice assistant is enabled (original behavior).

### Technical Details

**Library**: [openwakeword](https://github.com/dscripka/openWakeWord)
- Open source, no API keys required
- ONNX-based models for CPU efficiency
- ~10MB model downloads on first use
- 80ms audio chunks at 16kHz for detection

**Detection Process**:
1. Continuous audio stream from robot microphone
2. 80ms chunks processed through wake word model
3. Score computed for wake word presence (0.0-1.0)
4. If score > threshold, activate listening mode
5. Capture speech until silence detected (0.9s)
6. Process speech through STT → LLM → TTS
7. Return to wake word detection

**Performance**:
- Detection latency: <100ms
- CPU overhead: ~2-5% while waiting for wake word
- Memory: +50MB for wake word models

## API Endpoints

- `POST /api/voice/enable` - Start voice assistant
- `POST /api/voice/disable` - Stop voice assistant  
- `GET /api/voice/status` - Get conversation history
- `POST /api/voice/text_command` - Send text command
- `POST /api/voice/speak` - TTS only (no LLM)
- `GET /api/voice/wake-word/status` - Get wake word configuration
- `POST /api/voice/wake-word/configure` - Update wake word settings

## Performance Notes
- Models load into RAM on first use
- Conversation history kept to last 8 messages
- Audio processing runs in separate thread
- Camera auto-pauses during TTS playback
- Wake word detection adds ~2-5% CPU overhead when enabled
