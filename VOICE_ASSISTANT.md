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

## API Endpoints

- `POST /api/voice/enable` - Start voice assistant
- `POST /api/voice/disable` - Stop voice assistant  
- `GET /api/voice/status` - Get conversation history
- `POST /api/voice/text_command` - Send text command
- `POST /api/voice/speak` - TTS only (no LLM)

## Performance Notes
- Models load into RAM on first use
- Conversation history kept to last 6 messages
- Audio processing runs in separate thread
- Camera auto-pauses during TTS playback
