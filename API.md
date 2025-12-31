# API Documentation

Complete REST API reference for the Reachy Tracking & Control system.

## Base URL

```
http://localhost:8082
```

## Video Stream

### GET /video

Returns MJPEG video stream with annotated detections.

**Response**: `multipart/x-mixed-replace; boundary=frame`

**Usage**:

```html
<img src="http://localhost:8082/video" />
```

## Status & Configuration

### GET /api/status

Get current system status, telemetry, and backend information.

**Response**:

```json
{
  "status": "Tracking ID 3 (face)",
  "paused": false,
  "wiggle_enabled": true,
  "backend": {
    "python_version": "3.12.12",
    "robot_connected": true,
    "voice_enabled": true
  },
  "current_target_id": 3,
  "current_target": {
    "id": 3,
    "label": "face",
    "score": 320
  },
  "candidates": [
    {"id": 3, "label": "face", "score": 320}
  ],
  "volume": 50,
  "pose": {
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "head_roll": 0.0,
    "body_yaw": 0.0,
    "antenna_left": 0.0,
    "antenna_right": 0.0
  }
}
```

### POST /api/pause

Pause or resume tracking.

**Body**:

```json
{
  "paused": true
}
```

**Response**:

```json
{
  "status": "paused",
  "paused": true
}
```

### POST /api/wiggle

Enable or disable automatic antenna wiggles.

**Body**:

```json
{
  "enabled": true
}
```

**Response**:

```json
{
  "status": "ok",
  "wiggle_enabled": true
}
```

## Robot Control

### POST /api/move/head

Move the robot's head.

**Body**:

```json
{
  "pitch": 10.0,
  "yaw": -5.0,
  "roll": 0.0,
  "duration": 2.0
}
```

All angles in degrees. `duration` in seconds.

**Response**:

```json
{
  "status": "ok"
}
```

### POST /api/move/body

Move the robot's body yaw.

**Body**:

```json
{
  "yaw": 15.0,
  "duration": 2.0
}
```

**Response**:

```json
{
  "status": "ok"
}
```

### POST /api/move/antennas

Move the robot's antennas.

**Body**:

```json
{
  "left": 45.0,
  "right": -30.0,
  "duration": 1.0
}
```

Antenna values are raw degrees.

**Response**:

```json
{
  "status": "ok"
}
```

### POST /api/move/goto

Set absolute position for all joints.

**Body**:

```json
{
  "head": [10.0, -5.0, 2.0],
  "body": 15.0,
  "antennas": [45.0, -30.0],
  "duration": 2.0
}
```

Order: `[pitch, yaw, roll]` for head.

**Response**:

```json
{
  "status": "ok"
}
```

### POST /api/move/reset

Reset head and antennas to center (0, 0, 0).

**Response**:

```json
{
  "status": "ok"
}
```

### POST /api/motor_mode

Set motor compliance mode.

**Body**:

```json
{
  "mode": "stiff"
}
```

Values: `"stiff"`, `"soft"`, `"limp"`

**Response**:

```json
{
  "status": "ok",
  "mode": "stiff"
}
```

## Emotions

### POST /api/emote

Trigger an emotion animation.

**Body**:

```json
{
  "emotion": "happy"
}
```

**Available Emotions**:

- Basic: `happy`, `sad`, `surprised`, `angry`, `confused`, `scared`, `excited`, `bored`, `shy`
- Social: `greeting`, `waving`, `nodding`, `shaking_head`, `shrugging`
- Playful: `silly`, `curious`, `thinking`, `dancing`, `wiggle`
- Advanced: `love`, `sleepy`, `proud`, `disappointed`, `mischievous`, `focused`, `yawn`

**Response**:

```json
{
  "status": "ok",
  "emotion": "happy"
}
```

## Voice Assistant

### GET /api/voice/status

Get voice assistant status and model info.

**Response**:

```json
{
  "running": false,
  "listening": false,
  "models": {
    "whisper": {
      "name": "base",
      "dir": "C:\\...\\models\\whisper"
    },
    "llm": {
      "name": "google/gemma-2-2b-it",
      "dir": "C:\\...\\models\\llm",
      "device": "cpu",
      "dtype": "torch.float32"
    },
    "piper": {
      "path": "C:\\...\\models\\piper\\en_US-amy-medium.onnx"
    }
  },
  "error": null,
  "volume": 0.5
}
```

### POST /api/voice/start

Start voice listening.

**Response**:

```json
{
  "status": "ok",
  "listening": true
}
```

### POST /api/voice/stop

Stop voice listening.

**Response**:

```json
{
  "status": "ok",
  "listening": false
}
```

### GET /api/voice/audio-devices

Get list of available audio input devices.

**Response**:

```json
{
  "status": "ok",
  "devices": [
    {
      "id": 14,
      "name": "Headset Microphone (Plantronics BT600)",
      "channels": 1,
      "is_default": false,
      "is_selected": true
    }
  ],
  "selected_device_id": 14,
  "selected_device_name": "Headset Microphone (Plantronics BT600)"
}
```

### GET /api/voice/processing

Get current voice processing state (detecting wake word, listening, transcribing, generating response).

**Response**:

```json
{
  "is_processing": false
}
```

### GET /api/voice/transcript

Get last transcribed speech text (polling endpoint for chat UI).

**Response**:

```json
{
  "transcript": "Hello Reachy"
}
```

### GET /api/voice/response

Get last AI response text (polling endpoint for chat UI).

**Response**:

```json
{
  "response": "Hello! How can I help you?"
}
```

### POST /api/voice/listen_now

Trigger manual voice listening (Hold to Talk button).

**Response**:

```json
{
  "status": "ok",
  "message": "Listening for speech..."
}
```

### POST /api/voice/text_command

Send text command to LLM (optionally speak response).

**Body**:

```json
{
  "text": "Tell me a short story",
  "speak": true
}
```

**Response**:

```json
{
  "status": "ok",
  "response": "Once upon a time, in a quiet village..."
}
```

Note: Response returns immediately; TTS plays asynchronously in background.

### POST /api/voice/speak

Speak text directly without LLM processing.

**Body**:

```json
{
  "text": "Hello, I am Reachy!"
}
```

**Response**:

```json
{
  "status": "ok"
}
```

### POST /api/voice/volume

Set TTS volume.

**Body**:

```json
{
  "volume": 0.75
}
```

Value: 0.0 to 1.0 (0% to 100%)

**Response**:

```json
{
  "status": "ok",
  "volume": 0.75
}
```

### GET /api/voice/volume

Get current TTS volume.

**Response**:

```json
{
  "volume": 0.5
}
```

## LLM Configuration

### GET /api/llm/config

Get current LLM configuration.

**Response**:

```json
{
  "provider": "openai",
  "config": {
    "model": "gpt-5-nano",
    "api_key": "sk-...",
    "temperature": 0.7
  }
}
```

### POST /api/llm/config

Update LLM configuration.

**Body (OpenAI)**:

```json
{
  "provider": "openai",
  "model": "gpt-5-nano",
  "api_key": "sk-...",
  "temperature": 0.7
}
```

**Body (Ollama)**:

```json
{
  "provider": "ollama",
  "model": "llama2",
  "endpoint": "http://localhost:11434"
}
```

**Body (Local)**:

```json
{
  "provider": "local"
}
```

**Response**:

```json
{
  "status": "ok",
  "provider": "openai"
}
```

### POST /api/llm/test

Test current LLM configuration with a simple query.

**Response**:

```json
{
  "status": "ok",
  "response": "Hello! I'm working correctly."
}
```

## Tuning

### GET /api/tuning

Get current tuning parameters.

**Response**:

```json
{
  "detection_interval": 0.2,
  "command_interval": 1.2,
  "min_score_threshold": 250,
  "stream_fps_cap": 60,
  "detection_classes": ["face", "person"]
}
```

### POST /api/tuning

Update tuning parameters.

**Body**:

```json
{
  "detection_interval": 0.3,
  "command_interval": 1.5,
  "min_score_threshold": 300,
  "stream_fps_cap": 30
}
```

**Parameters**:

- `detection_interval`: Seconds between object detection runs (default: 0.2)
- `command_interval`: Minimum seconds between robot movements (default: 1.2)
- `min_score_threshold`: Minimum tracking score to select target (0-500, default: 250)
- `stream_fps_cap`: Maximum FPS for video stream (default: 60)
- `detection_classes`: Array of detection classes to enable

**Response**:

```json
{
  "status": "ok"
}
```

## WebSocket Events

### /ws

WebSocket connection for real-time updates.

**Server → Client Events**:

```json
{
  "type": "status_update",
  "data": {
    "status": "Tracking ID 3 (face)",
    "position": {...}
  }
}
```

```json
{
  "type": "voice_transcript",
  "data": {
    "text": "Hello Reachy"
  }
}
```

```json
{
  "type": "voice_response",
  "data": {
    "text": "Hello! How can I help you?"
  }
}
```

## Error Responses

All endpoints return standard error format:

```json
{
  "status": "error",
  "message": "Detailed error message"
}
```

HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Internal server error
- `503`: Service unavailable (robot not connected)

## Rate Limiting

No rate limiting currently implemented. Recommended client-side throttling:

- Status polling: Max 5 Hz (every 200ms)
- Manual control: Max 2 Hz (every 500ms)
- Voice commands: Sequential (wait for response)

## CORS

CORS is enabled for all origins in development. Configure for production use.
