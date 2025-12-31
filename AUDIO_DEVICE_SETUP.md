# Audio Device Configuration for Wake Word Detection

## Overview

The voice assistant now supports cross-platform audio device selection with automatic detection of the Reachy Mini Audio device. This ensures wake word detection uses the correct microphone on Windows, macOS, and Linux.

## Device Selection Priority

When the voice assistant starts, it follows this priority:

1. **User-Selected Device** - If a device has been configured via settings, it uses that first
2. **Reachy Device Detection** - Automatically finds "Reachy", "Echo", or "Speakerphone" devices
3. **Default System Device** - Falls back to the OS default input device
4. **First Available Device** - Uses the first available input device if nothing else works

## API Endpoints

### Get Available Audio Devices
```
GET /api/voice/audio-devices
```

Returns list of all available input devices:
```json
{
  "status": "ok",
  "devices": [
    {
      "id": 0,
      "name": "Microphone (USB Audio Device)",
      "channels": 2,
      "is_default": false,
      "is_selected": false
    },
    {
      "id": 14,
      "name": "Echo Cancelling Speakerphone (Reachy Mini Audio)",
      "channels": 2,
      "is_default": false,
      "is_selected": true
    }
  ],
  "selected_device_id": 14,
  "selected_device_name": "Echo Cancelling Speakerphone (Reachy Mini Audio)"
}
```

### Configure Wake Word with Audio Device
```
POST /api/voice/wake-word/configure
```

Set both wake word model and audio device:
```json
{
  "enabled": true,
  "wake_word": "oh_kay_computer",
  "threshold": 0.5,
  "timeout": 5.0,
  "audio_device_id": 14
}
```

### Get Wake Word Status
```
GET /api/voice/wake-word/status
```

Includes current audio device selection:
```json
{
  "enabled": true,
  "wake_word": "oh_kay_computer",
  "threshold": 0.5,
  "timeout": 5.0,
  "audio_device_id": 14,
  "audio_device_name": "Echo Cancelling Speakerphone (Reachy Mini Audio)",
  "available_models": ["alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "hay_ree_chee", "hey_ree_shee", "oh_kay_computer"],
  "model_loaded": true
}
```

## Code Changes

### voice_assistant.py

**New Attributes:**
- `audio_device_id` - Selected device ID (None = auto-detect)
- `audio_device_name` - Human-readable device name

**New Methods:**
- `get_available_audio_devices()` - List all input devices
- `set_audio_device(device_id)` - Set which device to use
- `_find_reachy_audio_device()` - Updated to respect user selection

**Updated Methods:**
- `_start_pc_microphone()` - Now uses `_find_reachy_audio_device()` with priority logic

### main.py

**New Endpoints:**
- `GET /api/voice/audio-devices` - List available audio devices

**Updated Endpoints:**
- `POST /api/voice/wake-word/configure` - Now accepts `audio_device_id` parameter
- `GET /api/voice/wake-word/status` - Returns audio device info

**Startup Sequence:**
- Loads saved audio device ID on application start
- Applies device selection before loading wake word model

**Settings Persistence:**
- Audio device ID saved in `.wake_word_config.json`
- Automatically restored on application restart

## Configuration File

Settings are saved in `.wake_word_config.json`:
```json
{
  "enabled": true,
  "wake_word": "oh_kay_computer",
  "threshold": 0.5,
  "timeout": 5.0,
  "audio_device_id": 14
}
```

## Cross-Platform Compatibility

The implementation uses `sounddevice` library which works on:
- **Windows** - DirectSound, WASAPI, MME
- **macOS** - Core Audio
- **Linux** - ALSA, PulseAudio, JACK

Device detection automatically works across all platforms by:
1. Querying `sounddevice.query_devices()`
2. Filtering devices with input capabilities
3. Searching for Reachy-related device names
4. Falling back to system defaults

## Testing

### List Available Devices
```python
from voice_assistant import VoiceAssistant

assistant = VoiceAssistant(robot_controller)
devices = assistant.get_available_audio_devices()
for device in devices:
    print(f"{device['id']}: {device['name']} - {device['channels']} ch")
```

### Set Specific Device
```python
assistant.set_audio_device(14)  # Device 14 is Reachy Mini Audio
```

### Verify Selection
```python
response = requests.get("http://localhost:8000/api/voice/audio-devices")
data = response.json()
print(f"Selected: {data['selected_device_name']}")
```

## Troubleshooting

### Wake word not detecting
1. Verify correct audio device is selected:
   ```
   GET /api/voice/audio-devices
   ```
2. Check device has input capability (channels > 0)
3. Verify device volume in system settings
4. Test with `test_wake_words.py` to confirm device works

### Device list shows "Stereo Mix" instead of microphone
- Stereo Mix is a virtual device for recording system audio
- Select the actual Microphone device instead
- On Reachy: select "Echo Cancelling Speakerphone (Reachy Mini Audio)"

### Device number changes after restart
- This is normal - device IDs can shift if USB devices are added/removed
- The application auto-detects Reachy device by name
- Manual selection persists in settings file
