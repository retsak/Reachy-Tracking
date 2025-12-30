# Troubleshooting Guide

Common issues and solutions for the Reachy Tracking & Control system.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Robot Connection Issues](#robot-connection-issues)
- [Camera Issues](#camera-issues)
- [Tracking Issues](#tracking-issues)
- [Voice Assistant Issues](#voice-assistant-issues)
- [Performance Issues](#performance-issues)
- [Audio Issues](#audio-issues)
- [Model Loading Issues](#model-loading-issues)

## Installation Issues

### Python Version Mismatch

**Symptom**: Import errors, module not found

**Solution**:

```bash
python --version  # Must be 3.10+
```

Reinstall with correct Python version:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### PowerShell Execution Policy

**Symptom**: Cannot activate virtual environment on Windows

**Error**: `cannot be loaded because running scripts is disabled`

**Solution**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Missing Build Tools

**Symptom**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:

Install Visual Studio Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Or use pre-built wheels:

```bash
pip install --only-binary :all: package-name
```

### Conflicting Dependencies

**Symptom**: Version conflicts during install

**Solution**:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --force-reinstall
```

## Robot Connection Issues

### Cannot Connect to Robot

**Symptom**: "Connecting to Robot..." never resolves

**Check**:

1. Robot daemon running?

```bash
curl http://localhost:8000/api/daemon/status
```

2. Correct host/port?

Edit `robot_controller.py` line ~15:

```python
def __init__(self, host='localhost'):
```

3. Firewall blocking?

```bash
# Windows
netsh advfirewall firewall add rule name="Reachy" dir=in action=allow protocol=TCP localport=8000

# Check if port is listening
netstat -an | findstr :8000
```

**Solution**: Ensure Reachy SDK daemon is running and accessible.

### Robot Moves Not Executing

**Symptom**: Commands sent but robot doesn't move

**Check**:

1. Is system paused?

Look for "Paused" status in dashboard. Click "Resume" button.

2. Are motors stiff?

Click "Motor Mode" → "Stiff" in dashboard.

3. Is robot SDK responding?

```bash
curl -X POST http://localhost:8000/api/move/goto \
  -H "Content-Type: application/json" \
  -d '{"head": [0, 0, 0], "duration": 1.0}'
```

### Manual Control Not Available

**Symptom**: Sliders disabled or not responding

**Solution**: System must be paused for manual control. Click "Pause" button first.

## Camera Issues

### Camera Not Detected

**Symptom**: Black video feed or "No camera detected"

**Check**:

1. Camera connected?

```bash
# Windows
Get-PnPDevice | Where-Object {$_.Class -eq "Camera"}

# Linux
ls -l /dev/video*
```

2. Camera index correct?

Edit `robot_controller.py` line ~110:

```python
camera_index = 1  # Try 0, 1, or 2
```

3. Camera in use by another app?

Close other apps using the camera (Zoom, Teams, etc.)

### Low Frame Rate / Laggy Video

**Symptom**: Choppy video, low FPS

**Solutions**:

1. Lower stream FPS cap:

Dashboard → Tuning → `stream_fps_cap` = 15-30

2. Reduce camera resolution:

Edit `robot_controller.py` line ~175:

```python
self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

3. Increase detection interval:

Dashboard → Tuning → `detection_interval` = 0.5

### Camera Too Dark/Bright

**Symptom**: Poor lighting in video feed

**Solution**: Adjust camera settings in `robot_controller.py` line ~20:

```python
# Too dark? Increase exposure or gain
self.win_exposure_value = -2  # less negative = brighter
self.win_gain = 15.0          # higher = brighter

# Too bright? Decrease
self.win_exposure_value = -6
self.win_gain = 8.0
```

### Camera Conflicts with Audio

**Symptom**: Camera freezes during TTS playback

**Cause**: SDK camera initialization conflict

**Solution**: Already handled in code. Camera pauses during audio. If still happening:

1. Check logs for OpenCV errors
2. Ensure SDK version is up to date
3. Restart application

## Tracking Issues

### No Detections

**Symptom**: No bounding boxes appear, robot doesn't track

**Check**:

1. Detection classes enabled?

Dashboard → Tuning → `detection_classes` includes target type

2. Score threshold too high?

Dashboard → Tuning → Lower `min_score_threshold` to 150-200

3. Lighting sufficient?

Improve lighting or adjust camera exposure

4. Object in view?

Ensure face/person clearly visible to camera

### Jittery Tracking

**Symptom**: Robot head shakes or moves erratically

**Solutions**:

1. Increase command interval:

Dashboard → Tuning → `command_interval` = 1.5-2.0

2. Increase score threshold:

Dashboard → Tuning → `min_score_threshold` = 300-350

3. Reduce detection frequency:

Dashboard → Tuning → `detection_interval` = 0.3-0.5

### Target Switching Too Often

**Symptom**: Robot switches between multiple targets

**Solution**:

1. Increase score threshold:

Dashboard → Tuning → `min_score_threshold` = 350

2. Limit detection classes:

Dashboard → Tuning → Select only one class (e.g., just "face")

3. Ensure single person in frame

### Robot Doesn't Recenter After Losing Target

**Symptom**: Robot stays in last position

**Expected Behavior**: System waits 5 seconds before recentering (idle timeout)

**Check**: Look for "Idle..." status after 5 seconds

**Solution**: Manually reset if needed:

```bash
curl -X POST http://localhost:8082/api/move/reset
```

### Tracking Fights with Speech

**Symptom**: Robot moves erratically while speaking

**Status**: Fixed in current version. Tracking automatically pauses during speech.

**If still occurring**: Check logs for errors, restart application.

## Voice Assistant Issues

### Voice Not Starting

**Symptom**: Clicking "Voice: Off" does nothing

**Check**:

1. Models downloaded?

```bash
python setup_model_assistant.py
```

2. Piper installed?

```bash
piper --version
```

3. Check logs for model loading errors

### No Speech Recognition

**Symptom**: Voice started but no transcription

**Check**:

1. Microphone working?

Test in system settings

2. ReachyMini MediaManager initialized?

Check logs for "Microphone recording active"

3. Speaking loud enough?

Energy threshold may need adjustment

### LLM Not Responding

**Symptom**: Transcription works but no response

**Check Provider**:

#### OpenAI

1. API key valid?

```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer sk-..."
```

2. Internet connection active?

3. API quota exceeded?

Check OpenAI dashboard

#### Ollama

1. Server running?

```bash
ollama list
curl http://localhost:11434/api/tags
```

2. Model pulled?

```bash
ollama pull llama2
```

#### Local

1. Enough RAM?

Check model size vs available memory

2. Hugging Face login (for Gemma)?

```bash
huggingface-cli login
```

3. Model downloaded?

First run downloads automatically. Check `models/llm/` directory.

### No TTS / Robot Silent

**Symptom**: Response shows in UI but robot doesn't speak

**Check**:

1. Piper voice installed?

Check `models/piper/` for `.onnx` and `.json` files

2. Piper CLI available?

```bash
piper --version
```

3. Volume not muted?

Dashboard → Voice panel → Volume slider

4. Check logs for TTS errors

### Conversation Memory Not Working

**Symptom**: Robot doesn't remember previous messages

**Check**:

1. Using correct LLM provider?

Local/Ollama should remember. OpenAI API has conversation history.

2. Check `conversation_history` length:

Should maintain last 8 messages

3. Review logs for context being sent

Look for `[LLM CONTEXT]` log entries

### Text Truncated in TTS

**Symptom**: Response cuts off mid-sentence

**Status**: Fixed in current version (removed 240-character limit)

**If still occurring**: Check logs, restart application

## Performance Issues

### High CPU Usage

**Symptoms**: Slow performance, fan noise

**Solutions**:

1. Reduce detection frequency:

```json
{
  "detection_interval": 0.5,
  "stream_fps_cap": 15
}
```

2. Lower camera resolution:

320x240 @ 15 FPS

3. Limit detection classes:

Enable only "face"

4. Close other applications

### High Memory Usage

**Symptoms**: System slow, swapping

**Solutions**:

1. Use smaller LLM model:

Switch to Qwen 2.5 0.5B or use OpenAI/Ollama

2. Restart application periodically

3. Close browser tabs

4. Check for memory leaks in logs

### Slow LLM Response

**Symptoms**: Long wait for text generation

**Solutions**:

1. Use cloud LLM:

OpenAI (gpt-5-nano) is fastest

2. Use Ollama with smaller model:

`ollama pull phi` (smaller than llama2)

3. Local optimization:

Ensure using quantized models (qint8)

4. Check CPU threads:

Should use all cores (logged at startup)

## Audio Issues

### Audio Overlapping

**Symptom**: Multiple TTS chunks play simultaneously

**Status**: Fixed in current version (proper chunk sequencing)

**If still occurring**:

1. Check `_audio_playing` event is working
2. Review logs for "Playing chunk X/Y" sequence
3. Restart application

### Audio Cutting Off

**Symptom**: TTS stops mid-sentence

**Status**: Fixed in current version (removed truncation, chunking improved)

**If still occurring**:

1. Check Piper not timing out
2. Review logs for "Piper TTS error"
3. Test Piper CLI directly:

```bash
echo "Test sentence" | piper --model models/piper/en_US-amy-medium.onnx --output_file test.wav
```

### Robotic/Distorted Audio

**Symptom**: TTS sounds wrong

**Causes**:

1. Sample rate mismatch
2. Audio buffer issues
3. Voice model corrupted

**Solutions**:

1. Re-download Piper voice
2. Check SDK audio settings
3. Try different voice model

### No Audio Output

**Symptom**: TTS plays but no sound

**Check**:

1. System volume not muted?

2. Robot speakers working?

Test with:

```bash
curl -X POST http://localhost:8082/api/voice/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Test"}'
```

3. Audio device selection:

ReachyMini SDK uses default audio device

### Robot Shakes After Speaking

**Symptom**: Jerky movement when TTS ends

**Status**: Fixed in current version (removed conflicting reset pose)

**If still occurring**: Check tracking not resuming too quickly

## Model Loading Issues

### Whisper Not Loading

**Error**: `Failed to load Whisper`

**Solutions**:

1. Check faster-whisper installed:

```bash
pip install faster-whisper
```

2. Clear cache and re-download:

```bash
rm -rf models/whisper
python setup_model_assistant.py
```

3. Check disk space

### LLM Not Loading

**Error**: `Failed to load LLM`

**Solutions**:

1. Check transformers installed:

```bash
pip install transformers torch
```

2. For Gemma, login to Hugging Face:

```bash
huggingface-cli login
```

3. Check RAM available:

```bash
# Windows
wmic OS get FreePhysicalMemory

# Linux/macOS
free -h
```

4. Try smaller model:

Force Qwen 0.5B by disabling others in `voice_assistant.py` line ~160

### Piper Not Working

**Error**: `Piper CLI not found`

**Solutions**:

1. Install Piper:

Windows: Download from https://github.com/rhasspy/piper/releases

Linux/macOS:

```bash
pip install piper-tts
```

2. Add to PATH

3. Test:

```bash
piper --version
```

### Model Download Stalled

**Symptom**: Setup script hangs during download

**Solutions**:

1. Check internet connection

2. Try Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. Manual download:

Visit Hugging Face model page and download manually to `models/` directory

4. Use setup script's retry logic

### CUDA/GPU Errors

**Error**: CUDA-related errors even though using CPU

**Solutions**:

1. Force CPU:

```python
# voice_assistant.py line ~150
self.device = "cpu"
```

2. Reinstall PyTorch CPU-only:

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. Set environment variable:

```bash
export CUDA_VISIBLE_DEVICES=""
```

## Getting Help

### Collecting Diagnostic Information

When reporting issues, include:

1. **System info**:

```bash
python --version
pip list
```

2. **Logs**: Console output with errors

3. **Configuration**:

```bash
cat .tuning_config.json
cat llm_config.json  # remove API keys!
```

4. **Model info**:

```bash
ls -lR models/
```

### Log Files

Check these locations:

- Console output (stdout/stderr)
- `voice_error.log` - Detailed LLM errors
- Browser console (F12) - UI errors

### Debug Mode

Enable verbose logging in `main.py` line ~17:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Community Support

- GitHub Issues: https://github.com/retsak/Reachy-Tracking/issues
- Include diagnostic info from above
- Describe expected vs actual behavior
- Include screenshots if applicable

### Known Limitations

1. **Windows camera backend**: DirectShow can be finicky
2. **Ollama streaming**: Not yet implemented
3. **Multi-camera**: Not supported
4. **GPU acceleration**: ONNX CPU-only for YOLOv8
5. **Rate limiting**: No API rate limiting implemented
