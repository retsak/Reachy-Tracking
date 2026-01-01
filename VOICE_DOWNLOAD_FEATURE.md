## Voice Download Feature - Implementation Summary

### Overview
Added automatic voice downloading functionality so users can select and download Piper TTS voices directly from the UI without manual file management.

### Components Implemented

#### 1. Backend (voice_assistant.py)
**New imports:**
- `json`, `requests`, `Tuple` type hint

**New method: `download_voice(voice_name: str) -> Tuple[bool, str]`**
- Downloads .onnx and .onnx.json files from Hugging Face
- Handles voice name parsing (en_US, en_GB, etc.)
- Constructs correct URLs based on language/region
- Streams download with progress tracking
- Error handling for timeouts and failed downloads
- Returns (success: bool, message: str)

**URL Format:**
```
https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{region}/{voice_name}.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{region}/{voice_name}.onnx.json
```

**Features:**
- Validates both .onnx and .onnx.json exist before declaring success
- Skips download if voice already exists (idempotent)
- Automatic fallback for unknown regions
- Detailed error messages with HTTP status codes
- Timeout handling (60s for ONNX, 30s for JSON)

#### 2. API Endpoints (main.py)
**New endpoint: `POST /api/voice/voices/download`**

Request:
```json
{"voice": "en_US-amy-medium"}
```

Success response:
```json
{
  "status": "ok",
  "message": "Successfully downloaded en_US-amy-medium",
  "voice": "en_US-amy-medium",
  "available_voices": ["en_US-amy-medium", "en_GB-alan-medium"]
}
```

Error response:
```json
{
  "status": "error",
  "message": "Failed to download en_US-amy-medium (HTTP 404)",
  "voice": "en_US-amy-medium"
}
```

**Behavior:**
- Initializes voice_assistant if needed
- Calls download_voice() method
- Auto-refreshes available voices list on success
- Returns updated voice list so UI can show new voices

#### 3. UI Updates (static/index.html)

**New HTML Elements:**
- Download button: `⬇️ Download Voice` (next to "💾 Set Voice")
- Progress container with:
  - Voice name display
  - Progress bar (0-100%)
  - Status text ("Please wait...")

**New JavaScript Functions:**

`downloadVoiceBtn.onclick` handler:
- Gets selected voice from dropdown
- Shows progress indicator
- Calls /api/voice/voices/download
- Hides progress on completion
- Reloads voice list
- Shows success/error alert

**Progress Display:**
```
Downloading: en_GB-alan-medium
[████████████████████████] 100%
Please wait...
```

### User Workflow

1. **Select Voice:**
   - User opens Voice Settings tab
   - Dropdown populates with available voices (grouped by region)
   - User selects a voice (e.g., "en_GB-alan-medium")

2. **Download (if needed):**
   - If voice not downloaded yet, click "⬇️ Download Voice"
   - Progress bar shows download status
   - "Please wait..." message displayed
   - Backend streams files from Hugging Face

3. **Use Voice:**
   - After download completes, click "💾 Set Voice"
   - Robot now uses the selected voice for TTS

4. **Repeat:**
   - Can download and switch between voices as desired

### Supported Voices
Currently optimized for English variants:
- **US English (en_US):** Amy (medium)
- **British English (en_GB):** Alan (medium), and others available

Available voices listed at:
- https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US
- https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB

### Error Handling

**Download Errors Handled:**
- ✓ Invalid voice name format
- ✓ HTTP 404 (voice not found on Hugging Face)
- ✓ HTTP 403 (access denied)
- ✓ Network timeout (60s limit)
- ✓ Partial download recovery (removes .onnx if .json fails)
- ✓ File system permissions

**User Feedback:**
- Alert messages for success/failure
- Console logging for debugging
- Progress indicator for long operations
- Status text updates

### Technical Details

**Dependencies:**
- `requests` - HTTP library for downloads (already in requirements.txt)
- `pathlib.Path` - Cross-platform file handling

**File Storage:**
- Location: `models/piper/{voice_name}.onnx` and `.onnx.json`
- No subdirectories (stored flat for simplicity)
- Automatic directory creation if missing

**Download Strategy:**
- Streaming download (memory efficient for large files)
- Chunk size: 8192 bytes
- Progress tracking in logs
- Both files required for voice to be valid

### Testing

Run test script:
```bash
python test_voice_download.py
```

Test output shows:
- Available voices enumeration
- Download method existence and format
- API endpoint structure
- Download URL format verification

### Future Enhancements

Possible additions:
1. **Progress percentage in UI** - Currently shows bar, could add %
2. **Cancel button** - Allow interrupting downloads
3. **Voice preview** - Play sample of voice before downloading
4. **Automatic download** - Pre-download common voices on startup
5. **Delete voice** - Remove downloaded voices from UI
6. **Voice ratings** - Show quality/variant info from Hugging Face
7. **Batch download** - Download multiple voices at once
8. **Persistent storage** - Save voice preferences to config file

### Files Modified

1. **voice_assistant.py**
   - Added imports: json, requests, Tuple
   - Added method: download_voice()
   - ~120 lines of new code

2. **main.py**
   - Added endpoint: POST /api/voice/voices/download
   - ~40 lines of new code

3. **static/index.html**
   - Added HTML: Download button, progress indicator
   - Added JavaScript: Download handler
   - ~40 lines of new code

4. **test_voice_download.py** (new)
   - Test script to verify functionality

### Deployment Notes

- Ensure `requests` module is installed: `pip install requests`
- No database changes needed
- No configuration changes needed
- Voice files stored in existing `models/piper/` directory
- Backward compatible with existing voice selection

### Troubleshooting

**Issue:** "No module named 'requests'"
**Solution:** `pip install requests`

**Issue:** Download times out
**Solution:** Check internet connection, Hugging Face may be slow

**Issue:** Voice downloaded but not appearing in list
**Solution:** Refresh page or click "Load Voices" button (might need to add explicit refresh)

**Issue:** "Voice not found" error
**Solution:** Check voice name format (en_US-name-variant) on Hugging Face

---

**Status:** ✅ Ready for deployment and testing
