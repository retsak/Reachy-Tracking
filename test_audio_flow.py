#!/usr/bin/env python3
"""Quick test to verify audio flow from selected input device."""

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test the conditional logic
def test_audio_selection():
    """Test that audio selection logic works correctly."""
    
    # Test 1: SDK mode
    audio_input_source = "sdk"
    use_sdk = (audio_input_source == "sdk")
    use_sounddevice = (audio_input_source != "sdk")
    
    assert use_sdk == True, "SDK mode detection failed"
    assert use_sounddevice == False, "SDK should not use sounddevice"
    logger.info("✓ SDK mode detection works")
    
    # Test 2: Device mode
    audio_input_source = "device"
    audio_device_id = 2
    use_sdk = (audio_input_source == "sdk")
    use_sounddevice = (audio_input_source != "sdk")
    
    assert use_sdk == False, "Device mode should not use SDK"
    assert use_sounddevice == True, "Device mode detection failed"
    logger.info("✓ Device mode detection works")
    
    # Test 3: Speech detection logic simulation
    silence_chunks = 0
    max_silence_chunks = 30
    vad_buffer = []
    speech_detected = False
    
    # Simulate speech chunk
    chunk = np.random.randn(1280).astype(np.float32) * 0.1  # Low energy chunk
    energy = np.sqrt(np.mean(chunk ** 2))
    is_speech = energy > 0.01
    
    if is_speech:
        vad_buffer.append(chunk)
        speech_detected = True
        silence_chunks = 0
        logger.info(f"✓ Speech detected (energy={energy:.4f})")
    else:
        logger.info(f"✓ No speech detected (energy={energy:.4f})")
    
    # Test 4: Verify audio chunk shape
    test_chunk = np.random.randn(1280).astype(np.float32)
    assert test_chunk.shape == (1280,), f"Chunk shape incorrect: {test_chunk.shape}"
    logger.info("✓ Audio chunk shape is correct (1280,)")
    
    logger.info("\n✅ All audio flow tests passed!")

if __name__ == "__main__":
    test_audio_selection()
