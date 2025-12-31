"""
Test wake word detection using openwakeword
"""
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_openwakeword():
    """Test openwakeword library"""
    try:
        import openwakeword
        from openwakeword.model import Model
        
        logger.info("✓ openwakeword installed successfully")
        
        # Download pre-trained models if needed
        logger.info("Downloading pre-trained wake word models...")
        from openwakeword import utils
        utils.download_models()
        
        logger.info("✓ Models downloaded")
        
        # Initialize model with default wake words
        # Available models: alexa, hey_jarvis, hey_mycroft, hey_rhasspy, timer
        owwModel = Model(
            wakeword_models=["alexa"],  # Start with alexa model (widely tested)
            inference_framework='onnx'
        )
        
        logger.info(f"✓ Model loaded. Available models: {list(owwModel.models.keys())}")
        
        # Test with silence (should not trigger)
        test_audio = np.zeros(1280, dtype=np.int16)  # 80ms of silence at 16kHz
        prediction = owwModel.predict(test_audio)
        
        logger.info(f"✓ Prediction on silence: {prediction}")
        logger.info("  (Scores should be close to 0.0 for silence)")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ openwakeword not installed: {e}")
        logger.info("Install with: pip install openwakeword")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing openwakeword: {e}", exc_info=True)
        return False

def test_pvporcupine():
    """Test Picovoice Porcupine library"""
    try:
        import pvporcupine
        
        logger.info("✓ pvporcupine installed successfully")
        
        # Note: Requires access key from Picovoice Console
        logger.info("! Porcupine requires API key from https://console.picovoice.ai/")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ pvporcupine not installed: {e}")
        logger.info("Install with: pip install pvporcupine")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing pvporcupine: {e}")
        return False

def test_vosk():
    """Test Vosk library for keyword spotting"""
    try:
        from vosk import Model as VoskModel, KaldiRecognizer
        
        logger.info("✓ vosk installed successfully")
        logger.info("! Vosk requires downloading language model (~50MB)")
        logger.info("  Download from: https://alphacephei.com/vosk/models")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ vosk not installed: {e}")
        logger.info("Install with: pip install vosk")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing vosk: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Wake Word Detection Library Test")
    logger.info("=" * 60)
    
    results = {}
    
    logger.info("\n[1/3] Testing openwakeword (recommended for open source)...")
    results['openwakeword'] = test_openwakeword()
    
    logger.info("\n[2/3] Testing pvporcupine (Picovoice - requires API key)...")
    results['pvporcupine'] = test_pvporcupine()
    
    logger.info("\n[3/3] Testing vosk (requires model download)...")
    results['vosk'] = test_vosk()
    
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info("=" * 60)
    for lib, available in results.items():
        status = "✓ Available" if available else "✗ Not installed"
        logger.info(f"{lib:15s}: {status}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Recommendation:")
    logger.info("=" * 60)
    if results.get('openwakeword'):
        logger.info("✓ openwakeword is ready to use!")
        logger.info("  - Fully open source")
        logger.info("  - CPU-friendly ONNX models")
        logger.info("  - Supports custom wake words")
        logger.info("  - No API key required")
    else:
        logger.info("Install openwakeword: pip install openwakeword")
