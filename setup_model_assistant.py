"""
Setup script for AI Models and Dependencies
Downloads and configures YOLOv8, Whisper, LLM, and Piper TTS into the project-local models folder
"""

import sys
import subprocess
from pathlib import Path
import urllib.request

# Project-local models directory
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
LLM_DIR = MODELS_DIR / "llm"
PIPER_DIR = MODELS_DIR / "piper"
YOLO_MODEL = MODELS_DIR / "yolov8n.onnx"
for d in [MODELS_DIR, WHISPER_DIR, LLM_DIR, PIPER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def download_yolo_model():
    """Download YOLOv8n ONNX model if not present"""
    print("\n📥 Checking YOLOv8 model...")
    if YOLO_MODEL.exists():
        print(f"✓ YOLOv8 model found at {YOLO_MODEL}")
        return True
    
    print("Downloading YOLOv8n ONNX model...")
    try:
        url = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"
        urllib.request.urlretrieve(url, str(YOLO_MODEL))
        print("✓ YOLOv8 model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading YOLOv8: {e}")
        print("   You can manually download from: https://github.com/ultralytics/assets/releases")
        return False

def check_and_install_package(package_name):
    """Check if package is installed and install if needed"""
    try:
        __import__(package_name.replace("-", "_"))
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            return False

def download_whisper_model():
    """Download Whisper base model"""
    print("\n📥 Downloading Whisper model...")
    try:
        from faster_whisper import WhisperModel
        print("Loading Whisper base model (this may take a moment)...")
        model = WhisperModel("base", device="cpu", compute_type="int8", download_root=str(WHISPER_DIR))
        print("✓ Whisper model ready")
        return True
    except Exception as e:
        print(f"✗ Error downloading Whisper: {e}")
        return False

def download_llm_model():
    """Download one of the preferred small instruction-tuned LLMs into the local cache.
    Tries a fallback list from smallest → larger for faster setup on CPU.
    """
    print("\n📥 Downloading LLM (preferred small instruct models)...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Prefer smaller, CPU-friendly models to reduce cold start and disk usage
        fallbacks = [
            "Qwen/Qwen2.5-0.5B-Instruct",      # ~0.5B, very light
            "google/gemma-2-2b-it",            # ~2B, still reasonable
            "microsoft/Phi-3-mini-4k-instruct",# ~3B, capable
            "microsoft/Phi-3.5-mini-instruct", # larger (~5GB shards), last resort
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # legacy fallback
        ]

        last_error = None
        for model_name in fallbacks:
            try:
                print(f"Downloading tokenizer for: {model_name}...")
                _ = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=str(LLM_DIR),
                    use_fast=True
                )
                print("Downloading model weights...")
                _ = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=str(LLM_DIR)
                )
                print(f"✓ LLM cached: {model_name}")
                return True
            except Exception as e:
                last_error = e
                print(f"✗ Failed to cache {model_name}: {e}")

        print(f"✗ Error downloading LLMs: {last_error}")
        return False
    except Exception as e:
        print(f"✗ Error initializing Transformers for LLM download: {e}")
        return False

def download_piper_voice():
    """Download Piper voice model if not present"""
    print("\n📥 Downloading Piper TTS voice model...")
    
    # Check if any .onnx model already exists
    onnx_models = list(PIPER_DIR.glob("*.onnx"))
    if onnx_models:
        print(f"✓ Piper voice found: {onnx_models[0].name}")
        return True
    
    print("Downloading en_US-amy-medium voice...")
    try:
        # Download both .onnx and .json files from Hugging Face
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
        voice_name = "en_US-amy-medium"
        
        onnx_url = f"{base_url}/{voice_name}.onnx"
        json_url = f"{base_url}/{voice_name}.onnx.json"
        
        onnx_path = PIPER_DIR / f"{voice_name}.onnx"
        json_path = PIPER_DIR / f"{voice_name}.onnx.json"
        
        print(f"  Downloading {voice_name}.onnx...")
        urllib.request.urlretrieve(onnx_url, str(onnx_path))
        
        print(f"  Downloading {voice_name}.onnx.json...")
        urllib.request.urlretrieve(json_url, str(json_path))
        
        print("✓ Piper voice downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading Piper voice: {e}")
        print("   You can manually download from: https://huggingface.co/rhasspy/piper-voices")
        return False

def setup_piper_tts():
    """Check Piper CLI installation"""
    print("\n📥 Checking Piper CLI...")
    try:
        subprocess.check_call(["piper", "--version"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        print("✓ Piper CLI found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Piper CLI not found in PATH")
        print("   Download from: https://github.com/rhasspy/piper/releases")
        print("   Or install via: pip install piper-tts")
        return False

def main():
    print("=" * 60)
    print("AI Models & Voice Assistant Setup")
    print("=" * 60)
    
    # Download YOLO model first
    if not download_yolo_model():
        print("⚠ YOLOv8 download failed. Detection may not work.")
    
    # Check core dependencies
    print("\n1. Checking core dependencies...")
    packages = [
        "faster-whisper",
        "transformers",
        "torch",
        "webrtcvad",
        "soundfile",
        "scipy",
        "numpy",
        # Enable accelerated Hub downloads when Xet Storage is configured
        "hf_xet"
    ]
    
    all_ok = True
    for pkg in packages:
        if not check_and_install_package(pkg):
            all_ok = False
    
    if not all_ok:
        print("\n⚠ Some packages failed to install. Please install manually.")
        return False
    
    # Download models
    print("\n2. Downloading AI models (into project-local cache)...")
    if not download_whisper_model():
        print("⚠ Whisper download failed, but you can retry later")
    if not download_llm_model():
        print("⚠ LLM download failed, it will download on first use")
    if not download_piper_voice():
        print("⚠ Piper voice download failed, TTS will be disabled")
    
    # Setup TTS
    print("\n3. Checking TTS setup...")
    setup_piper_tts()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the main server: python main.py")
    print("2. Open dashboard: http://localhost:8082")
    print("3. Click '🎤 Voice: Off' button to enable voice assistant")
    print("\n⚠ Note: First run may take longer as models are loaded into memory")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
