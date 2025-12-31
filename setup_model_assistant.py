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

def _get_hf_token():
    """Retrieve Hugging Face token from env or saved CLI login."""
    try:
        import os
        from huggingface_hub import HfFolder
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or HfFolder.get_token()
    except Exception:
        return None

def _ensure_hf_auth_interactive():
    """Attempt to run an interactive HF login if no token is present.
    Falls back to printing guidance if CLI isn't available or user cancels.
    """
    token = _get_hf_token()
    if token:
        return token
    print("\n⚠ Gemma access requires Hugging Face authentication.")
    print("   Launching login prompt... (you can skip if you prefer)")
    # Try new 'hf' CLI first, then legacy 'huggingface-cli'
    for cmd in ([sys.executable, "-m", "pip", "show", "huggingface_hub"],
                ["hf", "auth", "login"],
                ["huggingface-cli", "login"],):
        try:
            subprocess.run(cmd, check=True)
            break
        except Exception:
            continue
    # Re-check token after attempted login
    token = _get_hf_token()
    if token:
        print("✓ Hugging Face auth configured.")
    else:
        print("⚠ Hugging Face auth still missing. You can manually run:")
        print("   - hf auth login   (or)   huggingface-cli login")
        print("   And optionally set HF_TOKEN environment variable.")
    return token

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
            "google/gemma-2-2b-it",            # ~2B, preferred stable model
            "microsoft/Phi-3-mini-4k-instruct",# ~3B, capable
            "Qwen/Qwen2.5-0.5B-Instruct",      # ~0.5B, very light
            "microsoft/Phi-3.5-mini-instruct", # larger (~5GB shards), last resort
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # legacy fallback
        ]

        last_error = None
        hf_token = _get_hf_token()
        auth_kwargs = {"token": hf_token} if hf_token else {}

        for model_name in fallbacks:
            try:
                print(f"Downloading tokenizer for: {model_name}...")
                _ = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=str(LLM_DIR),
                    use_fast=True,
                    **auth_kwargs
                )
                print("Downloading model weights...")
                _ = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=str(LLM_DIR),
                    **auth_kwargs
                )
                print(f"✓ LLM cached: {model_name}")
                return True
            except Exception as e:
                last_error = e
                print(f"✗ Failed to cache {model_name}: {e}")
                # If Gemma fails due to missing auth, try to login interactively once and retry this model
                if model_name.startswith("google/gemma") and not hf_token:
                    hf_token = _ensure_hf_auth_interactive() or hf_token
                    auth_kwargs = {"token": hf_token} if hf_token else {}
                    if hf_token:
                        try:
                            print(f"Retrying with auth: {model_name}...")
                            _ = AutoTokenizer.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                cache_dir=str(LLM_DIR),
                                use_fast=True,
                                **auth_kwargs
                            )
                            _ = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                cache_dir=str(LLM_DIR),
                                **auth_kwargs
                            )
                            print(f"✓ LLM cached: {model_name}")
                            return True
                        except Exception as e2:
                            last_error = e2
                            print(f"✗ Still unable to cache {model_name}: {e2}")

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

def download_wake_word_models():
    """Download wake word detection models"""
    print("\n📥 Downloading wake word detection models...")
    try:
        from openwakeword import utils as oww_utils
        print("Downloading openwakeword models (this may take a moment)...")
        oww_utils.download_models()
        print("✓ Wake word models downloaded successfully")
        print("  Available wake words: alexa, hey_jarvis, hey_mycroft, hey_rhasspy")
        return True
    except ImportError:
        print("⚠ openwakeword not installed")
        print("   Installing openwakeword...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openwakeword"])
            from openwakeword import utils as oww_utils
            oww_utils.download_models()
            print("✓ Wake word models downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error installing/downloading wake word models: {e}")
            return False
    except Exception as e:
        print(f"✗ Error downloading wake word models: {e}")
        print("   Models will auto-download on first use")
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
        "huggingface-hub",
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
    
    # Check Hugging Face auth for gated models (e.g., Gemma)
    try:
        import os
        from huggingface_hub import HfFolder
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or HfFolder.get_token()
        if hf_token:
            print("\n✓ Hugging Face auth detected; gated models will be accessible.")
        else:
            print("\n⚠ No Hugging Face token found. If you need gated models like Gemma, run:")
            print("   - huggingface-cli login   (or)   python -m huggingface_hub login")
            print("   Then optionally set $env:HF_TOKEN for non-interactive environments.")
    except Exception:
        print("\n⚠ huggingface-hub not available; gated models may not be accessible.")

    # Download models
    print("\n2. Downloading AI models (into project-local cache)...")
    if not download_whisper_model():
        print("⚠ Whisper download failed, but you can retry later")
    if not download_llm_model():
        print("⚠ LLM download failed, it will download on first use")
    if not download_piper_voice():
        print("⚠ Piper voice download failed, TTS will be disabled")
    
    # Download wake word models
    print("\n3. Downloading wake word detection models...")
    if not download_wake_word_models():
        print("⚠ Wake word models will auto-download on first use")
    
    # Setup TTS
    print("\n4. Checking TTS setup...")
    setup_piper_tts()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the main server: python main.py")
    print("2. Open dashboard: http://localhost:8082")
    print("3. Click '🎤 Voice: Off' button to enable voice assistant")
    print("4. (Optional) Say 'Hey Jarvis' for hands-free activation")
    print("\n💡 Tips:")
    print("   - Configure wake word in Settings → Voice Settings")
    print("   - Choose from: Hey Jarvis, Alexa, Hey Mycroft, Hey Rhasspy")
    print("   - Adjust sensitivity and timeout to your environment")
    print("\n⚠ Note: First run may take longer as models are loaded into memory")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
