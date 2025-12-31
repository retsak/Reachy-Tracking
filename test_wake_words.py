#!/usr/bin/env python3
"""
Wake Word Model Testing Script
Tests wake word detection accuracy using the computer's built-in microphone.
"""

import os
import sys
import numpy as np
import sounddevice as sd
import queue
import time
from openwakeword.model import Model

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz
DETECTION_THRESHOLD = 0.5  # Default threshold

# Audio queue for processing
audio_queue = queue.Queue()

def list_audio_devices():
    """List all available audio devices"""
    print("\n" + "="*60)
    print("AVAILABLE AUDIO DEVICES")
    print("="*60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        marker = " ← DEFAULT INPUT" if i == sd.default.device[0] else ""
        print(f"{i}: {device['name']}")
        print(f"   Input channels: {device['max_input_channels']}, Output channels: {device['max_output_channels']}{marker}")
    print("="*60)
    return devices

def select_input_device():
    """Interactively select an input device"""
    devices = list_audio_devices()
    
    # Filter to devices with input
    input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    
    print(f"\nFound {len(input_devices)} input device(s)")
    
    if not input_devices:
        print("❌ No input devices found!")
        return None
    
    device_choice = input(f"Enter device number (default {sd.default.device[0]}): ").strip()
    
    if not device_choice:
        return sd.default.device[0]
    
    try:
        device_id = int(device_choice)
        if device_id in [i for i, _ in input_devices]:
            return device_id
        else:
            print(f"Invalid device. Using default device {sd.default.device[0]}")
            return sd.default.device[0]
    except ValueError:
        print(f"Invalid input. Using default device {sd.default.device[0]}")
        return sd.default.device[0]

def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream - pushes audio to queue"""
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    # Convert to mono if stereo
    if indata.shape[1] > 1:
        audio_data = np.mean(indata, axis=1)
    else:
        audio_data = indata[:, 0]
    audio_queue.put(audio_data.copy())

def list_available_models():
    """List all available wake word models"""
    custom_dir = os.path.join(os.path.dirname(__file__), "models", "openwakeword")
    
    print("\n" + "="*60)
    print("AVAILABLE WAKE WORD MODELS")
    print("="*60)
    
    # Built-in models
    print("\n📦 Built-in Models:")
    builtin = ["alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy"]
    for i, model in enumerate(builtin, 1):
        print(f"  {i}. {model}")
    
    # Custom models
    custom_models = []
    if os.path.exists(custom_dir):
        import glob
        custom_models = [os.path.splitext(os.path.basename(f))[0] 
                        for f in glob.glob(os.path.join(custom_dir, "*.onnx"))]
        if custom_models:
            print("\n🤖 Custom Models:")
            for i, model in enumerate(custom_models, len(builtin) + 1):
                print(f"  {i}. {model}")
    
    return builtin + custom_models

def load_model(model_name):
    """Load a wake word model by name or path"""
    custom_dir = os.path.join(os.path.dirname(__file__), "models", "openwakeword")
    custom_path = os.path.join(custom_dir, f"{model_name}.onnx")
    
    print(f"\n🔄 Loading model: {model_name}...")
    
    try:
        if os.path.exists(custom_path):
            # Load custom model by path
            print(f"   Using custom model from: {custom_path}")
            model = Model(wakeword_models=[custom_path], inference_framework='onnx')
        else:
            # Load built-in model by name
            print(f"   Using built-in model: {model_name}")
            model = Model(wakeword_models=[model_name], inference_framework='onnx')
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model keys: {list(model.models.keys())}")
        return model
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_model(model, model_name, threshold=DETECTION_THRESHOLD, duration=60, device=None):
    """Test wake word detection with live audio"""
    print("\n" + "="*60)
    print(f"TESTING: {model_name}")
    print("="*60)
    print(f"Threshold: {threshold}")
    print(f"Duration: {duration}s")
    print(f"Device: {device}")
    print("\n💬 Speak LOUDLY and CLEARLY for best results!")
    print("   Say the wake word multiple times")
    print("   (Press Ctrl+C to stop)")
    print("-"*60)
    
    # Clear queue
    while not audio_queue.empty():
        audio_queue.get()
    
    # Start audio stream
    try:
        stream = sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            latency='low'
        )
    except Exception as e:
        print(f"❌ Error opening audio stream: {e}")
        return
    
    audio_buffer = []
    detection_count = 0
    start_time = time.time()
    last_detection_time = 0
    check_count = 0
    max_score = 0.0
    
    try:
        with stream:
            print("🎤 Listening...")
            
            while time.time() - start_time < duration:
                try:
                    # Get audio chunk
                    chunk = audio_queue.get(timeout=1.0)
                    audio_buffer.append(chunk)
                    
                    # Process when we have enough samples
                    if sum(len(c) for c in audio_buffer) >= CHUNK_SIZE:
                        audio_data = np.concatenate(audio_buffer)
                        detection_chunk = audio_data[:CHUNK_SIZE]
                        
                        # Calculate audio level
                        rms = np.sqrt(np.mean(detection_chunk ** 2))
                        
                        # Convert to int16 for openwakeword
                        if detection_chunk.dtype in [np.float32, np.float64]:
                            detection_chunk = np.clip(detection_chunk, -1.0, 1.0)
                            detection_chunk = (detection_chunk * 32767).astype(np.int16)
                        
                        # Run detection
                        prediction = model.predict(detection_chunk)
                        
                        # Get score for this model
                        score = max(prediction.values()) if prediction else 0.0
                        max_score = max(max_score, score)
                        
                        # Keep remainder for next iteration
                        audio_buffer = [audio_data[CHUNK_SIZE:]] if len(audio_data) > CHUNK_SIZE else []
                        
                        # Display detection results - show every 10th check and all elevated scores
                        check_count += 1
                        current_time = time.time()
                        
                        if score > threshold:
                            if current_time - last_detection_time > 2.0:  # Debounce 2s
                                detection_count += 1
                                print(f"✨ DETECTED! Score: {score:.6f} RMS: {rms:.3f} (#{detection_count})")
                                last_detection_time = current_time
                        elif score > 0.01:  # Show significant scores
                            print(f"   Score: {score:.6f} RMS: {rms:.3f}")
                        elif rms > 0.01:  # Show when there's audio activity
                            print(f"   [Audio detected - RMS: {rms:.3f}, score: {score:.6f}]")
                        elif check_count % 20 == 0:  # Show heartbeat every 20 checks
                            print(f"   [Listening... score: {score:.6f}, RMS: {rms:.3f}]")
                
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        elapsed = time.time() - start_time
        print("\n" + "-"*60)
        print(f"📊 RESULTS:")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Detections: {detection_count}")
        print(f"   Max Score: {max_score:.6f}")
        print(f"   Rate: {detection_count / (elapsed / 60):.1f} detections/min")
        if max_score < threshold:
            print(f"\n⚠️  Max score ({max_score:.6f}) was below threshold ({threshold})")
            print(f"   Try: Speak louder, closer to mic, or lower threshold to {max_score * 1.5:.3f}")
        print("="*60)

def main():
    """Main testing loop"""
    print("\n🎤 Wake Word Model Tester")
    print("="*60)
    
    # Select audio device first
    device_id = select_input_device()
    if device_id is None:
        print("❌ No valid audio device selected. Exiting.")
        return
    
    # List available models
    available_models = list_available_models()
    
    if not available_models:
        print("\n❌ No models found!")
        return
    
    # Model selection
    print("\n" + "="*60)
    model_name = input("\nEnter model name to test (or 'quit' to exit): ").strip()
    
    if model_name.lower() == 'quit':
        return
    
    if model_name not in available_models:
        print(f"❌ Invalid model: {model_name}")
        print(f"   Available: {', '.join(available_models)}")
        return
    
    # Load model
    model = load_model(model_name)
    if not model:
        return
    
    # Get threshold
    threshold_input = input(f"\nEnter detection threshold (default {DETECTION_THRESHOLD}): ").strip()
    threshold = float(threshold_input) if threshold_input else DETECTION_THRESHOLD
    
    # Get duration
    duration_input = input(f"\nEnter test duration in seconds (default 60): ").strip()
    duration = int(duration_input) if duration_input else 60
    
    # Test model with selected device
    test_model(model, model_name, threshold, duration, device=device_id)
    
    # Ask to test another
    print("\n")
    again = input("Test another model? (y/n): ").strip().lower()
    if again == 'y':
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
