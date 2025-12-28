"""
Voice Assistant Module for Interactive Reachy Robot
Provides microphone input, STT (Whisper), LLM (Qwen3), and TTS (Piper) capabilities.
"""

import threading
import queue
import time
import logging
import numpy as np
import soundfile as sf
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Global state
_assistant_instance = None


class VoiceAssistant:
    """
    Interactive voice assistant integrating:
    - Microphone input via ReachyMini MediaManager
    - STT: faster-whisper for speech recognition
    - LLM: Qwen3-1.7B for conversational AI
    - TTS: Piper for natural voice synthesis
    - VAD: Voice activity detection for wake word and speech detection
    """
    
    def __init__(self, robot_controller=None):
        self.robot = robot_controller
        self.running = False
        self.listening = False
        # Project-local model/cache directories
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.whisper_dir = self.models_dir / "whisper"
        self.llm_dir = self.models_dir / "llm"
        self.piper_dir = self.models_dir / "piper"
        for d in [self.models_dir, self.whisper_dir, self.llm_dir, self.piper_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Audio configuration (match ReachyMini requirements)
        self.sample_rate = 16000  # 16kHz for Whisper
        self.chunk_duration = 0.03  # 30ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Buffers and queues
        self.audio_buffer = []
        self.text_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Models (lazy loaded)
        self._whisper_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        self._piper_model = None
        self._piper_model_path = None
        self.model_info = {"whisper": None, "llm": None, "piper": None}
        self.init_error = None
        
        # Conversation context
        self.conversation_history = []
        self.system_prompt = """You are a friendly and helpful robot assistant named Reachy. 
You are equipped with vision and can track people and objects. You are part of a surveillance 
and interaction system. Keep responses concise (1-2 sentences) and natural. Be helpful and engaging."""
        
        # Callbacks
        self.on_speech_detected: Optional[Callable[[str], None]] = None
        self.on_response_ready: Optional[Callable[[str], None]] = None
        
        # Threading
        self._listen_thread = None
        self._process_thread = None
        
        logger.info("VoiceAssistant initialized")
    
    def _load_whisper(self):
        """Load Whisper model for STT"""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info("Loading Whisper model (base)...")
                # Use base model for balance of speed and accuracy, cache locally
                self._whisper_model = WhisperModel(
                    "base", device="cpu", compute_type="int8", download_root=str(self.whisper_dir)
                )
                logger.info("Whisper model loaded successfully")
                self.model_info["whisper"] = {"name": "base", "dir": str(self.whisper_dir)}
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}", exc_info=True)
                raise
        return self._whisper_model
    
    def _load_llm(self):
        """Load Qwen3 model for conversational AI"""
        if self._llm_model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Public, lightweight chat models suitable for CPU
                fallbacks = [
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "Qwen/Qwen2.5-0.5B-Instruct",
                ]

                last_error = None
                selected = None
                for model_name in fallbacks:
                    try:
                        logger.info(f"Loading LLM model: {model_name}...")
                        self._llm_tokenizer = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True, cache_dir=str(self.llm_dir)
                        )
                        self._llm_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            dtype=torch.float32,
                            trust_remote_code=True,
                            cache_dir=str(self.llm_dir)
                        )
                        selected = model_name
                        logger.info(f"LLM loaded successfully: {model_name}")
                        self.model_info["llm"] = {"name": model_name, "dir": str(self.llm_dir)}
                        break
                    except Exception as e:
                        last_error = e
                        logger.error(f"Failed to load {model_name}: {e}")

                if self._llm_model is None:
                    # Propagate the last error
                    self.init_error = f"LLM init failed: {last_error}"
                    raise last_error if last_error else RuntimeError("Failed to load any LLM model")
                else:
                    logger.info(f"Using LLM model: {selected}")
            except Exception as e:
                logger.error(f"Failed to load Qwen3: {e}", exc_info=True)
                raise
        return self._llm_model, self._llm_tokenizer
    
    def _load_piper(self):
        """Load Piper TTS model"""
        if self._piper_model is None:
            try:
                logger.info("Configuring Piper TTS from local models directory...")
                # Find local .onnx voice model under models/piper
                onnx_models = list(self.piper_dir.glob("*.onnx"))
                if onnx_models:
                    self._piper_model_path = str(onnx_models[0])
                    self._piper_model = "local"
                    self.model_info["piper"] = {"path": self._piper_model_path}
                    logger.info(f"Piper TTS model found: {self._piper_model_path}")
                else:
                    self._piper_model = None
                    self._piper_model_path = None
                    self.model_info["piper"] = {"path": None}
                    logger.warning("No Piper .onnx model found under models/piper. TTS will be disabled.")
            except Exception as e:
                logger.error(f"Failed to configure Piper: {e}", exc_info=True)
                raise
        return self._piper_model
    
    def start_listening(self):
        """Start the voice assistant listening loop"""
        if self.running:
            logger.warning("Voice assistant already running")
            return
        
        self.running = True
        self.listening = True
        
        # Start processing threads
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self._listen_thread.start()
        self._process_thread.start()
        
        logger.info("Voice assistant started listening")
    
    def stop_listening(self):
        """Stop the voice assistant"""
        self.running = False
        self.listening = False
        logger.info("Voice assistant stopped")
    
    def _listen_loop(self):
        """Main listening loop - captures audio from microphone"""
        try:
            if not self.robot or not hasattr(self.robot, 'mini'):
                logger.error("Robot controller or MediaManager not available for voice input")
                self.running = False
                return
            
            logger.info("Starting microphone capture (Reachy built-in mic)...")
            mini = self.robot.mini  # Reuse robot's MediaManager
            
            try:
                mini.media.start_recording()
                logger.info("Microphone recording active (Reachy robot)")
                
                vad_buffer = []
                speech_detected = False
                silence_chunks = 0
                max_silence_chunks = 30  # ~0.9s of silence to finalize
                
                while self.running:
                    try:
                        # Get audio chunk from microphone
                        chunk = mini.media.get_audio_sample()
                        
                        if chunk is None or len(chunk) == 0:
                            time.sleep(0.01)
                            continue
                        
                        # Simple energy-based VAD
                        energy = np.sqrt(np.mean(chunk ** 2))
                        is_speech = energy > 0.01  # Threshold for speech detection
                        
                        if is_speech:
                            vad_buffer.append(chunk)
                            speech_detected = True
                            silence_chunks = 0
                        elif speech_detected:
                            vad_buffer.append(chunk)
                            silence_chunks += 1
                            
                            # If enough silence after speech, process buffer
                            if silence_chunks >= max_silence_chunks and len(vad_buffer) > 10:
                                audio_data = np.concatenate(vad_buffer)
                                self.text_queue.put(audio_data)
                                logger.info(f"Speech segment captured ({len(audio_data)} samples)")
                                
                                # Reset
                                vad_buffer = []
                                speech_detected = False
                                silence_chunks = 0
                    
                    except Exception as e:
                        logger.error(f"Audio capture error: {e}")
                        time.sleep(0.1)
                
                # Stop recording but don't close the MediaManager
                try:
                    mini.media.stop_recording()
                    logger.info("Microphone recording stopped")
                except Exception as e:
                    logger.warning(f"Error stopping recording: {e}")
            except Exception as e:
                logger.error(f"Microphone setup error: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Listen loop error: {e}", exc_info=True)
            self.running = False
    
    def _process_loop(self):
        """Processing loop - handles STT -> LLM -> TTS pipeline"""
        # Lazy load models
        try:
            whisper = self._load_whisper()
            llm, tokenizer = self._load_llm()
            piper = self._load_piper()
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.init_error = f"Model init failed: {e}"
            return
        
        while self.running:
            try:
                # Wait for audio input
                audio_data = self.text_queue.get(timeout=1.0)
                
                # Step 1: Speech-to-Text (Whisper)
                logger.info("Transcribing speech...")
                text = self._transcribe_audio(audio_data, whisper)
                
                if not text or len(text.strip()) < 3:
                    logger.info("No valid speech detected")
                    continue
                
                logger.info(f"Transcribed: {text}")
                
                # Notify callback
                if self.on_speech_detected:
                    self.on_speech_detected(text)
                
                # Step 2: Generate response (Qwen3)
                logger.info("Generating response...")
                response = self._generate_response(text, llm, tokenizer)
                logger.info(f"Response: {response}")
                
                # Notify callback
                if self.on_response_ready:
                    self.on_response_ready(response)
                
                # Step 3: Text-to-Speech (Piper)
                logger.info("Synthesizing speech...")
                self._speak(response)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Process loop error: {e}", exc_info=True)
    
    def _transcribe_audio(self, audio_data: np.ndarray, model) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1]
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Transcribe
            segments, info = model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect text from segments
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""
    
    def _generate_response(self, user_text: str, model, tokenizer) -> str:
        """Generate conversational response using Qwen3"""
        try:
            import torch
            
            # Build conversation
            self.conversation_history.append({"role": "user", "content": user_text})
            
            # Keep last 6 messages for context
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            # Format chat with system prompt
            messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
            
            # Generate
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer([text], return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # Add to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."
    
    def _speak(self, text: str):
        """Synthesize and play speech using Piper TTS"""
        try:
            import subprocess
            import tempfile
            import sounddevice as sd
            import soundfile as sf
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
            
            # Use piper CLI with local model path
            if not self._piper_model_path:
                logger.error("Piper TTS model not found in models/piper. Skipping TTS.")
                return
            cmd = [
                "piper",
                "--model", self._piper_model_path,
                "--output_file", output_path
            ]
            
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = proc.communicate(input=text)
            
            if proc.returncode != 0:
                logger.error(f"Piper TTS error: {stderr}")
                return
            
            # Play audio through robot speakers (async to avoid blocking)
            if self.robot:
                # Use threading to prevent blocking the voice processing loop
                playback_thread = threading.Thread(
                    target=lambda: self.robot.play_sound_from_file(output_path),
                    daemon=True
                )
                playback_thread.start()
                logger.info(f"TTS playback started: {text[:50]}...")
            else:
                logger.warning("Robot controller not available for audio playback")
            
            # Note: Don't delete file immediately, let playback thread finish
            # The temp file will be cleaned up eventually by the OS
        
        except FileNotFoundError:
            logger.error("Piper CLI not found. Install piper or ensure it's in PATH.")
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)

    def get_model_info(self):
        """Return a snapshot of loaded/selected models and errors"""
        return {
            "whisper": self.model_info.get("whisper"),
            "llm": self.model_info.get("llm"),
            "piper": self.model_info.get("piper"),
            "error": self.init_error,
        }
    
    def process_text_command(self, text: str) -> str:
        """Process a text command directly (bypass STT)"""
        try:
            llm, tokenizer = self._load_llm()
            response = self._generate_response(text, llm, tokenizer)
            return response
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            return "Sorry, I couldn't process that command."
    
    def speak_text(self, text: str):
        """Speak text directly (bypass LLM)"""
        self._speak(text)


def get_assistant(robot_controller=None) -> VoiceAssistant:
    """Get or create the global voice assistant instance"""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = VoiceAssistant(robot_controller)
    return _assistant_instance
