"""
Voice Assistant Module for Interactive Reachy Robot
Provides microphone input, STT (Whisper), LLM (Qwen3), and TTS (Piper) capabilities.
"""

import threading
import queue
import time
import logging
import re
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
        self.device = "cpu"
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
        self.system_prompt = """You are Reachy, a friendly robot assistant.
    Speak in plain English with 1–2 short sentences.
    Be relevant to the user’s message and this robot context.
    Never output code or markup; avoid repetition and filler."""
        
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
                import platform
                
                # Prefer small but higher-quality CPU-friendly models (smallest first)
                fallbacks = [
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "google/gemma-2-2b-it",
                    "microsoft/Phi-3-mini-4k-instruct",
                    "microsoft/Phi-3.5-mini-instruct",
                ]

                # Select device and dtype
                use_cuda = torch.cuda.is_available()
                use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                if use_cuda and platform.system() == "Windows":
                    self.device = "cuda"
                    load_dtype = torch.float16
                elif use_mps and platform.system() == "Darwin":
                    self.device = "mps"
                    load_dtype = torch.float32
                else:
                    self.device = "cpu"
                    load_dtype = torch.float32

                last_error = None
                selected = None
                for model_name in fallbacks:
                    try:
                        logger.info(f"Loading LLM model: {model_name}...")
                        self._llm_tokenizer = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True, cache_dir=str(self.llm_dir), use_fast=True
                        )
                        self._llm_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=load_dtype,
                            device_map=("auto" if self.device != "cpu" else "cpu"),
                            low_cpu_mem_usage=False,
                            trust_remote_code=True,
                            cache_dir=str(self.llm_dir)
                        )
                        # Ensure model is on expected device
                        if self.device == "cpu":
                            self._llm_model.to("cpu")
                        # Inference-only mode
                        self._llm_model.eval()

                        # Avoid cache-specific incompatibilities with some hub model code
                        try:
                            if hasattr(self._llm_model, "config"):
                                # Disable generation cache to avoid DynamicCache.seen_tokens issues
                                setattr(self._llm_model.config, "use_cache", False)
                                # Use eager attention implementation for broad compatibility
                                setattr(self._llm_model.config, "attn_implementation", "eager")
                        except Exception:
                            pass

                        # CPU threading: use all available cores
                        if self.device == "cpu":
                            try:
                                import os
                                num_threads = os.cpu_count() or 4
                                torch.set_num_threads(num_threads)
                            except Exception:
                                pass

                        # Dynamic quantization for Linear layers to speed CPU
                        if self.device == "cpu":
                            try:
                                import torch.nn as nn
                                from torch.ao.quantization import quantize_dynamic
                                self._llm_model = quantize_dynamic(
                                    self._llm_model, {nn.Linear}, dtype=torch.qint8
                                )
                                logger.info("Applied dynamic quantization (qint8) to Linear layers.")
                            except Exception as qe:
                                logger.warning(f"Quantization skipped: {qe}")
                        selected = model_name
                        logger.info(f"LLM loaded successfully: {model_name}")
                        self.model_info["llm"] = {"name": model_name, "dir": str(self.llm_dir), "device": self.device, "dtype": str(load_dtype)}
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
            # Fast intent short-circuit for low-power reliability
            intent = self._intent_response(user_text)
            if intent is not None:
                self.conversation_history.append({"role": "user", "content": user_text})
                self.conversation_history.append({"role": "assistant", "content": intent})
                return intent
            
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
                    max_new_tokens=40,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                )
            
            raw = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            response = self._postprocess_response(self._extract_assistant_content(raw))
            
            # Add to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."

    def _intent_response(self, text: str) -> Optional[str]:
        """Handle simple intents without invoking the LLM."""
        if not text:
            return None
        s = text.strip().lower()
        # Greetings / phatic
        if re.fullmatch(r"(hi|hello|hey|howdy|yo|sup|hola)[.!?\s]*", s):
            return "Hi there! How can I help today?"
        # "test" / "ping"
        if re.fullmatch(r"(test|ping|hello world)[.!?\s]*", s):
            return "I’m here and listening. What would you like me to do?"
        # Say hi "Name"
        m = re.search(r"say\s+hi\s+\"?([a-zA-Z][a-zA-Z\-\s']{0,30})\"?", s)
        if m:
            name = m.group(1).strip()
            # Title-case the name safely
            safe_name = re.sub(r"[^A-Za-z\-\s']", "", name).title()
            if safe_name:
                return f"Hi {safe_name}! Nice to meet you."
        return None
    
    def _strip_role_prefix(self, text: str) -> str:
        """Remove leading role prefixes like 'Assistant:' from model outputs."""
        if not text:
            return text
        s = text.strip()
        # Remove common chat role tokens
        s = s.replace("<|assistant|>", "").replace("<|im_start|>assistant", "").replace("<|im_end|>", "")
        s = s.replace("### Assistant:", "")
        # Regex to drop leading 'Assistant' label with punctuation
        s = re.sub(r"^(assistant|Assistant|ASSISTANT)\s*[:：\-—\.]*\s*", "", s)
        # Fallback removal for simple prefixes
        for pref in ("Assistant.", "assistant.", "Assistant:", "assistant:"):
            if s.startswith(pref):
                s = s[len(pref):].lstrip()
        return s.strip()

    def _extract_assistant_content(self, raw: str) -> str:
        """Extract only the assistant message body from raw decoded text.

        Handles templates like `<|im_start|>assistant ... <|im_end|>` and
        falls back to splitting on common role prefixes.
        """
        if not raw:
            return raw

        # Primary: capture between assistant start and end tags
        m = re.search(r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)", raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            body = m.group(1)
            return self._strip_role_prefix(body)

        # Fallback: after 'Assistant:' prefix up to next role marker
        m2 = re.search(r"(?:^|\n)(?:assistant|Assistant|ASSISTANT)\s*:\s*(.*)", raw, flags=re.DOTALL)
        if m2:
            body = m2.group(1)
            body = re.split(r"\n(?:User|System|assistant)\s*:\s*|<\|im_start\|>|<\|im_end\|>", body)[0]
            return body.strip()

        # Otherwise, just strip known tokens and return
        return self._strip_role_prefix(raw)

    def _postprocess_response(self, text: str) -> str:
        """Clean generated text: remove invalid chars, collapse repeats, and fallback."""
        if not text:
            return "I heard you. How can I help?"
        s = text.strip()
        # Remove replacement chars and non-printables
        s = s.replace("\ufffd", "")
        s = re.sub(r"[\x00-\x1F]", " ", s)
        # Collapse excessive repeats of short words
        s = re.sub(r"\b(\w{1,10})(?:\s+\1){2,}\b", r"\1", s)
        # If the text is mostly repeated 'len' or non-letters, fallback
        letters = re.findall(r"[A-Za-z]", s)
        if len(letters) < max(5, len(s) // 10):
            return "Thanks! I’m here. What would you like me to do?"
        # Trim overly long outputs
        s = s[:240]
        return s.strip()

    def _speak(self, text: str):
        """Synthesize and play speech using Piper TTS"""
        try:
            import subprocess
            import tempfile
            import sounddevice as sd
            import soundfile as sf
            # Ensure Piper is configured even when called outside the main loop
            if self._piper_model is None or not self._piper_model_path:
                try:
                    self._load_piper()
                except Exception:
                    pass
            # Clean and sanitize assistant text before TTS
            text = self._postprocess_response(self._extract_assistant_content(text))
            
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
                text=True,
                encoding="utf-8",
                errors="replace",
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
