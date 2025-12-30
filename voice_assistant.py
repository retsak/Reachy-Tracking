"""
Voice Assistant Module for Interactive Reachy Robot
Provides microphone input, STT (Whisper), LLM (Local/OpenAI/OLLAMA), and TTS (Piper) capabilities.
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
from llm_config import get_llm_config_manager

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

CRITICAL: Always remember and reference the conversation history above. If the user mentioned something earlier (like weather, names, topics), recall it naturally.

Rules:
- Answer directly and concisely (1-2 short sentences)
- If the user refers to something they said before, acknowledge it
- If unclear, ask one brief clarifying question
- Never invent facts; keep responses relevant
- Never output code or markup"""
        
        # Callbacks
        self.on_speech_detected: Optional[Callable[[str], None]] = None
        self.on_response_ready: Optional[Callable[[str], None]] = None
        
        # Threading
        self._listen_thread = None
        self._process_thread = None
        
        logger.info("VoiceAssistant initialized")

    def preload_models(self):
        """Preload models in a background thread to avoid latency on first use."""
        def _load():
            try:
                logger.info("[PRELOAD] Starting model preload in background...")
                self._load_whisper()
                self._load_piper()
                # Only load local LLM if it's the configured provider
                from llm_config import get_llm_config_manager
                llm_config = get_llm_config_manager()
                if llm_config.get_current_provider() == "local":
                    self._load_llm()
                    logger.info("[PRELOAD] Local LLM loaded.")
                else:
                    provider = llm_config.get_current_provider()
                    logger.info(f"[PRELOAD] Skipping local LLM load (using {provider} provider).")
                logger.info("[PRELOAD] All models preloaded successfully.")
            except Exception as e:
                logger.error(f"[PRELOAD] Background preload failed: {e}")

        # Only start if not already loaded/loading
        if not (self._llm_model and self._whisper_model):
            threading.Thread(target=_load, daemon=True).start()
    
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
        """Load local LLM model for conversational AI (only if local provider is configured)"""
        # Check if local provider is configured before loading
        try:
            from llm_config import get_llm_config_manager
            llm_config = get_llm_config_manager()
            if llm_config.get_current_provider() != "local":
                logger.info(f"[LLM] Skipping local model load (using {llm_config.get_current_provider()} provider)")
                return None, None  # Return tuple of Nones instead of None
        except Exception as e:
            logger.warning(f"[LLM] Could not check provider config, proceeding with local load: {e}")
        
        if self._llm_model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                import os
                try:
                    from huggingface_hub import HfFolder
                except Exception:
                    HfFolder = None
                import platform
                
                # Prefer small but higher-quality CPU-friendly models (smallest first)
                # Google Gemma 2 (2B) is the preferred stable model
                fallbacks = [
                    "google/gemma-2-2b-it",
                    "microsoft/Phi-3-mini-4k-instruct",
                    "Qwen/Qwen2.5-0.5B-Instruct",
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
                    load_dtype = torch.float16
                else:
                    self.device = "cpu"
                    load_dtype = torch.float32

                last_error = None
                selected = None
                # Optional Hugging Face auth token for gated models (e.g., Gemma)
                # Get HF token from env or saved CLI login
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                if not hf_token and HfFolder is not None:
                    try:
                        hf_token = HfFolder.get_token()
                    except Exception:
                        hf_token = None
                auth_kwargs = {"token": hf_token} if hf_token else {}

                # Memory-Aware Filtering
                try:
                    import psutil
                    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                    logger.info(f"System RAM detected: {total_ram_gb:.1f} GB")
                    
                    # Heuristic: Reserve 40% for OS + other apps, use 60% for model
                    safe_budget_gb = total_ram_gb * 0.6
                    
                    # Approximate size in GB (assuming float16 or int8 quantization effects)
                    model_sizes = {
                        "google/gemma-3-4b-it": 8.5,
                        "google/gemma-2-2b-it": 5.0,
                        "microsoft/Phi-3-mini-4k-instruct": 7.5,
                        "Qwen/Qwen2.5-0.5B-Instruct": 1.5,
                        "microsoft/Phi-3.5-mini-instruct": 8.0,
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.5
                    }
                    
                    filtered_fallbacks = []
                    for m in fallbacks:
                        size = model_sizes.get(m, 5.0) # default to 5GB if unknown
                        if size <= safe_budget_gb:
                            filtered_fallbacks.append(m)
                        else:
                            logger.warning(f"Skipping {m} (est {size}GB) due to memory budget ({safe_budget_gb:.1f}GB)")
                    
                    if filtered_fallbacks:
                        fallbacks = filtered_fallbacks
                        logger.info(f"Memory-safe model list: {fallbacks}")
                    else:
                        logger.warning("No models fit safe budget! Falling back to smallest unsafe option.")
                        fallbacks = [fallbacks[-1]] # Try the absolute smallest (Qwen/TinyLlama)
                        
                except ImportError:
                    logger.warning("psutil not installed, skipping memory checks")
                except Exception as e:
                    logger.error(f"Memory check error: {e}")

                for model_name in fallbacks:
                    try:
                        logger.info(f"Loading LLM model: {model_name}...")
                        self._llm_tokenizer = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True, cache_dir=str(self.llm_dir), use_fast=True, **auth_kwargs
                        )
                        
                        # Simplify loading: Avoid device_map="auto" on MPS/CPU to prevent Accelerate issues
                        # Only use device_map="auto" for CUDA unless explicitly needed
                        use_device_map = (self.device == "cuda")
                        
                        self._llm_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=load_dtype,
                            device_map="auto" if use_device_map else None,
                            low_cpu_mem_usage=(self.device == "cuda"), # Only True if using device_map
                            trust_remote_code=True,
                            cache_dir=str(self.llm_dir),
                            **auth_kwargs
                        )
                        
                        # Manual device placement for MPS/CPU
                        if not use_device_map:
                            self._llm_model.to(self.device)

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
                        # Write to detailed log for debugging
                        with open("voice_error.log", "a") as f:
                            import traceback
                            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error loading {model_name}:\n")
                            f.write(traceback.format_exc())

                if self._llm_model is None:
                    # Propagate the last error
                    self.init_error = f"LLM init failed: {last_error}"
                    raise last_error if last_error else RuntimeError("Failed to load any LLM model")
                else:
                    logger.info(f"Using LLM model: {selected}")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}", exc_info=True)
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
                
                # Step 2: Generate response - check LLM provider
                logger.info("Generating response...")
                llm_config = get_llm_config_manager()
                provider = llm_config.get_current_provider()
                
                if provider == "openai":
                    response = self._generate_response_openai(text)
                elif provider == "ollama":
                    response = self._generate_response_ollama(text)
                elif provider == "local":
                    # Only generate local response if models are loaded
                    if llm is None or tokenizer is None:
                        logger.error("Local LLM models not loaded but local provider is selected")
                        response = "I'm sorry, but the local AI model isn't ready yet. Please try again in a moment."
                    else:
                        response = self._generate_response(text, llm, tokenizer)
                else:
                    logger.error(f"Unknown provider: {provider}")
                    response = None
                
                if not response:
                    logger.error("Failed to generate response")
                    continue
                
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
    
    def _generate_response_openai(self, user_text: str) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            
            llm_config = get_llm_config_manager()
            api_key = llm_config.get_openai_key()
            config = llm_config.get_current_config()
            model = config.get("model", "gpt-3.5-turbo")
            
            if not api_key:
                logger.error("OpenAI API key not configured")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_text})
            
            # Keep last 8 messages for context
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            logger.info(f"OpenAI API call: model={model}")
            
            # GPT-5 and newer o-series models use max_completion_tokens instead of max_tokens
            use_completion_tokens = any(model.startswith(prefix) for prefix in ["gpt-5", "o1", "o3", "o4"])
            
            # GPT-5 and o-series models don't support custom temperature (only default=1)
            supports_temperature = not any(model.startswith(prefix) for prefix in ["gpt-5", "o1", "o3", "o4"])
            
            kwargs = {
                "model": model,
                "messages": self.conversation_history,
            }
            
            if supports_temperature:
                kwargs["temperature"] = 0.7
            
            if use_completion_tokens:
                # GPT-5 supports up to 128k output tokens; use 8k for excellent reasoning + context processing + output
                kwargs["max_completion_tokens"] = 8192
            else:
                kwargs["max_tokens"] = 256
            
            response = client.chat.completions.create(**kwargs)
            
            logger.info(f"OpenAI raw response: {response}")
            logger.info(f"Response choices: {response.choices}")
            
            if not response.choices:
                logger.error("No choices in OpenAI response")
                return None
            
            assistant_response = response.choices[0].message.content
            
            if not assistant_response:
                logger.error("Empty content in OpenAI response")
                return None
            
            assistant_response = assistant_response.strip()
            
            if not assistant_response:
                logger.error("OpenAI response was empty after stripping")
                return None
            
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            logger.info(f"OpenAI response: {assistant_response[:100]}...")
            return assistant_response
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return None
    
    def _generate_response_ollama(self, user_text: str) -> str:
        """Generate response using OLLAMA endpoint."""
        try:
            import requests
            
            llm_config = get_llm_config_manager()
            config = llm_config.get_current_config()
            endpoint = config.get("endpoint", "http://localhost:11434")
            model = config.get("model", "llama2")
            
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_text})
            
            # Keep last 8 messages for context
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            # Prepare prompt from conversation
            prompt = ""
            for msg in self.conversation_history:
                role = "You" if msg["role"] == "assistant" else "User"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "You: "
            
            logger.info(f"OLLAMA API call: endpoint={endpoint}, model={model}")
            url = f"{endpoint.rstrip('/')}/api/generate"
            response = requests.post(
                url,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30
            )
            
            if response.status_code == 200:
                assistant_response = response.json().get("response", "").strip()
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                logger.info(f"OLLAMA response: {assistant_response[:100]}...")
                return assistant_response
            else:
                logger.error(f"OLLAMA error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"OLLAMA error: {e}", exc_info=True)
            return None
    
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
            
            # Keep last 8 messages (4 exchanges) for context
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            # DEBUG: Log conversation context being sent to LLM
            logger.info(f"[LLM CONTEXT] Sending {len(self.conversation_history)} messages to LLM:")
            for i, msg in enumerate(self.conversation_history):
                logger.info(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")
            
            # Format: Inject history into SYSTEM PROMPT to force memory
            # This is more robust than chat templates for small/finicky models
            
            context_str = ""
            if len(self.conversation_history) > 1:
                context_str = "\n\n=== RECENT CONVERSATION LOG (YOU MUST REMEMBER THIS) ===\n"
                for m in self.conversation_history[:-1]: # All except current
                    r = m.get("role", "user").upper()
                    c = m.get("content", "")
                    context_str += f"{r}: {c}\n"
                context_str += "=== END LOG ===\n"

            # Combine static system prompt + dynamic context
            full_system_prompt = self.system_prompt + context_str
            
            # Current message
            current_user_text = self.conversation_history[-1]["content"]

            # Construct simple messages list for tokenizer (if it supports it)
            # or manual string build
            
            messages_for_token = [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": current_user_text}
            ]

            # Build prompt using tokenizer chat template if available
            try:
                if getattr(tokenizer, "chat_template", None):
                    text = tokenizer.apply_chat_template(
                        messages_for_token,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Manual format
                    text = f"{full_system_prompt}\n\nUSER: {current_user_text}\nASSISTANT:"
            except Exception:
                 text = f"{full_system_prompt}\n\nUSER: {current_user_text}\nASSISTANT:"

            inputs = tokenizer(text, return_tensors="pt")

            # Ensure inputs live on the same device as the model
            try:
                if hasattr(self, "device") and self.device != "cpu":
                    for k in list(inputs.keys()):
                        inputs[k] = inputs[k].to(self.device)
            except Exception:
                pass
            
            with torch.no_grad():
                # Resolve token IDs robustly
                eos_id = tokenizer.eos_token_id
                if eos_id is None and hasattr(model, "config"):
                    eos_id = getattr(model.config, "eos_token_id", None)
                pad_id = tokenizer.pad_token_id
                if pad_id is None and hasattr(model, "config"):
                    pad_id = getattr(model.config, "pad_token_id", eos_id)

                gen_kwargs = {
                    "max_new_tokens": 40,
                    "do_sample": False,
                    "use_cache": False,
                    "no_repeat_ngram_size": 2,
                    "repetition_penalty": 1.2,
                }
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id
                if pad_id is not None:
                    gen_kwargs["pad_token_id"] = pad_id

                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
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
        # Sanitize Unicode characters that Piper/espeak can't handle
        # Replace smart quotes with regular quotes
        s = s.replace('"', '"').replace('"', '"')  # Left and right double quotes
        s = s.replace(''', "'").replace(''', "'")  # Left and right single quotes
        # Replace em dashes and en dashes with regular hyphens
        s = s.replace('—', '-').replace('–', '-')
        # Replace ellipsis with periods
        s = s.replace('…', '...')
        
        # Remove any remaining problematic Unicode characters by encoding/decoding
        # This converts non-ASCII characters to their closest ASCII equivalent or removes them
        try:
            # First try: encode to ASCII, replacing unknown characters with '?'
            s = s.encode('ascii', errors='replace').decode('ascii')
        except Exception:
            # Fallback: remove all non-ASCII characters
            s = ''.join(char if ord(char) < 128 else '' for char in s)
        
        # Remove replacement chars and non-printables
        s = s.replace("\ufffd", "")
        s = re.sub(r"[\x00-\x1F]", " ", s)
        # Collapse excessive repeats of short words
        s = re.sub(r"\b(\w{1,10})(?:\s+\1){2,}\b", r"\1", s)
        # If the text is mostly repeated 'len' or non-letters, fallback
        letters = re.findall(r"[A-Za-z]", s)
        if len(letters) < max(5, len(s) // 10):
            return "Thanks! I’m here. What would you like me to do?"
        return s.strip()

    def _speak(self, text: str):
        """Synthesize and play speech using Piper TTS - splits long text into chunks"""
        try:
            import subprocess
            import tempfile
            import sounddevice as sd
            import soundfile as sf
            import re
            
            # Ensure Piper is configured even when called outside the main loop
            if self._piper_model is None or not self._piper_model_path:
                try:
                    self._load_piper()
                except Exception:
                    pass
            
            # Clean and sanitize assistant text before TTS
            text = self._postprocess_response(self._extract_assistant_content(text))
            
            # Split long text into sentences to avoid single-file length issues
            # Maximum ~500 characters per chunk (longer chunks are more efficient)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = ""
            chunk_max_len = 500  # Increased from 200 for more efficiency
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Calculate what the chunk would be if we add this sentence
                test_chunk = current_chunk + (" " if current_chunk else "") + sentence
                
                # If it fits, add it to current chunk
                if len(test_chunk) <= chunk_max_len:
                    current_chunk = test_chunk
                else:
                    # If current chunk has content, save it and start new one
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If no chunks were created, just use original text
            if not chunks:
                chunks = [text]
            
            logger.info(f"[TTS] Split text into {len(chunks)} chunks for playback")
            
            # Set speaking flag ONCE at the start of all chunks
            if self.robot:
                self.robot._is_speaking = True
                # Pause tracking during speech to prevent fighting with speech motion
                self.robot._pause_tracking = True
            
            # Generate and play each chunk sequentially
            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                logger.info(f"[TTS] Processing chunk {chunk_idx + 1}/{len(chunks)}: {chunk[:50]}...")
                
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    output_path = tmp.name
                
                # Use piper CLI with local model path
                if not self._piper_model_path:
                    logger.error("Piper TTS model not found in models/piper. Skipping TTS.")
                    if self.robot:
                        self.robot._is_speaking = False
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
                
                stdout, stderr = proc.communicate(input=chunk)
                
                if proc.returncode != 0:
                    logger.error(f"Piper TTS error for chunk {chunk_idx}: {stderr}")
                    continue
                
                # Ensure file is written and flushed to disk before playback
                # Wait up to 2 seconds for the file to exist and be readable
                import os
                file_ready = False
                for attempt in range(20):  # 20 * 0.1s = 2 seconds max
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        file_ready = True
                        break
                    time.sleep(0.1)
                
                if not file_ready:
                    logger.error(f"TTS audio file not created or empty: {output_path}")
                    continue
                
                # Play audio through robot speakers (async to avoid blocking)
                if self.robot:
                    # Wait for audio to finish before playing next chunk
                    logger.info(f"[TTS] Playing chunk {chunk_idx + 1}/{len(chunks)}")
                    try:
                        self.robot.play_sound_from_file(output_path)
                        # Wait for this chunk to finish before playing next one
                        self.robot._audio_playing.wait(timeout=60)  # 60 second timeout per chunk
                        # Don't delete file here - robot controller will handle it in background thread
                    except Exception as e:
                        logger.error(f"Error playing audio chunk {chunk_idx}: {e}")
                        # Clean up on error only
                        try:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                        except Exception:
                            pass
                else:
                    logger.warning("Robot controller not available for audio playback")
                    # Clean up if no robot
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except Exception:
                        pass
            
            # Set speaking flag to False ONLY after all chunks are done
            if self.robot:
                self.robot._is_speaking = False
                # Resume tracking after speech
                self.robot._pause_tracking = False
                # Reset to default position after speaking
                self.robot.recenter_head()
            
            logger.info(f"[TTS] All {len(chunks)} chunks played successfully")
        
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
            
            # Route through appropriate LLM provider
            llm_config = get_llm_config_manager()
            provider = llm_config.get_current_provider()
            
            if provider == "openai":
                response = self._generate_response_openai(text)
            elif provider == "ollama":
                response = self._generate_response_ollama(text)
            elif provider == "local":
                # Only generate local response if models are loaded
                if llm is None or tokenizer is None:
                    logger.error("Local LLM models not loaded but local provider is selected")
                    response = "I'm sorry, but the local AI model isn't ready yet. Please try again in a moment."
                else:
                    response = self._generate_response(text, llm, tokenizer)
            else:
                logger.error(f"Unknown provider: {provider}")
                response = None
            
            return response or "Sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"Command processing error: {e}", exc_info=True)
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
