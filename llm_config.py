"""
LLM Configuration Manager
Handles model provider selection, API key storage, and model listing.
Supports: Local, OLLAMA, OpenAI, and extensible to other providers.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import base64

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent / ".llm_config.json"
OPENAI_MODELS = [
    {"id": "gpt-5-nano", "name": "GPT-5 Nano (Ultra-Budget)", "input": 0.05, "output": 0.40, "category": "Ultra-Budget"},
    {"id": "gpt-5-mini", "name": "GPT-5 Mini (Budget)", "input": 0.25, "output": 2.00, "category": "Budget"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Fast)", "input": 0.15, "output": 0.60, "category": "Fast"},
    {"id": "gpt-5", "name": "GPT-5 (Advanced)", "input": 1.25, "output": 10.00, "category": "Advanced"},
    {"id": "gpt-4o", "name": "GPT-4o (Balanced)", "input": 2.50, "output": 10.00, "category": "Balanced"},
    {"id": "gpt-5.2", "name": "GPT-5.2 (Premium)", "input": 1.75, "output": 14.00, "category": "Premium"},
    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini (Fallback)", "input": 0.40, "output": 1.60, "category": "Fallback"},
    {"id": "o3-mini", "name": "o3 Mini (Reasoning)", "input": 1.10, "output": 4.40, "category": "Reasoning"},
    {"id": "o1-mini", "name": "o1 Mini (Advanced)", "input": 1.10, "output": 4.40, "category": "Advanced"},
]


class LLMConfigManager:
    """Manages LLM provider configuration and API key storage."""

    def __init__(self):
        self.config = self._load_config()
        self.encryption_key = os.environ.get("LLM_CONFIG_KEY", "default-key")

    def _load_config(self) -> Dict:
        """Load configuration from file or return defaults."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        return {
            "provider": "local",
            "local": {"model_id": "google/gemma-3-4b-it"},
            "ollama": {"endpoint": "http://localhost:11434", "model": "llama2"},
            "openai": {"api_key": None, "model": "gpt-3.5-turbo"}
        }

    def _save_config(self) -> bool:
        """Save configuration to file."""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def _encrypt_key(self, key: str) -> str:
        """Simple encryption for API keys (base64 + hash)."""
        if not key:
            return None
        try:
            encoded = base64.b64encode(key.encode()).decode()
            return encoded
        except Exception:
            return key

    def _decrypt_key(self, encrypted: str) -> str:
        """Decrypt API key."""
        if not encrypted:
            return None
        try:
            decoded = base64.b64decode(encrypted).decode()
            return decoded
        except Exception:
            return encrypted

    def get_current_provider(self) -> str:
        """Get the current active provider."""
        return self.config.get("provider", "local")

    def get_current_config(self) -> Dict:
        """Get the current provider's configuration."""
        provider = self.get_current_provider()
        return self.config.get(provider, {})

    def set_provider_config(self, provider: str, config: Dict) -> Tuple[bool, str]:
        """
        Set configuration for a provider.
        Args:
            provider: "local", "ollama", or "openai"
            config: Provider-specific configuration dict
        """
        if provider not in ["local", "ollama", "openai"]:
            return False, f"Unknown provider: {provider}"

        # Validate provider config
        if provider == "local":
            if "model_id" not in config:
                return False, "Local config requires 'model_id'"
        elif provider == "ollama":
            if "endpoint" not in config or "model" not in config:
                return False, "OLLAMA config requires 'endpoint' and 'model'"
        elif provider == "openai":
            if "api_key" not in config or "model" not in config:
                return False, "OpenAI config requires 'api_key' and 'model'"
            # Encrypt the API key
            config["api_key"] = self._encrypt_key(config["api_key"])

        self.config[provider] = config
        self.config["provider"] = provider

        if self._save_config():
            logger.info(f"LLM provider set to: {provider}")
            return True, "Configuration saved"
        return False, "Failed to save configuration"

    def get_openai_key(self) -> Optional[str]:
        """Get decrypted OpenAI API key."""
        encrypted = self.config.get("openai", {}).get("api_key")
        if encrypted:
            return self._decrypt_key(encrypted)
        
        # Fallback to environment variable
        return os.environ.get("OPENAI_API_KEY")

    def validate_openai_key(self, api_key: str) -> Tuple[bool, str]:
        """Validate OpenAI API key by making a test request."""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Try to list models (lightweight validation)
            models = client.models.list()
            return True, "Valid API key"
        except Exception as e:
            return False, str(e)

    def get_local_models(self) -> List[Dict]:
        """Get list of available local models."""
        fallbacks = [
            {"id": "google/gemma-3-4b-it", "name": "Gemma 3 4B", "current": True},
            {"id": "google/gemma-2-2b-it", "name": "Gemma 2 2B", "current": False},
            {"id": "microsoft/phi-3.5-mini-instruct", "name": "Phi-3.5 Mini", "current": False},
            {"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5 0.5B", "current": False},
        ]
        return fallbacks

    def get_openai_models(self, api_key: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Get list of OpenAI models and pricing.
        Returns: (models_list, pricing_list)
        """
        try:
            # For now, return static list; can be enhanced to fetch from API
            models = [
                {"id": m["id"], "name": m["name"]} for m in OPENAI_MODELS
            ]
            pricing = OPENAI_MODELS
            return models, pricing
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}")
            return [], []

    def get_ollama_models(self, endpoint: str) -> Tuple[List[str], str]:
        """
        Get list of available OLLAMA models.
        Returns: (model_list, error_message)
        """
        try:
            import requests
            url = f"{endpoint.rstrip('/')}/api/tags"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return models, None
        except requests.exceptions.RequestException as e:
            return [], f"Connection failed: {e}"
        except Exception as e:
            return [], f"Error: {e}"
        
        return [], "Unknown error"


# Global instance
_config_manager = None


def get_llm_config_manager() -> LLMConfigManager:
    """Get or create the global LLM config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = LLMConfigManager()
    return _config_manager
