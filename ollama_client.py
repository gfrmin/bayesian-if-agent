"""
Ollama Client

HTTP interface to local Ollama instance for LLM completions.
Uses MD5-based response caching.
"""

import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import requests


@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1  # Low temperature for consistency
    timeout: int = 30


class OllamaClient:
    """Client for local Ollama instance running Llama 3.1."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.cache: Dict[str, str] = {}

    def complete(self, prompt: str, use_cache: bool = True) -> str:
        """Get completion from Ollama."""
        cache_key = self._cache_key(prompt)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "temperature": self.config.temperature,
                "stream": False,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        result = response.json()["response"]

        if use_cache:
            self.cache[cache_key] = result

        return result

    def complete_json(self, prompt: str, use_cache: bool = True) -> Optional[Dict]:
        """Get JSON completion from Ollama."""
        full_prompt = prompt + "\n\nRespond with valid JSON only, no other text."

        response = self.complete(full_prompt, use_cache)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return None

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def clear_cache(self):
        self.cache = {}
