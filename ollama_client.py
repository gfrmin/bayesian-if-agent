"""
Ollama Client

HTTP interface to local Ollama instance for LLM completions.
"""

from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class OllamaConfig:
    model: str = "llama3.1:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    timeout: int = 30


class OllamaClient:
    """Client for local Ollama instance."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def complete(self, prompt: str) -> str:
        """Get completion from Ollama."""
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
        return response.json()["response"]
