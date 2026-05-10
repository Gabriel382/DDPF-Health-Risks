"""Ollama implementation of the DDPF LLM client interface."""

import json
import urllib.request

from ddpf.llm.base import BaseLLMClient


class OllamaLLMClient(BaseLLMClient):
    """LLM client for local Ollama models."""

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
    ) -> None:
        """Initialize the Ollama client.

        Args:
            model: Name of the Ollama model to use.
            host: Ollama server URL.
        """
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str) -> str:
        """Generate a response using Ollama.

        Args:
            prompt: Input prompt.

        Returns:
            Generated response text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        request = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        return data.get("response", "")