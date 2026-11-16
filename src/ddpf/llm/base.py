"""Abstract interfaces for LLM clients used across DDPF modules."""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    This interface is intentionally minimal so it can be reused by the
    Description, Detection, and Prediction modules.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response from a prompt.

        Args:
            prompt: Input prompt sent to the language model.

        Returns:
            Generated text response.
        """
        raise NotImplementedError