"""Unified LiteLLM client for local Ollama and OpenRouter models.

Uses `litellm.completion` directly, not the LiteLLM proxy. The only secret is
OPENROUTER_API_KEY read from .env. Do not log raw messages in production.
"""
from __future__ import annotations
from typing import Any, Callable
from ddpf.llm.base import BaseLLMClient
from ddpf.storyforest.settings import Settings

class LiteLLMClient(BaseLLMClient):
    def __init__(self, settings: Settings | None=None,
                 completion_fn: Callable[..., Any] | None=None):
        self.settings=settings or Settings.from_env()
        if completion_fn is None:
            from litellm import completion
            completion_fn=completion
        self._completion=completion_fn

    def chat(self, messages: list[dict[str,str]]) -> str:
        s=self.settings
        kwargs: dict[str,Any] = dict(model=s.litellm_model, messages=messages,
                                     temperature=s.temperature, max_tokens=s.max_tokens,
                                     timeout=s.timeout)
        if s.provider=='ollama':
            kwargs['api_base']=s.ollama_base_url
        else:
            kwargs['api_key']=s.openrouter_key
            kwargs['api_base']=s.openrouter_base_url
        response=self._completion(**kwargs)
        # LiteLLM ModelResponse uses attributes; support dict for test doubles.
        if isinstance(response,dict):
            return (response['choices'][0]['message']['content'] or '').strip()
        return (response.choices[0].message.content or '').strip()

    def generate(self, prompt: str) -> str:
        return self.chat([{'role':'user','content':prompt}])
