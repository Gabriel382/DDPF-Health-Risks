"""Project-root .env settings. Never print credentials."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for location in [Path.cwd(), *here.parents]:
        env_file = location / '.env'
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            return

@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    ollama_base_url: str
    openrouter_key: str
    openrouter_base_url: str
    temperature: float
    max_tokens: int
    timeout: int
    sqlite_path: str
    max_candidates: int

    @classmethod
    def from_env(cls, require_key: bool = True) -> 'Settings':
        load_project_env()
        provider = os.getenv('LLM_PROVIDER', 'ollama').lower().strip()
        if provider not in {'ollama', 'openrouter'}:
            raise ValueError('LLM_PROVIDER must be ollama or openrouter')
        model = os.getenv('OLLAMA_MODEL', 'qwen3:4b') if provider == 'ollama' else os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
        key = os.getenv('OPENROUTER_API_KEY', '').strip()
        if provider == 'openrouter' and require_key and not key:
            raise ValueError('OPENROUTER_API_KEY missing; set it in root .env (never commit it)')
        return cls(provider=provider, model=model,
                   ollama_base_url=os.getenv('OLLAMA_BASE_URL','http://localhost:11434').rstrip('/'),
                   openrouter_key=key,
                   openrouter_base_url=os.getenv('OPENROUTER_API_BASE','https://openrouter.ai/api/v1').rstrip('/'),
                   temperature=float(os.getenv('LLM_TEMPERATURE','0')),
                   max_tokens=int(os.getenv('LLM_MAX_TOKENS','600')),
                   timeout=int(os.getenv('LLM_TIMEOUT_SECONDS','120')),
                   sqlite_path=os.getenv('STORYFOREST_DB','outputs/storyforest/stories.sqlite3'),
                   max_candidates=max(1,int(os.getenv('STORYFOREST_TOP_K','8'))))

    @property
    def litellm_model(self) -> str:
        prefix='ollama' if self.provider=='ollama' else 'openrouter'
        return self.model if self.model.startswith(f'{prefix}/') else f'{prefix}/{self.model}'
