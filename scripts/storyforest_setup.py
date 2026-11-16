"""Non-destructive setup assistant: create .env from .env.example if absent."""
from __future__ import annotations
from pathlib import Path
import shutil

def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'.env.example';dest=root/'.env'
    if not source.exists():raise SystemExit('Missing .env.example. Extract ZIP into repository root first.')
    if not dest.exists():
        shutil.copyfile(source,dest)
        print('Created .env. Configure LLM_PROVIDER and OpenRouter key if needed.')
    else:print('Existing .env preserved (never overwritten).')
    print('Next: python -m pip install -e . && python -m pip install -r requirements-storyforest.txt')
    print('Then: python -m ddpf.storyforest.cli doctor')
if __name__=='__main__':main()
