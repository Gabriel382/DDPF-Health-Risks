#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
python -m unittest discover -s tests/realdata -p 'test_reviewer_v8_ddpf.py' -v
python -m scripts.realdata.reviewer_v8_ddpf.runner "$@"
