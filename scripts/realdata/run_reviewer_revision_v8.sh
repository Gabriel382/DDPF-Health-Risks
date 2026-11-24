#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$PWD/scripts/realdata:$PWD/src:${PYTHONPATH:-}"
"$PYTHON" -m unittest discover -s tests/realdata -p 'test_reviewer_revision_v8.py' -v
exec "$PYTHON" -u -m reviewer_v8.runner "$@"
