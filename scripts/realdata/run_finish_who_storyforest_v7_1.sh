#!/usr/bin/env bash
# No 4B/32B model calls. Existing V7 outputs are preserved; resumable 0.6B extraction backfill.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH="$PWD/src:$PWD/scripts/realdata${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests/realdata -p 'test_finish_who_storyforest_v7_1.py' -v
python -u scripts/realdata/finish_who_storyforest_v7_1.py "$@"
