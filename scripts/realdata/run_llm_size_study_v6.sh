#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests/realdata -p 'test_temporal_trajectory_v5.py' -v
python -m unittest discover -s tests/realdata -p 'test_llm_size_study_v6.py' -v
mkdir -p logs
python -u scripts/realdata/llm_size_study_v6.py "$@" 2>&1 | tee logs/llm_size_study_v6.log
