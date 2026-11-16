#!/usr/bin/env bash
set -euo pipefail

# Run this INSIDE predivid2. Does NOT create/kill tmux or Docker services.
mkdir -p logs
python -u scripts/realdata/temporal_trajectory_study.py \
  --max-llm-per-disease "${LLM_SAMPLE_PER_DISEASE:-20}" \
  --output outputs/realdata/temporal_trajectory_study \
  2>&1 | tee logs/temporal_trajectory_study.log
