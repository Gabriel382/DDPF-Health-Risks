#!/usr/bin/env bash
# One and only final batch for the existing V5 40-episode LLM study.
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p logs
log="logs/predivid_final_one_shot_audit.log"
{
  echo '=== FINAL TEMPORAL STUDY: ALL AUTOMATED TESTS IN ONE RUN ==='
  echo 'Existing V5 predictions and cached M3/M4 outputs are reused. No Ollama, no Neo4j.'
  python -m unittest discover -s tests/realdata -p 'test_*.py' -v
  echo '=== FINAL COUNTRY-BLOCK STATISTICS + KG PLACEBO ==='
  python -u scripts/realdata/final_temporal_evaluation.py \
    --bootstraps 2000 --country-signflips 5000 --kg-placebos 99
  echo '=== COMPLETE: the final report was saved ==='
  echo 'outputs/realdata/temporal_trajectory_study_v5/final_one_shot_report.md'
} 2>&1 | tee "$log"
