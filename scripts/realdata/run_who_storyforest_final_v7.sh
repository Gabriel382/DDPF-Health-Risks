#!/usr/bin/env bash
# Single reproducible documentary experiment (no time-based background magic).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f .venv/bin/activate ]]; then
  # Source .venv automatically so the script also works after new SSH login.
  source .venv/bin/activate
fi
if ! command -v python >/dev/null 2>&1; then
  echo 'ERROR: python not found; activate the project .venv first.' >&2
  exit 2
fi
export PYTHONPATH="$PWD/src:$PWD/scripts/realdata${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs outputs/realdata/who_storyforest_final_v7
printf '\n== WHO V7 documentary validation and regression tests ==\n'
python -m unittest discover -s tests/realdata -p 'test_*.py' -v
if [[ -d tests/storyforest ]]; then
  python -m unittest discover -s tests/storyforest -v
fi
printf '\n== WHO source preflight (fail closed BEFORE model execution) ==\n'
python -u scripts/realdata/who_storyforest_final_v7.py --audit-only \
  2>&1 | tee logs/who_storyforest_final_v7_sources.log
printf '\n== Checking required local Ollama models ==\n'
if ! command -v ollama >/dev/null 2>&1; then
  echo 'ERROR: Ollama not installed. Install/enable Ollama, rerun.' >&2
  exit 2
fi
for model in qwen3:0.6b qwen3:4b qwen3:32b; do
  if ! ollama show "$model" >/dev/null 2>&1; then
    ollama pull "$model"
  fi
done
printf '\n== WHO sources, extraction, fixed-pair linking, incremental StoryForest, paper export ==\n'
python -u scripts/realdata/who_storyforest_final_v7.py \
  --models qwen3:0.6b,qwen3:4b,qwen3:32b \
  --tree-model qwen3:0.6b \
  --top-k 4 \
  --timeout 1800 \
  2>&1 | tee logs/who_storyforest_final_v7.log
printf '\n== Final outputs ==\n'
ls -lh outputs/realdata/who_storyforest_final_v7/{final_documentary_report.md,paper_integrated_results_v7.md,paper_who_silver_table_v7.tex,human_adjudication.csv,study_manifest.json}
