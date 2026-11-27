#!/usr/bin/env bash
# Complete available HIST experiment. The main run uses six models; auxiliary encoders are baseline-only.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
BASE=outputs/realdata/reviewer_revision_v8_ddpf_final
RUN="$BASE/HIST_sentence_full"
GOLD=data/reviewer_v8_ddpf/gold_final/HIST_sentence
RUNNER=(python -m scripts.realdata.reviewer_v8_ddpf.runner)
# Supports extra flags shared across main and controls only through explicit individual commands.
# --pull-missing: download missing models; --preflight-only: do not infer; --main-only: skip auxiliary baseline runs.
pull=(); preflight_only=false; main_only=false
while (($#)); do
  case "$1" in
    --pull-missing) pull=(--pull-missing);;
    --preflight-only) preflight_only=true;;
    --main-only) main_only=true;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
  shift
done
python -m unittest discover -s tests/realdata -p test_reviewer_v8_ddpf.py -v
"${RUNNER[@]}" --preflight --neo4j --check-models --out "$RUN" "${pull[@]}"
# Keep reference data outside the inference process. Do not overwrite any annotation.
if [[ ! -e "$GOLD/annotator_A.csv" && ! -e "$GOLD/annotator_B.csv" && ! -e "$GOLD/adjudication.csv" ]]; then
  "${RUNNER[@]}" --review --out "$RUN" --gold-dir "$GOLD" --review-pairs 400
fi
if $preflight_only; then exit 0; fi
# The legacy source records are aggregate observations; explicitly label this run exploratory.
"${RUNNER[@]}" --full --exploratory --neo4j --out "$RUN" "${pull[@]}"
if ! $main_only; then
  "${RUNNER[@]}" --baselines-only --neo4j --embedding binary_bow --out "$BASE/HIST_sentence_BoW_baselines"
  # Qwen embedding is optional only if the model is unavailable. Errors are visible and do not masquerade as scores.
  if ! "${RUNNER[@]}" --baselines-only --neo4j --embedding ollama --embedding-model qwen3-embedding:4b --out "$BASE/HIST_sentence_QwenEmbedding_baselines" "${pull[@]}"; then
    echo 'Qwen3 embedding baseline failed; inspect log and run separately. Main run preserved.' >&2
  fi
  NODE2VEC=data/reviewer_v8_ddpf/sentence_node2vec_vectors.jsonl
  if [[ -f "$NODE2VEC" ]]; then
    "${RUNNER[@]}" --baselines-only --neo4j --embedding-jsonl "$NODE2VEC" --out "$BASE/HIST_sentence_Node2Vec_baselines"
  else
    echo 'Node2Vec archive not found at data/reviewer_v8_ddpf/sentence_node2vec_vectors.jsonl; no Node2Vec results fabricated.' >&2
  fi
fi
printf '\nMain outputs: %s\nBlind reference annotations: %s\n' "$RUN" "$GOLD"
