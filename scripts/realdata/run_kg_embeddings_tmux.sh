#!/usr/bin/env bash
# Safe to run from repository root inside tmux. Does not stop if server SSH disconnects.
set -euo pipefail

python scripts/realdata/recover_kg_observations.py --inspect

if [[ ! -f outputs/realdata/reconstruction_manifest.json ]]; then
  python scripts/realdata/recover_kg_observations.py
else
  echo 'Reconstruction manifest already exists: not overwriting data.'
fi

python scripts/realdata/embed_kg_observations.py \
  --models qwen3-embedding:0.6b embeddinggemma nomic-embed-text --features concepts --batch-size 8

python scripts/realdata/embed_kg_observations.py \
  --models qwen3-embedding:0.6b embeddinggemma nomic-embed-text --features concepts --mask-disease-terms --batch-size 8

python scripts/realdata/compare_kg_embeddings.py

echo "Finished. Check outputs/realdata/embedding_benchmark/results.csv"
