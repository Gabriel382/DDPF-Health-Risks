# PreDiViD: full executable experimental batch (v1)

This **drop-in overlay** extends, rather than replaces, the previously installed packages. It has been validated with 7 unit tests and an end-to-end structural/benchmark smoke run on synthetic observations (not scientific results):

- `PreDiViD_StoryForest_LiteLLM_DropIn.zip`
- `PreDiViD_KG_Embedding_Rebuild_DropIn.zip`

The `outputs/realdata/reconstructed_observations.jsonl` file must already exist. The Neo4j dump must already have been restored to Neo4j in the user's Linux environment. **Nothing in this package mutates Neo4j.**

The aim is to rerun every experiment *currently justified by the recovered aggregate data*, without falsely reporting clustering accuracy as epidemiological linking accuracy.

## Install (run in `predivid2` tmux on Linux)

The ZIP has `scripts/realdata/` and `tests/realdata/` directly at its root. Transfer it to the Linux project and extract **at the repository root**:

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
unzip -o PreDiViD_All_Experiments_Ready.zip
```

It does not overwrite your input observations, existing model matrices, Neo4j, or the Story Forest code. This overlay requires the previously installed `embed_kg_observations.py`.

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
python -m pip install numpy scipy scikit-learn neo4j python-dotenv
python -m pip install gensim  # optional for node2vec; recommended
# Install torch for your CPU/GPU following the official PyTorch instructions.
# Do not automatically reinstall torch if already present.
python -m unittest discover -s tests/realdata -p 'test_full_experiment_suite.py' -v
```

Check model availability:

```bash
ollama list
ollama pull qwen3-embedding:4b # OPTIONAL, ~2.5 GB additional
```

**Launch the complete suite, including Qwen3 Embedding 4B, from tmux:**

```bash
tmux attach -t predivid2
# once inside that tmux session:
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
mkdir -p logs
set -o pipefail
python -u scripts/realdata/run_all_predivid_experiments.py \
  --with-qwen4b --pull-missing \
  2>&1 | tee logs/predivid_all_experiments.log
```

`--pull-missing` asks Ollama to download any missing requested tags (can use bandwidth and disk). To run ONLY the three already-tested model families, omit `--with-qwen4b` and `--pull-missing`.

Detach with **Ctrl+B, then D**. Monitor progress from another SSH login:

```bash
tail -f /data/dataRapide/gabriel/DDPF-Health-Risks/logs/predivid_all_experiments.log
```

If `predivid2` is already attached to the current terminal, simply run commands inside it, rather than attaching again.

## Quick preflight

```bash
python scripts/realdata/run_all_predivid_experiments.py --help
python -m unittest discover -s tests/realdata -p 'test_full_experiment_suite.py' -v
ls -lh outputs/realdata/reconstructed_observations.jsonl
```

If these pass, run the long command above. The Neo4j audit uses `NEO4J_URI` in `.env`, which on your Docker host is likely `bolt://127.0.0.1:17687`, not the container's port 7687.

## What the suite does

### A. Text embeddings on recovered observations

Uses the *existing* `embed_kg_observations.py`, which caches each completed matrix, resumes partial runs and verifies input hash. Uses exactly the same input observations and their ordered identifiers for every method.

- 4 text variants: `concepts`, `concepts_country`, `concepts_time`, `concepts_country_time`;
- 2 lexical settings: original concepts and *remove explicit disease-name concepts*;
- 3 default local embedding models: `qwen3-embedding:0.6b`, `embeddinggemma`, `nomic-embed-text`;
- optional fourth: `qwen3-embedding:4b`.

This yields **24 text conditions by default**, or **32 with the 4B model**. Already completed matrices are reused, not recomputed.

The date-text ablation does not invent dates. If original `hasPresence` relationship properties contain no parsable dates, time variants can be identical to no-time variants and cannot establish a temporal benefit.

### B. Structural baselines (16 configurations)

`structural_observation_embeddings.py` builds a bounded bipartite graph from the reconstructed **observation-to-concept** associations. Critically, it **does not** project 13.9M BioPropaPhenKG nodes into RAM, nor use document/source/disease labels in the graph.

- `svd`: TF-IDF on observation/concept incidence followed by TruncatedSVD (true spectral projection).
- `fastrp_style`: fixed-seed random node features with multi-hop graph propagation. **FastRP-inspired**, not Neo4j GDS FastRP. Report this distinction.
- `node2vec`: biased random walks + skip-gram Word2Vec on the projected bipartite graph, with fixed seed, 1 worker. Needs `gensim`.
- `graphsage`: two-layer, neighborhood-aggregation model trained with self-supervised observed-edge vs non-edge reconstruction, with fixed seed. Needs `torch`. This is a **GraphSAGE-style local proxy**, not identical to the thesis's original GraphSAGE implementation.

Each algorithm runs with and without country nodes, and with and without explicitly disease-named concepts. That is 4 algorithms × 2 × 2 = 16 matrix configurations. **No result from these conditions is claimed to be the original FastRP, Node2Vec or GraphSAGE embedding of the entire BioPropaPhenKG**.

If `gensim` or `torch` is not installed, affected configurations are explicitly marked **skipped**, with a reason in `outputs/realdata/structural_embeddings/run_status.json`. To require every structural algorithm and instead stop when a dependency is missing, add `--strict-structural` to the suite invocation.

### C. Stronger, reproducible clustering evaluation

`extended_embedding_benchmark.py` aggregates binary BoW, TF-IDF BoW, completed Ollama matrices and completed projected structural matrices, verifying the original input SHA256 and precise row order. For each:

- K-Means with **five fixed random seeds**, on raw and unit-normalized vectors;
- optional cosine agglomerative clustering on normalized vectors;
- ARI and NMI, plus class-label *oracle-aligned* macro-F1 and balanced accuracy;
- 200 country-block bootstrap replicates to estimate descriptive uncertainty with **fixed** cluster assignments;
- 200 label permutations **within journal/medical/social source modalities**, as an exploratory chance-level check;
- automated diagnostics for country overlap, disease-word leakage and class coverage.

A standard deviation across optimization seeds is **not** an uncertainty interval for a dataset. Block-bootstrap CIs and permuted-label statistics also have limitations and are annotated separately.

**We do not claim prediction of epidemic spread.** These are retrospective separation metrics on COVID-19 and Mpox corpus labels, with temporal confounding.

### D. Story Forest and documentary provenance

`storyforest_source_audit.py` runs read-only Neo4j schema and sample queries (`CALL db.labels()`, `CALL db.relationshipTypes()`, sample `hasPresence` and `isReported`) to identify whether the restored dump contains plausible document and source properties.

**The 126 country-by-source observations are NOT individual events.** They lack independently verifiable source-document IDs and event-link labels. Consequently, the default suite **does not** create fake event dates, fabricate documentary evidence, or label Story Forest links as correct. This stage is reported as `blocked_missing_document_provenance_event_jsonl`, not a research result.

When a separately verified chronological document-level `events.jsonl` exists, run the same suite with:

```bash
python -u scripts/realdata/run_all_predivid_experiments.py \
  --skip-text --skip-structural --skip-benchmark --skip-source-audit \
  --real-events data/benchmark/verified_events.jsonl --allow-llm
```

This executes the five current conditions: temporal, KG-rules, LLM-only, LLM+KG and full. The input must have real event IDs, original document evidence IDs, source IDs, and timestamps. If independent pair labels exist, add `--gold path/to/independent_gold.jsonl`. **Having a gold file is not the same as showing its labels were created independently of input KG material**.

The included `config/storyforest_rules.json` is a configurable constraint example, not a complete OWL PropaPhen reasoner. Without independent labels, relations and link counts are functional outputs, not event-link precision or F1.

### E. Why full-scale GDS is not auto-installed

The server currently uses Neo4j **4.4.3**. Neo4j's current compatibility matrix lists **Neo4j 4.4.9 or later** for GDS 2.6. Running GDS FastRP/GraphSAGE on the full 13.9M-node / 54M-edge graph requires separately planned compatibility, memory estimates, and perhaps a sampled graph projection. The suite deliberately **does not** install an incompatible GDS plugin, allocate the full graph in RAM, or substitute a different algorithm while calling it GDS FastRP.

## Outputs

```text
outputs/realdata/
  reconstructed_observations.jsonl            # existing, unmodified
  embeddings/                                 # cached text matrices
  structural_embeddings/                      # structural matrices & run_status.json
  experiment_suites/<UTC-run-id>/
    suite_manifest.json                       # inputs, hashes, conditions, stage status
    logs/                                     # stdout/stderr per stage
    benchmark/results.csv                     # combined, fair aligned comparisons
    benchmark/results.json                    # all seeds, diagnostics and uncertainty
    benchmark/diagnostics.json                # source/year/country confounders
    storyforest_source_audit.json              # read-only Neo4j schema investigation
```

Each run creates a unique directory. Re-executing the suite will reuse completed model matrices. It will not erase or overwrite the original `outputs/realdata/embedding_benchmark/results.csv`.

## Inspect results

```bash
# latest timestamped suite
ls -td outputs/realdata/experiment_suites/* | head -1
LATEST=$(ls -td outputs/realdata/experiment_suites/* | head -1)
column -s, -t < "$LATEST/benchmark/results.csv" | less -S
cat "$LATEST/benchmark/diagnostics.json"
cat "$LATEST/suite_manifest.json"
```

If the bash pipeline fails, inspect `logs/predivid_all_experiments.log` and `suite_manifest.json`: the latter identifies the failed stage. Fix the issue and re-run; completed Ollama matrices and structural outputs will be reused.

## Testing without Neo4j or Ollama

```bash
python -m unittest discover -s tests/realdata -p 'test_full_experiment_suite.py' -v
```

Tests use synthetic, tiny observation-concept graphs to validate matrix shapes, masking, graph symmetry, correct hashes, deterministic propagation, blocked unsupported Story Forest input and uncertainty calculations. **They are not results on actual COVID/Mpox data**.

## Scientific scope

Current comparisons use 126 reconstructed observations, which are *country × source/corpus aggregates*. F1 is computed after assigning each unsupervised cluster to a disease label using the evaluation labels (oracle alignment), so it does not estimate the performance of an unattended disease classifier. Model selection on the same observations risks optimistic ranking. A robust article will need independent time/location splits, source checks, and document-level linking references for Story Forest.
