# PreDiViD: Aggregate Temporal Trajectory Study (M1–M4)

This self-contained drop-in extends the existing repository after
`build_weekly_episodes.py`. It **does not modify** Neo4j, Prediction, or the
documentary Story Forest code.

## Scientific scope

- Source: reconstructed `:isReported` aggregate country-week snapshots from
  the BioPropaPhenKG (COVID-19 and Mpox).
- Observed result: whether the **next reported snapshot value** is higher,
  equal, or lower than the current snapshot.
- This task is **NOT** incidence estimation, epidemiological transmission,
  event-link truth, or the original Story Forest paper's documentary story trees.
- Data must be ordered and consecutive weekly. The program fails on gaps.
- No target week's value is passed to the methods. An explicit chronological
  holdout starts on 2024-01-01 by default.
- Previous country-source concepts from `hasPresence` are static, with their
  collection dates retained, not interpreted as weekly observations.

## Methods

| ID | Description | Evaluation coverage |
| --- | --- | --- |
| M1 | Persistence of the last observed snapshot change | Full chronological holdout |
| M2 | k-nearest previous-week patterns + optional static KG concept Jaccard context | Full chronological holdout |
| M3 | LLM predicts next snapshot change from historical values only | Deterministic bounded sample |
| M4 | LLM + static KG concepts + typed/provenance input guard | Same bounded sample |

M4 is **not** an OWL reasoner: it only implements explicit schema and provenance
checks. It must not be presented as formal BioPropaPhen OWL validation. That
requires a separate implementation and benchmark.

Negative reporting corrections are labeled `revision`, preserved in outputs,
and **excluded** from the headline two-class increase-vs-flat F1 (counts
reported). Abstentions (`unknown`) lower coverage and are counted as wrong
in unconditional class F1.

## Run in your existing `predivid2` tmux

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
unzip -o PreDiViD_Temporal_Trajectory_Study.zip

python -m pip install numpy python-dotenv

python -m unittest discover -s tests/realdata     -p 'test_temporal_trajectory_study.py' -v

# Preflight: M1 and M2 only. No calls to Ollama.
python -u scripts/realdata/temporal_trajectory_study.py --skip-llm     --max-llm-per-disease 5     --output outputs/realdata/temporal_trajectory_study

# Bounded live test, up to 5 examples per disease, M3 and M4
mkdir -p logs
set -o pipefail
python -u scripts/realdata/temporal_trajectory_study.py     --max-llm-per-disease 5     --output outputs/realdata/temporal_trajectory_study     2>&1 | tee logs/temporal_study_smoke.log

# Full planned sample (20 per disease = 40 origins = 80 LLM judgments)
python -u scripts/realdata/temporal_trajectory_study.py     --max-llm-per-disease 20     --output outputs/realdata/temporal_trajectory_study     2>&1 | tee logs/temporal_study_full.log
```

`tmux` detach: Ctrl+B then D. Runs stay in foreground in tmux; no need to
start Docker again. You can use `bash scripts/realdata/run_temporal_study_tmux.sh`
for the full sample after smoke test.

**Important**: The script detects existing cached LLM outputs with different
model/prompt/dataset settings and rejects them. Use a new `--output` directory
for a different LLM or when changing the data. No secrets printed or stored.

## Required inputs

- `outputs/realdata/who_series/weekly/snapshots.csv`
- `outputs/realdata/reconstructed_observations.jsonl` (optional but needed to
  interpret M2 as KG-assisted and to get M4 coverage)
- `.env` with valid `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=qwen3:4b`
  (or valid OpenRouter settings), for M3/M4.

Verify `ollama list` before starting if using the local backend.

## Outputs

At `outputs/realdata/temporal_trajectory_study/`:

- `predictions.csv`: row-level future-observed-change labels and method output,
  with both original Neo4j relation IDs.
- `metrics.csv`: headline macro-F1, balanced accuracy, accuracy, coverage;
  `M1/M2__all_holdout` and `__common_sample`, and `M3/M4__common_sample`.
- `manifest.json`: hashes, split dates, counts, caveats, confusion matrices,
  exact sample strategy and graph-profile availability.
- `llm_predictions.jsonl`: append-only, resumable LLM outputs, model signature.
- `temporal_adjacency_not_link_gold.jsonl`: deterministic adjacency with
  `is_documented_event_continuation=false`.

## Methodological cautions

1. The BioPropaPhenKG `hasPresence` source dates are fixed corpus periods.
   The KG concepts are therefore static historical context, not weekly
   source-document observations.
2. WHO fields `Cumulative_cases` and `total_cases` were not independently
   proven equivalent. Compare changes **within each disease**, not between
   magnitudes of diseases.
3. COVID and Mpox cover different historical periods; country/date effects
   can confound disease classification. This benchmark predicts directions
   of within-disease reporting snapshots, not disease labels.
4. Country-week adjacency is a deterministic baseline, not evidence of
   linked outbreaks or documentary narrative transitions.
5. M2 uses labeled historical future-change training targets up to 2023;
   M1/M3/M4 use inference without those labels. This is a comparison of
   pipelines with different supervision, not an ablation isolating LLM alone.
6. An official OWL reasoner, source document metadata, gold event-pair labels,
   and a proper event-forest evaluation remain necessary for the primary
   *Discover Artificial Intelligence* Story Forest claim.
7. Results only count as real when the script is run on your remote datasets.
   Bundled tests use synthetic data, and do not establish epidemiological
   correctness.

## Reproducibility

- No scripts modify the Neo4j dump or source data.
- No randomized train/test sampling: holdout by calendar date, LLM sample by
  SHA256 of the origin episode ID, separately within each disease.
- A model change requires a new output directory, detected by cache signature.
- Does not download datasets or call external APIs unless LLM inference is
  deliberately enabled without `--skip-llm`.

