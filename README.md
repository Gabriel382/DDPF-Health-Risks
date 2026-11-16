# PreDiViD Temporal Trajectory V4: prevent Ollama output truncation

V3 passed the standalone native Ollama JSON probe but failed on a real benchmark example with `Ollama output truncated at num_predict=192`. This **is not a transport timeout**, and neither Neo4j nor the weekly dataset needs rebuilding.

## What V4 changes

- Requests **one classification only**, e.g. `{"prediction":"flat"}`; no natural-language `reason` in the generation request. The `reason` field in the saved cache is consequently empty and should not be interpreted as an explanation.
- Keeps `think: false`, `format: "json"`, `stream: false`, and direct local Ollama API `/api/chat`.
- On `done_reason: "length"`, discards that response and retries automatically with doubled output token budget (256 -> 512 -> 1024 by default). Even an apparently parseable truncated response is discarded.
- Never saves partial, invalid, or failed responses. Completed predictions are flushed and cached individually.
- Uses **separate V4 output directory** and a prompt/settings cache signature, so it does not mix V2/V3 outputs with V4 outputs.
- Keeps chronological train/holdout splits, M1 baseline, M2 noKG vs staticKG ablation, and shared, disease-balanced M3/M4 selection from V3.

**Important:** V4 has passed **12 mocked/synthetic local tests**. It has not been executed against your remote Ollama or actual local Neo4j data; that requires the commands below.

## Install and smoke-test (in your repository root)

```bash
unzip -o PreDiViD_Temporal_LLM_V4.zip
python -m unittest discover -s tests/realdata -p 'test_temporal_trajectory_v4.py' -v

mkdir -p logs
set -o pipefail
python -u scripts/realdata/temporal_trajectory_study_v4.py \
  --max-llm-per-disease 1 \
  --llm-max-tokens 256 \
  --llm-max-output-tokens 1024 \
  --llm-timeout-seconds 240 \
  2>&1 | tee logs/temporal_study_v4_smoke.log
```

The smoke test runs M3 and M4 on one shared KG-covered example per disease (up to 4 model calls, plus internal retries only if truncated).

When successful, run the same V4 program with `--max-llm-per-disease 5` to expand the sample, retaining other settings unchanged and **reusing the V4 cache**:

```bash
python -u scripts/realdata/temporal_trajectory_study_v4.py \
  --max-llm-per-disease 5 \
  --llm-max-tokens 256 \
  --llm-max-output-tokens 1024 \
  --llm-timeout-seconds 240 \
  2>&1 | tee logs/temporal_study_v4_small.log
cat outputs/realdata/temporal_trajectory_study_v4/metrics.csv
```

If even a one-field JSON response is truncated at 1024 output tokens, the script stops with an explicit error and does not save the truncated response. Then inspect the Ollama server version, model load, actual raw generation behavior and API token usage rather than silently modifying results. The `--llm-retries` option handles other exceptions, but the built-in truncation retries do not require it.

## Scientific caution

This is a retrospective **country-week reported aggregate direction** benchmark. M3/M4 predict `increase`, `flat`, or `revision` in the next weekly snapshot. The output is not case incidence, causal transmission, event identity, or documentary link ground truth. M4 validates typed/provenance properties in a limited fashion; it does not perform complete OWL entailment. To assess KG added value, compare M2 with- and without-KG variants on the same holdout; compare M3 vs M4 on their shared sample and report coverage.
