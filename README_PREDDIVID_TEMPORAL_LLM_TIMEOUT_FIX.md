# PreDiViD Temporal LLM V3: timeout and JSON backend fix

The V2 error `litellm.Timeout ... Timeout passed=120.0` means that LiteLLM did not receive a finished Ollama response within 120 seconds. It does not indicate a Neo4j, dataset, or KNN failure. The four prior unit tests passed.

## Changes

- Adds `scripts/realdata/temporal_trajectory_study_v3.py`, derived from V2 (V2 is left untouched).
- Default Ollama backend directly calls `POST /api/chat` with `think:false`, `format:"json"`, `stream:false`, and a short `options.num_predict` budget (192 tokens). This uses the Ollama-native API rather than relying on LiteLLM to forward thinking flags.
- Default per-request timeout 240 seconds, independently adjustable. A failed request is **never cached as `unknown`**. A later rerun resumes completed V3 cache entries, as long as input/prompt/model/decoding settings match.
- Adds `--llm-backend litellm` only for comparison/debugging; the native backend is the recommended one for the local Qwen3 test.
- Adds a short standalone probe, `scripts/realdata/probe_ollama_temporal.py`, before running the full study.
- Preserves V2 baselines and includes M2 with/without KG ablation. V3 uses a **separate output directory** and distinct LLM prediction signature. It does **not** re-use old V2 predictions under incompatible settings.

## Run (from the repository root)

```bash
unzip -o PreDiViD_Temporal_LLM_Timeout_Fix.zip

python -m unittest discover -s tests/realdata \
  -p 'test_temporal_trajectory_v3.py' -v

# Quick 60-second maximum Ollama check
python -u scripts/realdata/probe_ollama_temporal.py --timeout 60

# Two common KG-covered examples, one from each disease (4 LLM calls)
mkdir -p logs
set -o pipefail
python -u scripts/realdata/temporal_trajectory_study_v3.py \
  --max-llm-per-disease 1 \
  --llm-max-tokens 192 \
  --llm-timeout-seconds 240 \
  2>&1 | tee logs/temporal_study_v3_smoke.log

# Resume same V3 cache and expand to 5 per disease
python -u scripts/realdata/temporal_trajectory_study_v3.py \
  --max-llm-per-disease 5 \
  --llm-max-tokens 192 \
  --llm-timeout-seconds 240 \
  2>&1 | tee logs/temporal_study_v3_small.log

cat outputs/realdata/temporal_trajectory_study_v3/metrics.csv
```

If the probe times out, first check the runtime state:

```bash
ollama ps
ollama list
nvidia-smi   # when NVIDIA GPU is available
```

Check for another process using Ollama, an unloaded model, insufficient GPU memory, or unusually slow CPU inference. Do not run the large LLM evaluation until a quick probe succeeds. If `think:false` is unsupported by the installed Ollama/model version, the API error will be shown instead of being silently ignored. The native API supports `think` as a field but some versions/model combinations may behave differently.

### Troubleshooting

- `Ollama API HTTP 404`: model name or API endpoint incorrect. Check `.env` settings `OLLAMA_MODEL` and `OLLAMA_BASE_URL`, and `ollama list`.
- `Ollama API HTTP 400`: a model/server version may not support `think:false` or JSON format. Check Ollama version and model capabilities.
- `connection/timeout`: inspect `ollama ps` and `nvidia-smi`. A persistent failure indicates model/server runtime limitations rather than a benchmark metric.
- `Cache model/prompt/input mismatch`: use a new `--output` folder for different backend, token budget, or prompt. Do not delete scientific experiment caches.
- JSON invalid/truncated: inspect model compatibility; raise `--llm-max-tokens` to 320 **and use a different `--output`**, since that changes generation conditions.

### Scientific interpretation

These experiments predict the next direction of country-week reported aggregate values. They do **not** establish documentary event continuation, infection incidence, epidemiological transmission, or causal links. M4 checks static typed context/provenance, not full OWL entailment. Compare M1/M2 across full holdout, M3/M4 only on the common sampled holdout, and report coverage.
