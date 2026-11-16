# PreDiViD — Temporal Trajectory LLM V5: schema-constrained Ollama

## Why this fix?

V4's Ollama native probe succeeded but a benchmark sample failed with `Invalid Ollama response: invalid_response`, after an earlier `length` truncation at 256 tokens. This is consistent with the model returning JSON whose `prediction` is absent or not in the supported enum, **not** proof that all requested LLM classifications are wrong. The exact V4 response cannot be recovered from the supplied traceback because V4 did not log its content/shape.

V5 uses native `/api/chat` with a **JSON schema** specifying exactly one required field, `prediction`, whose value is `increase`, `flat`, or `revision`. The thinking switch is still explicitly `false`. The model is retried if the response is truncated or invalid. A response is considered valid only after explicit programmatic validation: JSON mode alone is not sufficient. No invalid output is converted to `flat`, `unknown`, or a correct label.

V5 **does not overwrite V4**. It bundles the original V4 script as a dependency and saves to `outputs/realdata/temporal_trajectory_study_v5` by default. Every successfully evaluated call is cached; failed calls are recorded separately in `llm_failures.jsonl` without storing any model reasoning text. Reruns with identical options preserve valid cached calls and retry missing calls.

Additional V5 `manifest.json` fields show `expected_llm_calls`, `completed_llm_calls`, `technical_failures_or_missing_calls`, `llm_study_complete`, and `paired_complete_examples`. A `paired_metrics.csv` file is generated only if both M3/M4 successfully completed for the same episode, to ensure fair comparison with M1/M2. The original `metrics.csv` may have incomplete M3/M4 sample rows after technical errors; consult `paired_metrics.csv` and the manifest instead.

## Run on remote repository (tmux session predivid2)

From the repository root:

```bash
unzip -o PreDiViD_Temporal_LLM_V5.zip
python -m unittest discover -s tests/realdata -p 'test_temporal_trajectory_v5.py' -v

mkdir -p logs
set -o pipefail
python -u scripts/realdata/temporal_trajectory_study_v5.py \
  --max-llm-per-disease 1 \
  --llm-max-tokens 256 \
  --llm-max-output-tokens 1024 \
  --llm-timeout-seconds 240 \
  2>&1 | tee logs/temporal_study_v5_smoke.log
```

If at least the small run works, reuse the cache by increasing the **number of examples only**, keeping the same model, prompt and generation limits:

```bash
python -u scripts/realdata/temporal_trajectory_study_v5.py \
  --max-llm-per-disease 5 \
  --llm-max-tokens 256 \
  --llm-max-output-tokens 1024 \
  --llm-timeout-seconds 240 \
  2>&1 | tee logs/temporal_study_v5_small.log
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/realdata/temporal_trajectory_study_v5')
manifest=json.loads((p/'manifest.json').read_text())
for key in ('expected_llm_calls','completed_llm_calls','technical_failures_or_missing_calls',
            'paired_complete_examples','llm_study_complete'):
    print(f'{key}: {manifest[key]}')
for file in ('paired_metrics.csv','llm_failures.jsonl'):
    if (p/file).exists():print(f'Output: {p/file}')
PY
```

**If all calls remain invalid**, check `llm_failures.jsonl` for `json_keys`, `prediction_value`, `done_reason`, and `eval_count`. A `json_keys` result such as `["thought"]` indicates that reasoning-shaped JSON may have been emitted despite `think: false`. Do not interpret it as a model prediction. Check `ollama --version` and the model details; if needed, try a model with reliable structured-output support as a separately labeled experiment.

The V5 tests (14 passed locally) use mocked native Ollama responses and a synthetic dataset; **no test was run against the user's remote Ollama or private KG**. The code preserves existing fixed chronological train/holdout evaluation, the M2 with/without KG ablation, the non-future-information rule, and the scientific warning that these are aggregate reported-value forecasts **not validated documentary Story Forest links**, causal transmission or clinical predictions.
