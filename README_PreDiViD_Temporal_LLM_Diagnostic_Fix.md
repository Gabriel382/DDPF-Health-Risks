# PreDiViD: M3/M4 Ollama abstention diagnostics and temporal study v2

This patch is an isolated **new script**, not a replacement for the original study. It leaves Neo4j, M1/M2 results, and existing cached LLM predictions untouched.

## Why patch?

- Prior results: M1/M2 produced 1,565 holdout forecasts; M3/M4 each returned `unknown` for 10/10 sampled episodes.
- M4 returns `missing_static_kg_profile` *before* calling Ollama when the country/disease profile is absent.
- M3's original prompt permitted `unknown` whenever it was uncertain, which might result in near-universal abstention.
- The original parser quietly turned invalid or empty Ollama content into an `unknown` and cached it, obscuring configuration failures.

The package includes an explicit diagnostic, a **forced-choice** prompt that still permits `unknown` for corrupted/insufficient history, an invalid-response hard failure, a more robust JSON parser, and an increased token allowance for reasoning models. It adds an **M2 KNN without KG** ablation. The default comparison sample is restricted to episodes whose origin has a static KG profile, so M3 and M4 can be meaningfully compared on the *same eligible sample*.

The `/no_think` instruction is a Qwen3-oriented prompt hint and is not guaranteed to disable all model reasoning; invalid/empty responses now fail loudly so they are not silently cached.

This is a **new experimental condition** (different prompt and sampling), not an equivalent rerun of the prior M3/M4 setup. M4 is only a typed/provenance guard, **not OWL reasoning**.

## Install (in the project root)

```bash
unzip -o PreDiViD_Temporal_LLM_Diagnostic_Fix.zip
python scripts/realdata/diagnose_temporal_llm.py
python -m unittest discover -s tests/realdata -p 'test_temporal_trajectory_v2.py' -v
```

### 1. Diagnostic only (no API calls)

```bash
python scripts/realdata/diagnose_temporal_llm.py \
  --output outputs/realdata/temporal_trajectory_study
```

Inspect the `reasons` and `manifest_context` sections to identify whether the problem was `invalid_json`, `missing_static_kg_profile`, or the LLM's intentional `unknown`.

### 2. New smoke test (5 KG-eligible examples per disease)

```bash
python -u scripts/realdata/temporal_trajectory_study_v2.py \
  --max-llm-per-disease 5 \
  --output outputs/realdata/temporal_trajectory_study_v2 \
  2>&1 | tee logs/temporal_study_v2_smoke.log
```

Cache goes into `outputs/realdata/temporal_trajectory_study_v2/`. **Never delete the old cache**; keep it for audit and comparison. The program rejects mismatched prompt/model/data signatures if you reuse an output directory.

If invalid LLM outputs are detected, the script **raises an exception instead of caching a fake abstention**. The cache remains resumable. If the script reports no KG-eligible episodes, inspect the profile IDs; don't fabricate or inject KG context.

### 3. Larger run after successful smoke test

```bash
python -u scripts/realdata/temporal_trajectory_study_v2.py \
  --max-llm-per-disease 20 \
  --output outputs/realdata/temporal_trajectory_study_v2 \
  2>&1 | tee logs/temporal_study_v2_full.log
```

### Interpretation caveats

- M1/M2 entire holdout can be compared under the same chronological split; M3/M4 are only on the shared KG-covered sample. Report **both coverage and metrics**.
- Static `hasPresence` concepts are corpus-time context only, **not weekly evidence**.
- Semantic benefit of M2 must be measured by comparing its `noKG` ablation; a higher score than persistence alone does not prove the KG helped.
- The target is direction of next reported cumulative/total value, not incidence, transmission, event continuity, or Story Forest gold links.
- Study v2 is for **method development**. Do not directly compare its M3/M4 to old M3/M4 without accounting for prompt and sample changes.
