# PreDiViD V6 — One-shot Qwen3 model-size experiment

This is **one final exploratory scaling experiment**, not a replacement for the V5 scientific audit. It compares the same **40 deterministic KG-covered** episodes (20 COVID-19 and 20 Mpox), each assessed by M3 (LLM-only) and M4 (LLM + static KG guard). The task is predicting the direction of next-week **aggregate reported values**, not biological transmission, incident cases, or documentary Story Forest links.

## What it tests

Same Qwen3 family, same 40 episodes, same prompt and JSON schema, same decoding (`temperature=0`, thinking off, `num_ctx=4096`, `num_predict=256` with retries up to 1024), same labels:

| Model | Approximate Ollama model download | Calls |
|---|---:|---:|
| `qwen3:0.6b` | 523 MB | 80 |
| `qwen3:1.7b` | 1.4 GB | 80 |
| `qwen3:4b` | 2.5 GB | **Reuse 80 V5 calls**, if the V5 hash/settings match |
| `qwen3:8b` | 5.2 GB | 80 |
| `qwen3:14b` | 9.3 GB | 80 |
| `qwen3:32b` | 20 GB | 80 |

Approximately **400 new calls** if the 4B cache matches; otherwise the script evaluates 4B anew. The new models are installed one by one via `ollama pull`, and the previous model is stopped to reclaim memory. Model download amounts do not equal RAM, VRAM, or peak inference requirements. Total downloads may require ~40 GB of disk before overhead. **125 GB host RAM is not 125 GB GPU VRAM**. A large model with CPU offload may be slow; there is no runtime guarantee.

Models are standard Ollama model tags; quantization and model digests are saved in the manifest. Compare model family rather than mixing unrelated models. `qwen3:30b` is deliberately **not** used because its Mixture-of-Experts architecture complicates a simple dense-parameter scaling comparison.

## Single command on the Linux server

Use the **same server** where V5 completed: `gabriel@192.168.20.14` (`pc-15rpqd3`). In the repository root:

```bash
unzip -o PreDiViD_Qwen3_Size_OneShot_V6.zip
bash scripts/realdata/run_llm_size_study_v6.sh
```

The shell script runs all included V5 and V6 tests, downloads models, performs **all** size tests, and prints a final report. There is no need to ask for individual follow-up scripts. Run inside tmux (`predivid2`) if you want the job to survive disconnects. It can be restarted with the **same command**; successful predictions are never rerun. Failures are explicitly recorded and are not silently converted into predictions.

Optional alternatives **only if necessary**:

```bash
# Models already installed (no downloads):
bash scripts/realdata/run_llm_size_study_v6.sh --skip-pull

# Skip 32B if CPU-only execution or disk constraints make it impractical.
# IMPORTANT: use a distinct output directory for a different model set.
bash scripts/realdata/run_llm_size_study_v6.sh \
  --models qwen3:0.6b,qwen3:1.7b,qwen3:4b,qwen3:8b,qwen3:14b \
  --output outputs/realdata/llm_size_study_v6_without_32b
```

## Outputs

In `outputs/realdata/llm_size_study_v6/`:

- `size_comparison_report.md`: **main report** with comparison, controls, trend and caveats.
- `study_manifest.json`: **full detailed results**, hashes, model metadata/quantization, completeness.
- `metrics_by_size.csv`: accuracy, macro-F1, increase recall by disease and by model, complete-sample and success-only estimates.
- `predictions_by_size.csv`: same paired examples, model predictions and true labels.
- `model_runtime.csv`: latency, output tokens and measured tokens/s.
- `predictions_qwen3_*.jsonl`: separate **resumable per-model caches**.
- `failures_qwen3_*.jsonl`: invalid responses or model/backend errors, **not** converted into model failures on the benchmark task.

`logs/llm_size_study_v6.log` is the combined terminal log.

The report includes **country-block bootstrap confidence intervals** (2,000 resamples), cluster sign-flip p-values and Holm correction for the two predefined largest-vs-smallest accuracy tests, plus an exploratory exact permutation Spearman correlation of model size versus macro-F1. These are **exploratory** (only six sizes and 40 selected examples); an apparent trend cannot be interpreted as causal because different model sizes also differ in training behavior and quantization.

### Download final results to Windows

From **Windows PowerShell**, outside SSH:

```powershell
scp `
  gabriel@192.168.20.14:/data/dataRapide/gabriel/DDPF-Health-Risks/outputs/realdata/llm_size_study_v6/study_manifest.json `
  gabriel@192.168.20.14:/data/dataRapide/gabriel/DDPF-Health-Risks/outputs/realdata/llm_size_study_v6/size_comparison_report.md `
  "$HOME\Downloads\"
```

## Methodological safeguards

- Requires the existing V5 `manifest.json`, `llm_predictions.jsonl`, `predictions.csv`, `snapshots.csv`, and original static profiles file. The script does **not** regenerate M1/M2, reconstruct KG/embeddings, call Neo4j, or alter V5 files.
- Checks SHA-256 of the exact V5 data, cutoff (`2024-01-01`), KG-covered sample and 40 labels before running.
- Reuses 4B cache only after the **exact V5 configuration signature** matches the current model/prompt/temperature/context/output schema. A mismatch triggers fresh 4B inference, never cross-experiment mixing.
- Per-model caches have their own signatures; any configuration change requires a new `--output` folder.
- Ollama returns only one `prediction` field with `increase`, `flat` or `revision`; invalid or truncated JSON cannot become a prediction.
- Missing outputs are reported in **coverage**; comparisons across models use the **intersection of episodes completed by every model for both M3 and M4**.
- Reports both overall and disease-specific results to reveal possible `flat` collapse (particularly Mpox).
- Tests run locally with mocked Ollama, not against the user's GPU/server; actual model availability and runtime are determined on the Linux server.

The prior final audit's negative shuffled-KG control remains scientifically important: the scale study does **not** establish that KG semantic content helps, even if any M4 model becomes better at this task.
