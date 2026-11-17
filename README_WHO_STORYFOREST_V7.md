# PreDiViD · WHO Documentary StoryForest Final Pilot V7

**Drop-in files, without overwriting any PreDiViD/StoryForest module or prior result.**

This is a self-contained **one-command** documentary evaluation that complements, rather than replaces, the completed V5/V6 next-week reported-value study.

## Execute in the existing Linux project checkout

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
unzip -o PreDiViD_WHO_StoryForest_Final_V7.zip
bash scripts/realdata/run_who_storyforest_final_v7.sh
```

The shell runner activates `.venv` automatically when it exists. For detachable SSH execution, start/attach your existing tmux session first. If you checked out the files from Git on Windows, commit and push as you did for V6, then `git pull` on the Linux server.

The script starts by running the new regression tests and available legacy StoryForest unit tests. It installs **no** Python packages and changes **no** Neo4j or MongoDB content. The existing `src/ddpf/storyforest` package must be present; the test suite verifies its APIs.

The script verifies or downloads three local Ollama models (`qwen3:0.6b`, `qwen3:4b`, `qwen3:32b`) and uses native `/api/chat` with temperature 0, `think:false`, schema-constrained JSON, adaptive output budgets, and a resumable **valid-output-only** cache.

### What it actually tests

1. **8 WHO Disease Outbreak News pages** (COVID-19 early 2020, mpox 2022): real URLs, downloaded into the user's ignored local `outputs/` folder, hashed and source-phrase checked. Download or evidence-anchor failure **fails closed** (not reported as verified).
2. **Direct LLM extraction** from WHO source text using the 3 Qwen sizes. This computes **reference date+event-type slot recall**. Extra extracted events are not assumed incorrect because the reference set is not exhaustive; we do not pretend to have extraction precision.
3. **Fixed candidate-pair link task** for all three sizes under `llm_only` and `full` (typed metadata + the configured StoryForest heuristic guard). Exactly **29 curated event cards, 39 curated reference pairs, 22 proposed positive core edges**. Scores apply *only* to these specified candidate pairs.
4. **Actual incremental StoryForest pipeline** for the selected 0.6B model with `process_event()` → SQLite `Store` → `forest()`, with `temporal`, `kg_rules`, `llm_only` and `full` methods. Tree structure diagnostics, silver-edge recall and top-k retrieval recall are saved.
5. **Extracted-event tree stage**, separately: uses Qwen3 0.6B events extracted directly from WHO pages, verifies that dates appear in source text, then constructs the real incremental forest. The generated trees are **qualitative/descriptive** and cannot receive a gold edge F1 without matching/adjudicating their nodes.
6. **Manuscript integration** from V5's `outputs/realdata/temporal_trajectory_study_v5/final_one_shot_audit.json` and V6's `outputs/realdata/llm_size_study_v6/study_manifest.json` plus `predictions_by_size.csv` when present. Does **not** rerun these experiments; recomputes binary macro-F1 consistently from raw V6 predictions if available.


### Crucial scientific boundaries

- **Curated WHO references are *silver*, not gold.** The 29 event cards and 39 pair labels were prepared using WHO pages and remain **unadjudicated**. WHO itself has not endorsed or annotated our tree relations. A small reviewer adjudication using `human_adjudication.csv` is required to claim independently reviewed gold labels. Even the cited WHO page anchor is only a source alignment check, not entailment.
- Link predictions are **conditional on the fixed, curated event cards**; the resulting StoryForest trees are **not end-to-end trees built from generated extractions**. The separate extraction track tests slot recall only.
- `full` exposes event type, location and disease CUI and applies the existing **heuristic** guard, not full OWL reasoning. It **does not query the BioPropaPhenKG**. Do not label this as a validated KG-grounding experiment; V5's KG feature ablations and negative control must remain separate.
- No gold documentary transmission links, no causal infection tree, no independently validated incidence; do not collapse them into the temporal weekly task.
- 8 selected documents do not support generalization or formal superiority of large models. Methods have different supervision budgets.
- All event cards contain **paraphrases**, not redistributed full WHO articles. The source articles are downloaded by the researcher at runtime from the official WHO pages and retain URL + hash provenance.

### Result files

`outputs/realdata/who_storyforest_final_v7/`

| File | Purpose |
|---|---|
| `final_documentary_report.md` | Main source-grounded WHO documentary report |
| `paper_integrated_results_v7.md` | Integrated V5 + V6 + V7 scientific results, explicitly separating tasks |
| `paper_who_silver_table_v7.tex` | Ready-to-review LaTeX table (silver pilot only) |
| `study_manifest.json` | Reproducibility metadata, annotation hashes, completion flags |
| `source_audit.json`, `sources/` | Download hashes and short evidence-anchor audits; raw WHO pages local only |
| `extracted_events.csv`, `extraction_metrics.csv` | Direct extraction results and reference slot recall |
| `pair_predictions.csv`, `pair_metrics.csv` | Fixed-candidate relation predictions and metrics |
| `tree_metrics.csv`, `trees/` | Oracle-card StoryForest incremental forest metrics, JSON trees and SQLite outputs |
| `generated_events.json`, `generated_event_audit.json`, `generated_trees/` | Separate documentary extraction-to-generated-tree pilot with date-level verification (no gold link score) |
| `human_adjudication.csv` | Expert-review form with *blank* reviewer label, no fake gold |
| `technical_failures.json`, `extraction_failures.json`, `tree_failures.json` | Explicit technical failures; never convert failures to a model label |
| `ollama_cache.jsonl` | Signed by input, prompt, schema, model and decoding settings; valid outputs only |

### Download all paper results from Linux to Windows PowerShell

```powershell
scp -r gabriel@192.168.20.14:/data/dataRapide/gabriel/DDPF-Health-Risks/outputs/realdata/who_storyforest_final_v7 "$HOME\Downloads\"
```

### First page retrieval if WHO blocks programmatic access

The script intentionally stops rather than substituting invented WHO texts. Inspect which `.html` is missing in `outputs/realdata/who_storyforest_final_v7/sources/`. Download the original WHO HTML on the Linux machine or transfer the original, unchanged pages from a browser there, preserving filenames such as `DON381.html`, then rerun. The file hash remains recorded. You may use `--offline` to explicitly verify only cached pages. **Do not fabricate page content to satisfy an anchor.**

### Sources

- WHO [COVID-19 — China, 12 January 2020](https://www.who.int/emergencies/disease-outbreak-news/item/2020-DON233)
- WHO [Thailand, 14 January 2020](https://www.who.int/emergencies/disease-outbreak-news/item/2020-DON234)
- WHO [Japan, 17 January 2020](https://www.who.int/emergencies/disease-outbreak-news/item/2020-DON237)
- WHO [Republic of Korea, 21 January 2020](https://www.who.int/emergencies/disease-outbreak-news/item/2020-DON238)
- WHO [United Kingdom mpox, 16 May 2022](https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON381)
- WHO [United Kingdom mpox, 18 May 2022](https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON383)
- WHO [Multi-country mpox, 21 May 2022](https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON385)
- WHO [Multi-country mpox update, 29 May 2022](https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON388)
