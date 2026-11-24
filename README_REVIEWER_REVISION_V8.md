# PreDiViD — consolidated reviewer revision suite (V8.1)

**Scientific target:** the first integrated presentation of PropaPhen + the complete Description–Detection–Prediction Framework (DDPF) + evidence-aware Story Forest. PropaPhen is the semantic foundation of the integrated contribution, not an isolated new ontology contribution. This patch does not change the 46-page manuscript, previous data, outputs, or source modules.

## 1. Installation and single execution

Unzip this archive at the root of the existing `DDPF-Health-Risks` checkout on the Linux server. It adds only `scripts/realdata/reviewer_v8/`, the new runner/test, `data/reviewer_v8/` (three frozen previously supplied study summaries), this README and the requirements file. The older V5–V7.1 code archives could not be retrieved in the authoring environment; the patch is a new compatible research extension, not a verified line-by-line modification of those modules. It reads the actual earlier files on the server and records their hashes. Existing outputs remain unchanged.

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
python -m pip install -r requirements-reviewer-v8.txt
mkdir -p logs
set -o pipefail
bash scripts/realdata/run_reviewer_revision_v8.sh --pull-missing 2>&1 | tee logs/reviewer_revision_v8.log
```

Run inside the existing `predivid2` tmux session; keep the Neo4j container active in the other session. The shell activates `.venv`, runs regression tests, checks both input tracks before inference, then runs the entire suite. Missing Ollama checkpoints are downloaded when `--pull-missing` is present. Default generation models: `qwen3:4b`, `qwen3:32b`, `gemma3:4b`, `gemma3:27b`. `embeddinggemma` is used for the documentary cosine baseline. Exact model digests and reported quantization are recorded. Near-size comparison is not a controlled comparison of identical parameter counts. Existing Ollama inference is sequential; avoid running a competing large model in parallel. `--context-tokens 16384` is default. Use `--ollama-url` to override the URL; `.env` is loaded. An optional `--provider litellm --models openrouter/...` accepts LiteLLM model identifiers and existing environment keys; it is not the primary locally matched family comparison, and remote provider behavior is untested. Use a separate output directory for different model lists or remote runs.

Preflight is automatic. To inspect inputs without model calls, use `bash scripts/realdata/run_reviewer_revision_v8.sh --preflight`. If a required input cannot be located or mapped, the run stops before costly inference and records the exact blocker in `automatic_preflight.json`. Do not substitute approximate datasets. The previous summaries are included in `data/reviewer_v8`, but the original data and scripts must exist on the server.

## 2. Experimental matrix

| Reviewer concern | New experiment / artifact | Scope |
|---|---|---|
| Full candidate coverage | All recovered 106 V7.1 candidate considerations, union with the 39 fixed reference pairs | Exact candidate schedule is recovered from JSON/SQLite if possible; otherwise a clearly marked new chronological top-4 schedule (also 106 for 29 cards). No claim that the two schedules are identical. |
| Complete documentary pipeline | Extraction from the original 8 archived WHO reports, validated source quotes and complete dates, followed by candidate generation, classification and single-parent forest insertion for each of 4 models | Separate from curated-card pair classification. Alignment uses unambiguous source/date/type slots; unmatched events/edges remain unadjudicated. |
| Cross-family comparison | Qwen3 4B / 32B and Gemma 3 4B / 27B | Same new prompts, same selected examples, same constraints, fixed temperature. Previous V6 prompts/results are retained separately, not merged as if identical. |
| Semantic ablation | Text-only, typed metadata, actual ontology/KG context, shuffled semantic context (when informative), each with/without post-decision guard | All predictions are reused for their guard variants, avoiding duplicated inference. Neo4j queries are read-only and allowlist static concept relations; no future `isReported`/`hasPresence` edges. If no real context is retrieved, the semantic condition is explicitly unavailable. |
| Semantic specificity | New logistic-regression refits with numeric history, numeric + country identity, numeric + actual CUI profile, and 19 within-disease profile permutations | Exact legacy profile hash and eligibility required. Conditional fixed-split sensitivity analysis, not proof of historical availability. |
| External temporal baselines | Persistence, always-flat, logistic regression, histogram gradient boosting | Same original 1,565 holdout targets, four historical snapshots, predeclared hyperparameters, no fitting on holdout. |
| External linking/forest baselines | Character TF-IDF cosine fixed threshold 0.35; embeddinggemma cosine fixed threshold 0.65; same insertion rule | Similarity-based Story Forest-style proxies, not a reimplementation of original Liu et al. EventX/Story Forest. Thresholds fixed before evaluation, not optimized using silver labels. |
| Metric consistency | Binary macro-F1 on flat/increase for all V6 aggregate metrics, including 8B revision predictions as errors; zero positive F1, undefined precision; per-disease recall; country-block paired intervals and Holm-adjusted p-values | Legacy V6 binary F1 reconstructed from known class supports, prediction marginals, correct counts and recall. No new old-model inference required. |
| DDPF and ontology evidence | Inventory existing detection, embedding, structural, and ontology artifacts; mapping audit, graph retrieval counts, ontology subclass/disjointness checks | No separate isolated PropaPhen contribution or invented OOPS!/FOOPS! result. Explicit ontology class mapping is optional; only verified existing URIs can be used. The guard implements limited explicit disjointness, chronology and evidence-ID checks, not complete OWL reasoning. |
| Human annotation | Two blinded pair reviewers, consensus sheet, reference and generated event review sheets; agreement/kappa, human-reviewed pair and forest scores | Human judgment is mandatory for independent reference validation. Official WHO documents are the source, not an automatic gold standard. |
| Reproducibility | Input hashes, legacy commit/source file hashes, model digests, full candidate predictions, graph JSON, metrics, errors, report, SQLite cache | A separate output directory; validated response cache keyed by model digest, prompt, schema and input. Prior caches cannot safely be reused for changed prompts. |

## 3. Required original inputs and scope

- `outputs/realdata/who_series/weekly/snapshots.csv` must match the frozen V6 SHA-256. The code reconstructs 26,205 eligible origin/target examples (24,640 train, 1,565 holdout) by requiring four consecutive historical snapshots and one consecutive target. It checks the holdout counts and 40 exact V6 IDs. It also checks earlier V5 IDs when recognizable in the original prediction CSV.
- The static profile file is located by the exact frozen V6 SHA-256. If the file cannot be parsed or the 578 eligible holdout examples and 40 selected IDs do not match, the temporal static context experiment is marked unavailable, not silently replaced by a new profile. The numerical and text-only comparisons remain valid. See `temporal/cohort_audit.json`.
- `data/realdata/who_storyforest_v7/silver_events.json` and `silver_pairs.json` (or uniquely discoverable equivalent names). Expected: 29 unique cards, 39 fixed pairs, 22 core positives. The original V7 HTML/text source files must be available in `outputs/realdata/who_storyforest_final_v7/`; the patch records their exact hashes. It does not download current web pages over archived ones. Recovered card excerpts are verified against the archived source text; if no excerpt can be verified the model receives the full archived source. Input budget failures are technical missing predictions.
- `outputs/realdata/who_storyforest_final_v7/final_v7_1/trees/` should contain the previous JSON or SQLite outputs. If the 106 candidate identities cannot be recovered, the replacement schedule is explicitly marked as new. This is a new transparent single-parent forest algorithm; it does not reuse the old `process_event` routine or produce the legacy Studio database format. Inspect graph JSON directly or use your own visualization. Same-event identity links are retained separately from continuations. The latest eligible chronological parent wins; all other accepted parents are counted as alternatives. No fabricated edge counts.
- `PropaPhen.owl` and `BioPropaPhen.rdf`, and an accessible Neo4j graph through the existing `.env` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`). A map from actual event-type vocabulary to ontology class URI may be supplied as `data/reviewer_v8/ontology_event_class_map.json` (a JSON object, each URI must be declared in the loaded ontology). No map is invented here. If exact labels do not align and no real KG context is retrieved, the relevant semantic claims remain blocked. Only static concept relation types in a conservative allowlist are queried.
- Reference cards and original detection annotations are distinct; the original detection study cannot be independently rescored from disease-group labels alone. The inventory identifies potential source/annotation files for author verification. No new human labels are automatically generated.

## 4. Human verification (after the single run)

The package produces blind CSV sheets in `outputs/realdata/reviewer_revision_v8/documentary/human_review/` with blank labels. Have two human reviewers independently complete `reviewer_A.csv` and `reviewer_B.csv`, using the linked WHO source evidence, not model outputs or the previous silver labels. Use one of `same_event`, `documented_continuation`, `related_only`, `unknown`. Positive links require an explicit supporting passage; chronological adjacency and common disease are not sufficient. Fill `consensus.csv` only for disagreements after discussion. Verify the event cards separately in the `event_reviewer_*.csv` files (yes/no/uncertain). The pooled `generated_event_reviewer_*.csv` files are anonymized with respect to generating model and permit further independent assessment of generated mentions. Unknowns remain unresolved. Report reviewer expertise, independence, adjudication procedure, and coverage. The event review summary reports agreement rather than inventing event-level gold labels.

After annotation, rescore existing predictions without LLM calls:

```bash
bash scripts/realdata/run_reviewer_revision_v8.sh --score-reviewed
```

## 5. Results and resume

All new results go to `outputs/realdata/reviewer_revision_v8/`. Main files:

- `run_manifest.json`, `automatic_preflight.json`, `reviewer_revision_report.md`, `model_metadata.json`, `inference_cache.sqlite3`, `llm_errors.jsonl`.
- `temporal/cohort_audit.json`, `legacy_v6_harmonized.csv`, `predictions.csv`, `metrics.json`, `paired_comparisons.json`, `refitted_profile_control.json`, `refitted_profile_permutations.csv`, `llm_profile_permutation.json`.
- `documentary/input_manifest.json`, `candidate_schedule.json`, `pair_predictions.csv`, `pair_metrics.csv`, `forest_metrics.csv`, `forests/*.json`, `semantic_audit.json`, `semantic_control.json`, `retrieved_context.json`, `end_to_end/*/metrics.json`, `human_review/`.
- `ddpf/ddpf_evidence_inventory.json`.

Rerun the **same command** after interruptions. Valid inference responses are cached transactionally. Failed/truncated/invalid JSON is never stored as successful. All numerical baselines may recompute; they are inexpensive relative to inference. Technical failure and abstention are distinct. A nonzero exit with `PARTIAL_OR_PENDING` does not erase finished outputs. Read the manifest to identify blocked components. New model versions, different prompts or changed inputs cannot reuse incompatible native Ollama responses. Change `--output` for a genuinely different protocol. A computationally complete status does not mean independent reference review is complete.

On Windows PowerShell, retrieve the result folder after the run:

```powershell
scp -r gabriel@192.168.20.14:/data/dataRapide/gabriel/DDPF-Health-Risks/outputs/realdata/reviewer_revision_v8 "$HOME\Downloads\"
```

For author review, retain the current 46-page manuscript, both appendices, all 122 references and figures; revise claims only after inspecting the actual measured results. Editorial work (abstract quantitative result, algorithms, final comparative synthesis, annotated example, declarations, references, archival identifier) is not performed automatically by this code. A complete independent review, original detection-reference provenance, comparison against a faithful original Story Forest implementation and generalization beyond the eight-document collection remain scientific/editorial decisions, not automatic guarantees of acceptance.

## 6. Local verification

14 offline regression tests passed in the package authoring environment, including a full mock 29-card/106-candidate documentary run, a blind-reference export, two-class metric harmonization, date handling, no future target in numerical history, guard behavior, one-parent acyclic forest, and model-digest-aware mock Ollama caching. The actual server data, real Gemma inference, actual graph retrieval, and original server module integration have not been executed in this environment. The tests are explicitly synthetic/mocked and are not publication results.
