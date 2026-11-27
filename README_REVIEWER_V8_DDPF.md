# PreDiViD V8 — DDPF-first Story Forest (27 November 2026)

## What changes

This is a non-destructive successor to `PreDiViD_Reviewer_Revision_V8_InputFix.zip`, based on the 17-page `PreDiViD_V8_Protocolo_Experimental_e_Revisao.pdf`. It does not replace the original reviewer_v8 package or its outputs. It copies and adapts the previous native Ollama/cache utilities into a new namespace. **The main inference input is DDPF observations and their concept/relationship subgraphs, not WHO event cards.** WHO/ECDC sources and annotations are evaluation-only. The earlier weekly reported-value study remains an auxiliary task, and the old WHO-to-WHO pilot remains development evidence. No inference cache from those tasks is reused for this new task.

The first executable stage recovers the real historical observations, creates traceable graph facts, reconstructs the Prediction representations and candidate lists, and audits what can be recovered. The historical server log confirmed a 126-record aggregate observation file with `observation_id`, `country_id`, `source`, `top_concepts`, `relation_dates`, and provenance. The new adapter uses that exact structure. This is not a document-level event corpus. The 339 document and 71 paragraph representations and the 271 count must still be reconciled with the original notebooks. They are not fabricated or automatically substituted by the sentence-level records.

## 1. Install and start

Extract the ZIP into the existing DDPF-Health-Risks repository (not into a nested directory). Commit and push/pull as usual. On the Linux server, in the existing tmux session:

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
python -m pip install -r requirements-reviewer-v8-ddpf.txt
mkdir -p logs
set -o pipefail
bash scripts/realdata/run_reviewer_v8_ddpf.sh --preflight --neo4j \
  2>&1 | tee logs/v8_ddpf_preflight.log
```

This runs the 18 regression tests, inventories available notebooks, data, embeddings and clusters, parses the real historical file, optionally verifies matching `hasPresence` edges against Neo4j, and builds the candidate plan. It does not call a generative LLM or download models. Default representation is a regenerated stateless hashed lexical subgraph vector; it is not misrepresented as the original saved matrix. `--hash-large` additionally streams hashes of large inventory files; use this when freezing the final experiment. `--skip-neo4j-audit` skips the edge-identity audit, not the later static context retrieval when `--neo4j` is requested. Neo4j is read-only. Uses the existing `.env` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, optional `NEO4J_DATABASE`). Do not change credentials or reset the database.

Inspect these files before committing to expensive inference:

```
outputs/realdata/reviewer_revision_v8_ddpf/HIST_sentence_pilot/
    inventory.json
    export_audit.json
    dataset_manifest.csv
    coverage.csv
    events.jsonl
    nodes.jsonl
    edges.jsonl
    neo4j_relationship_audit.jsonl   # if requested and successful
    events_with_prediction.jsonl
    embeddings.npy
    embedding_ids.jsonl
    clusters.jsonl
    candidates.jsonl
    eligible_pairs.jsonl
    frozen_config.json
    plan.json
```

The legacy adapter preserves observation IDs and the `top_concepts` intensity/source/month data as aggregate graph facts, with stable fact IDs. The Neo4j audit records the actual matching relationship IDs and their properties, but does not assume aggregate intensity equals any particular individual edge. Raw snippets, individual document IDs and dates of occurrence are not invented. The historical adapter uses collection-end as a **proxy** for availability, never as occurrence time. Inspect missing fact counts, granularity basis, provenance and known occurrence-order counts. The previous confirmed file hash is `738b3b304bd0cb4f2b1dd84c057fce8b02e645197d60d4932e2362aa01efa004`; a different file is recorded as such.

## 2. First timed pilot

After preflight succeeds:

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --pilot --neo4j \
  2>&1 | tee logs/v8_ddpf_pilot.log
```

This is a deterministic sample of up to 100 candidate pairs, with Qwen3 4B and C3 (plus reuse of the same raw results for C3-no-guard), and the numerical/lexical/similarity baselines on the same pair pool. This is an exploratory runtime and integration pilot, not a fully annotated study. It works without installing Gemma if Qwen3 4B is already present. Add `--pull-missing` only if you intend to download models. `--check-models` checks all six configured models before inference. The previous 412 error is an Ollama compatibility issue; Gemma 3 needs at least Ollama 0.6. Verify `ollama --version` and update only after any previous experiments finish. The runner checks actual server version and model manifests before a full generative run. The old V8 WHO job may finish, but its results cannot replace this new integrated benchmark.

Default six checkpoints for full runs:

| Scale | Qwen3 | Gemma 3 |
|---|---|---|
| Small | qwen3:4b | gemma3:4b |
| Medium | qwen3:14b | gemma3:12b |
| Large | qwen3:32b | gemma3:27b |

These are near-scale comparisons, not identical architectures or parameter counts. Actual model digests and metadata are recorded. Native Ollama requests use JSON schemas, temperature zero, seed 42, top-p 1.0, context 16384; Qwen3 thinking is disabled. Generation output budgets adapt on failure; only valid predictions are cached. Latency and token counts are collected from successful calls, separately indicating cache hits. GPU identity/total memory is recorded where `nvidia-smi` is available; peak VRAM, energy and monetary costs are not measured automatically. Inspect `llm_errors.jsonl` for failures. A missing model cannot silently be treated as an abstention.

## 3. Recover real DDPF embeddings and other granularities

The main adapter can use the original DDPF representation if a matrix and its exact row-to-observation mapping are available:

```bash
python scripts/realdata/export_ddpf_vectors_v8.py \
  --matrix /actual/path/to/ddpf_embeddings.npy \
  --ids /actual/path/to/observation_ids.csv \
  --output data/reviewer_v8_ddpf/ddpf_vectors.jsonl

bash scripts/realdata/run_reviewer_v8_ddpf.sh --preflight --neo4j \
  --embedding-jsonl data/reviewer_v8_ddpf/ddpf_vectors.jsonl \
  --out outputs/realdata/reviewer_revision_v8_ddpf/HIST_sentence_original_embeddings
```

The CSV must contain `observation_id` in exact matrix order (or a zero-based `row` column). JSONL is also accepted. No row order is guessed from file names or counts. External vectors must match the observation ID set exactly. Verify that the representation was fit without test/future information before calling it a prefix-safe representation. Alternatively `--embedding ollama --pull-missing` regenerates dense vectors from authorized flattened subgraph labels with `qwen3-embedding:0.6b`, one vector per observation; it is a separate representation configuration. `--config` can select an existing encoder. Default preflight hashing is deliberately fast and has no fitting leakage. Prefix KMeans and TF-IDF are fitted only to the currently available prefix. Original cluster IDs and embedding references supplied in a canonical export are retained as audit fields, but regenerated prefix cluster IDs are used for replay; they are not claimed to reproduce historical full-data clusters exactly.

Other levels can use `--observations /path/to/actual_recovered_records.jsonl --level paragraph` (or `document`) if the exact schema matches and provenance of that level is verified. For other schemas or EXT/EXT-3, export canonical inputs into a separate directory and use `--export-dir PATH --dataset EXT --level sentence`. No original detection or RE notebooks are overwritten or silently regenerated. Inspect the inventory and run the actual historical extraction/Prediction notebooks if the data cannot be recovered. Original corpus acquisition and expanded coverage require accessible sources, source licenses and original extraction settings; the protocol's 12-month and third-disease targets are not already implemented datasets.

### Canonical export contract

`events.jsonl`: one observation per record, including

- `observation_id`, `country_id`, `country_label`, `collection_start`, `collection_end`, `available_at`, `availability_basis` (`verified_publication` only with actual provenance), `occurrence_start`, `occurrence_end` (both nullable), `modality`, `RE_level`, `source_ids`, `source_corpus`, `source_text` (authorized only), `extraction_config_hash`, `subgraph_ref`, `fact_ids` (nonempty list), `provenance_status`.
- Optional `canonical_event_id`, `identity_basis`, `embedding_ref`, `cluster_id`. Singleton is the safe default; identity proposals are exported separately. The implementation does not automatically merge aggregate or ambiguously matched events. Source-based identity resolution beyond this conservative policy needs independently traceable evidence and remains a research task.

`nodes.jsonl`: `node_id`, `label`, `type`, `origin`; optional explicit `class_uri`.

`edges.jsonl`: `fact_id`, `subject`, `predicate`, `object`, `properties` (dictionary), `provenance` (dictionary), `kind` (`observed`, `extracted`, `observed_aggregate`, or explicitly identified formal fact), `available_at` (ISO date). All endpoints must exist. All event fact IDs must exist and be available by that event's availability date. The legacy aggregate fact ID is a deterministic derived ID, not an original Neo4j relationship ID. Country/month observation timestamps and occurrence timestamps are deliberately distinct.

Static context is retrieved only through an explicit allowlist of relation types. No `isReported` or WHO-specific facts are queried for semantic context. PropaPhen/BioPropaPhen RDF/XML files are discovered if unique, or supplied via repeated `--ontology PATH`. Named-class transitive subsumption is materialized; explicit ontology domain/range and class disjointness can invalidate a cited mapped fact. For a known concept/class mapping, supply `data/reviewer_v8_ddpf/ontology_node_map.json` mapping exact node IDs to ontology class URIs. The mapping must be justified independently; no fuzzy or invented match. Check `semantic_audit.json` for actual coverage; empty inference is not a successful formal-reasoning experiment. This is a limited auditable rule implementation, not a complete OWL reasoner, temporal geographic inference engine or SHACL suite.

## 4. Full experimental run — only once prerequisites pass

`--full` executes all six models × six principal conditions (C0–C5), repeats C5 for three fixed seeds, uses same raw outputs for C2-guard and C3-no-guard, and adds a C3-no-inference variant on the first model. Fixed-pool and natural-retrieval baseline runs are separate. Default K=10, window=730 days and threshold=.65 are **proposed defaults**, not fitted on test. The pilot or a separately exported development split must be used to select them before testing. `--top-k`, `--window-days`, `--threshold`, `--retrieval-mode`, `--all-pairs` and custom configuration enable retrieval/context sensitivity studies without changing a frozen run. Choose a new `--out` for each changed configuration; do not tune on test. Channel retrieval recall includes no-embedding, no-cluster, no-time and no-geography variants. Full additional ablations require separate explicit runs; the shell does not silently execute an enormous grid. `--all-pairs` uses the full eligible window, not all mathematical pairs without any window restriction.

```bash
# Prepare a dedicated full-run directory, with the verified export and frozen representations.
RUN=outputs/realdata/reviewer_revision_v8_ddpf/HIST_sentence_full
bash scripts/realdata/run_reviewer_v8_ddpf.sh --preflight --neo4j --out "$RUN"

# Only after provenance, splits, annotation protocol, and development choices are verified:
bash scripts/realdata/run_reviewer_v8_ddpf.sh --full --neo4j --pull-missing \
  --approval /actual/path/to/approval.json --out "$RUN" \
  2>&1 | tee logs/v8_ddpf_full.log
```

Use the same export/embedding/config arguments for both commands. The approval template is in `configs/reviewer_v8_ddpf/approval.template.json`. It binds the prepared export and plan hashes, config hash, and exact static context/violations hashes; fill those from generated files only after inspecting them. The first approval attempt can stop after generating the static context so it can be verified and its hashes approved before any generative model calls. The gate is an explicit human attestation, **not an automatic certificate of independence**. It refuses to certify legacy aggregate-only inputs with missing source IDs/text or proxy dates. EXT requires attested coverage and separate exports. Development and test splits must be prepared and frozen by the researcher; this patch does not create an automatically valid gold split or tune on the test set. For the existing historical data, exploratory full relation-only evaluation can be explicitly requested with `--full --exploratory --allow-no-ordered-pairs`, but when occurrence dates are unknown there may be no useful directed forest, so do not spend the full inference budget before recovering chronology/evidence. This command is not recommended as the first action.

Candidate retrieval uses similarity, time, country equality, concept overlap and prefix cluster membership; cross-country and cross-cluster candidates remain eligible. The hybrid selects top-K from a union of per-channel lists. True occurrence order controls accepted continuation direction, including when a later-published observation concerns an earlier occurrence. Availability controls what may enter each checkpoint. Related-only and same-event proposals do not create continuation edges. Maximum-weight branching yields a forest with optional roots, a DAG retains all thresholded chronological edges, and raw graph outputs preserve rejected continuations. The guard checks cited fact IDs, evidence from both observations, occurrence ordering and available explicit ontology contradictions. It cannot certify textual entailment or causal transmission: independent evidence support review remains necessary.

### Output and resume

The new output tree is `outputs/realdata/reviewer_revision_v8_ddpf/`, not the original `reviewer_revision_v8/`. Each run writes an SQLite cache, `run_manifest.json`, `run_report.md`, `model_run_progress.json`, `semantic_audit.json`, `static_context.jsonl`, and per-variant raw/validated decisions, graph/forest, replay checkpoints and evidence audit. A repeated identical command reuses validated predictions, frozen input files, model digests, code/config, and semantic context. New model digests, altered code or configuration require a new output directory. Successful cache keys bind model/digest, prompt, schema, payload, decoding settings, backend and budget. A model failure is logged and a missing decision stays missing. Partial stages can be rerun. The progress manifest is not a completed evaluation.

## 5. Independent reference construction and evaluation

Do not put the reference corpus in any DDPF source or embedding directory. The old WHO pilot must be development only if it influenced prompts, rules or model decisions. Create independent official reference event records before inspecting predictions. The utility `fetch_ddpf_reference_v8.py` optionally downloads a researcher-selected URL list to a gold-only directory, recording raw and text hashes. Input CSV columns: `source_id,url,publication_date,partition,license`. Supply `--dev-manifest` to reject exact source ID/URL/text overlap with the earlier pilot. Metadata and document selection are human-reviewed, not automatically verified. Example:

```bash
python scripts/realdata/fetch_ddpf_reference_v8.py \
  --urls /actual/path/to/official_reference_urls.csv \
  --gold-dir data/reviewer_v8_ddpf/gold/HIST_sentence_sources \
  --dev-manifest /actual/path/to/v7_development_reference_manifest.csv
```

The utility is optional and its live downloads have not been tested here. For PDFs, `pdftotext` must be installed. Existing text files may instead be listed in a CSV with `source_id,url,publication_date,text_path,sha256`; relative paths resolve from the manifest directory. Run the separate leakage audit:

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --audit-independence \
  --reference-manifest data/reviewer_v8_ddpf/gold/HIST_sentence_sources/reference_manifest.csv
```

This checks exact normalized text/source matches and 5-token shingle overlaps where actual input source text exists. It reports missing texts and cannot prove that the historical KG or static resources contain no copied facts. Human inspection of source/derived fact provenance and rebuilding affected inputs are still required.

Create independent annotation sheets before model results are examined:

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --review --review-pairs 400
```

The tool samples independently of retrieval scores, across time/country/modality/source strata, including pairs outside the retrieved list. Each annotator gets a blank copy. Populate the reference event inventory independently from the official documents, map DDPF IDs to reference events (including partial/ambiguous/unmappable), and annotate pair relations and evidence. Include all independently identified positives in the intended universe. For a real benchmark use two independent annotators and a third adjudicator. No LLM-derived gold labels are automatically filled. `insufficient_evidence` remains separate from negative `unrelated`. Fill `block_id` with independent trajectory/country-component groups for statistics, not one block per pair. Add evidence IDs, justification, and certainty. Review samples are not exhaustive and raw stratum proportions are not population estimates. The patch records strata sizes; weighted population estimation and power analysis require final annotation design. Run the sheets separately for each granularity and partition.

After prediction freeze and adjudication:

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --score --out "$RUN" \
  --gold-dir data/reviewer_v8_ddpf/gold/HIST_sentence
```

Scores are computed for judged pairs, plus retrieved-pair-only metrics, missing decisions, abstentions, per-class results, evidence ID/guard audit, kappa and raw agreement, Recall@K/MRR and per-channel retrieval ablations, judged-edge metrics with bounds for unknown edges, and conditional block bootstrap/accuracy sign-flip Holm correction for C3 vs C2. The pair sample does not justify a complete forest F1. For a declared exhaustive reference universe, provide `reference_events.csv`, `reference_edges.csv`, `mapping.csv` (rename populated templates), and `complete_universe.json` with `exhaustive_adjudicated:true`, a `universe_id`, and the scoped `observation_ids`. The scorer computes direct, ancestor and co-story F1 under exact unambiguous mapping and penalizes unmatched prediction edges. Partial/many-to-many identity resolution, true end-to-end detection recall beyond recovered inputs, full gold-graph temporal replay scoring, block CIs on exhaustive edge-F1 and independent semantic support error annotation remain additional tasks. The results are not silently called complete when only judged pair scores exist.

## 6. Protocol mapping and open tasks

| PDF section | Implemented in patch | Still requires evidence/work |
|---|---|---|
| pp.3–5 historical/technical integration | Inventory, exact legacy schema, canonical export, optional Neo4j edge audit, ID-keyed embeddings, prefix clusters, candidate plans | Actual 339/71 recovery, 271 reconciliation, original raw corpus IDs/text, validated availability and occurrence dates, original matrix/cluster lineage |
| pp.4,8–9 data/reference | Canonical EXT interface, coverage counts, reference downloader, overlap audit, blinded stratified sampling, mapping/reference templates, frozen approval | Actual EXT/EXT-3 corpora, source licensing, independent complete benchmark, adjudication, frozen trajectory/time/geographic splits and power analysis |
| pp.6–7 retrieval/forest | Hybrid and individual retrieval rankings, C0–C5, opaque aliases, shuffled controls, named-class subsumption, explicit disjointness checks, forest/DAG/raw links, availability-prefix outputs | Adequate ontology mappings/actual graph relation coverage, factual entailment review, independently supported event identity resolution; detailed spatial reasoning beyond country equality |
| pp.10–11 comparisons | B0/B1/B2/B3/B4/B5-adapted/B6, timeline; fixed and natural retrieval; 6 checkpoints, controls and separate configurable sensitivity runs | Official original Story Forest implementation is not bundled; B5 is explicitly a similarity-insertion proxy. Further input-selection and relation-attribute ablations, dev-only tuning and actual six-model runs |
| pp.12–13 evaluation | Pair/edge metrics, recall@K/MRR, kappa, graph metrics under exhaustive declaration, conditional block resampling and Holm accuracy tests, inference duration/tokens | Full primary EXT end-to-end confidence interval, actual human error/support analysis, non-inferiority and calibration, peak VRAM/energy, transfer/replay gold metrics and paper-ready figures |

This patch supplies an executable first integrated pipeline and the infrastructure for the new protocol. It does **not** assert all RQ1–RQ6 or G0–G7 are already satisfied. The required missing data and benchmark cannot be synthesized from the historical WHO cards. The first action is the preflight; use its actual outputs to decide what to recover before committing to the complete matrix.

## Local verification

18 unit/integration regression tests passed, including original schema parsing, no future fact access, explicit reference marker exclusion, evidence validation, prefix retrieval, candidate/forest invariants, shuffle, alias, ontology closure, blind-review and scoring. An additional command-line smoke test used a local mock native Ollama HTTP API: preflight passed, six fixture predictions were cached, repeat made zero new calls, and the no-chronology full-run gate blocked inference. See `TEST_RESULTS.txt` and `CLI_SMOKE_TEST.txt`. No real server Neo4j query, Gemma/Qwen inference, corpus download or gold annotation was executed here. The real server logs and results are authoritative.
