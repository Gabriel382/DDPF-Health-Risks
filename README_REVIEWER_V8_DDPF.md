# PreDiViD — V8.2 DDPF-first final integration patch

This patch supersedes the earlier DDPF-first V8 integration patch, without modifying previous result directories. It implements the **retrospective evolution of DDPF observations**, not the extraction of incident cards from WHO reports. The experimental protocol is the 27 November 2026 PDF. WHO/ECDC are evaluation-only. The earlier WHO-to-WHO and weekly time-series experiments are not part of this main pipeline.

## Quick start on the existing Linux server

Extract the ZIP in the repository root, not in a subdirectory. Commit/push/pull if appropriate. In the existing tmux session:

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
python -m pip install -r requirements-reviewer-v8-ddpf.txt
mkdir -p logs
set -o pipefail
bash scripts/realdata/run_reviewer_v8_ddpf_final.sh --pull-missing \
  2>&1 | tee logs/v8_ddpf_final.log
```

This runs tests, read-only Neo4j recovery/audit, the main embedding and candidate plan, checks all six generative checkpoints, creates independent blank annotation sheets, runs the six-model main experiment, then runs the binary BoW and Qwen3-Embedding retrieval/forest baseline comparisons. The Node2Vec comparison is run if its exact ID-keyed archived vector export exists. `--main-only` skips the auxiliary representation comparisons. `--preflight-only` stops after preparation/model checks and blank annotation generation. It can generate embedding vectors but does not call generative LLMs. `--pull-missing` authorizes downloads; omit it if the models are already present.

**Computational cost:** the main plan prints the actual number of pairs and maximum nominal calls (six checkpoints times eight variants with three C5 seeds, plus the designated no-inference control). Retries may increase actual API requests; cache hits and ineligible shuffle pairs reduce them. Do not equate this with the old 55,200-call estimate, because the candidate policy changed. Use the smaller pilot command below if you want to time the run. The GPU warning after Ollama installation warrants checking `nvidia-smi` and `ollama ps`; CPU inference of large models can be very slow. No device is forced by this patch.

### Optional timed pilot

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --pilot --neo4j --pull-missing \
  2>&1 | tee logs/v8_ddpf_pilot_final.log
```

The pilot is up to 100 selected pairs and Qwen3 4B C3, with baselines. Its outputs are separate; the one-shot full runner does not depend on it. The main complete computation is explicitly marked exploratory historical observation analysis pending an independently adjudicated reference. It is not a claim that the full PDF protocol, including EXT, has been fulfilled.

## 1. Recover existing DDPF observations, not raw documents

The known legacy file `outputs/realdata/reconstructed_observations.jsonl` has 126 country-by-source aggregate observations, previously associated with the sentence-level extraction experiment. The previously verified hash is `738b3b304bd0cb4f2b1dd84c057fce8b02e645197d60d4932e2362aa01efa004`. The adapter preserves their IDs and the previously selected top concepts. It reuses the known historical Neo4j `hasPresence` query and six thresholds from the 16 November reconstruction manifest:

| source key | historical source corpus | threshold |
|---|---|---:|
| Journal_COVID | Aylien | 5 |
| Medical_COVID | CORD-19 | 8 |
| Social_COVID | Reddit COVID | 10 |
| Journal_Monkeypox | BBC News | 5 |
| Medical_Monkeypox | PubMed | 3 |
| Social_Monkeypox | MonkeyPox2022 | 10 |

The exact previous server run recovered 9,822 threshold-filtered graph rows and 126 observations. The new runner re-queries the graph and records the actual counts, source-country groups and relationship property schema in `neo4j_recovery_inventory.json`; it saves the rows in `neo4j_notebook_threshold_rows.jsonl`. It checks all selected `hasPresence` relationships against Neo4j, preserving the original graph relationship IDs (local to this restored database), periods, individual intensities, aggregate intensity and observation ID. It fails on unmatched selected facts rather than silently making up evidence. The graph connection uses the existing `.env` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, optional `NEO4J_DATABASE`). All queries are read-only.

The date on the arcs is the observation collection period, the central country node supplies geographic identity. Raw documents, individual document IDs and individual incident-occurrence dates are not required to construct this observation-level forest. We do not invent them. Source-level provenance is preserved; exact document-level leakage/near-duplicate exclusion cannot be proven from source-level provenance alone.

**The 339 document-level and 71 paragraph-level observation sets:** the graph inventory checks whether extraction-level attributes are present, but counts alone cannot recover their membership. If `hasPresence` stores only source/date/intensity and all levels were aggregated into the same relationships, the exact extraction levels cannot be separated from that graph alone. Use the original notebook parameters or a recovered level-specific export. The 126 observations are not relabeled or copied to fake the other two sets. The code accepts both additional levels and any extended/external canonical export; it does not claim that those datasets exist.

For an actual separately recovered JSONL:

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --preflight --neo4j \
  --level paragraph --observations /actual/path/paragraph_observations.jsonl \
  --out outputs/realdata/reviewer_revision_v8_ddpf_final/HIST_paragraph_full
bash scripts/realdata/run_reviewer_v8_ddpf.sh --full --exploratory --neo4j \
  --level paragraph --observations /actual/path/paragraph_observations.jsonl \
  --out outputs/realdata/reviewer_revision_v8_ddpf_final/HIST_paragraph_full
```

Use the same arguments for document level with `--level document`. Supply only an independently verified export for that level. For different schema use `--export-dir` (see contract below). Actual new EXT/EXT-3 corpora must be acquired, licensed and exported separately.

## 2. The DDPF Prediction representation

The previous clustering comparison on 126 observations reported EmbeddingGemma with concepts + country + time as the strongest unmasked configuration (ARI approximately .817, macro-F1 .952); this is a clustering result, not a Story Forest result. The default is therefore `ddpf_best`, never a silent hashing BoW fallback.

The runner first checks explicitly named ID-keyed original exports:

* `data/reviewer_v8_ddpf/sentence_embeddinggemma_country_time_vectors.jsonl`
* `outputs/realdata/sentence_embeddinggemma_country_time_vectors.jsonl`

For other levels substitute the level in the filename. Each row has `observation_id`, `vector`, optional `method`, and optionally `cluster_id`. All observation IDs must match exactly; no inferred matrix row order. To use a historical matrix with an explicit row-to-ID mapping:

```bash
python scripts/realdata/export_ddpf_vectors_v8.py \
  --matrix /actual/path/embeddinggemma_matrix.npy \
  --ids /actual/path/observation_ids.csv \
  --output data/reviewer_v8_ddpf/sentence_embeddinggemma_country_time_vectors.jsonl
```

The CSV has `observation_id` in exact matrix order (or a zero-based `row` column). Use `--embedding-jsonl PATH` to select another exact vector export. `inventory.json` lists candidate artifacts and notebooks, including the previous November 16 experiment-suite directory if present. No arbitrary matrix is selected solely because its dimension/count matches. If there is no original keyed export, the runner **regenerates dense EmbeddingGemma representations of the DDPF concept labels, country and observation interval** using the local encoder. Its exact serialized template, model digest and hash are recorded; it is not claimed to be the exact original matrix or to reproduce the earlier .952 score. The main code uses L2 normalization and prefix KMeans, k=2 by default, to avoid fitting clusters on future observations. Original cluster IDs in canonical exports remain separate lineage fields; newly computed clusters are not claimed to be the old assignments. Verify original embeddings were obtained without future data before claiming strict replay with archived vectors.

Alternative encoders: `--embedding binary_bow` uses unit-normalized binary concept presence; `--embedding ollama --embedding-model qwen3-embedding:4b` uses the same concepts + country + time input as EmbeddingGemma. `--embedding hashing_bow` is an explicit old lexical control only. `--embedding-jsonl` supports archived Node2Vec or other DDPF vectors. For the automatic Node2Vec baseline, provide `data/reviewer_v8_ddpf/sentence_node2vec_vectors.jsonl` using the export utility. The auxiliary runs in the one-shot wrapper compare retrieval and baseline forests, not a full six-LLM factorial over all encoders. To run a C3 LLM comparison on the alternate encoder, choose a new output directory:

```bash
bash scripts/realdata/run_reviewer_v8_ddpf.sh --full --exploratory --neo4j \
  --conditions C3 --embedding binary_bow \
  --out outputs/realdata/reviewer_revision_v8_ddpf_final/HIST_sentence_BoW_C3
```

For other encoders substitute their arguments, keeping the six checkpoints, same observations, and independent reference. Retrieval ranks may differ, so compare common judged pairs separately from end-to-end retrieved results. The historical Node2Vec branch has a different training/projection pipeline; it is not claimed to be exactly regenerated here.

## 3. Story construction and models

* The default `--chronology collection` uses strict non-overlapping arc-date collection windows to order observations. Overlap/equal period does not imply direction; these pairs can still be `related_only`, `same_event`, `unrelated` or `insufficient_evidence`. The optional `--chronology occurrence` is a distinct individual incident task and needs actual occurrence dates. The code never copies collection dates into the occurrence fields.
* `available_at` is the recorded availability date, or the end of the observation collection period when publication dates are unavailable. Replay using proxy dates is retrospective window replay, not real-time operational surveillance.
* Prefix retrieval uses cosine, concept overlap, location, temporal distance, cluster membership and TF-IDF. To prevent within-month pairs from crowding out predecessors, each arrival has separate ranked pools: up to ten ordered predecessor proposals and three overlapping/unknown proposals. Counts and ranks are exported. `--top-k`, `--window-days`, `--threshold`, `--retrieval-mode`, `--all-pairs` and custom configuration enable sensitivity runs. Default window 730 days and threshold .65 are proposed settings, not tuned outcomes. Use a distinct output directory for changed parameters and tune only on development data.
* Relation output: `same_event`, `continuation`, `related_only`, `unrelated`, `insufficient_evidence`, score, evidence IDs, rationale and subtype. Continuation refers to evolution of the observations across collection windows; it does not establish transmission or individual incident chronology. Positive links need admissible evidence IDs; independently judge actual semantic support. No automatically forced links. Identity proposals are recorded, not blindly merged.
* The guard checks chronology, evidence IDs, both-side evidence, and available explicit ontology contradictions. Static context is retrieved from an allowlist; limited named-class subsumption is materialized when ontology files/mappings exist. Verify the actual semantic audit. It is not a complete OWL reasoner or human entailment verification.
* The outputs are raw graph, DAG, maximum-weight forest (at most one parent, no cycles, optional roots), and incremental replay. The static snapshot and original aggregate data impose real availability limits.

Six generative models: `qwen3:4b`, `qwen3:14b`, `qwen3:32b`, `gemma3:4b`, `gemma3:12b`, `gemma3:27b`. Ollama server version, tags and digests are recorded before generative inference. `embeddinggemma:latest` is accepted for the short model name `embeddinggemma`. The separate generative model checks prevent the previous missing-Gemma failure after hundreds of calls. Native Ollama JSON schema, temperature zero, seed 42, context 16384, Qwen thinking disabled, model/input/prompt/config keyed SQLite cache. Resume the same command after interruption; do not delete the cache. Technical failures are reported as missing decisions, not negative relations.

C0 metadata; C1 flattened labels/text; C2 structured triples; C3 KG context + available inference + guard; C4 anonymized C3 with shifted dates; C5 shuffled semantic context with three seeds. C2 guard and C3 no-guard reuse raw outputs; designated C3 no-inference is a further control. Baselines include no links, temporal/geographic, TF-IDF, Jaccard, DDPF cosine, hybrid, similarity-insertion Story Forest adaptation (not official reproduction), conservative rules, and timeline. Natural retrieval and fixed-pool evaluations remain separate.

## 4. Independent evaluation

The one-shot runner creates blank pair-annotation sheets (400 sampled pairs by default) and observation cards including their fact IDs, concept labels, country, period and corpus, before generative inference. These are stored in `data/reviewer_v8_ddpf/gold_final/HIST_sentence`. The old WHO event cards are not the gold for this experiment. Independently collect official reference reports, construct the reference inventory, map observations to reference events/periods (allow partial and ambiguous correspondence), and label relations between DDPF observations. Two independent annotators and adjudication are recommended; no LLM-generated reference labels are inserted. Include independently identified positives outside the retrieved candidates. Gold labels must not enter the inference inputs. Unknown evidence is not a negative label. Pair labels must refer to **observation-level evolution** under the same chronology basis; do not call a same-month cross-country relation a directed epidemiological transmission link.

The reference acquisition helper takes a researcher-selected URL manifest:

```bash
python scripts/realdata/fetch_ddpf_reference_v8.py \
  --urls /actual/path/official_reference_urls.csv \
  --gold-dir data/reviewer_v8_ddpf/gold_final/HIST_sentence_sources
```

Columns: `source_id,url,publication_date,partition,license`. Add `--dev-manifest` to exclude exact old development-source overlaps. Real downloads and coverage depend on the source. `--audit-independence --reference-manifest PATH` checks overlap where original input text exists; the historical aggregates lack document-level text, so this cannot prove independence. Keep WHO facts/derived features out of the model, and audit static KG context and source lineage independently.

After annotations and adjudication, run scoring (no new generative inference):

```bash
RUN=outputs/realdata/reviewer_revision_v8_ddpf_final/HIST_sentence_full
GOLD=data/reviewer_v8_ddpf/gold_final/HIST_sentence
bash scripts/realdata/run_reviewer_v8_ddpf.sh --score --out "$RUN" --gold-dir "$GOLD"
```

Repeat with alternate representation output dirs using the same gold. Metrics include judged-pair macro-F1, per-class results, decision coverage and missing/abstentions, candidate Recall@K/MRR (within the retrieval strata), channel ablations, judged-edge metrics with bounds for unjudged predictions, agreement/kappa, conditional paired block bootstrap and Holm-adjusted sign-flip accuracy contrasts. Provide `block_id` for independent trajectory groups. Complete-graph metrics require genuinely exhaustive scoped reference `reference_events.csv`, `reference_edges.csv`, `mapping.csv` and `complete_universe.json` with `exhaustive_adjudicated:true`, `universe_id`, `observation_ids`. Exact unambiguous mapping is implemented; partial matches stay unresolved. No automatic document-level detection recall is claimed. The main result is not global end-to-end F1 unless the exhaustive benchmark permits it.

For a formal frozen confirmatory run use a separate output directory, `--full --approval PATH`, and `configs/reviewer_v8_ddpf/approval.template.json` after independently verifying sources, static facts, embedding lineage, development settings and splits. The gate binds actual export, configuration, candidate plan and static context hashes. For aggregate historical inputs acknowledge limited provenance explicitly, with `aggregate_provenance_limitations_acknowledged:true`, only if the other attestation conditions are genuinely satisfied. It labels the resulting scope `aggregate_observation_study_attested_limited_provenance`, not a document-level independence certificate. Do not mark checklist items true just to bypass the gate. EXT requires actual coverage verification and a distinct export.

### Canonical export contract for additional granularities / EXT

`events.jsonl`: `observation_id`, `country_id`, `country_label`, `collection_start`, `collection_end`, `available_at`, `availability_basis`, nullable `occurrence_start`, `occurrence_end`, `modality`, `RE_level`, `source_ids`, `source_corpus`, optional authorized `source_text`, `extraction_config_hash`, `subgraph_ref`, nonempty `fact_ids`, `provenance_status`; optional `canonical_event_id`, `identity_basis`, `embedding_ref`, `cluster_id`. Dates ISO `YYYY-MM-DD`; the legacy adapter expands `YYYY-MM` into month bounds. `nodes.jsonl`: `node_id`, `label`, `type`, `origin`, optional `class_uri`. `edges.jsonl`: `fact_id`, `subject`, `predicate`, `object`, `properties`, `provenance`, `kind`, `available_at`. Use independently verified ID mapping. Do not put gold in this directory. One extraction level per run.

## 5. Output and verification

New results only in `outputs/realdata/reviewer_revision_v8_ddpf_final/` (or explicitly chosen `--out`). Previous V8 outputs and caches are not overwritten. The main `HIST_sentence_full` directory contains inventory/export audits, source and coverage manifests, graph rows, events/nodes/edges, embedding matrix and ID map, prefix clusters, candidate plans, per-variant raw/validated decisions, graphs/forests/replay, evidence audit, SQLite cache, model progress, run manifest and report; evaluation outputs appear after scoring. Changed code/config/model digests or input hashes require a new directory. Cache signatures include the new task, so old WHO predictions cannot be accidentally reused. Do not reuse the previous directory.

Local verification: 23 regression tests passed, including original schema, graph rehydration, observation chronology, predecessor quota, guard/forest, binary BoW, EmbeddingGemma view, prefix access, shuffle, blinded review and scoring. A mocked native Ollama HTTP CLI smoke test passed preflight, `:latest` alias, cache resume, collection full run, strict occurrence gate and baseline-only resume. Actual Neo4j queries, original-vector availability, six-model inference and external benchmark annotation **have not been executed in this environment**. The server logs determine their success. The full protocol's EXT/EXT-3 acquisition, original document/paragraph reconciliation, official Story Forest reproduction, complete event identity resolution, real-time prospective evaluation, full primary end-to-end confidence intervals, peak VRAM/energy and error adjudication remain dependent on actual data or further work.
