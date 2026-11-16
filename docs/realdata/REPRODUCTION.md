# PreDiViD: replay Prediction on the original COVID-19/Mpox data

**Scope:** A repeatable starting point for the original retrospective Prediction experiment and a *separate*, evidence-preserving preparation path for Story Forest. This package is a drop-in overlay, **not** a new implementation of the entire DDPF. Copy its contents to the repository root. It never replaces the source notebooks or the source ontologies. The 2.2 GB Neo4j dump is not included.

## 0. Evidence from the sources

Zenodo `10214515` (2023-11-28, v1.0.0) contains `PropaPhen.owl` (MD5 `1283bbe91596384af58e17395db200bd`) and the OnToology bundle.

Zenodo `10987743` (2024-04-17, v1.0.0) contains:

- `BioPropaPhen.rdf`, MD5 `1d211f9532efe1e6633dd33860abe08c`.
- `BioPropaPhenKG-COVIDvsMPOX.dump`, MD5 `4aced6b0b71bda7f301b81a60d84d307`, about 2.2 GB, created using **Neo4j 4.4.3**.

The record explicitly describes the WHO ground truth and extra `isReported` edges between CUI concepts and Country entities. These are not by themselves a per-document, evidence-linked story dataset.

The copy of `data/ontologies/PropaPhen/PropaPhen.owl` inside the supplied DDPF repository snapshot has MD5 `c2c7149926a27e79c986288fdfde8a58` and therefore is **not byte-identical** to the Zenodo `10214515` file. Preserve and document both versions; don't overwrite.

## 1. Download source data with checksums

From repository root:

```bash
python scripts/realdata/zenodo_fetch.py
# Downloads only PropaPhen.owl and BioPropaPhen.rdf (~2.7 MB total)

python scripts/realdata/zenodo_fetch.py --include-dump
# Optional, explicitly downloads the ~2.2 GB dump as well

python scripts/realdata/zenodo_fetch.py --check --include-dump
# Verify all expected checksums without network traffic
```

Expected:

```
data/zenodo/10214515/PropaPhen.owl
data/zenodo/10987743/BioPropaPhen.rdf
data/zenodo/10987743/BioPropaPhenKG-COVIDvsMPOX.dump
```

The download script resumes partial transfers when the server supports HTTP Range and checks MD5 before finalizing; it will not overwrite mismatched existing files. Record filenames and source versions in experiments.

## 2. Restore graph *in isolation*

Do **not** load this archive over your existing production Neo4j database. Work from a new instance with sufficient disk and memory, separate credentials, and a backup if needed.

The archive was created with Neo4j 4.4.3. Neo4j's official migration path from 4.4 to newer generations passes through **Neo4j 5.26 LTS**. For first inspection, the simplest compatibility strategy is a dedicated 4.4 instance; then plan an upgrade for a maintained instance as a separate task.

On **Neo4j 4.4**, the `neo4j-admin load` command supports `--from=<archive-path>` with Neo4j offline. Typical form (only if the dump's internal database name matches `neo4j`):

```bash
neo4j stop
neo4j-admin load --database=neo4j --from=/absolute/path/BioPropaPhenKG-COVIDvsMPOX.dump
neo4j start
```

**Check the output of `neo4j-admin load --help` and your edition/version before issuing the command.** This example must not be used to overwrite an important database. On a newer Neo4j version the command syntax and migration requirements differ.

For newer versions, see:
- https://neo4j.com/docs/upgrade-migration-guide/current/version-5/migration/
- https://neo4j.com/docs/upgrade-migration-guide/current/version-5/migration/migrate-databases/
- https://neo4j.com/docs/operations-manual/current/backup-restore/restore-dump/

Set `.env` locally (never commit credentials):

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j
```

Run the **read-only** schema/provenance audit:

```bash
python scripts/realdata/kg_provenance_audit.py --limit 20
```

It inspects graph labels, relation types and bounded samples of the historical WHO reference `isReported` links for `C5203670` (COVID) and `C0276180` (Mpox). No graph modification or event creation.

## 3. Replay the Prediction notebook in its original order

These are the source notebooks in the supplied repository snapshot:

**A. First, the historical *Observation* branch**
`Experiments/Prediction/Prediction Module - COVID vs Monkeypox Observation.ipynb`

- Cells 10–11: configure paths including **six input CSVs** (`covid_observations_{journal,medical,social}.csv` and `monkeypox_observations_{journal,medical,social}.csv`).
- Cells 14–15: load observations and prefix sources.
- Cells 16–20: parse lists, keep 25 concepts per observation, concatenate, create the observation vocabulary.
- Cells 22–27: generate **binary BoW** vectors from concept names.
- Cells 30–39: **K-Means k=2, random_state=9**, compare with disease labels. Cluster labels need a permutation-aware comparison.
- Cells 40–66: read **WHO `isReported`** from Neo4j to filter / compare spatiotemporal reference observations. The notebook contains hard-coded credentials and assumptions; configure them safely in your working copy.

**B. Then, full comparison of embeddings and spatiotemporal ablations**
`Experiments/Prediction/Prediction Module - Full Benchmark.ipynb`

- Cells 12–16: paths for gazetteers, observations and precomputed embedding tables.
- Cells 18–20: instantiate `PredictionModule`, all embedding methods and K-Means.
- Cells 22, 25, 28, 31: run the four combinations of spatial/temporal settings, with the Wang/K-Medoids benchmark.
- Cells 35–42: WHO-filtered comparisons. **Cell 36 issues Cypher `MERGE` writes. Do not run it on the published dump without a separate, explicitly recorded derived copy.**

**C. Optional later branches**
`Prediction Module - AIME 2025.ipynb`, `LLM Based Prediction Module.ipynb`. These need additional generated similarity/embedding matrices, which are NOT inside the ontology/dump files.

Check paths **before opening the notebooks**:

```bash
python scripts/realdata/notebook_preflight.py --mode all
python scripts/realdata/replay_bow.py --check
```

Crucially, the two Zenodo records above are **not** advertised as containing the six source-specific CSVs, the original documents, the gazetteers or all embedding tables. They may have to be recovered from local historical outputs or regenerated via Detection. The audit reports missing files rather than silently manufacturing them.

### A runnable, deterministic replay of the first BoW branch

After the six source CSVs exist under `data/neo4j`:

```bash
python -m pip install numpy scipy scikit-learn
python scripts/realdata/replay_bow.py --data-dir data/neo4j --outdir outputs/realdata/bow_kmeans
```

The script follows the source notebook's concept-selection and BoW/K-Means sequence, outputs a row-level assignment table, weighted F1 with cluster-label permutation, accuracy, ARI/AMI, SHA-256 input hashes and a parameter manifest. It fails instead of quietly losing rows if the original notebook's `System_Name` dictionary keys are duplicated.

**It does not reproduce the entire full-benchmark matrix.** It is the first faithful checkpoint; the full experiment requires its separate inputs.

## 4. Transfer to Story Forest only when real source evidence is available

Do not use graph `isReported` edges as `documented_continuation` ground truth. The graph can supply entity, place, date and WHO comparison evidence, but *story continuation* requires independent document-level support.

When a source CSV has actual document IDs, text, source IDs, occurrence dates and publication dates (not merely aggregate presence edges), convert it explicitly:

```bash
python scripts/realdata/documents_to_events.py \
  --input data/my_corpus/source_records.csv \
  --output outputs/realdata/storyforest_real_events.jsonl \
  --id-col document_id --text-col text --source-col source_id \
  --occurred-col occurred_at --published-col published_at \
  --location-col location --concept-col concept_ids
```

The field names above are **illustrative** and must be mapped to actual CSV headers, not inferred. The converter requires full ISO dates, real text, and provenance, and refuses to overwrite old exports.

Then run fresh method-specific databases; do not mix method runs in one database because the event table is shared:

```bash
python -m ddpf.storyforest.cli --db outputs/realdata/sf_kg_rules.sqlite3 \
  run --input outputs/realdata/storyforest_real_events.jsonl --method kg_rules
python -m ddpf.storyforest.cli --db outputs/realdata/sf_full.sqlite3 \
  run --input outputs/realdata/storyforest_real_events.jsonl --method full
```

Story Forest does **not automatically consume the BoW clustering assignments**. This is an intended separation: clustering is the historical benchmark; evidence-grounded links and incremental stories are the journal contribution. Connecting them requires a recorded event construction mapping and validation, not reinterpreting clusters as epidemiological transmission.

## 5. Evaluation caveats

- The WHO case edges are part of the published KG. Evaluating against those same edges without separate held-out references risks leakage.
- The disease datasets span different time periods (COVID 2019/2020, Mpox 2022), making raw date a strong potential confounder in a disease-label benchmark.
- Compare cluster results with permutation-invariant metrics such as ARI; keep clustering metrics separate from story edge precision/recall.
- The original LLM notebook relies on precomputed similarity matrices. Loading an ontology and a graph dump cannot regenerate them automatically.
- The standalone Python scripts are new reproducibility utilities. They are not claimed to have been present in or executed by the original thesis.

## 6. Source references

- https://zenodo.org/records/10214515
- https://zenodo.org/records/10987743
- Historical notebooks included in the user's `DDPF-Health-Risks-studio.zip`.
