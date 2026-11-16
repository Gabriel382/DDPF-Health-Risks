# PreDiViD Story Forest research overlay

**Drop-in overlay for [DDPF-Health-Risks](https://github.com/Gabriel382/DDPF-Health-Risks)** · LiteLLM (Ollama/OpenRouter) · research scaffold

> This package **does not replace the repository** and **does not contain finalized scientific results**. It adds new paths without touching the existing `Experiments/` notebooks, `src/ddpf/description/`, old `src/ddpf/llm/ollama.py`, `interface/index.html`, or the repository's `pyproject.toml`.
>
> The previous 131-item TODO is at `docs/storyforest/TODO_COMPLETO.md`; the gate-by-gate execution spec is in `docs/storyforest/IMPLEMENTATION_ORDER.md`.

## 0. Extraction into the EXISTING repository root

Extract the ZIP contents directly into the root of the DDPF checkout. **Do not create an extra nesting folder**. All paths in the ZIP are relative to the project root. Use your ZIP app's "extract here" option from within that folder. Make a Git branch/backup first.

This overlay only adds new files; see `PACKAGE_MANIFEST.txt`. It does **not** include `.env` with secrets.

## 1. Install / configure

Use Python 3.10–3.12 with Java/GATE optional for Detection. In project root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python scripts/storyforest_setup.py
python -m pip install -e .
python -m pip install -r requirements-storyforest.txt
python -m ddpf.storyforest.cli doctor
python -m unittest discover -s tests/storyforest -v
```

The setup script only creates `.env` when missing and **never overwrites** an existing one. Edit `.env` to select exactly one provider:

**Local Ollama**

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
```

Start Ollama separately (`ollama serve` and `ollama pull qwen3:4b`, adjusting to a model available on your host). No API key needed.

**OpenRouter**

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=YOUR_REAL_KEY_HERE
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

Only put your real key in `.env` (ignored by the existing `.gitignore`), **never** in `README`, GitHub, notebooks, command-line arguments, or papers. OpenRouter can incur charges. Ollama may require local GPU/RAM.

LiteLLM dispatches `ollama/<model>` to Ollama `OLLAMA_BASE_URL`, and `openrouter/<provider>/<model>` to the OpenRouter API. Both use `litellm.completion`, without starting a LiteLLM proxy. Sources: https://docs.litellm.ai/docs/providers/ollama and https://docs.litellm.ai/docs/providers/openrouter .

## 2. First test without Neo4j or API keys

This synthetic-only run proves the persistence and baseline plumbing, not scientific performance:

```bash
python -m ddpf.storyforest.cli --db outputs/storyforest/synthetic.sqlite3 run --input examples/storyforest/synthetic_events.jsonl --method kg_rules --output outputs/storyforest/synthetic_result.json
python -m ddpf.storyforest.cli --db outputs/storyforest/synthetic.sqlite3 evaluate --gold examples/storyforest/synthetic_gold.jsonl --method kg_rules
python -m ddpf.storyforest.cli --db outputs/storyforest/synthetic.sqlite3 serve --port 8765
```

Open **http://127.0.0.1:8765/storyforest** to inspect the event stories and candidate decisions. The evaluation should exclude all synthetic pairs rather than present fake real-data scores.

To test the real LLM provider, use a **different** empty database to avoid reusing persisted IDs:

```bash
python -m ddpf.storyforest.cli --db outputs/storyforest/llm_try.sqlite3 run --input examples/storyforest/synthetic_events.jsonl --method full --rules config/storyforest_rules.json --output outputs/storyforest/llm_try.json
```

The full method makes real API calls (Ollama or OpenRouter) for candidate pairs. Repeatability: use a new SQLite file for each run/condition or batch; snapshot keys and prompts in metadata. Model scores here are **not calibrated probabilities**.

## 3. Existing graph & ontology (important)

Data artifacts, not included in this ZIP:
- BioPropaPhen and KG: https://zenodo.org/records/10911980
- COVID-19/Mpox experimental snapshot: https://zenodo.org/records/10987743

Restore the existing database according to the Zenodo record instructions. Then use the read-only audit:

```bash
# Set NEO4J_* values in .env first
python -m ddpf.storyforest.cli audit-neo4j --limit 30 --output outputs/storyforest/neo4j_audit.json
```

**Important limitation:** the actual mapping from the existing Neo4j snapshot to verified individual `Event` records must be established from the audit; the overlay intentionally DOES NOT guess CUI/hasPresence schemas or fabricate provenance. It currently accepts researcher-approved events as JSONL (or per-event CSV with mandatory provenance fields via `scripts/import_event_csv.py`). The current `ontology.py` provides rule hooks, **not a complete OWL reasoner**; fill rules only after checking ontology IRIs.

## 4. Optional GATE headless incremental ingestion

Install GATE Java + a GATE `.xgapp` pipeline that contains the desired original biomedical and geospatial resources; `pip install gatenlp`. Set `GATE_HOME` and `GATE_PIPELINE_PATH` in `.env` (the shell utility reads `GATE_PIPELINE_PATH` from environment or uses `--gapp`; `.env` is loaded through the CLI settings in future integration; for this script, export the variables or pass `--gapp`).

```bash
python scripts/gate_headless.py --zip /path/to/texts.zip --gapp /path/to/PreDiViD.xgapp --output outputs/storyforest/gate_annotations.jsonl
```

The utility opens one ZIP text document at a time, runs a headless `GateWorker(start=True)`, appends raw annotations, commits processed hashes to SQLite, and closes the Java document handle. It **does not yet** translate GATE annotations into fully linked BioPropaPhenKG observations (requires your gazetteer and schema mapping). Do not call this an end-to-end rebuilt Detection module.

## 5. Study design, reproducibility, and what remains to implement

**Already executable:** LiteLLM switching, JSON event schema, bounded candidate heuristic, evidence-ID checks, temporal/type-rule hooks, persistent incremental storage, baseline/LLM modes, partial-label evaluator, bounded Neo4j audit, GATE raw-annotation stream, results API/visualizer and offline tests.

**Not implemented / must be completed before submitting a methods paper:** actual graph-backed event extraction from BioPropaPhenKG, complete ontology axiom checking, research-grade Story Forest duplicate merge/rewiring, faithfully adapted original Story Forest baseline, an independent real event-linking reference dataset, proper temporal held-out evaluation, published ablations, multi-run cost/memory profiling, interface integration into the existing main navigation, formal explainability/claim review, and a paper with *measured* results.

The scientific gates and all tasks are in:
- `docs/storyforest/TODO_COMPLETO.md` — original full checklist
- `docs/storyforest/IMPLEMENTATION_ORDER.md` — exact file map, implementation gates and stop conditions
- `docs/storyforest/BENCHMARK_SPEC.md` — data and evaluation rules
- `docs/storyforest/ARTICLE_OUTLINE.md` — manuscript skeleton without invented numbers

**Epidemiological caution:** thematic / temporal co-occurrence does not establish transmission or causality. Automated reference extraction and LLM-as-judge alone are not independent validated ground truth. No clinical or real-time surveillance claims.

## 6. Run a baseline / LLM method matrix (separate, auditable DBs)

```bash
# No billable LLM calls:
python scripts/run_storyforest_matrix.py --input examples/storyforest/synthetic_events.jsonl --gold examples/storyforest/synthetic_gold.jsonl --methods temporal,kg_rules

# After selecting provider and a real benchmark; this makes API/model calls:
python scripts/run_storyforest_matrix.py --input data/benchmark/events.jsonl --gold data/benchmark/pairs.jsonl --allow-billable-llm
```

This script creates a new time-stamped directory with separate SQLite databases and a manifest (no secret keys), so one experiment condition cannot silently reuse events from another.
