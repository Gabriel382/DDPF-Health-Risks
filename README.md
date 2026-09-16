# DDPF Health Risks

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DDPF is a **Description–Detection–Prediction Framework** for retrospective,
ontology-guided analysis of health-risk observations collected from
heterogeneous sources. The repository contains the framework code and the
research notebooks used for experiments with COVID-19 and mpox observations.

> **Scope.** The current experiments evaluate retrospective semantic
> organisation and clustering of previously collected observations. They do
> not constitute a validated real-time outbreak-detection or early-warning
> system.

## Framework overview

| Stage | Purpose | Main output |
|---|---|---|
| **Description** | Combine the PropaPhen ontology with biomedical and geographic knowledge. | BioPropaPhenKG / BioPropaPhenKG+ ontology and gazetteers |
| **Detection** | Match biomedical and geographic mentions and construct source-specific observations. | Observation tables and relation/similarity matrices |
| **Prediction** | Represent and cluster the observations for retrospective analysis. | Cluster assignments, evaluation metrics and visualisations |

The principal experiment compares COVID-19 and mpox observations at document,
paragraph and sentence granularity. It also compares several representation
pipelines, including ontology-based representations, biomedical sentence
embeddings, Node2Vec and LLM-derived embeddings.

## Data and archived experiment snapshots

Large data products are not stored in this Git repository. Preserve the
original archive names and record versions when downloading the Zenodo
snapshots.

| Record | Intended use |
|---|---|
| [Zenodo 10.5281/zenodo.10987743](https://doi.org/10.5281/zenodo.10987743) | BioPropaPhenKG+ snapshot for the COVID-19 versus mpox experiment. Use it to start from the prepared knowledge-graph/data products instead of rebuilding all earlier Description and Detection outputs. |
| [Zenodo 10.5281/zenodo.10997359](https://doi.org/10.5281/zenodo.10997359) | BioPropaPhenKG+ snapshot associated with the COVID-19 online-newspaper relation-extraction benchmark. |

After downloading a record:

1. Read its Zenodo manifest and note the record version and download date.
2. Extract the archive without changing its internal filenames.
3. Place its contents under `data/` using the layout expected by the target
   notebook, or edit the notebook's configuration cell to point to the actual
   extracted paths.
4. Do not mix files from different Zenodo records or locally regenerated runs
   without documenting which version supplied each input.

The notebooks still contain several historical relative paths. Therefore,
**the archive itself does not automatically configure the notebooks**: paths
must be checked before execution.

## Repository contents

```text
DDPF-Health-Risks/
├── Experiments/
│   ├── Description/
│   ├── Detection/
│   ├── Prediction/
│   └── Deprecated/
├── src/ddpf/
│   ├── description/          # maintained Description API
│   ├── detection/            # package scaffold
│   ├── prediction/           # package scaffold
│   ├── llm/                  # optional Ollama integration
│   └── deprecated/lib/       # legacy modules used by many notebooks
├── tests/
├── pyproject.toml
└── README.md
```

Many experiment notebooks depend on the modules under
`src/ddpf/deprecated/lib/`. The name indicates legacy organisation; it does
not mean that those modules are unused.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Gabriel382/DDPF-Health-Risks.git
cd DDPF-Health-Risks
```

### 2. Create a virtual environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install DDPF and notebook dependencies

The package metadata installs the maintained core dependencies:

```bash
python -m pip install -e .
```

The historical notebooks use additional research dependencies that are not
yet fully represented in `pyproject.toml`:

```bash
python -m pip install jupyterlab pandas numpy scipy tqdm matplotlib seaborn \
  scikit-learn scikit-learn-extra nltk gatenlp neo4j langchain-community
```

Some comparison notebooks additionally require PyTorch, DGL, PyKEEN or other
model-specific packages. Install those packages only for the notebook being
run and record their versions. GPU-specific PyTorch/DGL installation commands
depend on the operating system and CUDA version.

### 4. Download NLTK resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 5. Start Jupyter

```bash
jupyter lab
```

Start Jupyter from the repository root so that the path discovery below works
consistently.

## External services and resources

Depending on the chosen execution path, the notebooks may require:

- **GATE / GateWorker and ANNIE 8.6** for parts of the historical Detection
  pipeline;
- **Neo4j** for importing or querying graph outputs;
- licensed **UMLS** resources for rebuilding biomedical ontology inputs;
- **WorldKG/OpenStreetMap-derived** geographic resources;
- **Ollama** for the optional local-LLM Description helper.

The optional LLM helper proposes ontology enrichments. In the current code,
the suggestions are recorded separately and should not be described as
automatically modifying the graph unless that application step is explicitly
implemented and verified.

## Recommended data layout

The repository does not currently ship the complete datasets. A practical
local layout is:

```text
data/
├── ontologies/
│   ├── PropaPhen/PropaPhen.owl
│   └── propaphenplus/
│       ├── saved/
│       │   ├── umlsonto.owl
│       │   └── worldkg.owl
│       └── generated/
├── gazetteers/
├── neo4j/
├── relations/
├── embeddings/
└── results/
```

This is a recommendation, not a guarantee that every historical notebook
already uses these exact paths. Search the notebook's first configuration
cells for variables beginning with `path_to_` and update them before running.

## Notebook path compatibility

For notebooks that import the legacy modules, add the following setup cell
near the top. It discovers the repository root instead of relying on the
machine-specific paths saved in older notebooks.

```python
from pathlib import Path
import sys

REPO_ROOT = Path.cwd().resolve()
while not (REPO_ROOT / "pyproject.toml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

if not (REPO_ROOT / "pyproject.toml").exists():
    raise RuntimeError("Run the notebook from inside the DDPF repository.")

LEGACY_LIB = REPO_ROOT / "src" / "ddpf" / "deprecated" / "lib"
DATA_ROOT = REPO_ROOT / "data"
sys.path.insert(0, str(LEGACY_LIB))
```

Define all subsequent input and output paths from `DATA_ROOT` rather than from
an absolute user directory.

## Execution paths

### Path A — Minimal Description example

Use this path to test the maintained Description API and ontology composition:

1. Put the three ontology inputs at the locations expected in
   `Experiments/Description/Description.ipynb`:
   - `data/ontologies/PropaPhen/PropaPhen.owl`
   - `data/ontologies/propaphenplus/saved/umlsonto.owl`
   - `data/ontologies/propaphenplus/saved/worldkg.owl`
2. Open `Experiments/Description/Description.ipynb`.
3. Replace any saved machine-specific root path with `REPO_ROOT` from the
   compatibility cell.
4. Run the manual composition cells.
5. Verify the generated ontology under
   `data/ontologies/propaphenplus/generated/`.
6. Run the Ollama cells only if a compatible local model is installed.

### Path B — Fast reproduction from the Zenodo snapshot

Use this path when the goal is to reproduce the COVID-19 versus mpox
prediction experiment without rebuilding all ontology and detection products:

1. Download [Zenodo record 10987743](https://doi.org/10.5281/zenodo.10987743).
2. Extract it and inspect its files and metadata.
3. Map the supplied ontology, gazetteer, observation and/or matrix files to the
   path variables used by the target Prediction notebook.
4. Open
   `Experiments/Prediction/Prediction Module - COVID vs Monkeypox Observation.ipynb`.
5. Run only after every input-path assertion succeeds.
6. Clear old cell outputs before producing a new archived run.
7. Save the environment, Zenodo record/version, random seeds and generated
   output tables together with the executed notebook.

### Path C — Full Description → Detection → Prediction workflow

This is the longer workflow used when regenerating the intermediate products:

1. Prepare the PropaPhen, UMLS and WorldKG inputs.
2. Run `Experiments/Description/Description.ipynb` or the historical
   Description notebook required by the experiment.
3. Generate or provide the biomedical and geographic gazetteers.
4. Configure GATE/ANNIE and the source corpora.
5. Run `Experiments/Detection/Detection Module-COVIDvsMPOX.ipynb`.
6. Verify the six source/disease observation tables before proceeding:
   COVID-19 and mpox × journal, medical and social sources.
7. Run the corresponding Prediction notebook.
8. Export the executed notebook, metric tables and environment information.

The Detection notebook creates observations from biomedical and geographic
matches within document-, paragraph- or sentence-level units. The historical
COVID-19/mpox experiment uses source-specific occurrence thresholds, so those
values must be kept with the run rather than treated as universal defaults.

### Path D — COVID-19 online-newspaper benchmark

1. Download [Zenodo record 10997359](https://doi.org/10.5281/zenodo.10997359).
2. Configure the inputs for
   `Experiments/Detection/Detection Module - COVID on Newspapers.ipynb`.
3. Run the matching Prediction notebook:
   `Experiments/Prediction/Prediction Module - COVID on Newspapers.ipynb`.
4. Keep the Zenodo version and all local path changes with the executed run.

## Journal-extension embedding experiment

`Experiments/Prediction/LLM Based Prediction Module.ipynb` evaluates
precomputed embeddings/similarity matrices. It does **not** generate every
embedding from raw text by itself. To reproduce an encoder comparison, archive
the following together with each matrix:

- exact model checkpoint and revision;
- input template for Entity, Concept and Concepts configurations;
- embedding dimension and pooling strategy;
- maximum sequence length and truncation policy;
- normalisation and similarity calculation;
- batch size, software versions and hardware;
- output filename and checksum.

### Main clustering/evaluation protocol

The principal precomputed-matrix workflow in the current notebook uses:

1. distance conversion `D = 1 - S` from the similarity matrix;
2. global min–max scaling of `D` to `[0, 1]`;
3. K-Medoids with `k = 2`, `metric="precomputed"` and
   `random_state = 42`;
4. evaluation of both possible mappings between the two cluster identifiers
   and the two disease labels;
5. reporting of support-weighted class F1 for the better mapping.

The currently saved results are single-run results. They must not be presented
as means over repeated runs, and no standard deviation or confidence interval
should be claimed unless repeated runs or an appropriate bootstrap analysis
are actually performed.

### Node2Vec comparison

The historical Node2Vec branch follows a different downstream pipeline: graph
embeddings are aggregated, reduced with t-SNE and clustered with K-Means. It
should therefore be described as a **pipeline-level baseline**, not as a
strict encoder-only comparison under an identical clustering procedure.

## Reproducibility notes

- Some notebooks contain saved outputs from earlier executions. Treat those
  outputs as provenance clues, not as proof that the current files can be run
  without configuration.
- The saved mpox output in a notebook may be older than the corrected result.
  Clear the output and re-execute with the documented data version before
  archiving a reproduction.
- Embedding APIs do not use a text-generation temperature. If a generative
  model is used to create or transform text, record `temperature = 0` only
  where that parameter was explicitly set in the executed code or API call.
- Record all random seeds. At minimum, the current notebooks contain seed 42
  for K-Medoids, seed 99 for the t-SNE step in the Node2Vec branch and seed 0
  for its K-Means step.
- Do not compare results produced from silently different corpus snapshots,
  ontology versions, granularity levels or preprocessing rules.

## What is included and what is external

| Item | In this repository? | Where / action |
|---|---:|---|
| Maintained Description package | Yes | `src/ddpf/description/` |
| Optional Ollama integration | Yes | `src/ddpf/llm/` |
| Historical experiment notebooks | Yes | `Experiments/` |
| Legacy Detection/Prediction modules used by notebooks | Yes | `src/ddpf/deprecated/lib/` |
| Complete corpora and intermediate CSV files | No | Supply locally or retrieve from the applicable archive/source |
| Complete BioPropaPhenKG+ snapshots | No | Zenodo records above |
| UMLS source data | No | Obtain under the applicable UMLS licence |
| Exact environment lock file | No | Create one for each reproduced run |

## Pre-run checklist

Before running an experiment, verify:

- [ ] the intended Zenodo record and version are documented;
- [ ] all notebook input paths exist;
- [ ] the corpus, ontology and gazetteer versions match;
- [ ] observation granularity is fixed: document, paragraph or sentence;
- [ ] thresholds and filtering rules are recorded;
- [ ] model checkpoint, template and embedding settings are recorded;
- [ ] clustering algorithm, distance, `k`, seeds and label matching are recorded;
- [ ] old notebook outputs have been cleared;
- [ ] output directories are separate from downloaded source data;
- [ ] the executed notebook and environment package list are saved.

One simple environment capture is:

```bash
python -m pip freeze > environment-freeze.txt
```

## Troubleshooting

### `ModuleNotFoundError: detection` or `prediction`

Install the repository with `python -m pip install -e .` and add
`src/ddpf/deprecated/lib/` to `sys.path` using the compatibility cell above.

### A notebook points to another user's Windows directory

Replace the absolute path with `REPO_ROOT` / `DATA_ROOT` paths. Do not recreate
the original user's directory tree.

### A CSV, ontology or matrix is missing

The complete data products are not committed here. Check the appropriate
Zenodo archive and the notebook's configuration cells. If the file is not in
either source, it must be regenerated by an earlier pipeline stage.

### GATE/ANNIE cannot start

Confirm that Java, GATE, GateWorker and the ANNIE 8.6 resources are installed
and that the notebook points to their local locations.

### The notebook result differs from its saved output

First compare input checksums, data versions, preprocessing thresholds,
library versions and random seeds. Saved notebook output can reflect an older
run and should not override a clean, documented re-execution.

## Publications

- Project repository: <https://github.com/Gabriel382/DDPF-Health-Risks>
- Thesis: <https://theses.hal.science/tel-05056160>
- IEEE CBMS paper: <https://hal.science/hal-05398405>

When citing an experiment, cite both the relevant publication and the exact
Zenodo record/version used for its data products.

## License

This repository is released under the [MIT License](LICENSE). External
datasets, ontologies, models and software retain their own licences and terms
of use.

