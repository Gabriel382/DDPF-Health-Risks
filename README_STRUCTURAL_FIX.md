# PreDiViD projected structural embeddings: isolated-node and SVD-country fix

Patch for the `PreDiViD_All_Experiments_Ready.zip` suite. Extract in the **DDPF-Health-Risks project root**, overwriting the named script.

## Cause of the failure

Masking explicit disease terms can remove every concept edge of some observations. The original FastRP-style baseline raised `ValueError('Graph contains isolated nodes')`. This is an expected outcome of the controlled ablation, not an invalid input.

## What changed

- The FastRP-inspired random-propagation baseline uses a zero transition row for isolated vertices and yields zero embeddings for isolated observations.
- Node2Vec and GraphSAGE-style baselines also set isolated observation vectors to zero instead of preserving randomly initialized vectors or learned self-feature embeddings unsupported by the graph.
- Matrix row order and the original 126 observations are preserved, and the manifest reports isolated observation identifiers and counts.
- The SVD `with_country=True` variant previously **ignored** country, because only the observation-concept matrix `B` was passed to SVD. The corrected variant appends observed country one-hot adjacency columns. Previous SVD-country results are *archived* in `archive_svd_country_ignored_v1/` with `.legacy` filenames, not silently overwritten, then the corrected variants are regenerated. Previously completed SVD without country and other non-isolated model outputs are reused unchanged.
- The embedding folder names and output files expected by the benchmark remain unchanged.

This is still a **projected observation-concept graph benchmark**, not a rerun of the full BioPropaPhenKG GDS FastRP/GraphSAGE pipeline. An isolated, zero-information observation is deliberately retained as an all-zero vector for faithful comparison; the resulting ablation should be reported with its isolated count.

## Run

```bash
cd /data/dataRapide/gabriel/DDPF-Health-Risks
source .venv/bin/activate
unzip -o PreDiViD_Structural_Isolated_Nodes_Fix.zip
python -m unittest discover -s tests/realdata -p 'test_isolated_structural.py' -v

# Dedicated tmux window/session predivid2; all expensive Ollama embeddings are already done.
mkdir -p logs
set -o pipefail
python -u scripts/realdata/run_all_predivid_experiments.py --with-qwen4b --skip-text --strict-structural \
  2>&1 | tee logs/predivid_all_resume.log
```

The suite creates a **new** timestamped directory under `outputs/realdata/experiment_suites/`, and the 16 structural matrices are created/reused under `outputs/realdata/structural_embeddings/`. Text embeddings stay in `outputs/realdata/embeddings/` and are discovered by the benchmark.

To inspect output:

```bash
LATEST=$(find outputs/realdata/experiment_suites -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
echo "$LATEST"
cat "$LATEST/suite_manifest.json"
head -n 12 "$LATEST/benchmark/results.csv"
cat outputs/realdata/structural_embeddings/run_status.json
```

## Validation performed

- 6 new tests (including isolated vertices, no random evidence, SVD country feature and safe archive migration) passed.
- 7 original suite tests passed.
- All 16 structural configurations completed on a synthetic six-observation fixture containing isolated masked observations; two prior SVD-country matrices were safely archived and regenerated in a resume simulation.

**Not verified** on the user's remote Neo4j or actual 126 observations, which are not available in this environment.
