# PreDiViD: Real-data Prediction replay + Story Forest bridge

**Start here:** [`docs/realdata/REPRODUCTION.md`](docs/realdata/REPRODUCTION.md).

This is a non-destructive **overlay** for your `DDPF-Health-Risks` repository. Extract the contents directly into the repository root. No existing source files need to be overwritten.

This package includes:

- `scripts/realdata/zenodo_fetch.py` — exact Zenodo files, opt-in 2.2 GB dump download, MD5.
- `scripts/realdata/notebook_preflight.py` — find missing historical notebook inputs without executing cells.
- `scripts/realdata/kg_provenance_audit.py` — read-only Neo4j schema and `isReported` sampling.
- `scripts/realdata/replay_bow.py` — re-run the initial BoW + K-Means branch of the **original Prediction notebook**, when the 6 observation CSVs exist.
- `scripts/realdata/documents_to_events.py` — map real provenance-bearing documents to Story Forest JSONL with explicit field names.
- `Experiments/Prediction/RealData_Replay_Preflight.ipynb` — guided notebook in the original experiment sequence.
- `tests/realdata/test_realdata.py` — five offline unit tests.

From your repository root:

```bash
python scripts/realdata/zenodo_fetch.py --check
python scripts/realdata/notebook_preflight.py --mode all
python scripts/realdata/replay_bow.py --check
python -m unittest discover -s tests/realdata -v
```

**Honest boundary:** the archive does not contain the Zenodo 2.2 GB dump, six observation CSVs or all embedding matrices. Those inputs were not available locally when this package was created, so no real-data benchmark or story-linking results are claimed.
