# PreDiViD — One-shot final audit of the V5 temporal trajectory study

**One package, one command.** No more iterative Ollama smoke tests or additional manual queries for this experiment.

## Run (in the repository root, inside your existing `predivid2` tmux)

```bash
unzip -o PreDiViD_Final_OneShot.zip
bash scripts/realdata/run_predivid_final_audit.sh
```

This command runs the existing realdata unit tests together with the supplied final-audit tests, then consumes **already generated V5 predictions and Ollama caches**. It **does not** call Ollama, write Neo4j data, download files, retrain models, alter existing predictions or recompute embeddings. Requires Python and NumPy as already installed for V5.

**Precondition:** Your completed run is under `outputs/realdata/temporal_trajectory_study_v5` and your weekly snapshots remain under `outputs/realdata/who_series/weekly/snapshots.csv`. You already reported 80/80 cached LLM outputs and 40 paired test examples, so both conditions appear met.

## What runs, together

1. All tests named `test_*.py` in `tests/realdata` (including previously installed exporter, weekly snapshots, M1–M4/V5 and 12 new audit tests).
2. **Integrity & reproducibility**: snapshot SHA256 against V5 manifest, contiguous weekly series, correct countries/episodes and true labels recomputed from source values, source relationship provenance, chronological training/holdout timestamps, LLM cache agreement and all expected calls complete.
3. **Class imbalance controls**: always-flat and always-increase reference classifiers; macro-F1, balanced accuracy, precision/recall, accuracy, class frequencies, confusion matrices.
4. **Disease-stratified evaluations**: COVID-19, Mpox and both combined, on full test set / KG-covered / uncovered cohorts and the LLM sample.
5. **Paired ablations**: M2 without/with KG; M3 vs M4; M2-KG vs M3/M4, on identical held-out examples, with detailed discordant correctness counts.
6. **Cluster uncertainty**: 2,000 country-block bootstrap draws, 5,000 country-level sign-flip permutations per comparison, plus Holm adjustment across reported contrasts. Countries rather than weeks are independent resampling blocks.
7. **KG negative control**: reconstruct the original M2/KG predictions and verify perfect match with the saved results; then shuffle profiles between countries **within each disease** 99 times, preserving disease, KG availability and numeric KNN candidate pool. Report a descriptive empirical upper-tail p-value.
8. **Publication-ready audit summaries** plus explicit restrictions on what can and cannot be concluded.

The frozen `final_knn_reference_v4.py` in this package is the original V4 algorithm bundled with V5, copied under a new name **without overwriting** your experiment code.

## Outputs

- `logs/predivid_final_one_shot_audit.log`: complete automatic test and evaluation log.
- `outputs/realdata/temporal_trajectory_study_v5/final_one_shot_audit.json`: full evidence (including country intervals, disease-specific effects, negative control, adjusted p-values).
- `outputs/realdata/temporal_trajectory_study_v5/final_one_shot_report.md`: readable methods/results summary.
- `outputs/realdata/temporal_trajectory_study_v5/final_one_shot_performance.csv`: stratified metrics and baselines.

## Important scientific limits

- Evaluates **direction of the next aggregate reported count**: flat, increase or downward revision. The test holdout contains no revision examples; the main scores are therefore **increase vs flat**.
- These are **not** independently verified cases of documentary Story Forest event continuation, outbreak transmission, or valid incident case estimates.
- Static KG profiles originate from fixed source-collection periods, not individual weekly documents. They must not be misrepresented as time-local evidence.
- M2 is fitted on past labeled patterns; M3/M4 are prompted LLMs, a difference in supervision that must remain explicit.
- M4's checks are not an OWL reasoner. Evaluation of ontology-constrained documentary linking would require **new document-sourced events and gold annotations**, which the current data simply do not provide.
- LLM sample comprises 20 deterministic, KG-eligible origins for each disease, **not** a random or exhaustive LLM holdout evaluation.
- Bootstrap confidence intervals are **conditional on fixed model predictions**, not cross-run uncertainty. The country sign-flip test is approximate under cluster-label exchangeability. Exploratory negative control and post-hoc subgroup tests require cautious interpretation.
- Country profiles shuffled between countries within disease only test **sensitivity to biomedical content**, not a randomized clinical trial or causal hypothesis.
- Data (COVID cumulative cases vs Mpox total cases) have not been independently verified for equivalence.

**After this batch, the V5 temporal study is complete at its current scope.** No additional sequential benchmark requests are required to finish this experiment. A paper on genuine documentary Story Forest requires separate source document collection and gold relation validation, not another round of V5 tests.
