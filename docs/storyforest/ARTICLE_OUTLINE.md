# Manuscript outline — Discover Artificial Intelligence (candidate)

Working title: **Ontology-Guided LLM Event Linking and Incremental Story Construction for Retrospective Epidemic Surveillance**

1. **Abstract:** write only after evaluation; don't reuse F1 from disease clustering as link F1.
2. **Introduction:** missing reliable evidence-driven event story organization; prior DDPF vs new method; 3–4 research questions.
3. **Related work:** Story Forest / EventX, temporal event relation modeling, epidemic EBS, KG-grounded LLMs, ontology constraints, incremental story construction.
4. **Formal problem:** mention vs observation vs event vs story; time of occurrence vs availability; identity vs continuity vs association vs unsupported causality.
5. **Method:** existing BioPropaPhenKG input; candidate retrieval; relation assessment; deterministic constraints; incremental story forest; citation + abstention.
6. **Data:** DOI/version/checksum for Zenodo snapshots; granular provenance audit; independent benchmark, event-label quality and leakage risks.
7. **Evaluation:** fixed candidates and end-to-end; baselines (temporal, KG heuristics, LLM-only, LLM+KG, full), ablations (-ontology,-KG,-time,-location,-provenance), incremental replay and cost.
8. **Results:** generated from experiment files only. Tabulate candidate recall@K, grounded edge precision, class F1, abstention, temporal violations, variance, time and RAM.
9. **Studio:** illustrate a real research run and inspectable links; distinguish from clinical/operational validation.
10. **Discussion & limitations:** lack of human validation, incomplete silver labels, historical snapshots, ontology constraints not equivalent to transmission causality.
11. **Availability and declarations:** code revision, dataset DOIs, license constraints, prompts, keys excluded, statements required by journal on submission date.

## No fabricated results

| Experiment | Dataset | Baseline | Proposed | Difference | Status |
|---|---|---|---|---|---|
| Edge quality | TBD | TBD | TBD | TBD | NOT RUN |
| Ontology ablation | TBD | TBD | TBD | TBD | NOT RUN |
| Incremental overhead | TBD | TBD | TBD | TBD | NOT RUN |
| Evidence grounding | TBD | TBD | TBD | TBD | NOT RUN |
