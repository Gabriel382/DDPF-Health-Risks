# Coding agent execution contract: PreDiViD Story Forest

You are extending an existing research repository; the overlay is a starting scaffold, NOT a completed scientific paper. Do not modify original notebooks or prior published experiment outputs unless the researcher explicitly requests it.

1. Read `README_STORYFOREST.md`, `docs/storyforest/TODO_COMPLETO.md`, `docs/storyforest/IMPLEMENTATION_ORDER.md`, `docs/storyforest/BENCHMARK_SPEC.md` and the original `README.md`.
2. Start with `python -m unittest discover -s tests/storyforest -v`. Do not make live LLM calls in the test suite. Make key/provider switching correct for LiteLLM.
3. Connect existing source entities to the event schema only after Neo4j audit; write an explicit versioned adapter and provenance report. Never infer individually documented incidents from aggregated `(concept, country, date)` counts.
4. Implement candidate retrieval with read-only parameterized Cypher, paging, query limits, and no 13-million-node materialization.
5. Implement ontology/OWL alignment with actual `PropaPhen` IRIs. Test at least type/domain/range and time invariants; distinguish unsupported from explicitly forbidden.
6. Add a relation classifier with explicit structured JSON outputs, retries, abstention, constrained evidence IDs and versioned prompts. A valid cited ID is not proof of entailment: add support verification.
7. Design scientifically correct duplicate-event merge; current prototype reports same-event decisions but does not merge occurrences into one node.
8. Implement true story forest online insert/merge/extend; allow multiple hypotheses to remain unresolved, no invented causal edges.
9. Create independent public event-linking references (or report limits honestly); create ablations with fixed candidates, leave source/topic/period shortcuts out. No fabricated F1.
10. Add new endpoint/view to main Studio only after result formats stabilize; ensure source citation is inspectable.
11. Build all figures/tables from versioned experiment outputs; prepare manuscript and statistical comparisons only after real experiments.

Every change: include tests, a changelog note, updated config examples, and an explicit scientific claim audit. `.env` is private. OpenRouter keys must not be written to logs, benchmark artifacts, error traces, or screenshots. Do not claim clinical evaluation, causality or prospective outbreak predictions.

Work order is milestone gated, not weekly. Continue until actual blocker and report that blocker with reproducible commands.
