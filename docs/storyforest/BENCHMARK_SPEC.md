# Evidence-based benchmark protocol (not automatically created)

Define an **event** as a reported real-world incident, not a disease label,
country-wide observation total, or time window. Each must preserve a source,
publication date, event date (if recoverable), and evidence ID.

Gold record JSONL: `{ "parent_id": "...", "child_id": "...", "label": "same_event | documented_continuation | related_only | no_relation | unknown", "basis": "independent_documented | manual_gold", "reference_url": "...", "reference_quote": "..." }`.

Source-derived notes:
- Story Forest (Liu et al., 2020, TKDD) measures clustering separately from story-edge correctness and coherence; it used human judgments. It **does not** provide a no-human ground truth for BioPropaPhenKG.
- The PreDiViD graph snapshot may store aggregated observations; never silently relabel those aggregates as individually verified events.

Mandatory experiment conditions:
1. Prevent circular gold: data from WHO must not appear both as model evidence and independent reference for the same pair at the same timestamp.
2. Pair `unknown` and missing labels are **not negatives**. Evaluate precision, recall and F1 only on independently verified pair labels; report annotation coverage.
3. Keep synthetic counterfactuals separate from real-world results.
4. Split by original source group and chronological availability; remove duplicates and leaked disease identifiers.
5. Measure candidate recall@K before relation classifier, then exact relation accuracy/F1, abstention coverage, date violations, citation validity, latency and memory.
6. Rules from `config/storyforest_rules.json` are examples, not formal validation of PropaPhen; map ontology IRIs and test soundness before claiming ontology-driven reasoning.
7. Repeated LLM calls must record provider, model identifier, prompt revision, decoding parameters, timestamps, elapsed time and request counts (never credentials).
8. Need independent, real linked-event references before claiming epidemiological correctness; formal constraint checks alone prove only consistency.

**Current overlay:** functional experimental scaffold. No fully annotated event dataset, OWL reasoner, trained relationship classifier, or real snapshot-to-event mapping is provided. Those are substantive TODOs, not completed contributions.
