# Technical execution — gates, not weeks

**A · Data audit:** `python -m ddpf.storyforest.cli audit-neo4j ...`; inspect source granularity; produce an approved `data/benchmark/events.jsonl`. STOP if per-event provenance absent.

**B · Independent silver benchmark:** verified same-event and documented-continuation links, source+publication timestamp evidence, unknown retained. STOP if no independent evidence for real-edge evaluation.

**C · Research implementation:** replace heuristic candidate retrieval with indexed BioPropaPhenKG graph queries; create actual domain/range/time constraints from BioPropaPhen; incrementality with provenance; add proper duplicate-event merge and parent selection tests; persist complete experiment runs.

**D · Experiments:** compare 5 methods with fixed candidates; run no-KG, no-ontology, no-time and no-space ablations; no future data; profile RAM and token cost. STOP or reframe if no measurable contribution.

**E · DDPF modules:** Description lazy/read-only Neo4j access; Detection Java GATE `.xgapp` headless one file at a time with preserved offsets; Prediction import and testing pipeline from stable modules, keep notebooks as frozen legacy.

**F · UI integration:** existing `interface/index.html` remains untouched; validate new local Story Forest view (`http://localhost:8765/storyforest`) with actual event JSONL and at least one inspected evidence-backed relation.

**G · Paper:** experiments -> figures -> discussions -> limitations -> related work -> submission materials and open reproducibility bundle. Review actual journal author instructions before submission.

**File map**
- `src/ddpf/llm/litellm_client.py`: single LiteLLM entry point for Ollama/OpenRouter.
- `src/ddpf/storyforest/models.py`: structured event/evidence/link schemas.
- `src/ddpf/storyforest/candidates.py`: bounded retrieval baseline. KG-backed retrieval P0 TODO.
- `src/ddpf/storyforest/ontology.py`: deterministic rule hooks. Explicit OWL alignment P0 TODO.
- `src/ddpf/storyforest/classifier.py`: JSON evidence ID checks + LLM prompting.
- `src/ddpf/storyforest/pipeline.py`: incremental update.
- `src/ddpf/storyforest/storage.py`: persistent SQLite and story forest assembly.
- `src/ddpf/storyforest/audit_neo4j.py`: bounded read-only audit of existing snapshot.
- `src/ddpf/storyforest/detection_stream.py`: optional GATE `.xgapp` headless, raw annotations only.
- `src/ddpf/storyforest/evaluate.py`: conservative partial-label evaluation.
- `interface/storyforest.*`: separate working results explorer, no changes to existing Studio.
- `docs/storyforest/TODO_COMPLETO.md`: entire scientific checklist.
