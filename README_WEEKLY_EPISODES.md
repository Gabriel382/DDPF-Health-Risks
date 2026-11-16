# PreDiViD: weekly country-level observation episodes

## Why this stage exists

The BioPropaPhenKG `isReported` export contains longitudinal records (COVID-19
predominantly weekly, Mpox predominantly daily) but no individual news articles
or verified document-to-document event links. This stage creates **aggregate
country-week snapshots**, not confirmed epidemiological events or Story Forest
gold labels.

## Installation

Extract the ZIP directly inside the existing `DDPF-Health-Risks` repository.
It adds only a new script, tests and this documentation; it does not replace
existing code or modify Neo4j. Python standard library only.

```
cd /data/dataRapide/gabriel/DDPF-Health-Risks
unzip -o PreDiViD_Weekly_Episodes_Stage.zip
python -m unittest discover -s tests/realdata -p 'test_weekly_episodes.py' -v
python scripts/realdata/build_weekly_episodes.py
python -m json.tool outputs/realdata/who_series/weekly/quality.json
```

## Outputs

* `weekly/snapshots.csv`: one **last observed report** per disease/country/ISO week.
* `weekly/episodes.jsonl`: machine-readable aggregate observation snapshots,
  with source `:isReported` Neo4j relationship IDs, week, and evidence caveats.
* `weekly/adjacency_baseline.jsonl`: chronology within an individual
  country+ disease time series for **consecutive weeks only**. Not causal
  links, not documented continuation, not expert annotations.
* `weekly/quality.json`: checks and provenance incl. SHA-256 of input CSV,
  gaps, revisions, missing/invalid records, counts.

## Key methodological decisions

1. ISO weeks are Monday through Sunday, with week start used as the key.
2. For daily Mpox reports, use the **last available report in each week**, never
   sum cumulative counts. For weekly COVID reports, preserve the observed week.
3. If a week is absent, do **not** forward-fill, interpolate, or calculate a
   fictitious per-week change across the gap.
4. Differences are calculated only when snapshots exist for adjacent weeks.
   They are **differences in reported values**, not independently verified
   incidence; negative corrections are preserved as flags.
5. Static `hasPresence` profiles with a single collection date per source
   are **not** matched indiscriminately to every calendar week. They cannot be
   interpreted as time-specific documentary evidence.
6. The dataset cannot support a factual Story Forest `documented_continuation`
   gold standard. Do NOT feed `episodes.jsonl` directly into the existing
   document-grounded Story Forest CLI or treat adjacency as linking ground truth.
7. Neo4j relationship internal IDs are only stable inside the restored instance.
8. `Cumulative_cases` and `total_cases` semantic equivalence still requires
   documentary confirmation before using differences as true new cases.

## Future evaluation

With these snapshots, the next experimental step can evaluate **retrospective
aggregate trajectory methods** under chronological splits and a carefully
specified *independent* evaluation target. A document-grounded Story Forest
requires actual document-level sources and verified link labels. Temporal
adjacency itself is not an independent ground truth.
