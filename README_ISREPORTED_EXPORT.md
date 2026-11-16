# PreDiViD WHO-reference time series export

This read-only exporter handles the **two different `isReported` schemas** confirmed in your Neo4j 4.4.3 dump.

| Disease | UMLS ID | Date property | Value property |
|--|--|--|--|
| COVID-19 | C5203670 | `Date_reported` | `Cumulative_cases` |
| Mpox | C0276180 | `date` | `total_cases` |

## Installation in DDPF-Health-Risks

Unzip in the root of the repository, preserving `scripts/realdata` and `tests/realdata`. Dependencies: `neo4j`, `python-dotenv`.

```bash
python -m unittest discover -s tests/realdata -p 'test_export_isreported_series.py' -v
python scripts/realdata/export_isreported_series.py
```

With Neo4j reachable through the credentials in `.env`, this produces:

- `outputs/realdata/who_series/isreported_country_raw.csv`: **one row per original graph edge**, preserving source key names, values, and internal node/relationship IDs.
- `outputs/realdata/who_series/isreported_country_quality.json`: coverage, distinct time periods, missingness, duplicate country-date groups and conflicting values.

### Scientific interpretation

1. Values under `Cumulative_cases` and `total_cases` are not automatically assumed equivalent. Neither is transformed into incidence or summed.
2. All country-date duplicate edges are preserved; inspect the quality report before choosing any aggregation rule.
3. The export excludes villages, cities, and other geographic classes, preventing naive double-counting across geographic levels.
4. Reference dates do not establish individual source documents, transmission, or story continuation. A Story Forest experiment will need a separately validated event construction/evidence protocol.
5. The synthetic test does not validate any claim about the real dataset.
