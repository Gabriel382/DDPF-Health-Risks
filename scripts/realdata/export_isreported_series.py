#!/usr/bin/env python3
"""Read-only export of COVID-19 and Mpox country-level WHO-reference edges.

Neo4j 4.4-compatible. This does NOT produce document-level Story Forest events.
Preserves one CSV record per relationship, including potentially duplicated
(country, disease, date) keys, to enable an auditable reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, date
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path

COVID = "C5203670"
MPOX = "C0276180"

QUERY = """
MATCH (d)-[r:isReported]->(p:Country)
WHERE d.id IN $ids
RETURN d.id AS disease_id, d.name AS disease_node_name,
       id(d) AS disease_node_internal_id,
       p.id AS country_id, p.wkgs_nameEn AS country_raw_name,
       id(p) AS country_node_internal_id,
       CASE WHEN d.id = $covid THEN r.Date_reported ELSE r.date END AS reported_date,
       CASE WHEN d.id = $covid THEN r.Cumulative_cases ELSE r.total_cases END AS cases_raw,
       id(r) AS relation_internal_id
"""

FIELDS = [
    "disease_id", "disease", "disease_node_name", "disease_node_internal_id",
    "country_id", "country_name", "country_raw_name", "country_node_internal_id",
    "reported_date", "date_is_iso", "cases_raw", "cases_numeric",
    "original_date_property", "original_cases_property", "relation_internal_id",
]


def normalize_record(record):
    r = dict(record)
    disease_id = r["disease_id"]
    if disease_id not in (COVID, MPOX):
        raise ValueError(f"Unexpected disease_id: {disease_id!r}")
    raw_date = "" if r["reported_date"] is None else str(r["reported_date"]).strip()
    raw_cases = "" if r["cases_raw"] is None else str(r["cases_raw"]).strip()
    try:
        iso_date = date.fromisoformat(raw_date).isoformat()
        valid_date = True
    except ValueError:
        iso_date = raw_date
        valid_date = False
    try:
        numeric_cases = format(Decimal(raw_cases), "f") if raw_cases else ""
    except InvalidOperation:
        numeric_cases = ""
    country_raw_name = r["country_raw_name"]
    country_name = str(country_raw_name or "").strip().strip('"')
    return {
        "disease_id": disease_id,
        "disease": "COVID-19" if disease_id == COVID else "Mpox",
        "disease_node_name": r["disease_node_name"] or "",
        "disease_node_internal_id": r["disease_node_internal_id"],
        "country_id": r["country_id"] or "",
        "country_name": country_name,
        "country_raw_name": country_raw_name or "",
        "country_node_internal_id": r["country_node_internal_id"],
        "reported_date": iso_date,
        "date_is_iso": int(valid_date),
        "cases_raw": raw_cases,
        "cases_numeric": numeric_cases,
        "original_date_property": "Date_reported" if disease_id == COVID else "date",
        "original_cases_property": "Cumulative_cases" if disease_id == COVID else "total_cases",
        "relation_internal_id": r["relation_internal_id"],
    }


def report_from_rows(rows):
    by_disease = defaultdict(list)
    key_values = defaultdict(list)
    for row in rows:
        by_disease[row["disease"]].append(row)
        # Use graph node ID to disambiguate entities with missing/ambiguous country names
        key = (row["disease_id"], str(row["country_node_internal_id"]), row["reported_date"])
        key_values[key].append(row["cases_raw"])
    result = {}
    for disease, dr in sorted(by_disease.items()):
        dates = sorted({r["reported_date"] for r in dr if r["date_is_iso"]})
        country_keys = {str(r["country_node_internal_id"]) for r in dr}
        result[disease] = {
            "relationships": len(dr),
            "distinct_country_nodes": len(country_keys),
            "distinct_iso_dates": len(dates),
            "first_iso_date": dates[0] if dates else None,
            "last_iso_date": dates[-1] if dates else None,
            "missing_or_invalid_dates": sum(not r["date_is_iso"] for r in dr),
            "missing_or_invalid_values": sum(not r["cases_numeric"] for r in dr),
            "relationship_rows_with_blank_country_name": sum(not r["country_name"] for r in dr),
        }
    duplicates = {k: v for k, v in key_values.items() if len(v) > 1}
    conflicting = {k: v for k, v in duplicates.items() if len(set(v)) > 1}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Neo4j :isReported edges to :Country, unaggregated",
        "results_by_disease": result,
        "duplicate_disease_country_date_keys": len(duplicates),
        "extra_rows_with_duplicate_keys": sum(len(v) - 1 for v in duplicates.values()),
        "keys_with_conflicting_case_values": len(conflicting),
        "conflict_examples": [
            {"disease_id": k[0], "country_internal_id": k[1], "date": k[2], "values": v[:10]}
            for k, v in list(conflicting.items())[:10]
        ],
        "interpretation_warning": (
            "COVID Cumulative_cases and Mpox total_cases have NOT been verified as "
            "semantically equivalent. Duplicates are kept, no summation or "
            "difference is performed. These are aggregate counts, not "
            "source-backed individual epidemiological events."
        ),
        "neo4j_id_warning": "Neo4j id() values are only stable within this database instance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/realdata/who_series")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()
    from dotenv import load_dotenv
    from neo4j import GraphDatabase, READ_ACCESS
    load_dotenv(dotenv_path=args.env)
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        parser.error("NEO4J_PASSWORD is missing from the .env file")
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:17687")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "isreported_country_raw.csv"
    qa_path = out / "isreported_country_quality.json"
    normalized = []
    with GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USERNAME", "neo4j"), password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=db, default_access_mode=READ_ACCESS) as session:
            result = session.run(QUERY, ids=[COVID, MPOX], covid=COVID)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                for neo4j_record in result:
                    row = normalize_record(neo4j_record)
                    writer.writerow(row)
                    normalized.append(row)
            result.consume()
    qa = report_from_rows(normalized)
    qa["source_database"] = db
    qa["output_csv"] = str(csv_path)
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_csv": str(csv_path),
        "quality_report": str(qa_path),
        "results_by_disease": qa["results_by_disease"],
        "duplicate_keys": qa["duplicate_disease_country_date_keys"],
        "conflicting_keys": qa["keys_with_conflicting_case_values"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
