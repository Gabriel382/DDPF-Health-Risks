#!/usr/bin/env python3
"""Read-only, bounded inspection of BioPropaPhenKG before any event export.

Inspect a representative set of WHO isReported edges and graph labels. This
script never claims that each isReported edge is a document-level Event.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

CUI = {'COVID': 'C5203670', 'Monkeypox': 'C0276180'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--output', default='outputs/realdata/kg_provenance_audit.json')
    args = ap.parse_args()
    if not 1 <= args.limit <= 200:
        ap.error('--limit must be between 1 and 200')
    root = Path(args.root).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / '.env')
    except ImportError:
        pass
    from neo4j import GraphDatabase
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    if not password:
        ap.error('NEO4J_PASSWORD is missing (set it in .env)')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    report = {'record': '10987743', 'limit': args.limit,
              'database': database, 'uri': uri,
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'warning': 'isReported is aggregate WHO evidence, not necessarily a source-document event'}
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database, default_access_mode='READ') as session:
            report['node_labels'] = [r['label'] for r in session.run('CALL db.labels() YIELD label RETURN label')]
            report['relationship_types'] = [r['relationshipType'] for r in session.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType')]
            report['who_isreported_samples'] = {}
            for disease, cui in CUI.items():
                records = session.run('''
                    MATCH (source:CUI {id: $cui})-[r:isReported]->(country:Country)
                    RETURN source.id AS cui, source.name AS disease_name,
                           country.id AS country_id, country.name AS country_name,
                           r.date AS date, properties(r) AS relation_properties
                    LIMIT $limit
                ''', {'cui': cui, 'limit': args.limit})
                report['who_isreported_samples'][disease] = [dict(r) for r in records]
            report['generic_relation_samples'] = [dict(r) for r in session.run('''
                MATCH (a)-[r]->(b)
                RETURN labels(a) AS source_labels, type(r) AS relation_type,
                       labels(b) AS target_labels, keys(r) AS relation_property_keys
                LIMIT $limit
            ''', {'limit': args.limit})]
    dest = root / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding='utf-8')
    print(f'Saved read-only KG audit to {dest}')
    for disease, sample in report['who_isreported_samples'].items():
        print(f'{disease}: {len(sample)} bounded WHO isReported samples')
    print('Graph was NOT modified; no document-level Event rows were fabricated.')

if __name__ == '__main__':
    main()
