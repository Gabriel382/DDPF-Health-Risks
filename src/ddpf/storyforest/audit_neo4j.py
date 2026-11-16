"""Read-only bounded schema audit for the existing BioPropaPhenKG snapshot.

DOES NOT create `Event` records from graph rows: schema and provenance must
be inspected first. No writes or unrestricted graph fetches.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

def audit(output:str,limit:int=40):
    from .settings import load_project_env
    load_project_env()
    from neo4j import GraphDatabase
    uri=os.getenv('NEO4J_URI','bolt://localhost:7687')
    username=os.getenv('NEO4J_USERNAME','neo4j');password=os.getenv('NEO4J_PASSWORD','')
    if not password:raise ValueError('Set NEO4J_PASSWORD in local .env')
    report={'uri':uri,'limit':limit,'note':'SAMPLES ONLY: no inference about completeness/representativeness'}
    with GraphDatabase.driver(uri,auth=(username,password)) as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j'),default_access_mode='READ') as sess:
            report['labels']=[row['label'] for row in sess.run('CALL db.labels() YIELD label RETURN label')]
            report['relationship_types']=[row['relationshipType'] for row in sess.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType')]
            # Bounded samples: avoid huge result materialization in 13M-node KG.
            report['node_samples']=[dict(row) for row in sess.run('MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties LIMIT $limit',limit=limit)]
            report['edge_samples']=[dict(row) for row in sess.run('MATCH (a)-[r]->(b) RETURN type(r) AS type, properties(r) AS properties, labels(a) AS source_labels, labels(b) AS target_labels LIMIT $limit',limit=limit)]
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,default=str,ensure_ascii=False),encoding='utf-8')
    return str(dest)
