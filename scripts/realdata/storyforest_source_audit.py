#!/usr/bin/env python3
"""Read-only schema/evidence inventory of the restored BioPropaPhenKG.

Does NOT hallucinate event/document provenance and does NOT export synthetic
observations as real epidemiological events. Schema hints are not verification.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime,timezone
from pathlib import Path


def inventory(root):
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(root/'.env')
    uri=os.getenv('NEO4J_URI','bolt://127.0.0.1:17687')
    username=os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER','neo4j')
    pwd=os.getenv('NEO4J_PASSWORD')
    if not pwd:raise RuntimeError('NEO4J_PASSWORD missing from .env')
    database=os.getenv('NEO4J_DATABASE','neo4j')
    out={'datetime_utc':datetime.now(timezone.utc).isoformat(),
         'neo4j_uri':uri,'database':database,'read_only':True}
    with GraphDatabase.driver(uri,auth=(username,pwd)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database,default_access_mode='READ') as sess:
            out['labels']=[r['label'] for r in sess.run('CALL db.labels() YIELD label RETURN label ORDER BY label')]
            out['relationship_types']=[r['relationshipType'] for r in sess.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType')]
            for key,query in [
                ('hasPresence_samples','MATCH (a)-[r:hasPresence]->(b) RETURN labels(a) AS from_labels, keys(a) AS from_keys, labels(b) AS to_labels, keys(b) AS to_keys, keys(r) AS rel_keys, r.source AS source, r.date AS reported_date LIMIT 10'),
                ('isReported_samples','MATCH (a)-[r:isReported]->(b) RETURN labels(a) AS from_labels, keys(a) AS from_keys, labels(b) AS to_labels, keys(b) AS to_keys, keys(r) AS rel_keys LIMIT 10')]:
                try:
                    out[key]=[dict(r) for r in sess.run(query)]
                except Exception as ex:
                    out[key+'_error']=type(ex).__name__+': '+str(ex)[:300]
    tokens=('document','article','news','report','source','publication','text')
    out['potential_document_labels']=[s for s in out['labels'] if any(x in s.lower() for x in tokens)]
    out['documentary_events_ready']=False
    out['note']='Schema samples cannot establish original document evidence, publication time, event identity, or gold event relations. Real Story Forest remains gated until source-backed event JSONL is prepared and verified.'
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--out',type=Path,default=Path('outputs/realdata/storyforest_source_audit.json'))
    a=p.parse_args()
    result=inventory(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    print(json.dumps({'output':str(a.out),'possible_document_labels':result['potential_document_labels'],
                      'documentary_events_ready':False,'relationship_types':result['relationship_types']},indent=2))

if __name__=='__main__':main()
