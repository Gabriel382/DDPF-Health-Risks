#!/usr/bin/env python3
"""Recover the six original observation tables from BioPropaPhenKG.

Faithful query source: Detection Module-COVIDvsMPOX.ipynb, cells 57-59.
READ ONLY for Neo4j, stream edge rows into temporary SQLite staging on disk.
This recovers country-by-source concept observations, NOT original documents.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "Journal_COVID": ("covid_observations_journal.csv", 5, 0),
    "Medical_COVID": ("covid_observations_medical.csv", 8, 0),
    "Social_COVID": ("covid_observations_social.csv", 10, 0),
    "Journal_Monkeypox": ("monkeypox_observations_journal.csv", 5, 1),
    "Medical_Monkeypox": ("monkeypox_observations_medical.csv", 3, 1),
    "Social_Monkeypox": ("monkeypox_observations_social.csv", 10, 1),
}

# Predicates and thresholds match the original detection notebook, without mutations.
QUERY = """
MATCH (n:Country)<-[r:hasPresence]-(c)
WHERE r.source IN $sources
  AND toInteger(r.intensity) >= CASE r.source
    WHEN 'Journal_COVID' THEN 5
    WHEN 'Medical_COVID' THEN 8
    WHEN 'Social_COVID' THEN 10
    WHEN 'Journal_Monkeypox' THEN 5
    WHEN 'Medical_Monkeypox' THEN 3
    WHEN 'Social_Monkeypox' THEN 10
    ELSE 999999999 END
RETURN r.source AS source, n.wkgs_nameEn AS country_name,
       n.id AS country_id, c.name AS concept_name, c.id AS concept_id,
       r.intensity AS intensity, r.date AS relation_date
"""

SAMPLE_QUERY = """
MATCH (n:Country)<-[r:hasPresence]-(c)
WHERE r.source IN $sources
RETURN r.source AS source, n.wkgs_nameEn AS country_name,
       n.id AS country_id, c.name AS concept_name, c.id AS concept_id,
       r.intensity AS intensity, r.date AS relation_date
LIMIT $limit
"""


def load_env(root: Path):
    try:
        from dotenv import load_dotenv
        load_dotenv(root / '.env')
    except ImportError:
        pass


def connection(root: Path):
    load_env(root)
    from neo4j import GraphDatabase
    username = os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    if not password:
        raise RuntimeError('Set NEO4J_PASSWORD in .env (never put passwords in CLI or commits).')
    return GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
                                auth=(username,password)), os.getenv('NEO4J_DATABASE','neo4j')


def scalar(value):
    if value is None:
        return ''
    return str(value).strip()


def date_string(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return ','.join(map(str,value))
    return str(value)


def normalized_row(record):
    # Keep the canonical original field names as notebook read them.
    source = scalar(record['source'])
    country_id = scalar(record['country_id'])
    country_name = scalar(record['country_name']) or country_id
    concept_name = scalar(record['concept_name']) or scalar(record['concept_id'])
    concept_id = scalar(record['concept_id'])
    try: intensity = int(float(record['intensity']))
    except (ValueError, TypeError): return None
    if not source or not country_id or not concept_name:
        return None
    return (source,country_id,country_name,concept_name,concept_id,intensity,date_string(record['relation_date']))


def top_ranked(names, ids, intensities, limit=25):
    """Notebook cell 17 deduplicates names and retains top intensities.
    Deterministic ties here. Keep all source arrays in the six legacy CSVs.
    """
    best = {}
    for name, cid, score in zip(names,ids,intensities):
        if name not in best or score > best[name][1]:
            best[name] = (cid,score)
    chosen = sorted(best.items(), key=lambda kv:(-kv[1][1],kv[0]))[:limit]
    return [{'name':name,'id':pair[0],'intensity':pair[1]} for name,pair in chosen]


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def create_stage(stage: Path):
    db=sqlite3.connect(stage)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE observed (source TEXT,country_id TEXT,country_name TEXT,concept_name TEXT,concept_id TEXT,intensity INTEGER,relation_date TEXT)')
    return db


def write_files(db, output_root: Path, limit_top: int, sampled: bool, stats: dict):
    data_dir=output_root/'data'/'neo4j'
    data_dir.mkdir(parents=True,exist_ok=True)
    obs_path=output_root/'outputs'/'realdata'/'reconstructed_observations.jsonl'
    obs_path.parent.mkdir(parents=True,exist_ok=True)
    total=0
    by_source={}
    outputs=[]
    with obs_path.open('x',encoding='utf-8') as all_obs:
        for source,(filename,threshold,disease) in SOURCES.items():
            output=data_dir/filename
            outputs.append(output)
            n=0
            cursor=db.execute('''SELECT country_id,country_name,concept_name,concept_id,intensity,relation_date
                                 FROM observed WHERE source=?
                                 ORDER BY country_id ASC,intensity DESC,concept_name ASC,concept_id ASC''',(source,))
            with output.open('x',encoding='utf-8',newline='') as target:
                writer=csv.writer(target)
                writer.writerow(['System_Name','n.id','c.name','c.id','intensity','relation_dates'])
                current_id=None
                current_name=''
                names=[];ids=[];intensities=[];dates=[]
                def flush():
                    nonlocal n,total
                    if not names:return
                    writer.writerow([current_name,current_id,repr(names),repr(ids),repr(intensities),repr(dates)])
                    ranks=top_ranked(names,ids,intensities,limit_top)
                    obj={'observation_id':f'{source}::{current_id}',
                         'country_name':current_name,'country_id':current_id,
                         'source':source,'disease_label':disease,
                         'top_concepts':ranks,
                         'relation_dates':sorted({d for d in dates if d}),
                         'provenance':{'entity_granularity':'country_by_source',
                          'source':'BioPropaPhenKG-COVIDvsMPOX.dump',
                          'selection':'hasPresence with notebook original source-specific thresholds',
                          'limit_top_concepts':limit_top,
                          'aggregation':'by country and source; NOT by document',
                          'sampled':sampled}}
                    all_obs.write(json.dumps(obj,ensure_ascii=False)+'\n')
                    n+=1;total+=1
                for country_id,country_name,concept_name,concept_id,intensity,relation_date in cursor:
                    if current_id!=country_id:
                        flush()
                        current_id=country_id; current_name=country_name
                        names=[];ids=[];intensities=[];dates=[]
                    names.append(concept_name)
                    ids.append(concept_id)
                    intensities.append(intensity)
                    dates.append(relation_date)
                flush()
            by_source[source]={'csv':str(output),'n_country_observations':n,'threshold':threshold}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'source_notebook':'Experiments/Detection/Detection Module-COVIDvsMPOX.ipynb',
              'source_cells':[57,58,59], 'read_only_neo4j':True,
              'query':QUERY,'sampling':sampled,
              'graph_rows_seen':stats,'total_observations':total,
              'sources':by_source,
              'outputs_sha256':{str(p):sha256(p) for p in outputs+[obs_path]},
              'warning':'These are aggregate country-source observations, not source-document events. '
                        'The original CSVs can be reconstructed if the dump contains matching hasPresence edges, '
                        'but these do not establish document-level Story Forest evidence.'}
    report=output_root/'outputs'/'realdata'/'reconstruction_manifest.json'
    report.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest


def recover(root: Path, out: Path, max_relations: int=0, top: int=25, batch_size: int=10000):
    if max_relations < 0 or top<1:raise ValueError('Invalid sample cap or top.')
    obs=out/'outputs'/'realdata'/'reconstructed_observations.jsonl'
    manifest=out/'outputs'/'realdata'/'reconstruction_manifest.json'
    if obs.exists() or manifest.exists() or any((out/'data'/'neo4j'/v[0]).exists() for v in SOURCES.values()):
        raise FileExistsError('Output already exists. Use --out to make a new run directory. No files overwritten.')
    stage=out/'outputs'/'realdata'/'_neo4j_staging.sqlite3'
    stage.parent.mkdir(parents=True,exist_ok=True)
    if stage.exists():raise FileExistsError(f'Staging already exists: {stage}')
    db=create_stage(stage)
    driver,database=connection(root)
    seen=0;accepted=0;ignored=Counter();pending=[]
    try:
        with driver:
            driver.verify_connectivity()
            with driver.session(database=database,default_access_mode='READ',fetch_size=1000) as session:
                for rec in session.run(QUERY,sources=list(SOURCES)):
                    seen+=1
                    row=normalized_row(rec)
                    if row is None: ignored['missing_id_name_or_intensity']+=1
                    else:
                        pending.append(row);accepted+=1
                    if len(pending)>=batch_size:
                        db.executemany('INSERT INTO observed VALUES (?,?,?,?,?,?,?)',pending)
                        db.commit();pending=[]
                        if accepted%100000<batch_size:print(f'processed {seen:,} graph rows',flush=True)
                    if max_relations and seen>=max_relations:break
        if pending:
            db.executemany('INSERT INTO observed VALUES (?,?,?,?,?,?,?)',pending)
            db.commit()
        if accepted == 0:
            raise ValueError('No matching hasPresence rows. Run --inspect, check graph restore and source values.')
        db.execute('CREATE INDEX observed_sort ON observed(source,country_id,intensity DESC,concept_name,concept_id)')
        db.commit()
        report=write_files(db,out,top,bool(max_relations),{'read':seen,'kept':accepted,'excluded':dict(ignored)})
        print(json.dumps({'rows':seen,'observation_count':report['total_observations'],
            'sources':{k:v['n_country_observations'] for k,v in report['sources'].items()},
            'manifest':str(manifest)},indent=2))
    finally:
        db.close()
        stage.unlink(missing_ok=True)


def inspect(root: Path, sample_size: int):
    driver,database=connection(root)
    with driver:
        driver.verify_connectivity()
        with driver.session(database=database,default_access_mode='READ') as s:
            samples=[dict(r) for r in s.run(SAMPLE_QUERY,sources=list(SOURCES),limit=sample_size)]
            rel_types=[r['relationshipType'] for r in s.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType')]
            labels=[r['label'] for r in s.run('CALL db.labels() YIELD label RETURN label')]
    print(json.dumps({'sample_count':len(samples),'hasPresence_in_type_catalog':'hasPresence' in rel_types,
        'Country_in_labels':'Country' in labels,'source_values_seen':list(sorted({s['source'] for s in samples})),
        'samples':samples},indent=2,default=str,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('.'))
    ap.add_argument('--inspect',action='store_true',help='Only inspect up to --sample-size read-only rows')
    ap.add_argument('--sample-size',type=int,default=6)
    ap.add_argument('--max-relations',type=int,default=0,help='Use only for smoke tests; outputs marked partial')
    ap.add_argument('--top',type=int,default=25)
    a=ap.parse_args()
    if a.inspect:inspect(a.root,a.sample_size)
    else:recover(a.root,a.out,a.max_relations,a.top)

if __name__=='__main__':main()
