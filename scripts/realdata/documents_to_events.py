#!/usr/bin/env python3
"""Evidence-preserving CSV -> Story Forest Event JSONL converter.

Only works with actual document text, explicit source identifiers and ISO dates.
Does not infer epidemiological event-to-event relations from KG proximity.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path


def parse_values(value: str) -> list[str]:
    if not value or value.lower() in {'nan', 'none'}:
        return []
    try:
        import ast
        data = ast.literal_eval(value)
        if isinstance(data, (list,tuple,set)):
            return [str(v) for v in data if str(v).strip()]
        if isinstance(data, str): return [data] if data.strip() else []
    except (SyntaxError,ValueError):
        pass
    return [v.strip() for v in value.split('|') if v.strip()]


def convert(input_csv: Path, out: Path, *, id_col: str, text_col: str,
            source_col: str, occurred_col: str, published_col: str,
            location_col: str = '', concept_col: str = '', ontology_type_col: str = '',
            url_col: str = '', description_col: str = '', max_events: int = 0):
    total = 0
    out.parent.mkdir(parents=True,exist_ok=True)
    # Use exclusive creation so a rerun never silently mixes experiment versions.
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}. Choose a fresh output path.')
    try:
        with input_csv.open('r',encoding='utf-8-sig',newline='') as f, out.open('x',encoding='utf-8') as fo:
            reader = csv.DictReader(f)
            required = [id_col,text_col,source_col,occurred_col,published_col]
            optional = [v for v in (location_col,concept_col,ontology_type_col,url_col,description_col) if v]
            missing = [k for k in required+optional if k not in (reader.fieldnames or [])]
            if missing: raise ValueError(f'CSV missing mapped columns: {missing}; columns are {reader.fieldnames}')
            seen_ids=set()
            for line_no,row in enumerate(reader,start=2):
                for key in required:
                    if not (row.get(key) or '').strip():
                        raise ValueError(f'Line {line_no}: missing required {key}; no evidence or dates may be invented')
                occurred = date.fromisoformat(row[occurred_col][:10]).isoformat()
                published = date.fromisoformat(row[published_col][:10]).isoformat()
                doc_id = row[id_col].strip()
                source = row[source_col].strip()
                # Keep original dates, including pre-announcement documents, without inventing a correction.
                identity=hashlib.sha256((source+'\0'+doc_id).encode()).hexdigest()[:20]
                event_id = f'REAL-{identity}'
                if event_id in seen_ids:
                    raise ValueError(f'Line {line_no}: repeated source+document id {source}/{doc_id}; merge or disambiguate upstream')
                seen_ids.add(event_id)
                evid_id = f'REAL-E-{identity}'
                text = row[text_col].strip()
                event = {'event_id':event_id,
                         'description':(row[description_col].strip() if description_col else text[:500]),
                         'occurred_at':occurred,'published_at':published,
                         'location':row[location_col].strip() if location_col else '',
                         'concept_ids':parse_values(row[concept_col]) if concept_col else [],
                         'ontology_types':parse_values(row[ontology_type_col]) if ontology_type_col else [],
                         'evidence':[{'evidence_id':evid_id,'source_id':source+'#'+doc_id,
                                      'text':text,'published_at':published,
                                      'url':row[url_col].strip() if url_col else ''}],
                         'metadata':{'source_file':str(input_csv),'source_line':line_no,
                                     'event_granularity':'one_evidence_row_per_event',
                                     'validation':'unvalidated_event_candidate',
                                     'published_before_occurrence': published < occurred}}
                fo.write(json.dumps(event,ensure_ascii=False)+'\n')
                total += 1
                if max_events and total >= max_events: break
    except Exception:
        out.unlink(missing_ok=True)
        raise
    print(f'Wrote {total} provenance-bearing event CANDIDATES to {out}')
    print('These are document-level candidates; linkage and epidemiological validity remain unevaluated.')
    return total


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--id-col',required=True)
    p.add_argument('--text-col',required=True)
    p.add_argument('--source-col',required=True)
    p.add_argument('--occurred-col',required=True)
    p.add_argument('--published-col',required=True)
    p.add_argument('--location-col',default='')
    p.add_argument('--concept-col',default='')
    p.add_argument('--ontology-type-col',default='')
    p.add_argument('--url-col',default='')
    p.add_argument('--description-col',default='')
    p.add_argument('--max-events',type=int,default=0)
    a=p.parse_args()
    convert(Path(a.input),Path(a.output),id_col=a.id_col,text_col=a.text_col,source_col=a.source_col,
            occurred_col=a.occurred_col,published_col=a.published_col,location_col=a.location_col,
            concept_col=a.concept_col,ontology_type_col=a.ontology_type_col,
            url_col=a.url_col,description_col=a.description_col,max_events=a.max_events)

if __name__=='__main__':main()
