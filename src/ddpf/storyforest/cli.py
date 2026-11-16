"""CLI for incremental story experiments, audit and evaluation.

Example: python -m ddpf.storyforest.cli run --input data/benchmark/events.jsonl --method full
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .settings import Settings
from .models import Event
from .storage import Store,forest
from .pipeline import process_event,METHODS
from .ontology import OntologyGuard
from .evaluate import read_jsonl,evaluate_pairs

def save_json(path:str,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main(argv=None)->int:
    p=argparse.ArgumentParser(prog='ddpf-storyforest',description='Experimental, NOT clinically validated')
    p.add_argument('--db',default=None,help='SQLite DB (overrides STORYFOREST_DB)')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('doctor')
    pr=sub.add_parser('run');pr.add_argument('--input',required=True,help='Event JSONL, streamed');pr.add_argument('--method',choices=sorted(METHODS),default='full');pr.add_argument('--rules',default='config/storyforest_rules.json');pr.add_argument('--output',default='outputs/storyforest/result.json');pr.add_argument('--top-k',type=int,default=None);pr.add_argument('--max-events',type=int,default=0)
    pe=sub.add_parser('evaluate');pe.add_argument('--gold',required=True);pe.add_argument('--method',choices=sorted(METHODS),default='full');pe.add_argument('--output',default='outputs/storyforest/evaluation.json')
    pa=sub.add_parser('audit-neo4j');pa.add_argument('--limit',type=int,default=40);pa.add_argument('--output',default='outputs/storyforest/neo4j_audit.json')
    sub.add_parser('serve').add_argument('--port',default=8765,type=int)
    a=p.parse_args(argv)
    if a.cmd=='doctor':
        s=Settings.from_env(require_key=False)
        print(json.dumps({'provider':s.provider,'model':s.litellm_model,'api_base':s.ollama_base_url if s.provider=='ollama' else s.openrouter_base_url,'openrouter_key_present':bool(s.openrouter_key), 'database':a.db or s.sqlite_path},indent=2))
        return 0
    if a.cmd=='audit-neo4j':
        from .audit_neo4j import audit
        print(audit(a.output,min(max(a.limit,1),200)));return 0
    if a.cmd=='serve':
        if a.db:
            import os
            os.environ['STORYFOREST_DB']=a.db
        import uvicorn
        from .api import app
        uvicorn.run(app,host='127.0.0.1',port=a.port);return 0
    s=Settings.from_env(require_key=False);dbpath=a.db or s.sqlite_path
    if a.cmd=='evaluate':
        with Store(dbpath) as store:
            results=evaluate_pairs(store.links(a.method),a.gold,method=a.method)
        save_json(a.output,results);print(json.dumps(results,indent=2));return 0
    llm=None
    if a.method in {'llm_only','llm_kg','full'}:
        from ddpf.llm.litellm_client import LiteLLMClient
        llm=LiteLLMClient(Settings.from_env(require_key=True))
    guard=OntologyGuard.from_file(a.rules)
    processed=0;created=0;decisions=[]
    with Store(dbpath) as store:
        for row in read_jsonl(a.input):
            event=Event.from_dict(row)
            outcome=process_event(store,event,a.method,llm,guard,
                                  top_k=a.top_k or s.max_candidates)
            processed+=1;created+=int(outcome.created)
            if outcome.decision:decisions.append(outcome.decision.to_dict())
            if a.max_events and processed>=a.max_events:break
        events=store.list_events(limit=500) # summary sample only
        links=store.links(a.method)
        result={'method':a.method,'input_events_seen':processed,'events_created':created,
                'total_events_in_db':store.count_events(),
                'stories_sample':forest(events,links),
                'sample_capped_at':500,'links_count':len(links),
                'note':'No claim of epidemiologically correct links. Inspect evidence and gold references.'}
    save_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='stories_sample'},indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
