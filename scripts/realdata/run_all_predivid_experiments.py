#!/usr/bin/env python3
"""Reproducible, resumable, log-backed PreDiViD experiment suite.

Run from repository root inside tmux. Performs text ablations, projected KG
structural baselines, robust clustering benchmarks, and read-only evidence audit.
Story linking on real data is blocked unless a DOCUMENTED event JSONL exists.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen

DEFAULT_MODELS=['qwen3-embedding:0.6b','embeddinggemma','nomic-embed-text']
FEATURES=['concepts','concepts_country','concepts_time','concepts_country_time']
METHODS=['temporal','kg_rules','llm_only','llm_kg','full']


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def ollama_tags(base):
    with urlopen(base.rstrip('/')+'/api/tags',timeout=15) as resp:
        return {x['name'] for x in json.load(resp).get('models',[])}


def preflight_events(path):
    if not path:return 'no_real_event_jsonl_supplied'
    dates=[];n=0;source_set=set();evidence_n=0
    with Path(path).open(encoding='utf8') as f:
        from datetime import date
        for line in f:
            if not line.strip():continue
            e=json.loads(line)
            for key in ('event_id','description','occurred_at','published_at'):
                if not isinstance(e.get(key),str) or not e[key].strip():raise ValueError('Event missing '+key)
            for key in ('occurred_at','published_at'):
                date.fromisoformat(e[key][:10])
            n+=1;dates.append(e['published_at'])
            for evidence in e.get('evidence',[]):
                if evidence.get('source_id') and str(evidence.get('text') or '').strip() and evidence.get('published_at'):
                    date.fromisoformat(str(evidence['published_at'])[:10])
                    evidence_n+=1;source_set.add(evidence['source_id'])
    if n<2 or evidence_n<2 or len(source_set)<2:
        raise ValueError('Real Story Forest requires >=2 events with >=2 documentary evidence records from >=2 sources. Not satisfied.')
    if dates!=sorted(dates):
        raise ValueError('Event JSONL must be chronological by publication date to support incremental evaluation.')
    return {'events':n,'evidence_records':evidence_n,'sources':len(source_set),
            'caution':'Evidence presence does not imply independent verification of link correctness.'}


def runner(cmd,logs,slug,root):
    path=logs/(slug+'.log')
    print('\n===',slug,'===',flush=True)
    print('COMMAND:', ' '.join(map(str,cmd)),flush=True)
    with path.open('w',encoding='utf8') as log:
        log.write('COMMAND: '+' '.join(map(str,cmd))+'\n');log.flush()
        proc=subprocess.Popen(list(map(str,cmd)),cwd=root,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,bufsize=1,text=True,errors='replace')
        try:
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line);log.flush()
            rc=proc.wait()
        except BaseException:
            proc.terminate()
            raise
    if rc:raise RuntimeError(f'Stage {slug} failed with exit code {rc}; see {path}')
    return {'cmd':list(map(str,cmd)),'log':str(path),'status':'ok'}


def run(a):
    root=Path.cwd().resolve()
    if not (root/'src/ddpf/storyforest').exists() or not (root/'scripts/realdata/embed_kg_observations.py').is_file():
        raise FileNotFoundError('Run this from DDPF-Health-Risks root with the previous StoryForest and embedding packages installed')
    source=root/'outputs/realdata/reconstructed_observations.jsonl'
    if not source.is_file():raise FileNotFoundError(str(source)+' missing; run recover_kg_observations.py first')
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    base=os.getenv('OLLAMA_BASE_URL','http://127.0.0.1:11434')
    labels=DEFAULT_MODELS[:]
    if a.with_qwen4b:labels.append('qwen3-embedding:4b')
    if a.models:labels=a.models
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=(root/a.output_root/stamp).resolve()
    out.mkdir(parents=True,exist_ok=False)
    logs=out/'logs';logs.mkdir()
    status={'timestamp_utc':stamp,'project_root':str(root),'python':sys.version,
            'input_path':str(source),'input_sha256':hash_file(source),
            'models':labels,'stages':{},'storyforest_real_status':'not_evaluated',
            'warnings':['This reuses country-by-source aggregates, NOT extracted source-document events.',
                        'FastRP-style and GraphSAGE-proxy operate on a bounded projection; they are not original Neo4j GDS results.',
                        'No epidemiological link accuracy is claimed without independent documented gold.']}
    file=out/'suite_manifest.json'
    def persist():
        file.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
    persist()
    try:
        if not a.skip_text:
            installed=ollama_tags(base)
            missing=[x for x in labels if x not in installed and x+':latest' not in installed]
            if missing and not a.pull_missing:
                raise RuntimeError(f'Missing Ollama models: {missing}. Run `ollama pull ...` or rerun with --pull-missing.')
            for model in missing:
                status['stages']['pull_'+model]=runner(['ollama','pull',model],logs,'pull_'+re.sub(r'[^A-Za-z0-9]','_',model),root);persist()
            for model in labels:
              for feature in FEATURES:
                for masked in [False,True]:
                    name='text__'+re.sub(r'[^A-Za-z0-9]','_',model)+'__'+feature+('__masked' if masked else '')
                    cmd=[sys.executable,'scripts/realdata/embed_kg_observations.py','--models',model,
                         '--features',feature,'--batch-size',str(a.batch_size)]
                    if masked:cmd+=['--mask-disease-terms']
                    status['stages'][name]=runner(cmd,logs,name,root);persist()
        else:
            status['stages']['text']='skipped_by_user';persist()
        if not a.skip_structural:
            cmd=[sys.executable,'scripts/realdata/structural_observation_embeddings.py']
            if a.strict_structural:cmd+=['--strict']
            status['stages']['structural']=runner(cmd,logs,'structural',root);persist()
        else:
            status['stages']['structural']='skipped_by_user';persist()
        if not a.skip_benchmark:
            cmd=[sys.executable,'scripts/realdata/extended_embedding_benchmark.py',
                 '--out',str(out/'benchmark'), '--bootstrap',str(a.bootstrap),
                 '--permutations',str(a.permutations)]
            status['stages']['benchmark']=runner(cmd,logs,'benchmark',root);persist()
            status['results_csv']=str(out/'benchmark/results.csv');persist()
        if not a.skip_source_audit:
            cmd=[sys.executable,'scripts/realdata/storyforest_source_audit.py',
                 '--out',str(out/'storyforest_source_audit.json')]
            status['stages']['source_audit']=runner(cmd,logs,'source_audit',root);persist()
        if a.real_events:
            event_info=preflight_events(a.real_events)
            if any(x in a.story_methods for x in ['llm_only','llm_kg','full']) and not a.allow_llm:
                raise ValueError('LLM methods require --allow-llm (OpenRouter could incur charges).')
            status['storyforest_real_input']={'path':str(a.real_events),'sha256':hash_file(a.real_events),**event_info}
            status['storyforest_real_status']='engineering_run_not_independent_evaluation'
            persist()
            for method in a.story_methods:
                command=[sys.executable,'-m','ddpf.storyforest.cli',
                         '--db',str(out/f'storyforest_{method}.sqlite3'), 'run','--input',str(a.real_events),
                         '--method',method,'--output',str(out/f'storyforest_{method}.json')]
                if a.max_story_events:command+=['--max-events',str(a.max_story_events)]
                status['stages']['storyforest_'+method]=runner(command,logs,'storyforest_'+method,root);persist()
                if a.gold:
                    eval_cmd=[sys.executable,'-m','ddpf.storyforest.cli',
                              '--db',str(out/f'storyforest_{method}.sqlite3'), 'evaluate','--gold',str(a.gold),
                              '--method',method,'--output',str(out/f'storyforest_{method}_eval.json')]
                    status['stages']['storyforest_evaluate_'+method]=runner(eval_cmd,logs,'storyforest_evaluate_'+method,root);persist()
        else:
            status['storyforest_real_status']='blocked_missing_document_provenance_event_jsonl'
            status['stages']['storyforest_real']='not_run_no_document_level_event_dataset'
            persist()
        status['run_state']='completed';persist()
    except BaseException as ex:
        status['run_state']='failed'
        status['last_error']=type(ex).__name__+': '+str(ex)
        persist();raise
    print('\nALL AVAILABLE EXPERIMENTS COMPLETED:',file,flush=True)
    if status.get('results_csv'):print('BENCHMARK:',status['results_csv'],flush=True)
    if not a.real_events:print('STORY FOREST REAL: skipped (document-level event evidence not supplied)',flush=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',default='outputs/realdata/experiment_suites')
    p.add_argument('--models',nargs='+',default=None)
    p.add_argument('--with-qwen4b',action='store_true')
    p.add_argument('--pull-missing',action='store_true')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--strict-structural',action='store_true')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--permutations',type=int,default=200)
    p.add_argument('--skip-text',action='store_true')
    p.add_argument('--skip-structural',action='store_true')
    p.add_argument('--skip-benchmark',action='store_true')
    p.add_argument('--skip-source-audit',action='store_true')
    p.add_argument('--real-events',type=Path,default=None)
    p.add_argument('--gold',type=Path,default=None)
    p.add_argument('--story-methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--allow-llm',action='store_true')
    p.add_argument('--max-story-events',type=int,default=0)
    a=p.parse_args()
    if a.gold and not a.real_events:p.error('--gold requires --real-events')
    if a.max_story_events<0 or a.batch_size<1:p.error('Invalid batch size or cap')
    run(a)

if __name__=='__main__':main()
