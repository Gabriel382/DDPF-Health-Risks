#!/usr/bin/env python3
"""Generate reproducible observation embedding matrices from the KG aggregate observations.

Uses only Ollama's official /api/embed interface, not chat completions.
Never sends disease ground truth or r.source to the embedding model.
Resumable per model/feature/observation-set; atomic progress checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FEATURES=('concepts','concepts_country','concepts_time','concepts_country_time')
DISEASE_TERMS = re.compile(r'(?i)\b(?:covid(?:[\s-]?19)?|sars[\s-]?cov[\s-]?2|coronavirus|monkeypox|mpox|orthopoxvirus)\b')


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for buf in iter(lambda:stream.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()


def date_tokens(raw):
    text=' '.join(map(str,raw)) if isinstance(raw,list) else str(raw or '')
    return sorted(set(re.findall(r'\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b',text)))


def observation_text(obj,features,mask_disease_terms=False):
    # Never include source or disease label, which are available only for evaluation.
    concepts=[str(c.get('name') or c.get('id') or '').strip() for c in obj.get('top_concepts',[])]
    concepts=[c for c in concepts if c]
    if mask_disease_terms:
        concepts=[c for c in concepts if not DISEASE_TERMS.search(c)]
    if not concepts and not mask_disease_terms:
        raise ValueError(f"No concepts for {obj.get('observation_id','<unknown>')}")
    # If masking removes every concept, use a neutral placeholder rather than drop rows.
    parts=['Biomedical concepts: '+ ('; '.join(concepts) if concepts else '[none]')]
    if features in ('concepts_country','concepts_country_time'):
        parts.append('Country: '+str(obj.get('country_name') or obj.get('country_id') or 'unknown'))
    if features in ('concepts_time','concepts_country_time'):
        dates=date_tokens(obj.get('relation_dates',[]))
        if dates:parts.append('Reported periods: '+', '.join(dates))
    return '. '.join(parts)+'.'


def load_observations(path, limit):
    observations=[];ids=set()
    with Path(path).open('r',encoding='utf-8') as f:
        for line_no,line in enumerate(f,start=1):
            if not line.strip():continue
            obj=json.loads(line)
            oid=obj['observation_id']
            if oid in ids:raise ValueError(f'Duplicate observation_id {oid} in {path}')
            ids.add(oid)
            observations.append(obj)
            if limit and len(observations)>=limit:break
    if not observations:raise ValueError(f'No observations in {path}')
    return observations


def post_json(url,payload,timeout=180,attempts=4):
    data=json.dumps(payload).encode('utf-8')
    for attempt in range(attempts):
        try:
            with urlopen(Request(url,data=data,headers={'Content-Type':'application/json'},method='POST'), timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as e:
            detail=e.read().decode('utf-8','replace')[:500]
            if e.code not in (429,500,502,503,504) or attempt+1>=attempts:
                raise RuntimeError(f'HTTP {e.code} calling Ollama {url}: {detail}') from e
        except (URLError,TimeoutError) as e:
            if attempt+1>=attempts:raise RuntimeError(f'Ollama connection failed at {url}: {e}') from e
        time.sleep(min(2**attempt,10))
    raise RuntimeError('Exhausted Ollama retry budget.')


def get_tags(url):
    with urlopen(url+'/api/tags',timeout=25) as response:
        data=json.loads(response.read().decode('utf-8'))
    return {item['name']:item for item in data.get('models',[])}


def model_slug(name):
    return re.sub(r'[^a-zA-Z0-9._-]+','_',name)


def meta_key(path_hash, observed_ids, model, features, truncate,mask):
    payload={'sha256':path_hash,'ids':observed_ids,'model':model,'features':features,'truncate':truncate,'mask_disease_terms':mask,'version':1}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def embed_one(observations,source_file:Path,model:str,features:str,outroot:Path,url:str,batch_size:int,truncate:bool,timeout:int,mask_disease_terms=False):
    import numpy as np
    if features not in FEATURES:raise ValueError(f'Unknown features {features}')
    texts=[observation_text(o,features,mask_disease_terms) for o in observations]
    ids=[o['observation_id'] for o in observations]
    key=meta_key(file_hash(source_file),ids,model,features,truncate,mask_disease_terms)
    out=outroot/model_slug(model)/(features+('_masked' if mask_disease_terms else ''))
    out.mkdir(parents=True,exist_ok=True)
    matrix=out/'embeddings.npy'
    meta=out/'manifest.json'
    progress=out/'progress.json'
    ids_csv=out/'observations.csv'
    if matrix.exists() and not (meta.exists() or progress.exists()):
        raise FileExistsError(f'Orphaned matrix {matrix} without progress or manifest. Investigate before retrying.')
    if meta.exists():
        old=json.loads(meta.read_text(encoding='utf-8'))
        if old.get('key') != key:raise ValueError(f'Existing output has a different input or configuration: {out}')
        if not matrix.exists():raise FileNotFoundError(f'Missing matrix for completed run {out}')
        print(f'SKIP verified completed {model} {features}: {out}',flush=True)
        return out
    state={'key':key,'completed':0,'dimension':0}
    if progress.exists():
        state=json.loads(progress.read_text(encoding='utf-8'))
        if state.get('key')!=key:raise ValueError(f'Partial output {out} belongs to another dataset/configuration')
        if not matrix.exists():raise FileNotFoundError(f'Missing partial matrix {matrix}')
    completed=state['completed'];arr=None
    if state['dimension']:
        arr=np.lib.format.open_memmap(matrix,mode='r+')
        if arr.shape!=(len(observations),state['dimension']):raise ValueError('Resumed matrix dimensions mismatch')
    print(f'Embedding {model} | {features} | {len(ids)} rows | resuming at {completed}',flush=True)
    for start in range(completed,len(texts),batch_size):
        end=min(start+batch_size,len(texts))
        response=post_json(url+'/api/embed',{'model':model,'input':texts[start:end],
                                             'truncate':truncate},timeout)
        vectors=np.asarray(response.get('embeddings',[]),dtype=np.float32)
        if vectors.ndim!=2 or vectors.shape[0]!=(end-start) or vectors.shape[1]<1:
            raise ValueError(f'Ollama returned wrong matrix shape {vectors.shape}, expected {end-start} rows')
        if not np.isfinite(vectors).all():raise ValueError('NaN or infinity in model embeddings')
        if arr is None:
            arr=np.lib.format.open_memmap(matrix,mode='w+',dtype='float32',shape=(len(ids),vectors.shape[1]))
            state['dimension']=int(vectors.shape[1])
        elif arr.shape[1]!=vectors.shape[1]:raise ValueError('Ollama embedding dimension changed mid-run')
        arr[start:end]=vectors
        arr.flush()
        # Atomic checkpoint only after flushed data.
        state['completed']=end
        tmp=progress.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
        tmp.replace(progress)
        print(f'{model} {features}: {end}/{len(ids)}',flush=True)
    if arr is None:raise ValueError('No vectors produced')
    with ids_csv.open('w',encoding='utf-8',newline='') as f:
        import csv
        writer=csv.writer(f)
        writer.writerow(['row','observation_id','country_id','source','disease_label','date_tokens'])
        for i,o in enumerate(observations):
            writer.writerow([i,o['observation_id'],o.get('country_id',''),o.get('source',''),
                             o.get('disease_label',''),';'.join(date_tokens(o.get('relation_dates',[])))])
    with source_file.open('r',encoding='utf-8') as f:
        input_rows=sum(1 for line in f if line.strip())
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'key':key,
              'method':'ollama_text_embedding','model':model,'features':features,
              'ollama_url':url,'source_jsonl':str(source_file),'source_sha256':file_hash(source_file),
              'n_observations':len(observations),'dimension':int(arr.shape[1]),
              'output_matrix':str(matrix),'row_alignment':str(ids_csv),
              'truncate':truncate,'mask_disease_terms':mask_disease_terms,'sample':len(observations)!=input_rows,
              'notes':['Disease label and source are not provided in the embedding input.',
                       'These are country-source aggregate observations, not individual events.',
                       'They are text embeddings, not structural graph embeddings such as FastRP.']}
    tmp=meta.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    tmp.replace(meta)
    progress.unlink(missing_ok=True)
    print('DONE',model,features,arr.shape,flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/realdata/reconstructed_observations.jsonl'))
    p.add_argument('--outroot',type=Path,default=Path('outputs/realdata/embeddings'))
    p.add_argument('--models',nargs='+',required=True,help='Exact Ollama tags, e.g. qwen3-embedding:0.6b')
    p.add_argument('--features',choices=FEATURES,default='concepts')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int,default=0,help='Smoke test cap, recorded as sample in manifest')
    p.add_argument('--truncate',action='store_true',help='Allow truncation of overly long inputs')
    p.add_argument('--mask-disease-terms',action='store_true',help='Exclude explicit COVID/Mpox concept names as leakage ablation')
    p.add_argument('--timeout',type=int,default=240)
    a=p.parse_args()
    if a.batch_size<1 or a.limit<0:p.error('Invalid batch or limit')
    try:
        from dotenv import load_dotenv
        load_dotenv(Path('.env'))
    except ImportError:
        pass
    url=os.getenv('OLLAMA_BASE_URL','http://localhost:11434').rstrip('/')
    tags=get_tags(url)
    for model in a.models:
        if model not in tags and not (model+':latest' in tags):
            raise SystemExit(f'Ollama model not installed: {model}. Run: ollama pull {model}\nAvailable: {list(tags)}')
    rows=load_observations(a.input,a.limit)
    for model in a.models:
        embed_one(rows,a.input,model,a.features,a.outroot,url,a.batch_size,a.truncate,a.timeout,a.mask_disease_terms)

if __name__=='__main__':main()
