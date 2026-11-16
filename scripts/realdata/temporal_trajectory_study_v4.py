#!/usr/bin/env python3
"""Temporal trajectory STUDY on aggregated country-week Neo4j observations.

This is NOT a documentary Story Forest evaluation. We compare the ability of
temporal/KG/LLM models to anticipate whether the NEXT observed weekly snapshot
will have a higher, equal or lower reported_value.

Input features contain only the history available at the origin week. The
target value is never passed to a model. Labels refer to observed changes,
NOT validated incidence, transmission, disease classification or event links.

Uses exact snapshots.csv fields from build_weekly_episodes.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import socket
import time

import numpy as np

CLASSES = ('increase', 'flat', 'revision')
PRIMARY = ('increase', 'flat')
SNAPSHOT_FIELDS = {
    'episode_id','disease','disease_id','country_id','country_name',
    'week_start','observation_date','reported_value','source_relation_id',
    'previous_episode_id','difference_from_previous_snapshot'
}
PROMPT_SYSTEM = (
    'Task: predict whether the NEXT weekly reported aggregate count will '
    'increase, stay flat, or decrease due to a reporting revision. '
    'Choose exactly one of increase, flat, revision. Use ONLY the historic '
    'reported values in the input. No future values are given. '
    'A forecast is uncertain, but still choose the most plausible class. '
    'Use unknown ONLY if fewer than two valid historic values are given. '
    'Do not claim infection incidence, transmission, causation, or document '
    'continuation. Return ONLY one short JSON object with one key: '
    '{"prediction":"flat"}. Do not include a reason or any other field. '
    '/no_think'
)


@dataclass(frozen=True)
class Snapshot:
    episode_id: str
    disease: str
    disease_id: str
    country_id: str
    country: str
    week: date
    observation_date: str
    value: float
    source_relation_id: str
    previous_episode_id: str

@dataclass
class Example:
    origin: Snapshot
    target: Snapshot
    history: list[Snapshot]
    target_label: str
    features: list[float]
    semantic_keys: set[str]
    semantic_names: list[str]
    semantic_sources: tuple[str,...]

def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for b in iter(lambda: fh.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load_snapshots(path: Path) -> dict[tuple[str,str],list[Snapshot]]:
    series = defaultdict(list)
    seen = set()
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        missing = SNAPSHOT_FIELDS-set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing columns: '+repr(sorted(missing)))
        for r in reader:
            d=date.fromisoformat(r['week_start'])
            val=float(r['reported_value'])
            if not math.isfinite(val) or val<0:
                raise ValueError('Invalid snapshot value '+r['episode_id'])
            if d.weekday()!=0:
                raise ValueError('Not a Monday: '+r['episode_id'])
            k=(r['disease_id'],r['country_id'],d)
            if k in seen:raise ValueError('Duplicate episode: '+repr(k))
            seen.add(k)
            series[k[:2]].append(Snapshot(
                r['episode_id'],r['disease'],r['disease_id'],r['country_id'],
                r['country_name'],d,r['observation_date'],val,
                r['source_relation_id'],r['previous_episode_id']))
    for k,v in series.items():
        v.sort(key=lambda x:x.week)
        for i,s in enumerate(v):
            if i and (s.week-v[i-1].week).days != 7:
                raise ValueError('Gap in '+repr(k)+' at '+s.week.isoformat()+
                                 '; no interpolation permitted')
            if i and s.previous_episode_id != v[i-1].episode_id:
                raise ValueError('Incorrect previous_episode_id: '+s.episode_id)
    return dict(series)

def load_static_profiles(path: Path) -> dict[tuple[str,str],dict]:
    """Country + disease static profiles collected in 2019/2020 or 2022.

    NEVER attributed to a particular week. Can be used only for periods
    after all source collection dates, as historical/static context.
    """
    profiles=defaultdict(lambda:{'ids':set(),'names':[],'sources':set(),'dates':set()})
    if not path.exists():return {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            disease='C5203670' if str(row['disease_label'])=='0' else 'C0276180'
            k=(disease,str(row['country_id']))
            p=profiles[k]
            p['sources'].add(row['source'])
            for d in row.get('relation_dates',[]):
                if isinstance(d,str) and re.fullmatch(r'\d{4}-\d{2}',d):
                    p['dates'].add(d)
            for c in row.get('top_concepts',[]):
                if c.get('id'):
                    p['ids'].add(str(c['id']))
                if c.get('name') and len(p['names'])<30:
                    p['names'].append(str(c['name'])[:80])
    return {k:dict(v) for k,v in profiles.items()}

def lookup_profile(origin: Snapshot, profiles:dict) -> tuple[set[str],list[str],tuple[str,...]]:
    p=profiles.get((origin.disease_id,origin.country_id))
    if p is None:return set(), [], tuple()
    # Date restriction: no future-derived static features. Source-specific
    # relation_dates are metadata and may be absent. Max known date verified
    # against the origin date if present.
    origin_month=origin.week.isoformat()[:7]
    if p['dates'] and max(p['dates']) > origin_month:
        return set(), [], tuple()
    return set(p['ids']), list(dict.fromkeys(p['names']))[:20], tuple(sorted(p['sources']))

def direction(diff:float) -> str:
    if diff < -1e-9:return 'revision'
    if diff > 1e-9:return 'increase'
    return 'flat'

def history_features(history:list[Snapshot]) -> list[float]:
    """Features up to and INCLUDING origin, never the next-week target."""
    curr=history[-1]
    changes=[history[i].value-history[i-1].value for i in range(1,len(history))]
    recent=(changes[-4:] if changes else [])
    pos=sum(v>0 for v in recent)
    zero=sum(abs(v)<1e-9 for v in recent)
    neg=sum(v<0 for v in recent)
    latest=changes[-1] if changes else 0.
    older=changes[-2] if len(changes)>1 else 0.
    return [
        math.log1p(curr.value),
        math.copysign(math.log1p(abs(latest)), latest),
        math.copysign(math.log1p(abs(older)), older),
        (pos/max(len(recent),1)),
        (zero/max(len(recent),1)),
        (neg/max(len(recent),1)),
        math.log1p(len(history)),
        # Calendar features available as of origin.
        math.sin(2*math.pi*curr.week.timetuple().tm_yday/365.25),
        math.cos(2*math.pi*curr.week.timetuple().tm_yday/365.25),
    ]

def make_examples(series:dict, profiles:dict, min_history:int=4) -> list[Example]:
    out=[]
    for key,rows in sorted(series.items()):
        for i in range(min_history-1,len(rows)-1):
            origin,target=rows[i],rows[i+1]
            if (target.week-origin.week).days != 7:continue
            hist=rows[:i+1]
            sem, names, sources=lookup_profile(origin,profiles)
            out.append(Example(
                origin,target,hist,direction(target.value-origin.value),
                history_features(hist),sem,names,sources))
    return sorted(out,key=lambda x:(x.target.week,x.target.disease_id,x.origin.country_id))

def m1_persistence(ex: Example) -> str:
    h=ex.history
    if len(h)<2:return 'unknown'
    old=direction(h[-1].value-h[-2].value)
    # A past negative revision is not evidence of another future revision.
    return 'flat' if old=='revision' else old

def m2_knn_predict(train:list[Example], test:list[Example], k:int=9, use_semantic=True):
    """Country-independent past-only kNN, with optional STATIC KG overlap.

    Not an implementation of Story Forest. A supervised temporal trajectory
    comparator. Training examples have target weeks < holdout start.
    """
    if not train:raise ValueError('No training examples for temporal KNN')
    X=np.array([x.features for x in train],dtype=np.float64)
    mean=X.mean(axis=0)
    sd=X.std(axis=0)
    sd[sd<1e-8]=1.
    X=(X-mean)/sd
    Y=np.array([v.features for v in test],dtype=np.float64)
    Y=(Y-mean)/sd
    by_disease=defaultdict(list)
    for i,ex in enumerate(train):by_disease[ex.origin.disease_id].append(i)
    predictions=[]
    for test_ix,ex in enumerate(test):
        indexes=by_disease.get(ex.origin.disease_id,[])
        if not indexes:
            predictions.append('unknown')
            continue
        arr=np.asarray(indexes,dtype=int)
        dist=np.linalg.norm(X[arr]-Y[test_ix],axis=1)/math.sqrt(X.shape[1])
        order=np.argsort(dist,kind='stable')[:min(max(k*15,80),len(dist))]
        scored=[]
        for o in order:
            j=int(arr[o])
            rank=float(dist[o])
            # If either profile missing, do not invent a similarity score.
            if use_semantic and ex.semantic_keys and train[j].semantic_keys:
                a,b=ex.semantic_keys,train[j].semantic_keys
                rank += 0.3 * (1-len(a&b)/max(len(a|b),1))
            scored.append((rank,j))
        choices=[train[j].target_label for _,j in sorted(scored)[:k]]
        votes=Counter(choices)
        # Only report revision if observed in neighbors; rare class.
        result=sorted(votes,key=lambda label:(-votes[label],('flat','increase','revision').index(label)))[0]
        predictions.append(result)
    return predictions

def stable_sample(examples:list[Example],per_disease:int)->list[Example]:
    by=defaultdict(list)
    for x in examples:by[x.origin.disease_id].append(x)
    chosen=[]
    for disease,examples_d in sorted(by.items()):
        examples_d.sort(key=lambda e:hashlib.sha256(e.origin.episode_id.encode()).hexdigest())
        chosen.extend(examples_d[:per_disease])
    return sorted(chosen,key=lambda x:(x.origin.disease_id,x.target.week,x.origin.country_id))

def safe_json_response(content: str) -> dict:
    """Parse one valid JSON response including fenced or reasoning-prefixed text."""
    cleaned = (content or '').strip()
    if not cleaned:
        return {'prediction': 'unknown', 'reason': 'empty_model_content'}
    decoder = json.JSONDecoder()
    valid = []
    # Exact JSON is preferred; otherwise locate embedded JSON objects.
    try:
        out = json.loads(cleaned)
        valid = [out]
    except (ValueError, TypeError):
        for m in re.finditer(r'\{', cleaned):
            try:
                out, end = decoder.raw_decode(cleaned, m.start())
            except ValueError:
                continue
            # Reject if two distinct JSON objects appear in the answer.
            if isinstance(out, dict):
                valid.append(out)
        if len(valid) != 1:
            return {'prediction': 'unknown', 'reason': 'invalid_or_ambiguous_json'}
    out = valid[0]
    if not isinstance(out, dict) or out.get('prediction') not in (*CLASSES, 'unknown'):
        return {'prediction': 'unknown', 'reason': 'invalid_response'}
    return {'prediction': out['prediction'], 'reason': str(out.get('reason', ''))[:300]}

def payload(ex:Example, with_kg:bool=False)->dict:
    h=ex.history[-6:]
    result={
        'task':'anticipate_next_week_change_of_aggregate_reporting_snapshot',
        'disease':ex.origin.disease,
        'country':ex.origin.country,
        'origin_week':ex.origin.week.isoformat(),
        'history':[{'week':v.week.isoformat(),'reported_value':v.value}
                   for v in h],
        'target_week':ex.target.week.isoformat(),
        'target_value':'HIDDEN_FOR_EVALUATION',
        'warning':'Reported values not independently validated as incidence.'
    }
    if with_kg:
        result['graph_context']={
            'country_id':ex.origin.country_id,
            'disease_concept_id':ex.origin.disease_id,
            'static_historical_concepts':ex.semantic_names,
            'static_sources':list(ex.semantic_sources),
            'graph_context_is_week_specific':False,
            'current_snapshot_relation_id':ex.origin.source_relation_id,
            'granularity':'country_week_reporting_snapshot'
        }
        result['guardrails']=[
            'Cross-disease event linking prohibited',
            'No forecast using future observed values',
            'No claim of causal propagation or documentary continuity',
            'This is NOT OWL reasoning or a clinical claim'
        ]
    return result

def llm_predict(ex:Example,method:str,llm: Any):
    with_kg=method=='M4_llm_kg_guard'
    if with_kg and not ex.semantic_keys:
        return {'prediction':'unknown','reason':'missing_static_kg_profile'}
    raw=llm.chat([
        {'role':'system','content':PROMPT_SYSTEM},
        {'role':'user','content':json.dumps(payload(ex,with_kg),ensure_ascii=False)}
    ])
    parsed=safe_json_response(raw)
    if parsed['prediction'] == 'unknown' and parsed['reason'] in (
            'empty_model_content', 'invalid_or_ambiguous_json', 'invalid_response'):
        raise RuntimeError('Invalid Ollama response: ' + parsed['reason'] +
                           '. Review model thinking/content settings before retrying.')
    if with_kg:
        # The actual ontology is NOT reasoned over. Verify only explicit
        # typed constraints and source relationship provenance.
        if not ex.origin.disease_id or not ex.origin.country_id or not ex.origin.source_relation_id:
            return {'prediction':'unknown','reason':'missing_required_typed_provenance'}
    return parsed

def load_cache(path:Path)->dict:
    cache={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item=json.loads(line)
                cache[(item['method'],item['episode_id'])]=item
    return cache

class OllamaNativeClient:
    """Native Ollama with a compact classification response and safe retry.

    On a `length` stop, retry with a larger budget. A truncated response is
    NEVER used and never cached, even if it happens to contain valid JSON.
    """
    def __init__(self, settings: Any, timeout_seconds: int, max_tokens: int,
                 num_ctx: int = 4096, max_output_tokens: int = 1024):
        if settings.provider != 'ollama':
            raise ValueError('Native Ollama backend requires LLM_PROVIDER=ollama')
        if max_output_tokens < max_tokens:
            raise ValueError('max_output_tokens must be >= max_tokens')
        self.url = settings.ollama_base_url.rstrip('/') + '/api/chat'
        self.model = settings.model.removeprefix('ollama/')
        self.temperature = settings.temperature
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.num_ctx = num_ctx
        self.last_stats = {}

    def request_body(self, messages: list[dict[str, str]], budget: int | None = None) -> dict:
        return {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'think': False,
            'format': 'json',
            'keep_alive': '10m',
            'options': {
                'temperature': self.temperature,
                'num_predict': self.max_tokens if budget is None else budget,
                'num_ctx': self.num_ctx,
            },
        }

    def _post(self, body: dict) -> dict:
        req = Request(self.url, data=json.dumps(body).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read(400).decode('utf-8', errors='replace')
            raise RuntimeError(f'Ollama API HTTP {exc.code}: {detail}') from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f'Ollama API connection/timeout error after up to '
                               f'{self.timeout_seconds}s: {exc}') from exc
        try:
            result = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise RuntimeError('Ollama returned malformed response JSON') from exc
        if not isinstance(result, dict) or not isinstance(result.get('message'),dict):
            raise RuntimeError('Ollama response lacks message')
        return result

    def chat(self, messages: list[dict[str, str]]) -> str:
        budget = self.max_tokens
        attempted = []
        while True:
            result = self._post(self.request_body(messages,budget))
            attempted.append(budget)
            self.last_stats = {k: result.get(k) for k in (
                'done_reason','total_duration','load_duration','prompt_eval_count',
                'prompt_eval_duration','eval_count','eval_duration')}
            self.last_stats['num_predict_used'] = budget
            self.last_stats['budgets_attempted'] = list(attempted)
            content = result['message'].get('content') or ''
            if result.get('done_reason') == 'length':
                if budget >= self.max_output_tokens:
                    raise RuntimeError('Ollama output truncated even at num_predict=' +
                                       str(budget) + '; incomplete output not cached')
                next_budget = min(budget * 2, self.max_output_tokens)
                print(f'Ollama response truncated at {budget} tokens; retrying with '
                      f'{next_budget} (discarding incomplete output)',flush=True)
                budget = next_budget
                continue
            return content.strip()


def run_llm(examples:list[Example],out:Path,skip_llm=False,signature_context=None,
            llm_max_tokens:int=192, llm_timeout_seconds:int=240,
            ollama_num_ctx:int=4096, llm_backend:str='ollama_native',
            llm_retries:int=0, llm_max_output_tokens:int=1024):
    if skip_llm:
        return {}
    cache_file=out/'llm_predictions.jsonl'
    from ddpf.storyforest.settings import Settings
    from dataclasses import replace
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path('.env'),override=False)
    settings=replace(Settings.from_env(require_key=True), max_tokens=llm_max_tokens,
                     timeout=llm_timeout_seconds)
    if llm_backend=='ollama_native':
        llm=OllamaNativeClient(settings,timeout_seconds=llm_timeout_seconds,
                               max_tokens=llm_max_tokens,num_ctx=ollama_num_ctx,
                               max_output_tokens=llm_max_output_tokens)
    elif llm_backend=='litellm':
        from ddpf.llm.litellm_client import LiteLLMClient
        llm=LiteLLMClient(settings)
    else:
        raise ValueError('Unknown LLM backend: '+llm_backend)
    model_signature={
        'study_version':'v4_short_json_adaptive_budget',
        'backend':llm_backend,
        'think':False if llm_backend=='ollama_native' else 'backend_default',
        'format':'json' if llm_backend=='ollama_native' else 'backend_default',
        'num_ctx':ollama_num_ctx if llm_backend=='ollama_native' else None,
        'max_tokens':settings.max_tokens,
        'max_output_tokens':llm_max_output_tokens if llm_backend=='ollama_native' else None,
        'provider':settings.provider,
        'model':settings.litellm_model,
        'temperature':settings.temperature,
        'task':'next_week_reported_value_direction',
        'system_prompt_sha256':hashlib.sha256(PROMPT_SYSTEM.encode()).hexdigest(),
        **(signature_context or {})
    }
    signature=hashlib.sha256(json.dumps(model_signature,sort_keys=True).encode()).hexdigest()
    cache=load_cache(cache_file)
    if cache:
        signatures={v.get('signature') for v in cache.values()}
        if signatures != {signature}:
            raise RuntimeError('Cache model/prompt/input mismatch. Use NEW --output directory; '
                               'no stale LLM predictions reused.')
    print(f'LLM backend={llm_backend}, model={settings.litellm_model}, '
          f'num_predict_initial={llm_max_tokens}, adaptive_max={llm_max_output_tokens}, timeout={llm_timeout_seconds}s, '
          f'cached={len(cache)}',flush=True)
    with cache_file.open('a',encoding='utf-8') as f:
        for ex in examples:
            for method in ('M3_llm_only','M4_llm_kg_guard'):
                key=(method,ex.origin.episode_id)
                if key in cache: continue
                for attempt in range(llm_retries+1):
                    started=time.monotonic()
                    try:
                        ans=llm_predict(ex,method,llm)
                        break
                    except Exception as err:
                        if attempt >= llm_retries:
                            raise RuntimeError(
                                f'LLM failed at {key} after {attempt+1} attempt(s): {err}. '
                                'Completed predictions remain cached; rerun the same command '
                                'after checking Ollama.') from err
                        print(f'LLM attempt {attempt+1} failed for {key}: {err}; '
                              'retrying...',flush=True)
                elapsed=round(time.monotonic()-started,3)
                record={'method':method,'episode_id':ex.origin.episode_id,
                        'target_episode_id':ex.target.episode_id,
                        'prediction':ans['prediction'],'reason':ans['reason'],
                        'signature':signature,'model':settings.litellm_model,
                        'backend':llm_backend,'elapsed_seconds':elapsed}
                if isinstance(llm,OllamaNativeClient):
                    # Metadata only; do not store model internal thinking.
                    record['ollama_stats']=llm.last_stats
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
                cache[key]=record
                print(f'{method}: {len(cache)} cached, prediction={ans["prediction"]}, '
                      f'elapsed={elapsed}s, reason={ans["reason"][:70]!r}, '
                      f'origin={ex.origin.episode_id}',flush=True)
    return cache

def metrics(rows:list[dict],method:str)->dict:
    result={'method':method,'n_total':len(rows),
            'n_true_revision':sum(r['target_label']=='revision' for r in rows),
            'n_abstain':sum(r['prediction']=='unknown' for r in rows)}
    result['coverage']=round(1-result['n_abstain']/len(rows),6) if rows else None
    primary=[r for r in rows if r['target_label'] in PRIMARY]
    result['n_primary']=len(primary)
    cm={l:{p:0 for p in (*PRIMARY,'revision','unknown')} for l in PRIMARY}
    for r in primary:
        cm[r['target_label']][r['prediction']]+=1
    result['confusion']=cm
    if not primary:
        result.update(macro_f1=None,balanced_accuracy=None,accuracy=None)
        return result
    f1=[];recall=[]
    for label in PRIMARY:
        tp=cm[label][label]
        fn=sum(cm[label].values())-tp
        fp=sum(cm[other][label] for other in PRIMARY if other!=label)
        f1.append(2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0.)
        recall.append(tp/(tp+fn) if tp+fn else 0.)
    result.update(
        macro_f1=round(sum(f1)/len(f1),6),
        balanced_accuracy=round(sum(recall)/len(recall),6),
        accuracy=round(sum(r['prediction']==r['target_label'] for r in primary)/len(primary),6))
    return result

def save_csv(path:Path,rows:list[dict]):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshots',default='outputs/realdata/who_series/weekly/snapshots.csv')
    ap.add_argument('--profiles',default='outputs/realdata/reconstructed_observations.jsonl')
    ap.add_argument('--output',default='outputs/realdata/temporal_trajectory_study_v4')
    ap.add_argument('--holdout-start',default='2024-01-01')
    ap.add_argument('--max-llm-per-disease',type=int,default=20)
    ap.add_argument('--skip-llm',action='store_true')
    ap.add_argument('--knn-k',type=int,default=9)
    ap.add_argument('--llm-max-tokens',type=int,default=256,
                    help='Initial output token cap for short JSON class')
    ap.add_argument('--llm-max-output-tokens',type=int,default=1024,
                    help='Upper bound for automatic retries after truncated output')
    ap.add_argument('--llm-timeout-seconds',type=int,default=240)
    ap.add_argument('--ollama-num-ctx',type=int,default=4096)
    ap.add_argument('--llm-backend',choices=['ollama_native','litellm'],default='ollama_native')
    ap.add_argument('--llm-retries',type=int,default=0,
                    help='Extra tries on exception; completed outputs are always cached')
    ap.add_argument('--sample-all',action='store_true',help='Disable KG-coverage filtering; M4 may abstain')
    args=ap.parse_args(argv)
    if args.max_llm_per_disease<0:ap.error('max-llm-per-disease must be >= 0')
    if args.llm_max_tokens<64:ap.error('llm-max-tokens must be >= 64')
    if args.llm_max_output_tokens < args.llm_max_tokens:ap.error('llm-max-output-tokens must be >= llm-max-tokens')
    if args.llm_timeout_seconds < 10:ap.error('llm-timeout-seconds must be >= 10')
    if args.ollama_num_ctx < 512:ap.error('ollama-num-ctx must be >= 512')
    if args.llm_retries < 0:ap.error('llm-retries must be >= 0')
    inp=Path(args.snapshots);profile_path=Path(args.profiles);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    cutoff=date.fromisoformat(args.holdout_start)
    series=load_snapshots(inp)
    profiles=load_static_profiles(profile_path)
    examples=make_examples(series,profiles)
    training=[ex for ex in examples if ex.target.week < cutoff]
    holdout=[ex for ex in examples if ex.target.week >= cutoff]
    if not training or not holdout:
        raise RuntimeError('Need nonempty chronological train and holdout.')
    # Keep only training labels whose target weeks precede the holdout.
    m2_all=m2_knn_predict(training,holdout,k=args.knn_k)
    m2_no_kg=m2_knn_predict(training,holdout,k=args.knn_k,use_semantic=False)
    preds={(x.origin.episode_id,'M1_persistence'):m1_persistence(x) for x in holdout}
    preds.update({(x.origin.episode_id,'M2_temporal_knn_noKG'):y
                  for x,y in zip(holdout,m2_no_kg)})
    preds.update({(x.origin.episode_id,'M2_temporal_staticKG_knn'):y
                  for x,y in zip(holdout,m2_all)})
    kg_eligible=[ex for ex in holdout if ex.semantic_keys]
    sample_pool=holdout if args.sample_all else kg_eligible
    if args.max_llm_per_disease and not sample_pool:
        raise RuntimeError('No holdout episodes with KG profiles. Examine static profile ID coverage or pass --sample-all.')
    selected=stable_sample(sample_pool,args.max_llm_per_disease)
    selected_ids={x.origin.episode_id for x in selected}
    llm_cache=run_llm(selected,out,args.skip_llm,signature_context={
        'input_sha256':hash_file(inp),
        'profile_sha256':hash_file(profile_path) if profile_path.exists() else None,
        'holdout_start':args.holdout_start,
        'sample_all':args.sample_all
    },llm_max_tokens=args.llm_max_tokens, llm_timeout_seconds=args.llm_timeout_seconds,
      ollama_num_ctx=args.ollama_num_ctx,llm_backend=args.llm_backend,
      llm_retries=args.llm_retries,llm_max_output_tokens=args.llm_max_output_tokens)
    rows=[]
    for ex in holdout:
        for method in ('M1_persistence','M2_temporal_knn_noKG','M2_temporal_staticKG_knn'):
            rows.append({
                'method':method,'split':'holdout_all',
                'origin_id':ex.origin.episode_id,'target_id':ex.target.episode_id,
                'disease':ex.origin.disease,'country':ex.origin.country,
                'origin_week':ex.origin.week.isoformat(),'target_week':ex.target.week.isoformat(),
                'target_label':ex.target_label,
                'prediction':preds[(ex.origin.episode_id,method)],
                'has_static_kg_context':int(bool(ex.semantic_keys)),
                'source_relation_id_origin':ex.origin.source_relation_id,
                'source_relation_id_target':ex.target.source_relation_id
            })
        if ex.origin.episode_id not in selected_ids:continue
        for method in ('M3_llm_only','M4_llm_kg_guard'):
            record=llm_cache.get((method,ex.origin.episode_id))
            if not record:continue
            rows.append({
                'method':method,'split':'holdout_sample',
                'origin_id':ex.origin.episode_id,'target_id':ex.target.episode_id,
                'disease':ex.origin.disease,'country':ex.origin.country,
                'origin_week':ex.origin.week.isoformat(),'target_week':ex.target.week.isoformat(),
                'target_label':ex.target_label,'prediction':record['prediction'],
                'has_static_kg_context':int(bool(ex.semantic_keys)),
                'source_relation_id_origin':ex.origin.source_relation_id,
                'source_relation_id_target':ex.target.source_relation_id
            })
    save_csv(out/'predictions.csv',rows)
    subset=[r for r in rows if r['origin_id'] in selected_ids]
    summaries=[]
    for method in ('M1_persistence','M2_temporal_knn_noKG','M2_temporal_staticKG_knn','M3_llm_only','M4_llm_kg_guard'):
        all_method=[r for r in rows if r['method']==method]
        if method.startswith(('M1','M2')):
            summaries.append(metrics(all_method,method+'__all_holdout'))
        summaries.append(metrics([r for r in subset if r['method']==method],method+'__common_sample'))
    save_csv(out/'metrics.csv',[{
        k:json.dumps(v) if isinstance(v,dict) else v
        for k,v in s.items() if k!='confusion'
    } for s in summaries])
    manifest={
        'input':str(inp),'sha256_snapshots':hash_file(inp),
        'profiles':str(profile_path),'sha256_profiles':hash_file(profile_path) if profile_path.exists() else None,
         'sample_selection':'SHA256(origin episode ID), disease-balanced; KG-covered holdout only unless --sample-all; no future labels used',
        'sample_all':args.sample_all,
        'n_kg_eligible_holdout':len(kg_eligible),
        'n_kg_eligible_by_disease':dict(Counter(x.origin.disease for x in kg_eligible)),
        'n_llm_selected_by_disease':dict(Counter(x.origin.disease for x in selected)),
        'n_series':len(series),'n_snapshots':sum(map(len,series.values())),
        'n_train_examples':len(training),'n_holdout_examples':len(holdout),
        'n_llm_selected':len(selected),'n_holdout_revisions':sum(x.target_label=='revision' for x in holdout),
        'cutoff_date':args.holdout_start,'last_train_target':max(x.target.week for x in training).isoformat(),
        'first_holdout_target':min(x.target.week for x in holdout).isoformat(),
        'M2_uses_static_KG_context_when_available':True,
        'M2_noKG_ablation_included':True,
        'llm_backend':args.llm_backend,
        'llm_max_tokens':args.llm_max_tokens,
        'llm_max_output_tokens':args.llm_max_output_tokens,
        'llm_timeout_seconds':args.llm_timeout_seconds,
        'ollama_thinking_explicitly_disabled':args.llm_backend=='ollama_native',
        'n_train_static_profiles':sum(bool(x.semantic_keys) for x in training),
        'n_holdout_static_profiles':sum(bool(x.semantic_keys) for x in holdout),
        'M4_guard_is_OWL_reasoner':False,
        'is_document_grounded_Story_Forest':False,
        'meaning_of_truth':'next observed weekly reported_value increase/flat/revision; NOT incident cases or link truth',
        'no_document_link_gold':True,
        'no_temporal_leakage':'target values excluded from all model inputs; training target weeks precede cutoff',
        'warnings':[
            'Only country-level aggregated snapshots. No transmission, causality, or documentary event links inferred.',
            'M4 uses static KG context and explicit provenance/type checks, NOT OWL reasoning.',
            'COVID and Mpox case fields have not been independently validated as semantically equivalent.',
            'Snapshots are reporting observations, not confirmed incidence.',
            'Primary macro-F1 excludes negative revisions, but revisions are recorded and counted.',
            'Only M1/M2 are evaluated on full holdout; LLM methods on deterministic shared sample.',
            'M3/M4 can abstain. Compare sample metrics and coverage, not full holdout to sample directly.'
        ],
        'metrics':summaries
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # Deterministic observed temporal adjacency, for auditing ONLY.
    with (out/'temporal_adjacency_not_link_gold.jsonl').open('w',encoding='utf-8') as f:
        for ex in holdout:
            f.write(json.dumps({
                'from_episode_id':ex.origin.episode_id,
                'to_episode_id':ex.target.episode_id,
                'relation':'consecutive_country_disease_weekly_snapshots',
                'is_documented_event_continuation':False,
                'source_relation_ids':[ex.origin.source_relation_id,ex.target.source_relation_id]
            },ensure_ascii=False)+'\n')
    print(json.dumps({
        'output':str(out),'n_train':len(training),'n_holdout':len(holdout),
        'n_llm_sample':len(selected),'n_llm_cached':len(llm_cache),
        'n_kg_eligible_holdout':len(kg_eligible),
        'metrics':[dict((k,v) for k,v in x.items() if k!='confusion') for x in summaries]
    },indent=2))

if __name__=='__main__':
    main()
