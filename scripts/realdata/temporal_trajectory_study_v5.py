#!/usr/bin/env python3
"""V5: schema-constrained native Ollama, failure diagnostics and paired LLM metrics.

Extends the immutable V4 experiment (bundled), retains chronological holdout,
KG ablation and the country-week scientific limitations. Native Ollama only.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temporal_trajectory_study_v4 as base

SCHEMA = {
    'type': 'object',
    'properties': {'prediction': {'type':'string', 'enum':list(base.CLASSES)}},
    'required': ['prediction'],
    'additionalProperties': False,
}

base.PROMPT_SYSTEM = (
    'Predict the direction of the next weekly reported aggregate value using '
    'ONLY previous reported values. It is a forecast, not a known future fact. '
    'Choose increase (higher), flat (equal), or revision (lower). '
    'Return ONLY JSON with precisely one key prediction and exactly one of '
    'those three string values. No explanations, no reasoning, no extra keys. '
    'Do not make clinical, causal, transmission or documentary claims. /no_think'
)

class ModelOutputError(RuntimeError):
    def __init__(self, category: str, diagnostics: dict):
        self.category = category
        self.diagnostics = diagnostics
        super().__init__('Ollama invalid output: '+category)


class StrictOllamaClient(base.OllamaNativeClient):
    """Validate a one-field enum schema, retry incomplete/bad outputs once.

    The output of any failed attempt is excluded from predictions and cache.
    Failure diagnostics contain only JSON field names/size, not raw reasoning.
    """
    def request_body(self, messages: list[dict[str,str]], budget: int|None=None) -> dict:
        body = super().request_body(messages,budget)
        body['format'] = SCHEMA
        return body

    @staticmethod
    def diagnostic(content:str,result:dict,budget:int)->dict:
        out={
            'done_reason':result.get('done_reason'),
            'num_predict':budget,
            'eval_count':result.get('eval_count'),
            'content_chars':len(content),
            'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
        }
        try:
            parsed=json.loads(content)
            if isinstance(parsed,dict):
                out['json_keys']=sorted(str(k)[:50] for k in parsed.keys())[:16]
                val=parsed.get('prediction')
                if isinstance(val,str): out['prediction_value']=val[:45]
                else: out['prediction_type']=type(val).__name__
            else:out['json_type']=type(parsed).__name__
        except (ValueError,TypeError):out['json_parse_failed']=True
        return out

    def chat(self, messages:list[dict[str,str]])->str:
        budget=self.max_tokens
        repaired=False
        attempts=[]
        while True:
            outgoing=list(messages)
            if repaired:
                outgoing.append({
                    'role':'user',
                    'content':'Your output did not match the schema. Reply with ONLY '
                              'one JSON object, e.g. {"prediction":"flat"}; replace '
                              'flat with increase or revision if appropriate. No other '
                              'keys or prose. /no_think',
                })
            result=self._post(self.request_body(outgoing,budget))
            content=str(result['message'].get('content') or '').strip()
            diag=self.diagnostic(content,result,budget)
            attempts.append(diag)
            self.last_stats={k:result.get(k) for k in (
                'done_reason','total_duration','load_duration','prompt_eval_count',
                'prompt_eval_duration','eval_count','eval_duration')}
            self.last_stats.update({
                'num_predict_used':budget,
                'budgets_attempted':[a['num_predict'] for a in attempts],
                'schema':'one_field_enum_v5',
            })
            if result.get('done_reason')=='length':
                diag['failure']='truncated'
                if budget>=self.max_output_tokens:
                    raise ModelOutputError('truncated_at_max_budget',{'attempts':attempts})
                next_budget=min(budget*2,self.max_output_tokens)
                print(f'Ollama truncated at {budget}; retrying with {next_budget}',flush=True)
                budget=next_budget
                continue
            try: obj=json.loads(content)
            except (ValueError,TypeError):obj=None
            if isinstance(obj,dict) and set(obj)=={'prediction'} and obj['prediction'] in base.CLASSES:
                self.last_stats['validation_attempts']=len(attempts)
                return content
            diag['failure']='invalid_schema_or_label'
            print('Invalid Ollama JSON classification: '
                  f'keys={diag.get("json_keys")}, label={diag.get("prediction_value")}, '
                  f'done_reason={diag["done_reason"]}; '
                  + ('retrying with constrained prompt' if not repaired else 'giving up this item'),flush=True)
            if repaired:
                raise ModelOutputError('invalid_response_after_repair',{'attempts':attempts})
            repaired=True


# The original V4 remains available and unchanged. Only its native client is
# replaced in the new V5 process.
base.OllamaNativeClient=StrictOllamaClient


def save_csv_union(path:Path, rows:list[dict]):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)

base.save_csv=save_csv_union


def run_llm_isolated(examples:list[base.Example],out:Path,skip_llm=False,signature_context=None,
                     llm_max_tokens:int=256,llm_timeout_seconds:int=240,
                     ollama_num_ctx:int=4096,llm_backend:str='ollama_native',
                     llm_retries:int=0,llm_max_output_tokens:int=1024):
    if skip_llm:return {}
    if llm_backend!='ollama_native':
        raise RuntimeError('V5 requires native Ollama for schema-constrained evaluation')
    from dotenv import load_dotenv
    from ddpf.storyforest.settings import Settings
    load_dotenv(dotenv_path=Path('.env'),override=False)
    settings=replace(Settings.from_env(require_key=True),
                     max_tokens=llm_max_tokens,timeout=llm_timeout_seconds)
    llm=StrictOllamaClient(settings,timeout_seconds=llm_timeout_seconds,
                           max_tokens=llm_max_tokens,num_ctx=ollama_num_ctx,
                           max_output_tokens=llm_max_output_tokens)
    cfg={
        'study_version':'v5_schema_enum_isolated_failures',
        'backend':llm_backend,'model':settings.litellm_model,
        'provider':settings.provider,'temperature':settings.temperature,
        'think':False,'output_schema':SCHEMA,
        'prompt_sha256':hashlib.sha256(base.PROMPT_SYSTEM.encode()).hexdigest(),
        'num_ctx':ollama_num_ctx,'max_tokens':llm_max_tokens,
        'max_output_tokens':llm_max_output_tokens,
        **(signature_context or {}),
    }
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    cache_file=out/'llm_predictions.jsonl'
    errors_file=out/'llm_failures.jsonl'
    cache=base.load_cache(cache_file)
    if cache and {row.get('signature') for row in cache.values()}!={signature}:
        raise RuntimeError('Cache signature mismatch: use a new --output directory')
    print(f'V5 native Ollama, model={settings.litellm_model}, schema enum, '
          f'budget={llm_max_tokens}..{llm_max_output_tokens}, '
          f'timeout={llm_timeout_seconds}s, cached={len(cache)}',flush=True)
    with cache_file.open('a',encoding='utf-8') as f, errors_file.open('a',encoding='utf-8') as errout:
        for ex in examples:
            for method in ('M3_llm_only','M4_llm_kg_guard'):
                key=(method,ex.origin.episode_id)
                if key in cache:continue
                started=time.monotonic()
                answer=None
                failure=None
                for retry in range(llm_retries+1):
                    try:
                        answer=base.llm_predict(ex,method,llm)
                        break
                    except (RuntimeError,ModelOutputError) as exc:
                        failure=exc
                        if retry<llm_retries:
                            print(f'{method} attempt failed, retrying: {type(exc).__name__}',flush=True)
                elapsed=round(time.monotonic()-started,2)
                if answer is None:
                    # Technical failure is NOT a model abstention (unknown).
                    errrecord={
                        'method':method,'episode_id':ex.origin.episode_id,
                        'signature':signature,'elapsed_seconds':elapsed,
                        'error_type':type(failure).__name__,
                        'error_category':getattr(failure,'category','backend_error'),
                        'diagnostics':getattr(failure,'diagnostics',None),
                        'ollama_stats':llm.last_stats,
                        'is_prediction':False,
                    }
                    errout.write(json.dumps(errrecord,ensure_ascii=False)+'\n')
                    errout.flush();os.fsync(errout.fileno())
                    print(f'FAILED {method} {ex.origin.episode_id}: '
                          f'{errrecord["error_category"]}; see {errors_file}',flush=True)
                    continue
                item={
                    'method':method,'episode_id':ex.origin.episode_id,
                    'target_episode_id':ex.target.episode_id,
                    'prediction':answer['prediction'],'reason':answer['reason'],
                    'signature':signature,'model':settings.litellm_model,
                    'backend':'ollama_native','elapsed_seconds':elapsed,
                    'ollama_stats':llm.last_stats,
                }
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
                f.flush();os.fsync(f.fileno())
                cache[key]=item
                print(f'OK {method}: {len(cache)} cached, prediction={answer["prediction"]}, '
                      f'elapsed={elapsed}s, origin={ex.origin.episode_id}',flush=True)
    return cache

base.run_llm=run_llm_isolated


def supplement_manifest(output:Path,skip_llm:bool):
    manifest_path=output/'manifest.json'
    report=json.loads(manifest_path.read_text(encoding='utf-8'))
    n_selected=int(report['n_llm_selected'])
    expected=0 if skip_llm else n_selected*2
    cached=base.load_cache(output/'llm_predictions.jsonl')
    predictions=[]
    with (output/'predictions.csv').open(encoding='utf-8',newline='') as f:
        predictions=list(csv.DictReader(f))
    # Recreate the exact deterministic selection, independent of any additional
    # cached results from a previous run with a larger sample size.
    series=base.load_snapshots(Path(report['input']))
    profiles=base.load_static_profiles(Path(report['profiles']))
    candidates=base.make_examples(series,profiles)
    candidates=[ex for ex in candidates
                if ex.target.week>=date.fromisoformat(report['cutoff_date'])
                and (report['sample_all'] or bool(ex.semantic_keys))]
    by_disease=defaultdict(list)
    for ex in candidates:by_disease[ex.origin.disease].append(ex)
    selected=set()
    for disease, group in by_disease.items():
        n=int(report['n_llm_selected_by_disease'].get(disease,0))
        group.sort(key=lambda ex:hashlib.sha256(ex.origin.episode_id.encode()).hexdigest())
        selected.update(ex.origin.episode_id for ex in group[:n])
    if len(selected)!=n_selected:
        raise RuntimeError('Selected LLM example reconstruction differs from manifest')
    paired={ep for ep in selected if all(
        (method,ep) in cached for method in ('M3_llm_only','M4_llm_kg_guard'))}
    completed=sum((method,ep) in cached for ep in selected
                  for method in ('M3_llm_only','M4_llm_kg_guard'))
    paired_rows=[r for r in predictions if r['origin_id'] in paired]
    paired_metrics=[]
    if paired:
        for method in ('M1_persistence','M2_temporal_knn_noKG',
                       'M2_temporal_staticKG_knn','M3_llm_only','M4_llm_kg_guard'):
            subset=[r for r in paired_rows if r['method']==method]
            paired_metrics.append(base.metrics(subset,method+'__paired_complete'))
        save_csv_union(output/'paired_metrics.csv',[
            {k:v for k,v in item.items() if k!='confusion'} for item in paired_metrics])
    report.update({
        'study_version':'v5_schema_enum_isolated_failures',
        'ollama_output_format':'JSON schema enum, not unrestricted json',
        'expected_llm_calls':expected,
        'completed_llm_calls':completed,
        'technical_failures_or_missing_calls':max(expected-completed,0),
        'llm_study_complete':completed==expected,
        'paired_complete_examples':len(paired),
        'paired_comparison_valid_only_on':'paired_metrics.csv for episodes with both M3 and M4 outputs',
        'failures_are_neither_predictions_nor_abstentions':True,
        'invalid_response_diagnostics_file':'llm_failures.jsonl',
        'scientific_warning':'Aggregate direction prediction; not documentary Story Forest link evaluation',
    })
    manifest_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({
        'output':str(output),'llm_expected':expected,'llm_completed':completed,
        'llm_technical_missing':expected-completed,
        'llm_paired_completed_examples':len(paired),
        'status':'COMPLETE' if completed==expected else 'INCOMPLETE',
        'diagnostics':str(output/'llm_failures.jsonl'),
    },indent=2),flush=True)
    if completed!=expected:
        print('WARNING: incomplete LLM sample; do not compare unpaired scores. '
              'Retrying the same command reuses valid cached predictions.',flush=True)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not any(a=='--output' or a.startswith('--output=') for a in args):
        args=['--output','outputs/realdata/temporal_trajectory_study_v5',*args]
    output=Path(args[args.index('--output')+1]) if '--output' in args else Path(next(
        a.split('=',1)[1] for a in args if a.startswith('--output=')))
    skip_llm='--skip-llm' in args
    base.main(args)
    supplement_manifest(output,skip_llm)


if __name__=='__main__':main()
