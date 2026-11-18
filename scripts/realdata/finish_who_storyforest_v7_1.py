#!/usr/bin/env python3
"""Post-process WHO V7 using signed LLM caches. Does not re-run 4B/32B inference.

Two duties: retry only previously failed short extraction requests (0.6B), and
reconstruct conservative incremental trees with the *already cached* fixed-pair
predictions. Pairs not previously queried are ABSTENTIONS, not negatives.
The resulting trees are a curated-card, partial-coverage replay pilot, NOT
end-to-end document-event forest evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/realdata'))
import who_storyforest_final_v7 as v7
from ddpf.storyforest.ontology import OntologyGuard
from ddpf.storyforest.pipeline import process_event
from ddpf.storyforest.storage import Store, forest

MODEL_NAMES=('qwen3:0.6b','qwen3:4b','qwen3:32b')


def canonical_date(raw: str) -> str:
    """Only unambiguous full English/ISO dates. Never infer day, month or year."""
    if not isinstance(raw,str):raise ValueError('date must be a string')
    raw=raw.strip()
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',raw):
        return date.fromisoformat(raw).isoformat()
    if not re.fullmatch(r'[A-Za-z0-9,\s-]{10,35}',raw):
        raise ValueError('date format not safely recoverable: '+raw[:50])
    # Explicit English month dictionary: independent of French/English Linux locale.
    months={name:i for i,name in enumerate(('january','february','march','april',
        'may','june','july','august','september','october','november','december'),1)}
    months.update({name[:3]:i for name,i in list(months.items())})
    m=re.fullmatch(r'(\d{1,2}) ([A-Za-z]+) (\d{4})',raw)
    if m:
        d,month,y=m.groups()
    else:
        m=re.fullmatch(r'([A-Za-z]+) (\d{1,2}), (\d{4})',raw)
        if not m:raise ValueError('date format not safely recoverable: '+raw[:50])
        month,d,y=m.groups()
    if month.lower() not in months:raise ValueError('Unknown month: '+month)
    return date(int(y),months[month.lower()],int(d)).isoformat()


def normalize_validate_extraction(o, changes=None):
    if not isinstance(o,dict) or set(o)!={'events'} or not isinstance(o['events'],list):
        raise ValueError('Extraction schema invalid')
    if len(o['events'])>25:raise ValueError('Too many extracted events')
    # Transactional: no partly normalized object when another date is bad.
    replacements=[]
    for item in o['events']:
        if not isinstance(item,dict) or set(item)!={'event_date','event_type','description'}:
            raise ValueError('Bad event fields')
        new=canonical_date(item['event_date'])
        if item['event_type'] not in v7.EVENT_TYPES or not isinstance(item['description'],str) or not item['description'].strip():
            raise ValueError('Bad event type/description')
        replacements.append((item,new))
    for item,new in replacements:
        old=item['event_date']
        item['event_date']=new
        if old!=new and changes is not None:changes.append({'raw_date':old,'iso_date':new})
    return None


def safe_normalize_dates(changes):
    def validate(o):return normalize_validate_extraction(o,changes)
    v7.validate_extraction=validate


class CacheOnlyLinker:
    """Feeds existing signed Ollama outputs to actual process_event; never calls network."""
    def __init__(self,model,original_output):
        self.model=model
        self.path=Path(original_output)/'ollama_cache.jsonl'
        self.cache={}
        self.hit=0;self.not_queried=0
        if not self.path.exists():raise FileNotFoundError(self.path)
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r.get('model')==model and r.get('kind')=='link':
                self.cache[r['key']]=r['output']
        if len(self.cache)<39:
            raise ValueError(f'Only {len(self.cache)} cached {model} links; expected at least 39')
    @staticmethod
    def _sig(model,messages):
        return {'v':'who_v7_prompt_1','model':model,'messages':messages,
                'schema':v7.LINK_SCHEMA,'kind':'link','budget_initial':256,
                'num_ctx':8192,'temperature':0,'think':False}
    def chat(self,messages):
        key=v7.hash_obj(self._sig(self.model,messages))
        if key in self.cache:
            self.hit+=1
            return json.dumps(self.cache[key],ensure_ascii=False)
        self.not_queried+=1
        return json.dumps({'relation':'unknown','evidence_ids':[],
          'rationale':'unqueried_candidate_in_partial_cache_replay', 'score':0.0})


def tree_replay(events,refs,out,original_output,top_k=4):
    """Replays only candidate decisions already scored; unknown candidates abstain.

    The silver labels are used solely to score after the forest is constructed,
    NEVER to select, filter, order or predict links.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    guard=OntologyGuard.from_file(ROOT/'config/storyforest_rules.json' if (ROOT/'config/storyforest_rules.json').exists() else None)
    ordered=sorted(events,key=lambda e:(e.published_at,e.occurred_at,e.event_id))
    result=[]
    for model in ('qwen3:4b','qwen3:32b'):
        for method in ('llm_only','full'):
            client=CacheOnlyLinker(model,original_output)
            sqlite=out/'trees'/f'{model.split(":")[1]}_{method}.sqlite3'
            sqlite.parent.mkdir(parents=True,exist_ok=True)
            if sqlite.exists():sqlite.unlink()
            errors=[]
            with Store(str(sqlite)) as store:
                for e in ordered:
                    try:process_event(store,e,method,client,guard,top_k=top_k)
                    except Exception as ex:
                        errors.append({'event_id':e.event_id,'error':str(ex)[:250]})
                        store.insert_event(e)
                links=store.links(method)
                trees=forest(store.list_events(limit=len(ordered)),links)
            mt=v7.tree_metrics(trees,ordered)
            mt.update(v7.tree_reference_scores(trees,refs))
            row={'model':model,'method':method,'scope':'cached_fixed_pair_candidate_replay',
                 'cached_decisions':client.hit,'unqueried_candidates_abstained':client.not_queried,
                 'candidate_decisions':len(links), 'n_errors':len(errors),
                 'n_accepted_continuations':sum(x.status=='accepted' and x.relation=='documented_continuation' for x in links),
                 **mt}
            result.append(row)
            v7.dump(out/'trees'/f'{model.split(":")[1]}_{method}.json',{
                'metadata':{'scope':'CACHED_PREDICTIONS_ONLY','model':model,'method':method,
                  'unqueried_candidates_are_abstentions':True,
                  'not_full_candidate_matrix':True,'silver_labels_not_used_for_predictions':True},
                'links':[x.to_dict() for x in links], 'trees':trees,'errors':errors})
            print('REPLAY',model,method,'edges=',mt['n_edges'],'cached=',client.hit,
                  'unqueried=',client.not_queried,flush=True)
    v7.save_csv(out/'cached_tree_replay_metrics.csv',result)
    return result


def summarize_extraction(rows,models,sources):
    by={}
    for model in models:
        d=[r for r in rows if r['model']==model]
        ok=[r for r in d if r['status']=='OK']
        total=sum(int(x['gold_slots']) for x in ok)
        matched=sum(int(x['covered']) for x in ok)
        by[model]={'documents_ok':len(ok),'documents_total':len(sources),
          'reference_slots_in_completed_documents':total,'matched_slots':matched,
          'slot_recall_conditional_on_completed':round(matched/total,6) if total else None,
          'failures':len(d)-len(ok)}
    return by


def get_pair_metrics(path):
    with open(path,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)<8:raise ValueError('Expected all original fixed-pair metrics; use existing V7 outputs')
    return rows


def write_summary(report,extraction,tree_rows,pair_rows,issues,provenance):
    lines=['# PreDiViD WHO documentary V7.1 — cache-preserving final audit','',
      '**STATUS: '+('TECHNICALLY_INCOMPLETE' if issues else 'CONTROLLED_PILOT_COMPLETE_WITH_LIMITATIONS')+'**','',
      '> IMPORTANT: This is an exploratory **silver** documentary pilot, not independently adjudicated event/link gold.','',
      '## Direct extraction from 8 WHO reports','',
      '| Model | Documents completed | Dated reference slots recovered | Conditional slot recall |','|---|---:|---:|---:|']
    for model,r in extraction.items():
        val=r['slot_recall_conditional_on_completed']; score='NA' if val is None else f'{val:.3f}'
        lines.append(f'| {model} | {r["documents_ok"]}/8 | {r["matched_slots"]}/{r["reference_slots_in_completed_documents"]} | {score} |')
    lines+=['','Slot coverage is NOT event-level precision, nor an end-to-end forest F1.','',
      '## Fixed-pair documentary relation classification (unchanged V7, no new calls)','',
      '| Model | Method | n | Core-link F1 | Missing |', '|---|---|---:|---:|---:|']
    for r in pair_rows:
        if r['model']=='heuristic':continue
        lines.append(f'| {r["model"]} | {r["method"]} | {r["n_labeled"]} | '+
                     (f'{float(r["f1"]):.3f}' if r['f1'] else 'Undefined')+
                     f' | {r["technical_missing"]} |')
    lines+=['','## Incremental StoryForest — strictly cache-limited candidate replay','',
      '**This is not a full end-to-end forest benchmark.** The original V7 forest with 0.6B '+
      'had 29 roots and 0 edges. The new trees replay only independently obtained V7 '+
      'fixed-pair LLM predictions using `process_event`, SQLite Store, and `forest()`. '+
      'Candidate pairs without a cached prediction abstain and are not interpreted as '+
      'negative. Silver labels are not used to choose edges.','',
      '| Model | Method | Cached candidate decisions | Unqueried abstentions | Forest edges | Silver direct-edge recall* |',
      '|---|---|---:|---:|---:|---:|']
    for r in tree_rows:
        lines.append(f'| {r["model"]} | {r["method"]} | {r["cached_decisions"]} | {r["unqueried_candidates_abstained"]} | {r["n_edges"]} | {r["direct_edge_recall_silver"]:.3f} |')
    lines+=['','*Silver direct-edge recall is conditional on 21 proposed continuation labels. '+
      'A valid transitive path is not counted as a direct match; not all graph edges have been adjudicated.','',
      '## What remains outside the evidence','',
      '- Actual document-to-generated-event-to-LLM-tree end-to-end link quality is **not measured**. '+
      'The original V7 generated-event stage was incomplete; do not claim it passed.',
      '- `full` is typed-metadata guard, not BioPropaPhenKG retrieval or OWL reasoning.',
      '- WHO source anchoring does not prove each extracted event description or causal link is true.',
      '- The 29 curated cards / 39 selected pairs are an exploratory, non-random silver set.',
      '- These results do not replace V5/V6 aggregate weekly-direction evaluation.','',
      '## Provenance','',
      f'- WHO source audit: `{provenance["source_audit"]}`',
      f'- Original V7 cache SHA256: `{provenance["v7_cache_sha256"]}`',
      '- Original fixed-pair predictions were reused without changes.',
      '- Only previously uncached extraction documents may have caused new 0.6B inference.']
    if issues:lines+=['','## Remaining technical issues']+['- '+s for s in issues]
    Path(report).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    original=Path(args.original_output).resolve()
    final=original/'final_v7_1'
    if not original.is_dir():raise FileNotFoundError(original)
    sources,ann,pairs=[v7.load(v7.REF/f'{n}.json') for n in ('sources','silver_events','silver_pairs')]
    v7.validate_reference(sources,ann,pairs)
    texts=v7.verify_sources(sources,ann,original,offline=True)
    if v7.load(original/'source_audit.json')['status']!='PASS':raise RuntimeError('Original WHO source audit not PASS')
    pair_rows=get_pair_metrics(original/'pair_metrics.csv')
    if not (original/'ollama_cache.jsonl').exists():raise FileNotFoundError('Original V7 cache missing')
    if not (original/'pair_predictions.csv').exists():raise FileNotFoundError('Fixed pair predictions missing')
    final.mkdir(parents=True,exist_ok=True)
    changes=[]
    issues=[]
    if args.no_extraction_retry:
        with (original/'extraction_metrics.csv').open(newline='',encoding='utf-8') as f:extrows=list(csv.DictReader(f))
        if any(r['status']!='OK' for r in extrows):issues.append('Some extraction documents remain technically missing (retry disabled)')
    else:
        safe_normalize_dates(changes)
        extrows,failed=v7.extraction_run(MODEL_NAMES,sources,texts,ann,original,args.timeout,args.retries)
        if failed:issues.append(f'{len(failed)} extraction requests still failed')
    v7.dump(final/'date_normalizations.json',{'unambiguous_full_dates_only':True,'changes':changes})
    events=v7.cards(sources,ann,texts)
    trees=tree_replay(events,pairs,final,original,args.top_k)
    if any(r['n_errors'] for r in trees):issues.append('Cache-replay incremental processing errors')
    extraction=summarize_extraction(extrows,MODEL_NAMES,sources)
    prov={'source_audit':'PASS','v7_cache_sha256':v7.sha_file(original/'ollama_cache.jsonl')}
    manifest={'status':'TECHNICALLY_INCOMPLETE' if issues else 'CONTROLLED_PILOT_COMPLETE_WITH_LIMITATIONS',
      'independently_adjudicated_gold':False,
      'full_end_to_end_generated_forest_validated':False,
      'scope':'cache-only fixed-pair candidate replay; unqueried pairs abstain',
      'extraction':extraction, 'tree_replay':trees, 'issues':issues,'provenance':prov,
      'date_normalizations':changes}
    v7.dump(final/'manifest_v7_1.json',manifest)
    write_summary(final/'final_article_ready_caveated_report.md',extraction,trees,pair_rows,issues,prov)
    print(json.dumps({'status':manifest['status'],'output':str(final),'new_normalized_dates':len(changes),
                     'tree_edges':{r['model']+'__'+r['method']:r['n_edges'] for r in trees},
                     'issues':issues},indent=2),flush=True)
    return 2 if issues else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-output',default=str(v7.DEFAULT_OUTPUT))
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--retries',type=int,default=2)
    p.add_argument('--top-k',type=int,default=4)
    p.add_argument('--no-extraction-retry',action='store_true',
      help='Fully offline operation: preserve existing missing extraction documents')
    return run(p.parse_args(argv))

if __name__=='__main__':sys.exit(main())
