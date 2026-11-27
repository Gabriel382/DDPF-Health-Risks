"""Run the DDPF-first pipeline. Gold is accessed only in explicit review/score commands."""
import argparse,sys,os,platform,subprocess,time
from pathlib import Path
from .data import *
from .experiment import *


def cli():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for flag in ('preflight','pilot','full','baselines-only','review','score','audit-independence'):mode.add_argument('--'+flag,action='store_true')
    p.add_argument('--root',default='.');p.add_argument('--out');p.add_argument('--dataset',choices=['HIST','EXT','EXT-3'],default='HIST');p.add_argument('--level',choices=sorted(LEVELS),default='sentence');p.add_argument('--config');p.add_argument('--observations');p.add_argument('--export-dir');p.add_argument('--embedding-jsonl');p.add_argument('--embedding',choices=['ddpf_best','binary_bow','hashing_bow','ollama']);p.add_argument('--embedding-model');p.add_argument('--chronology',choices=['collection','occurrence']);p.add_argument('--conditions',nargs='+',choices=['C0','C1','C2','C3','C4','C5']);p.add_argument('--models',nargs='+');p.add_argument('--neo4j',action='store_true');p.add_argument('--ontology',action='append',default=[]);p.add_argument('--pull-missing',action='store_true');p.add_argument('--ollama-url',default=None);p.add_argument('--exploratory',action='store_true');p.add_argument('--approval');p.add_argument('--gold-dir');p.add_argument('--review-pairs',type=int,default=400);p.add_argument('--reference-manifest');p.add_argument('--check-models',action='store_true');p.add_argument('--skip-neo4j-audit',action='store_true');p.add_argument('--allow-no-ordered-pairs',action='store_true');p.add_argument('--top-k',type=int);p.add_argument('--window-days',type=int);p.add_argument('--threshold',type=float);p.add_argument('--retrieval-mode',choices=['hybrid','cosine','jaccard','geo','time','tfidf','no_embedding','no_cluster','no_time','no_geography']);p.add_argument('--all-pairs',action='store_true');p.add_argument('--hash-large',action='store_true');return p.parse_args()


def approval_check(a,out,report,cfg,plan):
    if not a.full or a.exploratory:return 'pilot' if a.pilot else 'exploratory'
    if not a.approval:raise ValueError('Full confirmatory run requires --approval signed provenance/development freeze JSON; use --exploratory only for a clearly non-confirmatory full run')
    x=load(a.approval)
    required=['independent_reference_verified','source_derivatives_excluded','static_knowledge_availability_verified','embedding_provenance_verified','development_only_tuning','splits_frozen','annotation_protocol_frozen','test_reference_isolated']
    if not all(x.get(k) is True for k in required) or not x.get('reviewer') or not x.get('date'):raise ValueError('approval checklist incomplete')
    if x.get('export_hashes')!=report['hashes'] or x.get('config_sha256')!=stable(cfg) or x.get('plan_hashes')!=plan['hashes']:raise ValueError('approval does not bind to exact prepared exports/config/plan')
    if a.dataset in ('EXT','EXT-3') and not x.get('coverage_verified'):raise ValueError('extended/external coverage must be verified')
    if report['availability_proxy_count'] or report['missing_source_ids'] or report['missing_source_text']:
        if not x.get('aggregate_provenance_limitations_acknowledged'):raise ValueError('Historical aggregate provenance requires explicit attestation of limitations; cannot assert document-level independence or real-time replay.')
        return 'aggregate_observation_study_attested_limited_provenance'
    if a.dataset in ('EXT','EXT-3') and not x.get('coverage_verified'):raise ValueError('extended/external coverage must be verified')
    return 'confirmatory_inputs_attested_by_user'


def precheck_models(client,models,pull):
    v=requests.get(client.base+'/api/version',timeout=30);v.raise_for_status();version=v.json().get('version','unknown');print('Ollama server version:',version,flush=True)
    if any(m.startswith('gemma3:') for m in models):
        match=re.match(r'(\d+)\.(\d+)',version)
        if match and tuple(map(int,match.groups()))<(0,6):raise RuntimeError('Gemma 3 requires Ollama >=0.6. Update after other running experiments finish; no inference started.')
    failures={}
    for m in dict.fromkeys(models):
        try:client.prepare(m)
        except Exception as e:failures[m]=str(e)[:400]
    if failures:raise RuntimeError('Missing/unavailable models before inference: '+json.dumps(failures)+'; check Ollama compatibility and disk space')
    return version


def report(out,m):
    lines=['# DDPF-first V8 run','',f"Scope: **{m['scope']}**",f"Status: **{m['status']}**",'',f"Observations: {m['plan']['n_observations']}; candidate pairs: {m['n_pairs_run']}; eligible universe: {m['plan']['n_eligible_pairs']}",f"Known observation-window order among selected candidates: {m['plan']['known_observation_order_pairs']}; known individual occurrence order: {m['plan']['known_occurrence_order_pairs']}",'','| Variant | Pairs | Missing | Forest | DAG |','|---|---:|---:|---:|---:|']
    for n,v in m['variants'].items():lines.append(f"| {n} | {v['pairs']} | {v['technical_missing']} | {v['forest_edges']} | {v['dag_edges']} |")
    lines+=['','No official reference documents or reference labels were read by the inference worker. This is code-level separation, not proof of source independence.','Collection intervals order aggregate DDPF observations, not individual epidemiological incidents. Links require relation decisions, not chronology alone. Replay is retrospective collection-window replay when availability is proxied.','Run --review and --score separately with independently constructed reference annotations.']
    (out/'run_report.md').write_text('\n'.join(lines)+'\n')


def inference(a,root,out,cfg):
    # Preflight and run always re-validate export; no old WHO calls are imported into the cache.
    old_manifest=load(out/'run_manifest.json') if (out/'run_manifest.json').exists() else None
    if a.baselines_only and old_manifest and old_manifest.get('scope')!='representation_baseline_only':raise ValueError('Choose a separate --out for baseline-only runs')
    if old_manifest and old_manifest.get('scope')=='representation_baseline_only' and not (a.baselines_only or a.preflight):raise ValueError('Baseline-only output cannot be resumed as generative run; choose --out')
    if old_manifest and (old_manifest.get('config')!=cfg or old_manifest.get('scope')=='pilot' and a.full or old_manifest.get('scope')!='pilot' and a.pilot):raise ValueError('Run configuration changed; choose new --out')
    if a.dataset!='HIST' and not a.export_dir and not old_manifest:raise ValueError('EXT/EXT-3 require explicit canonical export directory')
    if old_manifest:
        report0=load(out/'export_audit.json')
        if old_manifest['export']['hashes']!={n:sha(out/n) for n in report0['hashes']}:raise ValueError('Prepared input files changed; choose new --out')
        if a.export_dir or a.observations:print('Resuming frozen exports; supplied input paths ignored for this existing run.',flush=True)
    else:report0=prepare(root,out,a,cfg)
    ei,fi,ni=load_export(out)
    models=cfg['models'];client=None
    try:
        if (not a.preflight and not a.baselines_only) or a.check_models or cfg['embedding'] in ('ollama','ddpf_best') and not a.embedding_jsonl:
            base=a.ollama_url or os.getenv('OLLAMA_HOST','http://localhost:11434');base=base if '://' in base else 'http://'+base
            client=Client(out,base=base,timeout=cfg['timeout'],ctx=cfg['context_tokens'],pull=a.pull_missing)
            version=None
            if cfg['embedding'] in ('ollama','ddpf_best') and not a.embedding_jsonl and not old_manifest:client.prepare(cfg['embedding_model'])
        else:version=None
        if old_manifest:
            plan=load(out/'plan.json')
            if old_manifest['plan']['hashes']!={n:sha(out/n) for n in plan['hashes']}:raise ValueError('Frozen plan files changed; choose new --out')
        else:plan=create_plan(out,ei,fi,ni,cfg,client,a.embedding_jsonl)
        dump(out/'frozen_config.json',{'config':cfg,'config_sha256':stable(cfg),'input_hashes':report0['hashes'],'plan_hashes':plan['hashes']})
        print(json.dumps({'export':report0,'plan':plan},indent=2,ensure_ascii=False),flush=True)
        if a.preflight:
            if a.check_models:precheck_models(client,models+([cfg['embedding_model']] if cfg['embedding'] in ('ollama','ddpf_best') and not a.embedding_jsonl else []),a.pull_missing)
            return
        if a.baselines_only:
            pairs=read_jsonl(out/('eligible_pairs.jsonl' if a.all_pairs else 'candidates.jsonl'))
            variants={name:write_variant(out,name,ei,rows,cfg) for name,rows in baseline_rows(ei,pairs,cfg,fi).items()}
            variants.update({name:write_variant(out,name,ei,rows,cfg) for name,rows in natural_baselines(ei,read_jsonl(out/'eligible_pairs.jsonl'),cfg,fi).items()})
            m={'status':'COMPLETE','scope':'representation_baseline_only','config':cfg,'export':report0,'plan':plan,'n_pairs_run':len(pairs),'variants':variants,'models':client.meta if client else {}}
            dump(out/'baseline_manifest.json',m);report(out,m)
            # Keep scoring interface compatible, never overwrite a generative run.
            dump(out/'run_manifest.json',m);print('Baseline representation run complete:',out,flush=True);return
        if a.full and not plan['known_observation_order_pairs'] and not (a.allow_no_ordered_pairs and a.exploratory):raise ValueError('No selected candidates with strict ordering in the chosen chronology basis. Full forest run would be uninformative; use --exploratory --allow-no-ordered-pairs only for a relation-only run.')
        scope=approval_check(a,out,report0,cfg,plan)
        version=precheck_models(client,(models[:1] if a.pilot and not a.check_models else models)+([cfg['embedding_model']] if cfg['embedding'] in ('ollama','ddpf_best') and not a.embedding_jsonl else []),a.pull_missing)
        pairs=read_jsonl(out/('eligible_pairs.jsonl' if a.all_pairs else 'candidates.jsonl'))
        if a.pilot:
            pairs=sorted(random.Random(cfg['seed']).sample(pairs,min(cfg['pilot_pairs'],len(pairs))),key=lambda p:p['pair_id'])
        jsonl(out/'run_pairs.jsonl',pairs)
        ont=a.ontology
        if not ont:
            for name in ('PropaPhen.owl','BioPropaPhen.rdf'):
                try:
                    p=locate(root,name,required=False)
                    if p:ont.append(str(p))
                except ValueError as ex:print('Ambiguous ontology; supply --ontology explicitly:',ex,flush=True)
        if old_manifest:
            if old_manifest['static_sha256']!=sha(out/'static_context.jsonl') or old_manifest['formal_violations_sha256']!=sha(out/'formal_fact_violations.json'):raise ValueError('Frozen semantic context changed; choose new --out')
            static={r['observation_id']:r['facts'] for r in read_jsonl(out/'static_context.jsonl')};violations=load(out/'formal_fact_violations.json')
        else:
            ss=StaticContext(root,cfg,a.neo4j,ont);static={}
            try:
                for i,e in ei.items():static[i]=ss.for_event(e,fi)
                violations={id:r for id,f in fi.items() if (r:=ss.check_fact(f,ni))}
            finally:ss.close()
            dump(out/'formal_fact_violations.json',violations)
            jsonl(out/'static_context.jsonl',[{'observation_id':i,'facts':x} for i,x in static.items()]);ss.audit.update(events_with_static_facts=sum(bool(x) for x in static.values()),events_with_inferred_facts=sum(any(r['kind']=='inferred_rdfs' for r in x) for x in static.values()),scope='allowlisted snapshot context; temporal independence requires attestation');dump(out/'semantic_audit.json',ss.audit)
        if scope in ('confirmatory_inputs_attested_by_user','aggregate_observation_study_attested_limited_provenance'):
            ap=load(a.approval)
            if ap.get('static_context_sha256')!=sha(out/'static_context.jsonl') or ap.get('formal_violations_sha256')!=sha(out/'formal_fact_violations.json'):raise ValueError('Approval must bind exact static context/violations hashes; inspect prepared files, then update approval and rerun')
        run_cfg=dict(cfg,models=cfg['models'][:1],conditions=['C3']) if a.pilot else cfg
        meta={'status':'RUNNING','scope':scope,'config':cfg,'actual_model_conditions':{'models':run_cfg['models'],'conditions':run_cfg['conditions']},'export':report0,'plan':plan,'static_sha256':sha(out/'static_context.jsonl'),'formal_violations_sha256':sha(out/'formal_fact_violations.json'),'run_pairs_sha256':sha(out/'run_pairs.jsonl'),'n_pairs_run':len(pairs),'models':client.meta,'ollama_version':version,'python':sys.version,'platform':platform.platform(),'variants':{},'approval_sha256':sha(a.approval) if a.approval else None}
        if old_manifest and {k:v.get('digest') for k,v in old_manifest['models'].items()}!={k:v.get('digest') for k,v in meta['models'].items()}:raise ValueError('Model digests changed; choose new --out (cache entries keyed by digest)')
        meta['code_hashes']={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))};meta['hardware']={'cpu_count':os.cpu_count(),'platform':platform.platform()}
        try:meta['hardware']['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
        except (OSError,subprocess.TimeoutExpired):pass
        if old_manifest and old_manifest.get('code_hashes')!=meta['code_hashes']:raise ValueError('Code changed; choose new --out')
        dump(out/'run_manifest.json',meta)
        start=time.monotonic()
        for name,rows in baseline_rows(ei,pairs,cfg,fi).items():meta['variants'][name]=write_variant(out,name,ei,rows,cfg)
        if a.full:
            for name,rows in natural_baselines(ei,read_jsonl(out/'eligible_pairs.jsonl'),cfg,fi).items():meta['variants'][name]=write_variant(out,name,ei,rows,cfg)
        meta['variants'].update(run_models(root,out,ei,fi,ni,run_cfg,client,pairs,static,scope,violations))
        meta['status']='COMPLETE_WITH_MISSING' if any(v['technical_missing'] for v in meta['variants'].values()) else 'COMPLETE';meta['elapsed_seconds_this_execution']=time.monotonic()-start;meta['new_llm_calls']=client.calls
        dump(out/'run_manifest.json',meta);report(out,meta);print(json.dumps({'status':meta['status'],'output':str(out),'new_llm_calls':client.calls,'elapsed_seconds':meta['elapsed_seconds_this_execution']},indent=2))
    finally:
        if client:client.close()


def main():
    a=cli();root=Path(a.root).resolve();os.chdir(root)
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    cfg=load(a.config or root/'configs/reviewer_v8_ddpf/default.json')
    if a.models:cfg['models']=a.models
    if a.embedding:cfg['embedding']=a.embedding
    if a.embedding_model:cfg['embedding_model']=a.embedding_model
    if a.chronology:cfg['chronology_basis']=a.chronology
    if a.conditions:cfg['conditions']=a.conditions
    # Prefer an exact, explicitly named ID-keyed historical export, never guess row order.
    if not (a.review or a.score or a.audit_independence) and cfg['embedding']=='ddpf_best' and not a.embedding_jsonl:
        from .vectors import discover_vectors
        a.embedding_jsonl=discover_vectors(root,a.level)
    if a.embedding_jsonl:cfg['embedding_input_sha256']=sha(a.embedding_jsonl)
    for k in ('top_k','window_days','threshold','retrieval_mode'):
        if getattr(a,k) is not None:cfg[k]=getattr(a,k)
    cfg['all_pairs']=a.all_pairs
    if cfg['top_k']<1 or cfg['window_days']<0 or not 0<=cfg['threshold']<=1:raise ValueError('invalid retrieval settings')
    mode='baseline' if a.baselines_only else ('full' if a.full else 'pilot');out=Path(a.out or root/'outputs/realdata/reviewer_revision_v8_ddpf_final'/f'{a.dataset}_{a.level}_{mode}').resolve();out.mkdir(parents=True,exist_ok=True)
    gold=Path(a.gold_dir or root/'data/reviewer_v8_ddpf/gold_final'/f'{a.dataset}_{a.level}').resolve()
    if a.review or a.score or a.audit_independence:
        from .evaluation import sample_review,evaluate,audit_independence
        if a.review:result=sample_review(out,gold,a.review_pairs,cfg['seed'])
        elif a.score:result=evaluate(out,gold)
        else:result=audit_independence(out,Path(a.reference_manifest),gold)
        print(json.dumps(result,indent=2,ensure_ascii=False));return
    inference(a,root,out,cfg)

if __name__=='__main__':
    try:main()
    except Exception as e:print('BLOCKED:',type(e).__name__,str(e),file=sys.stderr,flush=True);raise
