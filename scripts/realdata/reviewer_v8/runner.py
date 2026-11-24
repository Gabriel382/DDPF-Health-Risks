"""One-shot orchestration; no previous artifacts are mutated."""
import argparse,os,sys,traceback,platform,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from .core import *
from . import temporal,documentary,ddpf_audit

def capture_legacy(root,out):
    files=[]
    names=['temporal_trajectory_study_v5.py','llm_size_study_v6.py','who_storyforest_final_v7.py','finish_who_storyforest_v7_1.py','build_weekly_episodes.py','export_isreported_series.py','recover_kg_observations.py']
    for name in names:
        p=root/'scripts'/'realdata'/name
        if p.is_file():files.append({'path':str(p.relative_to(root)),'sha256':sha(p)})
    for p in (root/'src'/'ddpf'/'storyforest').glob('*.py'):files.append({'path':str(p.relative_to(root)),'sha256':sha(p)})
    def git(*args):
        z=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True);return z.stdout.strip() if z.returncode==0 else None
    return {'commit':git('rev-parse','HEAD'),'branch':git('rev-parse','--abbrev-ref','HEAD'),'working_tree_status':git('status','--porcelain'),'legacy_sources':files,'python':sys.version,'platform':platform.platform()}

def report(out,manifest):
    lines=['# PreDiViD reviewer revision experiments — V8.1','','Integrated contribution: PropaPhen + full DDPF + evidence-aware Story Forest. Ontology checks are evaluated in the integrated workflow, not as a new isolated ontology contribution.','','**Run status:** '+manifest['status'],'','| Stage | Status |','|---|---|']
    for k,v in manifest['stages'].items():lines.append(f"| {k} | {v.get('status','UNKNOWN')} |")
    lines+=['','## Temporal comparison (exact original cohort; new shared prompt)','']
    p=out/'temporal'/'metrics.json'
    if p.exists():
        lines+=['| Model | Condition | Cohort | n | Accuracy | Macro-F1 | Coverage |','|---|---|---|---:|---:|---:|---:|']
        for r in load(p):
            if r['disease']!='ALL':continue
            def f(k):return 'NA' if r[k] is None else f'{r[k]:.4f}'
            lines.append(f"| {r['model']} | {r['condition']} | {r['cohort']} | {r['n']} | {f('accuracy')} | {f('macro_f1')} | {f('coverage')} |")
    p=out/'documentary'/'forest_metrics.csv'
    lines+=['','## Documentary forest construction','']
    if p.exists():
        lines+=['| Model | Condition | Candidate coverage | Edges | Reference matches | Direct recall |','|---|---|---:|---:|---:|---:|']
        for r in csvread(p):lines.append(f"| {r['model']} | {r['condition']} | {r['candidate_coverage']} | {r['n_edges']} | {r['matched']} | {r['direct_edge_recall']} |")
    lines+=['','## Scientific boundaries and reviewer tasks','', '- Do not describe a different candidate schedule as the original 106 pairs. Check `documentary/input_manifest.json`.','- No gold labels are manufactured. Reviewer files are blank and require independent human annotation and consensus.','- Static KG context is not historical epidemiological evidence. Inspect `semantic_audit.json` and `semantic_control.json`; no semantic benefit is claimed if retrieval fails or the control is non-informative.','- The end-to-end experiment matches generated cards by source, date, and type; unresolved identity, unmatched edges, and unlabelled predictions remain separate.','- The new forest implementation is a fully specified chronological single-parent insertion baseline/extension; it is not a claim to reproduce the exact V7.1 SQLite insertion function.','- Inspect all stages, complete human review, and rerun `--score-reviewed` before calling the revision complete.','','## Artifacts','', 'The machine-readable reports and cached predictions in this output directory are the authoritative results. Do not infer missing metrics from this summary.']
    (out/'reviewer_revision_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--output',default='outputs/realdata/reviewer_revision_v8');p.add_argument('--models',nargs='+',default=['qwen3:4b','qwen3:32b','gemma3:4b','gemma3:27b']);p.add_argument('--provider',choices=['ollama','litellm'],default='ollama');p.add_argument('--ollama-url',default=None);p.add_argument('--timeout',type=int,default=600);p.add_argument('--context-tokens',type=int,default=16384);p.add_argument('--pull-missing',action='store_true');p.add_argument('--skip-neo4j',action='store_true');p.add_argument('--embedding-model',default='embeddinggemma');p.add_argument('--preflight',action='store_true');p.add_argument('--score-reviewed',action='store_true');p.add_argument('--only',choices=['all','temporal','documentary'],default='all');a=p.parse_args()
    root=Path(a.root).resolve();out=(root/a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(root/'.env')
    except ImportError:pass
    base=a.ollama_url or os.getenv('OLLAMA_BASE_URL') or os.getenv('OLLAMA_API_BASE') or 'http://localhost:11434';base=base.rstrip('/')
    if base.endswith('/api'):base=base[:-4]
    os.chdir(root);manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'scope':'reviewer revision V8.1','config':vars(a),'legacy':capture_legacy(root,out),'stages':{}};client=None;sem=None
    # Review-only must not overwrite previous run provenance; create a separate manifest.
    mpath=out/('human_review_run_manifest.json' if a.score_reviewed else 'preflight_manifest.json' if a.preflight else 'run_manifest.json')
    if not a.preflight and not a.score_reviewed:client=Client(out,a.provider,base,a.timeout,a.context_tokens,a.pull_missing)
    def stage(name,func):
        print('\n===',name,'===',flush=True)
        try:manifest['stages'][name]=func()
        except Exception as e:
            manifest['stages'][name]={'status':'BLOCKED_OR_FAILED','error':str(e),'traceback':traceback.format_exc()};print(traceback.format_exc(),flush=True)
        dump(mpath,manifest)
    # Check both data tracks before allocating hours to inference. Fail closed on input mismatches.
    if not a.preflight and not a.score_reviewed:
        checks={}
        if a.only in ('all','temporal'):
            try:checks['temporal']=temporal.run(root,out/'temporal',None,a.models,preflight=True)
            except Exception as e:checks['temporal']={'status':'BLOCKED_OR_FAILED','error':str(e)}
        if a.only in ('all','documentary'):
            try:checks['documentary']=documentary.run(root,out/'documentary',None,a.models,None,preflight=True)
            except Exception as e:checks['documentary']={'status':'BLOCKED_OR_FAILED','error':str(e)}
        dump(out/'automatic_preflight.json',checks)
        if any(v.get('status')!='PREFLIGHT_OK' for v in checks.values()):
            manifest.update(stages=checks,status='PREFLIGHT_BLOCKED',finished_utc=datetime.now(timezone.utc).isoformat());dump(mpath,manifest);report(out,manifest)
            if client:client.close()
            print(json.dumps({'status':'PREFLIGHT_BLOCKED','checks':checks},indent=2,default=str),flush=True);return 2
    try:
        if not a.score_reviewed and not a.preflight and a.only=='all':stage('ddpf_evidence_inventory',lambda:ddpf_audit.run(root,out/'ddpf'))
        if not a.score_reviewed and a.only in ('all','temporal'):stage('temporal',lambda:temporal.run(root,out/'temporal',client,a.models,preflight=a.preflight))
        if a.only in ('all','documentary') or a.score_reviewed:
            def docs():
                nonlocal sem
                d=out/'documentary';d.mkdir(exist_ok=True)
                if not a.preflight and not a.score_reviewed:sem=documentary.Semantic(root,d,not a.skip_neo4j)
                return documentary.run(root,d,client,a.models,sem,review_only=a.score_reviewed,preflight=a.preflight,embedding_model=a.embedding_model)
            stage('documentary',docs)
    finally:
        if sem:sem.close()
        if client:
            dump(out/'model_metadata.json',client.meta);client.close()
    statuses=[v.get('status') for v in manifest['stages'].values()];manifest['finished_utc']=datetime.now(timezone.utc).isoformat();manifest['status']='PREFLIGHT_OK' if a.preflight and all(s=='PREFLIGHT_OK' for s in statuses) else 'HUMAN_REVIEW_COMPLETE' if a.score_reviewed and all(s=='HUMAN_REVIEW_COMPLETE' for s in statuses) else 'COMPUTATIONAL_STAGES_COMPLETE' if statuses and all(s in ('COMPLETE','INVENTORY_COMPLETE') for s in statuses) else 'PARTIAL_OR_PENDING';dump(mpath,manifest)
    if not a.score_reviewed and not a.preflight:report(out,manifest)
    print(json.dumps({'status':manifest['status'],'output':str(out),'stages':{k:v.get('status') for k,v in manifest['stages'].items()}},indent=2),flush=True)
    return 0 if manifest['status'] in ('PREFLIGHT_OK','HUMAN_REVIEW_COMPLETE','COMPUTATIONAL_STAGES_COMPLETE') else 2
if __name__=='__main__':sys.exit(main())
