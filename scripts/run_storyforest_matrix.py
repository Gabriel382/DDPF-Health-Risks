"""Run independent research conditions with separate SQLite stores.

Cloud LLM methods require explicit --allow-billable-llm; API keys are NEVER
written to the run manifest. This is orchestration, not final publication.
"""
from __future__ import annotations
import argparse,subprocess,sys,json,os,datetime,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--gold',default=None)
 p.add_argument('--methods',default='temporal,kg_rules,llm_only,llm_kg,full')
 p.add_argument('--allow-billable-llm',action='store_true',help='Explicitly allow network/model calls')
 p.add_argument('--output-dir',default='outputs/storyforest/experiments');p.add_argument('--max-events',type=int,default=0)
 a=p.parse_args();methods=[m.strip() for m in a.methods.split(',')]
 supported={'temporal','kg_rules','llm_only','llm_kg','full'}
 if any(m not in supported for m in methods):p.error('Invalid method')
 if any(m in {'llm_only','llm_kg','full'} for m in methods) and not a.allow_billable_llm:
  p.error('LLM runs can cost money; pass --allow-billable-llm to confirm intentional calls')
 base=Path(a.output_dir);base.mkdir(parents=True,exist_ok=True)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 run=base/stamp
 if run.exists():p.error('Run directory already exists')
 run.mkdir()
 manifest={'created_utc':stamp,'input':str(Path(a.input).resolve()),
   'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
   'methods':methods,'max_events':a.max_events,'commands':[],
   'warning':'No end-to-end epidemiological claim; gold/reference coverage must be audited.'}
 for method in methods:
  db=str(run/f'{method}.sqlite3')
  command=[sys.executable,'-m','ddpf.storyforest.cli','--db',db,'run','--input',a.input,'--method',method,'--output',str(run/f'{method}.json')]
  if a.max_events:command+=['--max-events',str(a.max_events)]
  print('RUN:',method,flush=True)
  subprocess.run(command,check=True)
  manifest['commands'].append({'method':method,'db':db,'returncode':0})
  if a.gold:
   subprocess.run([sys.executable,'-m','ddpf.storyforest.cli','--db',db,'evaluate','--gold',a.gold,'--method',method,'--output',str(run/f'{method}_eval.json')],check=True)
 (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('Completed: ',run)
if __name__=='__main__':main()
