import sys,os,subprocess,tempfile,json,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
root=Path(__file__).resolve().parents[2];env=dict(os.environ,PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,v):
        data=json.dumps(v).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        self.reply({'version':'0.6.10'} if self.path=='/api/version' else {'models':[{'name':'qwen3:4b','digest':'fixture','details':{'quantization_level':'Q4_K_M'}},{'name':'embeddinggemma:latest','digest':'embeddingfixture'}]})
    def do_POST(self):
        x=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path=='/api/embed':self.reply({'embeddings':[[1.,2.,3.]]});return
        assert 'DON233' not in json.dumps(x);assert x['format']['properties']['relation']['enum'];self.reply({'message':{'content':json.dumps({'relation':'related_only','subtype':'other','score':.8,'evidence_ids':[],'rationale':'fixture','abstain_reason':''})},'done_reason':'stop','prompt_eval_count':100,'eval_count':20,'total_duration':1000})
s=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=s.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory() as d:
    r=Path(d);p=r/'outputs/realdata/reconstructed_observations.jsonl';p.parent.mkdir(parents=True)
    rows=[{'observation_id':f'Journal_COVID::wkg:{i}','country_name':f'Country {i}','country_id':f'wkg:{i}','source':'Journal_COVID','disease_label':'0','top_concepts':[{'id':'C0000001','name':'Fever','intensity':3}], 'relation_dates':['2019-11' if i<2 else '2019-12'],'provenance':{'entity_granularity':'country_by_source'}} for i in range(4)]
    p.write_text('\n'.join(map(json.dumps,rows)))
    cmd=[sys.executable,'-m','scripts.realdata.reviewer_v8_ddpf.runner','--root',d,'--config',str(root/'configs/reviewer_v8_ddpf/default.json'),'--models','qwen3:4b','--ollama-url',f'http://127.0.0.1:{s.server_port}']
    for flag in ('--preflight','--pilot','--pilot'):
        z=subprocess.run(cmd+[flag],env=env,capture_output=True,text=True);print(flag,z.returncode,z.stdout[-220:]);assert z.returncode==0,z.stderr+z.stdout
    m=json.loads((r/'outputs/realdata/reviewer_revision_v8_ddpf_final/HIST_sentence_pilot/run_manifest.json').read_text());assert m['new_llm_calls']==0;assert m['status']=='COMPLETE';assert m['variants']['qwen3_4b__C3']['cache_hits']==6
    z=subprocess.run(cmd+['--full','--chronology','occurrence'],env=env,capture_output=True,text=True);assert z.returncode!=0 and 'No selected candidates with strict ordering' in z.stderr
    z=subprocess.run(cmd+['--full','--exploratory','--conditions','C3'],env=env,capture_output=True,text=True);assert z.returncode==0,z.stderr+z.stdout
    for _ in range(2):
        z=subprocess.run(cmd+['--baselines-only','--embedding','binary_bow'],env=env,capture_output=True,text=True);assert z.returncode==0,z.stderr+z.stdout
    print('CLI embeddinggemma:latest, preflight, pilot cache resume, collection full run, occurrence gate, baseline-only resume: PASS')
s.shutdown()
