"""Local offline V7.1 regression tests, no WHO or Ollama network calls."""
import csv
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/realdata'))
p=ROOT/'scripts/realdata/finish_who_storyforest_v7_1.py'
spec=importlib.util.spec_from_file_location('finish_who_storyforest_v7_1',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from ddpf.storyforest.models import Event,Evidence
from ddpf.storyforest.classifier import SYS, event_context


class Dates(unittest.TestCase):
 def test_iso(self):self.assertEqual(m.canonical_date('2022-05-21'),'2022-05-21')
 def test_date_don385(self):self.assertEqual(m.canonical_date('21 May 2022'),'2022-05-21')
 def test_date_don388(self):self.assertEqual(m.canonical_date('29 May 2022'),'2022-05-29')
 def test_month_first(self):self.assertEqual(m.canonical_date('May 21, 2022'),'2022-05-21')
 def test_ambiguous_refused(self):
  for text in ('05/06/2022','yesterday','May 2022','35 May 2022','2022-13-15'):
   with self.subTest(text=text),self.assertRaises(ValueError):m.canonical_date(text)
 def test_atomic(self):
  obj={'events':[{'event_date':'21 May 2022','event_type':'outbreak_update','description':'x'},
                 {'event_date':'May 2022','event_type':'outbreak_update','description':'y'}]}
  with self.assertRaises(ValueError):m.normalize_validate_extraction(obj)
  self.assertEqual(obj['events'][0]['event_date'],'21 May 2022')
 def test_invoke_normalizes_before_cache(self):
  with tempfile.TemporaryDirectory() as d:
   changes=[];m.safe_normalize_dates(changes)
   source={'source_id':'DON385','title':'WHO','published_at':'2022-05-21'}
   fake={'events':[{'event_date':'21 May 2022','event_type':'outbreak_update','description':'WHO update'}]}
   with patch.object(m.v7.OllamaJSON,'_request',return_value=(fake,{'elapsed_s':0.01})) as call:
    client=m.v7.OllamaJSON('qwen3:0.6b',Path(d))
    result=client.extract(source,'Source text 21 May 2022')
    self.assertEqual(result[0]['event_date'],'2022-05-21')
    self.assertEqual(call.call_count,1)
   self.assertEqual(changes,[{'raw_date':'21 May 2022','iso_date':'2022-05-21'}])
   with patch.object(m.v7.OllamaJSON,'_request',side_effect=AssertionError('No call allowed')):
    another=m.v7.OllamaJSON('qwen3:0.6b',Path(d))
    self.assertEqual(another.extract(source,'Source text 21 May 2022')[0]['event_date'],'2022-05-21')


def ev(name,when):
 return Event(event_id=name,description='patient event '+name,
   occurred_at=when,published_at='2022-05-21',location='United Kingdom',
   concept_ids=['C0276180'],ontology_types=['lab_confirmation'],
   evidence=[Evidence(evidence_id='WHO-'+name,source_id='DON385',text='WHO patient',published_at='2022-05-21')])


def cache_file(out,events,all_methods=True):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 lines=[]
 for model in ('qwen3:4b','qwen3:32b'):
  # V7.1 requires a complete batch of original 39 fixed pair caches.
  for k in range(39):
   lines.append({'model':model,'kind':'link','key':f'DUMMY-{model}-{k}',
                'output':{'relation':'unknown','evidence_ids':[],'rationale':'dummy','score':0.0}})
  for method in ('llm_only','full'):
   a,b=events
   context={'event_a':event_context(a,method!='llm_only'),
            'event_b':event_context(b,method!='llm_only')}
   messages=[{'role':'system','content':SYS},
             {'role':'user','content':json.dumps(context,ensure_ascii=False)}]
   key=m.v7.hash_obj(m.CacheOnlyLinker._sig(model,messages))
   lines.append({'model':model,'kind':'link','key':key,
     'output':{'relation':'documented_continuation','evidence_ids':['WHO-'+a.event_id],
               'rationale':'Both mentions are explicitly the same patient','score':0.85}})
 with (out/'ollama_cache.jsonl').open('w',encoding='utf-8') as f:
  for item in lines:f.write(json.dumps(item)+'\n')


class Replay(unittest.TestCase):
 def test_cache_only_miss_abstains_no_network(self):
  with tempfile.TemporaryDirectory() as d:
   cache_file(d,[ev('A','2022-05-01'),ev('B','2022-05-02')])
   client=m.CacheOnlyLinker('qwen3:32b',d)
   output=json.loads(client.chat([{'role':'user','content':'not cached'}]))
   self.assertEqual(output['relation'],'unknown');self.assertEqual(client.not_queried,1)
 def test_incremental_real_store_accepts_cached_link(self):
  with tempfile.TemporaryDirectory() as d:
   a,b=ev('A','2022-05-01'),ev('B','2022-05-02')
   cache_file(d,[a,b])
   pairs=[{'parent_id':'A','child_id':'B','label':'documented_continuation'}]
   out=Path(d)/'results'
   rows=m.tree_replay([a,b],pairs,out,d,top_k=4)
   self.assertEqual(len(rows),4)
   self.assertTrue(all(x['n_edges']==1 and x['direct_edge_recall_silver']==1 for x in rows))
   self.assertTrue(all(x['cached_decisions']==1 for x in rows))
   self.assertTrue(all(x['unqueried_candidates_abstained']==0 for x in rows))
 def test_unqueried_pair_not_mistaken_as_negative(self):
  with tempfile.TemporaryDirectory() as d:
   a,b=ev('A','2022-05-01'),ev('B','2022-05-02')
   cache_file(d,[a,b])
   c=ev('C','2022-05-03')
   out=Path(d)/'results'
   rows=m.tree_replay([a,b,c],[],out,d,top_k=4)
   self.assertEqual(len(rows),4)
   self.assertTrue(all(x['unqueried_candidates_abstained']>=1 for x in rows))
   self.assertTrue(all(x['n_edges']==1 for x in rows))
   for r in rows:
    record=m.v7.load(out/'trees'/f'{r["model"].split(":")[1]}_{r["method"]}.json')
    unknown=[x for x in record['links'] if x['child_id']=='C']
    self.assertTrue(unknown and all(x['relation']=='unknown' for x in unknown))
 def test_replay_no_silver_prediction_leak(self):
  with tempfile.TemporaryDirectory() as d:
   a,b=ev('A','2022-05-01'),ev('B','2022-05-02')
   cache_file(d,[a,b])
   positive=m.tree_replay([a,b],[{'parent_id':'A','child_id':'B','label':'documented_continuation'}],Path(d)/'p',d)
   negative=m.tree_replay([a,b],[{'parent_id':'A','child_id':'B','label':'related_only'}],Path(d)/'n',d)
   self.assertEqual([x['n_edges'] for x in positive],[x['n_edges'] for x in negative])
   self.assertNotEqual([x['direct_edge_recall_silver'] for x in positive],[x['direct_edge_recall_silver'] for x in negative])


class Report(unittest.TestCase):
 def test_report_marks_silver_and_not_end_to_end(self):
  with tempfile.TemporaryDirectory() as d:
   ext={'qwen3:0.6b':{'documents_ok':6,'documents_total':8,
        'matched_slots':1,'reference_slots_in_completed_documents':27,'slot_recall_conditional_on_completed':.037}}
   tree=[{'model':'qwen3:32b','method':'full','cached_decisions':4,
          'unqueried_candidates_abstained':2,'n_edges':2,'direct_edge_recall_silver':.25}]
   pair=[{'model':'qwen3:32b','method':'full','n_labeled':'39','f1':'0.89','technical_missing':'0'}]
   f=Path(d)/'report.md'
   m.write_summary(f,ext,tree,pair,[],{'source_audit':'PASS','v7_cache_sha256':'x'})
   text=f.read_text()
   self.assertIn('not a full end-to-end forest benchmark',text)
   self.assertIn('silver',text.lower())
   self.assertIn('unqueried',text.lower())
 def test_source_fixture_offline_end_to_end(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'who';sourcepath=out/'sources';sourcepath.mkdir(parents=True)
   sources,ann,pairs=[m.v7.load(m.v7.REF/f'{n}.json') for n in ('sources','silver_events','silver_pairs')]
   for s in sources:
    anchors=' '.join('WHO documented '+e['evidence_anchor']+' during testing.' for e in ann if e['source_id']==s['source_id'])
    html=f'<html><main><h2>Situation at a glance</h2><p>{s["published_at"]} {anchors}</p><p>{"test corpus "*80}</p></main></html>'
    (sourcepath/(s['source_id']+'.html')).write_text(html,encoding='utf-8')
   cache_file(out,[ev('A','2022-05-01'),ev('B','2022-05-02')])
   with (out/'pair_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['model','method','n_labeled','f1','technical_missing']);w.writeheader()
    for model in ('heuristic','qwen3:0.6b','qwen3:4b','qwen3:32b'):
     for method in ('llm_only','full'):
      w.writerow({'model':model,'method':method,'n_labeled':39,'f1':'0.7','technical_missing':0})
   (out/'pair_predictions.csv').write_text('model,method\n')
   with (out/'extraction_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['model','source_id','status','gold_slots','covered']);w.writeheader()
    for model in m.MODEL_NAMES:
     for s in sources:w.writerow({'model':model,'source_id':s['source_id'],
       'status':'OK','gold_slots':2,'covered':1})
   rc=m.run(SimpleNamespace(original_output=str(out),timeout=3,retries=0,top_k=4,no_extraction_retry=True))
   self.assertEqual(rc,0)
   result=m.v7.load(out/'final_v7_1'/'manifest_v7_1.json')
   self.assertFalse(result['full_end_to_end_generated_forest_validated'])
   self.assertEqual(result['status'],'CONTROLLED_PILOT_COMPLETE_WITH_LIMITATIONS')
   self.assertTrue((out/'final_v7_1'/'cached_tree_replay_metrics.csv').exists())

if __name__=='__main__':unittest.main()
