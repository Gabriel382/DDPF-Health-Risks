"""Unit + mock end-to-end tests. NO WHO website or Ollama is contacted."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sp=ROOT/'scripts/realdata/who_storyforest_final_v7.py'
spec=importlib.util.spec_from_file_location('who_storyforest_final_v7',sp)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReferenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.sources,cls.events,cls.pairs=[m.load(m.REF/f'{x}.json') for x in
     ('sources','silver_events','silver_pairs')]
 def test_reference_integrity(self):self.assertTrue(m.validate_reference(self.sources,self.events,self.pairs))
 def test_no_silver_group_leak_to_event_models(self):
  t={s['source_id']:' '.join([e['evidence_anchor'] for e in self.events if e['source_id']==s['source_id']]) for s in self.sources}
  E=m.cards(self.sources,self.events,t)
  for e in E:
   self.assertNotIn('episode_group',e.to_dict()['metadata'])
   self.assertNotIn('episode_group',e.to_dict())
 def test_source_date_anchor_filter(self):
  src=self.sources[0]
  s={src['source_id']:'WHO noticed cases on 31 December 2019; 7 January 2020.'}
  with tempfile.TemporaryDirectory() as d:
   rows=[{'model':'fake','source_id':src['source_id'],'event_date':'2019-12-31',
          'event_type':'outbreak_report','description':'Wuhan cluster announced'},
         {'model':'fake','source_id':src['source_id'],'event_date':'2020-09-11',
          'event_type':'outbreak_report','description':'Unverified event'}]
   result=m.build_generated_events('fake',[src],s,rows,Path(d))
   self.assertEqual(len(result),1)
   self.assertTrue(result[0].metadata['date_anchor_verified'])
   audit=m.load(Path(d)/'generated_event_audit.json')
   self.assertEqual(audit['excluded'],1)
 def test_8_who_sources(self):
  self.assertEqual(len(self.sources),8)
  self.assertTrue(all('who.int' in s['url'] for s in self.sources))
 def test_silver_not_gold(self):
  self.assertTrue(all(e['reference_status']=='curated_silver_unadjudicated' for e in self.events))
  self.assertTrue(all('manual_review_needed' in p['annotation_basis'] for p in self.pairs))
 def test_29_events_39_pairs(self):
  self.assertEqual(len(self.events),29);self.assertEqual(len(self.pairs),39)
  self.assertEqual(sum(m.pair_is_strong(p['label']) for p in self.pairs),22)
 def test_future_reference_rejected(self):
  ev=[dict(e) for e in self.events];ev[0]['occurred_at']='2030-01-01'
  with self.assertRaisesRegex(ValueError,'Future event'):m.validate_reference(self.sources,ev,self.pairs)
 def test_non_who_reference_rejected(self):
  sources=[dict(s) for s in self.sources];sources[0]['url']='https://example.org'
  with self.assertRaisesRegex(ValueError,'non-WHO'):m.validate_reference(sources,self.events,self.pairs)
 def test_no_negative_by_absence(self):
  result=m.score_pairs([],self.pairs)
  self.assertEqual(result['n_positive'],22)
  self.assertEqual(result['fn'],0)
  self.assertEqual(result['technical_missing'],39)
  self.assertEqual(result['n_completed'],0)
 def test_core_metrics_perfect(self):
  got=[{'parent_id':p['parent_id'],'child_id':p['child_id'],'prediction':p['label']} for p in self.pairs]
  r=m.score_pairs(got,self.pairs)
  self.assertEqual(r['f1'],1.0)
  self.assertEqual(r['exact_multiclass_accuracy'],1.0)
 def test_unverified_positive_not_tp(self):
  got=[{'parent_id':p['parent_id'],'child_id':p['child_id'],'prediction':'unknown'} for p in self.pairs]
  r=m.score_pairs(got,self.pairs);self.assertEqual(r['tp'],0)
  self.assertEqual(r['fn'],22)
 def test_extraction_slot_recall_not_precision(self):
  r=m.extraction_reference_recall([{'event_date':'2019-12-31','event_type':'outbreak_report','description':'x'}],self.events,'DON233')
  self.assertEqual(r['covered'],1)
  self.assertLess(r['recall'],1)
 def test_schema_relation_invalid(self):
  with self.assertRaises(ValueError):m.validate_link({'relation':'causes','evidence_ids':[],'rationale':'x','score':0.7})
 def test_schema_extra_field_invalid(self):
  with self.assertRaises(ValueError):m.validate_link({'relation':'same_event','evidence_ids':[],'rationale':'x','score':0.7,'thinking':'secret'})
 def test_schema_valid(self):
  self.assertIsNone(m.validate_link({'relation':'unknown','evidence_ids':[],'rationale':'n/a','score':0.0}))
 def test_extraction_invalid_calendar_date(self):
  with self.assertRaises(ValueError):m.validate_extraction({'events':[{'event_date':'2020-99-99','event_type':'symptom_onset','description':'x'}]})
 def test_extraction_valid(self):
  self.assertIsNone(m.validate_extraction({'events':[{'event_date':'2020-01-01','event_type':'outbreak_report','description':'x'}]}))
 def test_fake_html_skips_scripts(self):
  t=m.clean_who_html(b'<html><main><script>IGNORE ME</script><h1>Situation at a glance</h1><p>WHO data 2022 and disease</p></main></html>')
  self.assertNotIn('IGNORE ME',t)
  self.assertIn('WHO data',t)
 def test_anchor_span(self):
  self.assertIsNone(m.anchor_span('some text','missing'))
  self.assertIn('13 May',m.anchor_span('WHO reported on 13 May','13 May'))
 def test_tree_structure(self):
  from ddpf.storyforest.models import Event
  ev=[Event(event_id='a',description='a',occurred_at='2020-01-01',published_at='2020-01-01')]
  t=m.tree_metrics([{'nodes':[{'event_id':'a','parent_id':None}]}],ev)
  self.assertTrue(t['acyclic']);self.assertTrue(t['all_nodes_included'])
 def test_cache_reuses_valid_output(self):
  with tempfile.TemporaryDirectory() as d:
   client=m.OllamaJSON('fake',Path(d))
   obj={'relation':'related_only','evidence_ids':[],'rationale':'x','score':0.2}
   with patch.object(m.OllamaJSON,'_request',return_value=(obj,{'elapsed_s':0.01})) as call:
    a=client.chat([{'role':'user','content':'test'}]);b=client.chat([{'role':'user','content':'test'}])
    self.assertEqual(a,b);self.assertEqual(call.call_count,1)
   reloaded=m.OllamaJSON('fake',Path(d))
   with patch.object(m.OllamaJSON,'_request',side_effect=AssertionError('network should not be called')):
    reloaded.chat([{'role':'user','content':'test'}])
 def test_errors_not_cached(self):
  with tempfile.TemporaryDirectory() as d:
   client=m.OllamaJSON('fake',Path(d),retries=0)
   with patch.object(m.OllamaJSON,'_request',side_effect=RuntimeError('timeout')):
    with self.assertRaises(RuntimeError):client.chat([{'role':'user','content':'test'}])
   self.assertFalse(client.path.exists())

class OtherHelpers(unittest.TestCase):
 def test_paired_block_signflip_perfect_tie(self):
  from ddpf.storyforest.models import Event
  e1=Event(event_id='x',description='x',occurred_at='2020-01-01',published_at='2020-01-02',metadata={'source_document':'DON233'})
  e2=Event(event_id='y',description='y',occurred_at='2020-01-02',published_at='2020-01-02',metadata={'source_document':'DON233'})
  p=[{'parent_id':'x','child_id':'y','label':'documented_continuation'}]
  pred=[{'model':'a','method':'llm_only','parent_id':'x','child_id':'y','prediction':'documented_continuation'},
        {'model':'b','method':'llm_only','parent_id':'x','child_id':'y','prediction':'documented_continuation'},
        {'model':'a','method':'full','parent_id':'x','child_id':'y','prediction':'documented_continuation'},
        {'model':'b','method':'full','parent_id':'x','child_id':'y','prediction':'documented_continuation'}]
  r=m.paired_llm_comparisons(pred,p,[e1,e2],['a','b'])
  self.assertEqual(len(r),4)
  self.assertTrue(all(x['cluster_signflip_two_sided_p']==1.0 for x in r))
 def test_technical_prediction_excluded(self):
  ref=[{'parent_id':'x','child_id':'y','label':'related_only'}]
  mtr=m.score_pairs([{'parent_id':'x','child_id':'y','prediction':'technical_missing'}],ref)
  self.assertEqual(mtr['tn'],0)
  self.assertEqual(mtr['coverage'],0.0)
 def test_tree_reference_counts(self):
  ref=[{'parent_id':'a','child_id':'b','label':'documented_continuation'},
       {'parent_id':'a','child_id':'c','label':'related_only'}]
  tree=[{'nodes':[{'event_id':'a','parent_id':None},
                  {'event_id':'b','parent_id':'a'},
                  {'event_id':'c','parent_id':None}]}]
  got=m.tree_reference_scores(tree,ref)
  self.assertEqual(got['retrieved_gold_edges_in_forest'],1)
  self.assertEqual(got['known_negative_edges_in_forest'],0)

class EndToEndTests(unittest.TestCase):
 def test_synthetic_extraction_to_generated_forest(self):
  import re
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'generated';sourcepath=out/'sources';sourcepath.mkdir(parents=True)
   sources,ev,pairs=[m.load(m.REF/f'{x}.json') for x in ('sources','silver_events','silver_pairs')]
   for ss in sources:
    anchors=' '.join('WHO documented '+e['evidence_anchor']+' during test. ' for e in ev if e['source_id']==ss['source_id'])
    html=f'<html><main><h2>Situation at a glance</h2><p>{ss["published_at"]} {anchors}</p><p>{"test article "*80}</p></main></html>'
    (sourcepath/(ss['source_id']+'.html')).write_text(html,encoding='utf-8')
   def fake_request(self,messages,schema,budget):
    if 'events' in schema['properties']:
     matches=re.findall(r'\b20\d\d-\d\d-\d\d\b',messages[-1]['content'])
     assert matches
     return {'events':[{'event_date':matches[0],'event_type':'outbreak_update','description':'A dated WHO situation update'}]}, {'elapsed_s':0.01}
    return {'relation':'related_only','evidence_ids':[],'rationale':'mock','score':0.3}, {'elapsed_s':0.01}
   with patch.object(m.OllamaJSON,'_request',new=fake_request):
    rc=m.main(['--models','fake:0.6b','--tree-model','fake:0.6b',
               '--output',str(out),'--offline','--timeout','1'])
   self.assertEqual(rc,0)
   self.assertTrue((out/'generated_trees'/'trees'/'full.json').exists())
   audit=m.load(out/'generated_event_audit.json')
   self.assertEqual(audit['accepted_date_anchored'],8)
   self.assertTrue((out/'paper_integrated_results_v7.md').exists())
 def test_synthetic_docs_all_pipeline_no_network(self):
  # Local fake pages are not WHO validation; the test only asserts plumbing and report.
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'out';sourcepath=out/'sources';sourcepath.mkdir(parents=True)
   sources,ev,pairs=[m.load(m.REF/f'{x}.json') for x in ('sources','silver_events','silver_pairs')]
   for s in sources:
    anchors=' '.join('WHO documented '+e['evidence_anchor']+' during testing. ' for e in ev if e['source_id']==s['source_id'])
    html=f'<html><main><h2>Situation at a glance</h2><p>{s["published_at"]} {anchors}</p><p>{"test corpus "*80}</p></main></html>'
    (sourcepath/(s['source_id']+'.html')).write_text(html,encoding='utf-8')
   result={'relation':'related_only','evidence_ids':[],'rationale':'mock non-link','score':0.3}
   with patch.object(m.OllamaJSON,'_request',return_value=(result,{'elapsed_s':0.01})):
    # no extraction: the mock contains only the link schema
    rc=m.main(['--models','fake:0.6b','--tree-model','fake:0.6b',
               '--output',str(out),'--offline','--skip-extraction','--timeout','1'])
   self.assertEqual(rc,0)
   self.assertEqual(m.load(out/'study_manifest.json')['status'],'COMPLETE_WITH_LIMITATIONS')
   self.assertTrue((out/'final_documentary_report.md').exists())
   self.assertTrue((out/'human_adjudication.csv').exists())
   with (out/'pair_metrics.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),4)
   self.assertTrue((out/'trees'/'full.json').exists())
   self.assertEqual(m.load(out/'source_audit.json')['status'],'PASS')

if __name__=='__main__':unittest.main()
