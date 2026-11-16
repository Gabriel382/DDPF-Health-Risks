"""Offline integration tests; no Java, GPU, Neo4j, keys, or network required."""
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ddpf.storyforest.models import Event,Link
from ddpf.storyforest.candidates import retrieve
from ddpf.storyforest.ontology import OntologyGuard
from ddpf.storyforest.classifier import classify,parse_response
from ddpf.storyforest.storage import Store,forest
from ddpf.storyforest.pipeline import process_event
from ddpf.storyforest.evaluate import evaluate_pairs
from ddpf.storyforest.detection_stream import documents_from_zip
from ddpf.storyforest.settings import Settings
from ddpf.llm.litellm_client import LiteLLMClient

def event(id_,date_,pub=None,place='City A',concept='SYN:c',evidence=True):
 d={'event_id':id_,'description':'Fictional event '+id_,'occurred_at':date_,
    'published_at':pub or date_,'location':place,'concept_ids':[concept],
    'evidence':[{'evidence_id':'E'+id_,'source_id':'source_'+id_,
                 'published_at':pub or date_,'text':'TEST ONLY, synthetic evidence'}] if evidence else []}
 return Event.from_dict(d)

class MockLLM:
 def __init__(self,response):self.response=response
 def chat(self,messages):return json.dumps(self.response)

class StoryForestTests(unittest.TestCase):
 def test_owl_guard_temporal_violation(self):
  a=event('a','2024-02-10');b=event('b','2024-02-02',pub='2024-02-11')
  self.assertFalse(OntologyGuard().assess(a,b,'documented_continuation')[0])

 def test_unknown_evidence_abstains(self):
  a=event('a','2024-02-01');b=event('b','2024-02-02')
  llm=MockLLM({'relation':'documented_continuation','evidence_ids':['INVENTED'], 'score':0.9,'rationale':'unsupported'})
  link=classify(a,b,'full',llm,OntologyGuard())
  self.assertEqual(link.relation,'unknown');self.assertIn('unrecognized_evidence_ids',link.reasons)

 def test_documented_link_with_known_ids(self):
  a=event('a','2024-02-01');b=event('b','2024-02-02')
  llm=MockLLM({'relation':'documented_continuation','evidence_ids':['Ea','Eb'],'score':0.82,'rationale':'both references'})
  link=classify(a,b,'full',llm,OntologyGuard())
  self.assertEqual(link.relation,'documented_continuation')
  # Citation ID validity does not guarantee that its prose entails the link.

 def test_unknown_when_no_evidence(self):
  a=event('a','2024-02-01',evidence=False);b=event('b','2024-02-02',evidence=False)
  link=classify(a,b,'full',MockLLM({'relation':'same_event','evidence_ids':[], 'score':1,'rationale':''}),OntologyGuard())
  self.assertEqual(link.relation,'unknown')

 def test_invalid_json_abstain(self):
  self.assertEqual(parse_response('No idea')['relation'],'unknown')

 def test_partial_label_evaluation(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'gold.jsonl'
   path.write_text('\n'.join([json.dumps({'parent_id':'a','child_id':'b','label':'documented_continuation','basis':'independent_documented'}),json.dumps({'parent_id':'a','child_id':'c','label':'unknown','basis':'independent_documented'}),json.dumps({'parent_id':'x','child_id':'y','label':'same_event','basis':'synthetic_only'})]),encoding='utf-8')
   link=Link('a','b','documented_continuation',['Ea'],'test',0.9,'full','accepted')
   r=evaluate_pairs([link],str(path),'full')
   self.assertEqual(r['verified_pairs'],1);self.assertEqual(r['excluded_unknown_or_unverified'],2);self.assertEqual(r['f1'],1)

 def test_pipeline_persistence_and_forest(self):
  with tempfile.TemporaryDirectory() as tmp:
   with Store(str(Path(tmp)/'db.sqlite')) as s:
    a=event('a','2024-02-01');b=event('b','2024-02-02')
    client=MockLLM({'relation':'documented_continuation','evidence_ids':['Ea'],'score':0.8,'rationale':'test'})
    self.assertTrue(process_event(s,a,'full',client).created)
    self.assertTrue(process_event(s,b,'full',client).created)
    self.assertFalse(process_event(s,b,'full',client).created)
    trees=forest(s.list_events(),s.links('full'))
    self.assertEqual(len(trees),1);self.assertEqual(len(trees[0]['nodes']),2)

 def test_bounded_zip_processing(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'texts.zip'
   with zipfile.ZipFile(path,'w') as z:
    z.writestr('a.txt','short text');z.writestr('b.txt','x'*500);z.writestr('other.exe','ignore')
   docs=list(documents_from_zip(str(path),max_bytes=200))
   self.assertEqual(docs,[('a.txt','short text')])

 def test_litellm_provider_switch(self):
  for provider,prefix in [('ollama','ollama/'),('openrouter','openrouter/')]:
   captured={}
   def fake(**kw):captured.update(kw);return {'choices':[{'message':{'content':'ok'}}]}
   env={'LLM_PROVIDER':provider,'OPENROUTER_API_KEY':'test-secret','OLLAMA_MODEL':'qwen3:4b','OPENROUTER_MODEL':'openai/gpt-4o-mini'}
   with patch.dict(os.environ,env):
    config=Settings.from_env()
    self.assertEqual(LiteLLMClient(config,completion_fn=fake).generate('hello'),'ok')
    self.assertTrue(captured['model'].startswith(prefix))
    if provider=='openrouter':self.assertEqual(captured['api_key'],'test-secret')
    else:self.assertNotIn('api_key',captured)

 def test_missing_openrouter_key(self):
  with patch.dict(os.environ,{'LLM_PROVIDER':'openrouter','OPENROUTER_API_KEY':''}):
   with self.assertRaisesRegex(ValueError,'OPENROUTER_API_KEY'):
    Settings.from_env()

if __name__=='__main__':unittest.main()
