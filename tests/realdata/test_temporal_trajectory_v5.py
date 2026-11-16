import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/realdata/temporal_trajectory_study_v5.py'
spec=importlib.util.spec_from_file_location('temporal_trajectory_study_v5',SCRIPT)
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

class Reply:
    def __init__(self,content,done_reason='stop'):
        self.data=json.dumps({'message':{'content':content},'done_reason':done_reason,
                              'eval_count':21}).encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return self.data

class TestV5(unittest.TestCase):
    def client(self,initial=256,maximum=1024):
        cfg=SimpleNamespace(provider='ollama',model='qwen3:4b',
            ollama_base_url='http://localhost:11434',temperature=0.0)
        return mod.StrictOllamaClient(cfg,240,initial,max_output_tokens=maximum)

    def test_schema_required_prediction(self):
        body=self.client().request_body([{'role':'user','content':'hello'}])
        self.assertEqual(body['format']['required'],['prediction'])

    def test_schema_enum(self):
        body=self.client().request_body([])
        self.assertEqual(body['format']['properties']['prediction']['enum'],
                         ['increase','flat','revision'])

    def test_schema_no_extra_fields(self):
        body=self.client().request_body([])
        self.assertIs(body['format']['additionalProperties'],False)

    def test_think_false(self):
        self.assertIs(self.client().request_body([])['think'],False)

    def test_native_valid(self):
        with patch.object(mod.base,'urlopen',return_value=Reply('{"prediction":"flat"}')):
            self.assertEqual(json.loads(self.client().chat([]))['prediction'],'flat')

    def test_invalid_label_repair(self):
        with patch.object(mod.base,'urlopen',side_effect=[Reply('{"prediction":"stable"}'),
             Reply('{"prediction":"increase"}')]) as req:
            value=self.client().chat([])
        self.assertEqual(json.loads(value)['prediction'],'increase')
        self.assertEqual(req.call_count,2)

    def test_thought_field_is_not_prediction(self):
        with patch.object(mod.base,'urlopen',side_effect=[Reply('{"thought":"private reasoning"}'),
             Reply('{"thought":"private reasoning"}')]):
            with self.assertRaises(mod.ModelOutputError) as e:self.client().chat([])
        self.assertEqual(e.exception.category,'invalid_response_after_repair')
        self.assertEqual(e.exception.diagnostics['attempts'][0]['json_keys'],['thought'])
        self.assertNotIn('private reasoning',str(e.exception.diagnostics))

    def test_extra_field_rejected(self):
        with patch.object(mod.base,'urlopen',side_effect=[
                Reply('{"prediction":"flat","thought":"a"}'),
                Reply('{"prediction":"flat","thought":"b"}')]):
            with self.assertRaises(mod.ModelOutputError):self.client().chat([])

    def test_truncation_budget_retry(self):
        with patch.object(mod.base,'urlopen',side_effect=[
                Reply('{"prediction":','length'),Reply('{"prediction":"flat"}')]) as mock:
            client=self.client(192,384)
            self.assertEqual(json.loads(client.chat([]))['prediction'],'flat')
        budgets=[json.loads(x.args[0].data)['options']['num_predict'] for x in mock.call_args_list]
        self.assertEqual(budgets,[192,384])

    def test_truncated_at_max_is_error(self):
        with patch.object(mod.base,'urlopen',return_value=Reply('{','length')):
            with self.assertRaises(mod.ModelOutputError) as e:self.client(192,384).chat([])
        self.assertEqual(e.exception.category,'truncated_at_max_budget')

    def test_http_timeout_not_prediction(self):
        with patch.object(mod.base,'urlopen',side_effect=URLError('timeout')):
            with self.assertRaisesRegex(RuntimeError,'connection/timeout'):
                self.client().chat([])

    def test_missing_kg_is_abstention_not_forged(self):
        example=SimpleNamespace(semantic_keys=set())
        self.assertEqual(mod.base.llm_predict(example,'M4_llm_kg_guard',None)['reason'],
                         'missing_static_kg_profile')

    def test_diagnostic_does_not_log_output(self):
        info=self.client().diagnostic('{"prediction":"stable","thought":"something"}',
                                      {'done_reason':'stop'},256)
        self.assertEqual(info['json_keys'],['prediction','thought'])
        self.assertNotIn('something',str(info))


if __name__=='__main__':unittest.main()

class EndToEndTests(unittest.TestCase):
    def test_no_llm_synthetic_pipeline_and_manifest(self):
        import csv
        import tempfile
        from datetime import date, timedelta
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            f=root/'snapshots.csv'
            fields=['episode_id','disease','disease_id','country_id','country_name',
                    'week_start','observation_date','reported_value','source_relation_id',
                    'previous_episode_id','difference_from_previous_snapshot']
            with f.open('w',newline='') as writer_file:
                w=csv.DictWriter(writer_file,fieldnames=fields);w.writeheader()
                for disease_id,disease in [('C5203670','COVID-19'),('C0276180','Mpox')]:
                    previous=''
                    for i in range(35):
                        dt=date(2023,10,2)+timedelta(weeks=i)
                        eid=f'{disease_id}:nation:{dt.isoformat()}'
                        w.writerow(dict(episode_id=eid,disease=disease,disease_id=disease_id,
                            country_id='nation',country_name='Nation',week_start=dt.isoformat(),
                            observation_date=(dt+timedelta(days=6)).isoformat(),
                            reported_value=10+i,source_relation_id=str(i),
                            previous_episode_id=previous,difference_from_previous_snapshot=1))
                        previous=eid
            profiles=root/'profiles.jsonl'
            with profiles.open('w') as out:
                for label in ['0','1']:
                    out.write(json.dumps({'disease_label':label,'country_id':'nation',
                        'source':'Social_COVID','relation_dates':['2020-02'],
                        'top_concepts':[{'id':'C123','name':'Fever'}]})+'\n')
            output=root/'result'
            mod.main(['--snapshots',str(f),'--profiles',str(profiles),
                      '--output',str(output),'--skip-llm','--max-llm-per-disease','1'])
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['expected_llm_calls'],0)
            self.assertEqual(manifest['technical_failures_or_missing_calls'],0)
            self.assertEqual(manifest['paired_complete_examples'],0)
            self.assertTrue(manifest['M2_noKG_ablation_included'])
