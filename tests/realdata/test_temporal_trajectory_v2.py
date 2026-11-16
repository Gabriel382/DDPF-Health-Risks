import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/realdata/temporal_trajectory_study_v2.py'
spec = importlib.util.spec_from_file_location('predivid_trajectory_v2', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class V2Tests(unittest.TestCase):
    def test_json_with_reasoning_prefix(self):
        text = '<think>My internal text</think>\n```json\n{"prediction":"increase","reason":"x"}\n```'
        self.assertEqual(mod.safe_json_response(text)['prediction'], 'increase')
        self.assertEqual(mod.safe_json_response('')['reason'], 'empty_model_content')
        self.assertEqual(mod.safe_json_response('{}')['reason'], 'invalid_response')

    def test_json_errors_fail_loudly(self):
        class Mock:
            def chat(self, messages):
                return 'not JSON'
        class Ex:
            semantic_keys = {'cui'}
            origin = type('O', (), {'disease_id':'C', 'disease':'COVID-19',
                                    'country_id':'P', 'country':'Country',
                                    'source_relation_id':'2', 'week':date(2024,1,1)})()
            target = type('T', (), {'week':date(2024,1,8)})()
            history = [type('H', (), {'week':date(2024,1,1), 'value':3.0})()]
        with self.assertRaisesRegex(RuntimeError, 'Invalid Ollama response'):
            mod.llm_predict(Ex(), 'M3_llm_only', Mock())

    def test_no_kg_is_not_auto_forged(self):
        class Ex:
            semantic_keys = set()
        class Mock:
            def chat(self, messages):
                return '{"prediction":"flat","reason":"x"}'
        self.assertEqual(mod.llm_predict(Ex(), 'M4_llm_kg_guard', Mock())['reason'],
                         'missing_static_kg_profile')

    def test_v2_baselines_and_sampling(self):
        import csv
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_file = root / 'snapshots.csv'
            fields = ['episode_id','disease','disease_id','country_id','country_name',
                      'week_start','observation_date','reported_value','source_relation_id',
                      'previous_episode_id','difference_from_previous_snapshot']
            with csv_file.open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                for disease_id, disease in [('C5203670','COVID-19'), ('C0276180','Mpox')]:
                    for country in ['one', 'two']:
                        prev=''
                        for week_ix in range(30):
                            d=date(2023,10,2)+timedelta(weeks=week_ix)
                            eid=f'{disease_id}:{country}:{d.isoformat()}'
                            writer.writerow({
                                'episode_id':eid,'disease':disease,'disease_id':disease_id,
                                'country_id':country,'country_name':country,
                                'week_start':d.isoformat(),'observation_date':(d+timedelta(days=6)).isoformat(),
                                'reported_value':str(10+week_ix),'source_relation_id':str(week_ix),
                                'previous_episode_id':prev,'difference_from_previous_snapshot':1})
                            prev=eid
            profiles=root/'profiles.jsonl'
            with profiles.open('w') as f:
                for label in ['0','1']:
                    for country in ['one','two']:
                        f.write(json.dumps({'disease_label':label,'country_id':country,
                                            'source':'Social_COVID', 'relation_dates':['2020-02'],
                                            'top_concepts':[{'id':'C123','name':'Fever'}]})+'\n')
            out=root/'res'
            mod.main(['--snapshots',str(csv_file),'--profiles',str(profiles),'--output',str(out),
                      '--max-llm-per-disease','1','--skip-llm'])
            report=json.loads((out/'manifest.json').read_text())
            assert report['n_llm_selected']==2
            assert report['n_kg_eligible_holdout'] > 0
            assert report['M2_noKG_ablation_included'] is True
            names=[r['method'] for r in report['metrics']]
            assert 'M2_temporal_knn_noKG__all_holdout' in names

if __name__ == '__main__':
    unittest.main()
