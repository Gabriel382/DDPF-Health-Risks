import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date,timedelta
from pathlib import Path

HERE=Path(__file__).resolve().parents[2]
SCRIPT=HERE/'scripts/realdata/temporal_trajectory_study.py'
spec=importlib.util.spec_from_file_location('predivid_trajectory',SCRIPT)
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

COLS=['episode_id','disease','disease_id','country_id','country_name',
      'week_start','observation_date','reported_value','source_relation_id',
      'previous_episode_id','difference_from_previous_snapshot']

def create_csv(path, gap=False):
    start=date(2023,10,2)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS)
        w.writeheader()
        for disease_id,disease in (('C5203670','COVID-19'),('C0276180','Mpox')):
            for country in ('a','b','c'):
                prev=''
                for week in range(25):
                    if gap and country=='a' and week==8:
                        continue
                    d=start+timedelta(weeks=week)
                    key=f'{disease_id}:{country}:{d}'
                    val=100+week*(1 if country=='a' else 0)+max(0,week-10)*(country=='c')
                    w.writerow(dict(episode_id=key,disease=disease,disease_id=disease_id,
                        country_id=country,country_name=country,week_start=str(d),
                        observation_date=str(d+timedelta(days=6)),reported_value=val,
                        source_relation_id='rel:'+key,previous_episode_id=prev))
                    prev=key

class TemporalStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.input=self.root/'snapshots.csv'
        create_csv(self.input)

    def test_no_future_in_payload(self):
        series=mod.load_snapshots(self.input)
        examples=mod.make_examples(series,{})
        ex=examples[-1]
        p=mod.payload(ex,False)
        self.assertEqual(p['target_value'],'HIDDEN_FOR_EVALUATION')
        self.assertEqual(p['history'][-1]['reported_value'],ex.origin.value)
        self.assertFalse(any(entry['week']==ex.target.week.isoformat() for entry in p['history']))

    def test_train_test_chronology_and_shapes(self):
        ex=mod.make_examples(mod.load_snapshots(self.input),{})
        cutoff=date(2024,1,1)
        train=[x for x in ex if x.target.week<cutoff]
        test=[x for x in ex if x.target.week>=cutoff]
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertLess(max(x.target.week for x in train),min(x.target.week for x in test))
        self.assertEqual(len(mod.m2_knn_predict(train,test,k=3)),len(test))

    def test_gap_fail_closed(self):
        create_csv(self.input,gap=True)
        with self.assertRaisesRegex(ValueError,'Gap'):
            mod.load_snapshots(self.input)

    def test_static_profiles_and_missing(self):
        path=self.root/'static.jsonl'
        path.write_text(json.dumps({'disease_label':0,'country_id':'a',
            'source':'Social_COVID','relation_dates':['2020-02'],
            'top_concepts':[{'id':'C123','name':'Fever'}]})+'\n')
        profiles=mod.load_static_profiles(path)
        e=mod.make_examples(mod.load_snapshots(self.input),profiles)
        match=next(x for x in e if x.origin.country_id=='a' and x.origin.disease_id=='C5203670')
        self.assertIn('C123',match.semantic_keys)
        self.assertIn('Fever',match.semantic_names)
        absent=next(x for x in e if x.origin.country_id=='b')
        self.assertFalse(absent.semantic_keys)

    def test_llm_json_parsing(self):
        self.assertEqual(mod.safe_json_response('garbled')['prediction'],'unknown')
        self.assertEqual(mod.safe_json_response('{"prediction":"flat","reason":"x"}')['prediction'],'flat')
        self.assertEqual(mod.safe_json_response('{"prediction":"false"}')['prediction'],'unknown')

    def test_llm_guards_and_context(self):
        e=mod.make_examples(mod.load_snapshots(self.input),{})
        ex=e[-1]
        class Mock:
            def chat(self,messages):
                return '{"prediction":"increase","reason":"uncertain"}'
        self.assertEqual(mod.llm_predict(ex,'M3_llm_only',Mock())['prediction'],'increase')
        self.assertEqual(mod.llm_predict(ex,'M4_llm_kg_guard',Mock())['prediction'],'unknown')
        self.assertFalse(mod.payload(ex,True)['graph_context']['graph_context_is_week_specific'])

    def test_metrics_abstain_and_revision(self):
        rows=[dict(target_label='increase',prediction='increase'),
              dict(target_label='flat',prediction='unknown'),
              dict(target_label='revision',prediction='revision')]
        m=mod.metrics(rows,'test')
        self.assertEqual(m['n_primary'],2)
        self.assertEqual(m['n_true_revision'],1)
        self.assertEqual(m['n_abstain'],1)
        self.assertAlmostEqual(m['coverage'],2/3,places=6)
        self.assertTrue(m['macro_f1']<1.0)

    def test_full_dry_run_and_paths(self):
        out=self.root/'res'
        mod.main(['--snapshots',str(self.input),'--profiles',str(self.root/'none.jsonl'),
                  '--output',str(out),'--max-llm-per-disease','2','--skip-llm'])
        self.assertTrue((out/'manifest.json').exists())
        self.assertTrue((out/'metrics.csv').exists())
        self.assertTrue((out/'temporal_adjacency_not_link_gold.jsonl').exists())
        manifest=json.loads((out/'manifest.json').read_text())
        self.assertFalse(manifest['is_document_grounded_Story_Forest'])
        self.assertEqual(manifest['n_llm_selected'],4)

if __name__=='__main__':
    unittest.main()
