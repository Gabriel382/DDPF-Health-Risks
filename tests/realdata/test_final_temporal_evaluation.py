import csv
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'/'realdata'))
import final_temporal_evaluation as audit
import final_knn_reference_v4 as base

class FinalAuditTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.st=self.root/'study';self.st.mkdir()
        self.snapshots=self.root/'snapshots.csv'
        self.profiles=self.root/'profiles.jsonl'
        cases=[
            ('C5203670','COVID-19','wkg:11','Alpha',[1,2,2,4,4,4,5,7,7],'0','C01'),
            ('C0276180','Mpox','wkg:22','Beta',[1,1,1,1,2,2,3,3,3],'1','C02'),
        ]
        start=date(2023,11,13)  # Mondays; holdout target weeks Jan 1 onwards
        rows=[];profile_rows=[];series=[]
        for did, disease,cid,country,values,lab,concept in cases:
            prev=''
            for i, v in enumerate(values):
                week=start+timedelta(days=i*7)
                eid=f'KGWEEK:{did}:{cid}:{week.isoformat()}'
                row={'episode_id':eid,'disease':disease,'disease_id':did,
                     'country_id':cid,'country_name':country,'week_start':week.isoformat(),
                     'week_end':(week+timedelta(days=6)).isoformat(),
                     'observation_date':week.isoformat(), 'reported_value':float(v),
                     'number_of_reports_in_week':1,'source_relation_id':f'source:{did}:{i}',
                     'previous_episode_id':prev,'previous_observation_date':'',
                     'weeks_since_previous_snapshot':1 if i else '',
                     'difference_from_previous_snapshot':'','difference_kind':'','quality_flags':''}
                rows.append(row);prev=eid
            profile_rows.append({'disease_label':lab,'country_id':cid,'source':'Social',
                                 'relation_dates':['2020-02' if lab=='0' else '2022-05'],
                                 'top_concepts':[{'id':concept,'name':concept}]})
        with self.snapshots.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        self.profiles.write_text(''.join(json.dumps(x)+'\n' for x in profile_rows))
        series=base.load_snapshots(self.snapshots)
        profiles=base.load_static_profiles(self.profiles)
        examples=base.make_examples(series,profiles)
        training=[e for e in examples if e.target.week < date(2024,1,1)]
        holdout=[e for e in examples if e.target.week >= date(2024,1,1)]
        m2n=base.m2_knn_predict(training,holdout,k=9,use_semantic=False)
        m2k=base.m2_knn_predict(training,holdout,k=9,use_semantic=True)
        # Two episodes selected, one from each disease; exhaustive 1 paired sample.
        sample={next(e.origin.episode_id for e in holdout if e.origin.disease==d)
                for d in ('COVID-19','Mpox')}
        predictions=[];cache=[]
        for i,e in enumerate(holdout):
            for method, y in [(audit.M1,base.m1_persistence(e)),(audit.M2N,m2n[i]),(audit.M2K,m2k[i])]+(
                  [(audit.M3,'flat'),(audit.M4,'flat')] if e.origin.episode_id in sample else []):
                predictions.append({'method':method,'split':'holdout_sample' if method in (audit.M3,audit.M4) else 'holdout_all',
                    'origin_id':e.origin.episode_id,'target_id':e.target.episode_id,
                    'disease':e.origin.disease,'country':e.origin.country,
                    'origin_week':e.origin.week.isoformat(),'target_week':e.target.week.isoformat(),
                    'target_label':e.target_label,'prediction':y,
                    'has_static_kg_context':str(int(bool(e.semantic_keys))),
                    'source_relation_id_origin':e.origin.source_relation_id,
                    'source_relation_id_target':e.target.source_relation_id})
                if method in (audit.M3,audit.M4):
                    cache.append({'method':method,'episode_id':e.origin.episode_id,'prediction':y})
        with (self.st/'predictions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        (self.st/'llm_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in cache))
        m={'cutoff_date':'2024-01-01','last_train_target':'2023-12-25','first_holdout_target':'2024-01-01',
           'sha256_snapshots':audit.file_sha256(self.snapshots),'sha256_profiles':audit.file_sha256(self.profiles),
           'input':str(self.snapshots),'profiles':str(self.profiles),
           'n_holdout_examples':len(holdout),'completed_llm_calls':4,'expected_llm_calls':4,
           'paired_complete_examples':2,'knn_k':9}
        (self.st/'manifest.json').write_text(json.dumps(m))
        self.rows=predictions

    def test_dataset_integrity(self):
        state,rows,_=audit.validate_dataset(self.st,self.snapshots)
        self.assertEqual(state['status'],'PASS')
        self.assertEqual(state['pairwise_llm_episodes'],2)
        self.assertEqual(state['llm_cache_entries'],4)
        self.assertTrue(state['train_target_before_holdout'])

    def test_tampered_target_detected(self):
        p=self.st/'predictions.csv'
        data=audit.read_csv(p)
        data[0]['target_label']='revision' if data[0]['target_label']!='revision' else 'increase'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        with self.assertRaisesRegex(ValueError,'Target label'):
            audit.validate_dataset(self.st,self.snapshots)

    def test_hash_mismatch_detected(self):
        self.snapshots.write_text(self.snapshots.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            audit.validate_dataset(self.st,self.snapshots)

    def test_kg_placebo_reproduces_original(self):
        p=audit.kg_negative_control(self.st,self.rows,permutations=3,seed=2026)
        self.assertTrue(p['verified_exact_reproduction_of_original_M2KG_predictions'])
        self.assertEqual(p['n_kg_eligible_episodes'],4)
        self.assertEqual(p['placebo_permutations'],3)

    def test_country_signflip_repeatability(self):
        pairs=audit.make_pairs(audit.read_predictions_from_rows(self.rows),audit.M3,audit.M4)
        a=audit.cluster_signflip_accuracy(pairs,200,42)
        b=audit.cluster_signflip_accuracy(pairs,200,42)
        self.assertEqual(a,b)

    def test_class_baselines_and_paired_cohorts(self):
        table=audit.cohort_table(self.rows)
        self.assertTrue(any(x['method']=='always_flat' for x in table))
        self.assertTrue(any(x['method']==audit.M2K and x['cohort']=='kg_eligible' for x in table))
        pairs=audit.paired_comparisons(self.rows,10,10,42)
        self.assertEqual(pairs[0]['n'],4)
        self.assertEqual(pairs[3]['n'],2)

if __name__=='__main__':unittest.main()
