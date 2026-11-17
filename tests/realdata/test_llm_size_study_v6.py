import csv
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/realdata'))
import llm_size_study_v6 as v6


def row(label='increase', prediction='increase', episode='x', country='a'):
    return dict(target_label=label,prediction=prediction,
                episode_id=episode,country_id=country)


class UnitTests(unittest.TestCase):
    def test_tag_slug(self):
        self.assertEqual(v6.tag_id('qwen3:0.6b'),'qwen3_0_6b')
        self.assertNotEqual(v6.tag_id('qwen3:0.6b'), v6.tag_id('qwen3:1.7b'))

    def test_model_parameter_size(self):
        self.assertEqual(v6.model_billions('qwen3:32b'),32.)
        self.assertEqual(v6.model_billions('qwen3:0.6b'),0.6)
        self.assertIsNone(v6.model_billions('qwen3:instruct'))

    def test_score_perfect(self):
        r=v6.metric_rows([row('flat','flat'),row('increase','increase')])
        self.assertEqual(r['accuracy'],1.)
        self.assertEqual(r['macro_f1'],1.)
        self.assertEqual(r['recall_increase'],1.)

    def test_score_imbalanced_flat_does_not_hide_zero_increase_recall(self):
        r=v6.metric_rows([row('flat','flat') for _ in range(9)] + [row('increase','flat')])
        self.assertEqual(r['accuracy'],0.9)
        self.assertLess(r['macro_f1'],0.6)
        self.assertEqual(r['recall_increase'],0.)

    def test_empty_metrics_are_null(self):
        r=v6.metric_rows([])
        self.assertIsNone(r['accuracy'])
        self.assertIsNone(r['macro_f1'])

    def test_rank_ties(self):
        self.assertEqual(v6.rank([2,1,2,3]),[2.5,1.,2.5,4.])

    def test_spearman_exact_strong_direction(self):
        r=v6.spearman_exact([1,2,3,4],[.2,.3,.4,.5])
        self.assertEqual(r['rho'],1.)
        self.assertGreater(r['p_two_sided'],0)
        self.assertLess(r['p_two_sided'],.1)

    def test_spearman_constant_score(self):
        r=v6.spearman_exact([1,2,3],[.5,.5,.5])
        self.assertIsNone(r['rho'])

    def test_bootstrap_identical(self):
        aa=[row(episode=str(i),country=f'c{i//3}') for i in range(9)]
        r=v6.country_bootstrap_paired(aa,aa,resamples=40)
        self.assertEqual(r['delta_accuracy'],0.)
        self.assertEqual(r['accuracy_95pct_CI'],[0.,0.])

    def test_bootstrap_advantage(self):
        aa=[row('increase','flat',str(i),f'c{i//2}') for i in range(12)]
        bb=[row('increase','increase',str(i),f'c{i//2}') for i in range(12)]
        r=v6.country_bootstrap_paired(aa,bb,resamples=40)
        self.assertEqual(r['delta_accuracy'],1.)
        self.assertEqual(r['countries'],6)
        self.assertEqual(r['macro_f1_95pct_CI'],[1.,1.])

    def test_load_cache_rejects_other_signature(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.jsonl'
            v6.append_jsonl(path,dict(signature='wrong',method='M3_llm_only',episode_id='e',
                                     target_episode_id='t',prediction='flat'))
            with self.assertRaisesRegex(RuntimeError,'incompatible cache'):
                v6.load_model_cache(path,'right',[])

    def test_load_cache_rejects_invalid_label(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.jsonl'
            v6.append_jsonl(path,dict(signature='s',method='M3_llm_only',episode_id='e',
                                     target_episode_id='t',prediction='unknown'))
            with self.assertRaisesRegex(RuntimeError,'Invalid cached'):
                v6.load_model_cache(path,'s',[])

    def test_load_cache_rejects_wrong_target_episode(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.jsonl'
            v6.append_jsonl(path,dict(signature='s',method='M3_llm_only',episode_id='e',
                                     target_episode_id='bad',prediction='flat'))
            sample=[SimpleNamespace(origin=SimpleNamespace(episode_id='e'),
                                    target=SimpleNamespace(episode_id='good'))]
            with self.assertRaisesRegex(RuntimeError,'not in current sample'):
                v6.load_model_cache(path,'s',sample)

    def test_v5_import_rejects_mismatched_signature(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            v6.append_jsonl(root/'llm_predictions.jsonl',dict(
                method='M3_llm_only',episode_id='e',target_episode_id='t',
                model='ollama/qwen3:4b',signature='wrong',prediction='flat'))
            sample=[SimpleNamespace(origin=SimpleNamespace(episode_id='e'),
                                    target=SimpleNamespace(episode_id='t'))]
            dest=root/'v6.jsonl'
            self.assertFalse(v6.import_4b_cache(root,dest,sample,'s','good',{}))
            self.assertFalse(dest.exists())

    def test_same_messages_and_schema_as_v5(self):
        settings=SimpleNamespace(provider='ollama',model='ollama/qwen3:8b',
                                 ollama_base_url='http://localhost:11434',temperature=0.)
        client=v6.v5.StrictOllamaClient(settings,240,256,num_ctx=4096,max_output_tokens=1024)
        req=client.request_body([{'role':'system','content':v6.base.PROMPT_SYSTEM}])
        self.assertEqual(req['model'],'qwen3:8b')
        self.assertFalse(req['think'])
        self.assertEqual(req['format'],v6.v5.SCHEMA)
        self.assertEqual(req['options']['temperature'],0.)


class SyntheticEndToEnd(unittest.TestCase):
    def test_one_shot_resumes_and_imports_4b_without_inference(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            snap=root/'snapshots.csv'
            fields=sorted(v6.base.SNAPSHOT_FIELDS)
            with snap.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                for concept,disease in [('C5203670','COVID-19'),('C0276180','Mpox')]:
                    prev=''
                    for i in range(50):
                        week=(date(2023,10,2)+timedelta(weeks=i)).isoformat()
                        eid=f'{concept}:x:{week}'
                        writer.writerow(dict(episode_id=eid,disease=disease,disease_id=concept,
                            country_id='x',country_name='Mockland',week_start=week,
                            observation_date=week,reported_value=i*3,source_relation_id=str(i),
                            previous_episode_id=prev,difference_from_previous_snapshot=3))
                        prev=eid
            prof=root/'profiles.jsonl'
            with prof.open('w') as f:
                for lab in ('0','1'):
                    f.write(json.dumps(dict(disease_label=lab,country_id='x',source='mock',
                        relation_dates=['2020-02'],top_concepts=[{'id':'C1','name':'Fever'}]))+'\n')
            out_v5=root/'v5'
            out_v5.mkdir()
            data=dict(input=str(snap),profiles=str(prof),
                      sha256_snapshots=v6.base.hash_file(snap),
                      sha256_profiles=v6.base.hash_file(prof),
                      cutoff_date='2024-01-01',sample_all=False,
                      n_llm_selected=40,llm_study_complete=True)
            (out_v5/'manifest.json').write_text(json.dumps(data))
            all_examples=v6.base.make_examples(v6.base.load_snapshots(snap),
                                               v6.base.load_static_profiles(prof))
            pool=[x for x in all_examples if x.target.week >= date(2024,1,1) and x.semantic_keys]
            selected=v6.base.stable_sample(pool,20)
            self.assertEqual(len(selected),40)
            sig=v6.original_v5_signature(data,0.,256,1024,4096)
            for ex in selected:
                for method in v6.METHODS:
                    v6.append_jsonl(out_v5/'llm_predictions.jsonl',dict(
                        signature=sig,model='ollama/qwen3:4b',method=method,
                        episode_id=ex.origin.episode_id,target_episode_id=ex.target.episode_id,
                        prediction='flat',elapsed_seconds=2.,ollama_stats={'eval_count':8}))
            with (out_v5/'predictions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['method','origin_id','prediction'])
                w.writeheader()
                for ex in selected:
                    for method in v6.BASELINES:
                        w.writerow(dict(method=method,origin_id=ex.origin.episode_id,prediction='flat'))
            out=root/'v6'
            fake_reply={'message':{'content':'{"prediction":"flat"}'},'done_reason':'stop','eval_count':12,
                        'eval_duration':1_000_000_000}
            with patch.object(v6.v5.StrictOllamaClient,'_post',return_value=fake_reply) as post, \
                 patch.object(v6,'model_metadata',return_value={'parameter_size':'8B','quantization_level':'Q4_K_M'}), \
                 patch.object(v6,'free_previous_model'):
                status=v6.main(['--v5-output',str(out_v5),'--output',str(out),
                                '--models','qwen3:4b,qwen3:8b','--skip-pull'])
                self.assertEqual(status,0)
                self.assertEqual(post.call_count,80)
                first_manifest=json.loads((out/'study_manifest.json').read_text())
                self.assertEqual(first_manifest['n_common_complete_all_models_both_methods'],40)
                self.assertEqual(len(v6.load_jsonl(out/'predictions_qwen3_4b.jsonl')),80)
                self.assertEqual(len(v6.load_jsonl(out/'predictions_qwen3_8b.jsonl')),80)
                again=v6.main(['--v5-output',str(out_v5),'--output',str(out),
                               '--models','qwen3:4b,qwen3:8b','--skip-pull'])
                self.assertEqual(again,0)
                self.assertEqual(post.call_count,80)
                self.assertIn('M3_llm_only', (out/'size_comparison_report.md').read_text())
                self.assertEqual(len(v6.load_jsonl(out/'predictions_qwen3_8b.jsonl')),80)
                self.assertEqual(first_manifest['status'],'COMPLETE')


if __name__=='__main__':
    unittest.main()
