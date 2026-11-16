import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]/'scripts'/'realdata'

def load(path):
    spec=importlib.util.spec_from_file_location(path,BASE/(path+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

recover=load('recover_kg_observations')
embed=load('embed_kg_observations')
benchmark=load('compare_kg_embeddings')

class ReconstructionTests(unittest.TestCase):
    def test_normalization_fallback(self):
        r=dict(source='Journal_COVID',country_id='wkg:1',country_name=None,concept_name='fever',concept_id='C0015967',intensity='7',relation_date='2020-02')
        self.assertEqual(recover.normalized_row(r),('Journal_COVID','wkg:1','wkg:1','fever','C0015967',7,'2020-02'))
        self.assertIsNone(recover.normalized_row({**r,'country_id':None}))

    def test_top25_best_tie(self):
        ranked=recover.top_ranked(['Fever','Fever','Cough'],['C1','C1','C2'],[3,8,7],2)
        self.assertEqual([x['name'] for x in ranked],['Fever','Cough'])
        self.assertEqual(ranked[0]['intensity'],8)

    def test_reconstruction_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stage=root/'stage.db'
            db=recover.create_stage(stage)
            rows=[('Journal_COVID','wkg:11','France','fever','C1',5,'2020-02'),
                  ('Journal_COVID','wkg:11','France','cough','C2',8,'2020-03'),
                  ('Journal_Monkeypox','wkg:12','Spain','rash','C3',6,'2022-06')]
            db.executemany('INSERT INTO observed VALUES (?,?,?,?,?,?,?)',rows)
            db.commit()
            info=recover.write_files(db,root,25,False,{'read':3,'kept':3})
            db.close()
            self.assertEqual(info['total_observations'],2)
            csv_path=root/'data/neo4j/covid_observations_journal.csv'
            self.assertTrue(csv_path.is_file())
            obs=(root/'outputs/realdata/reconstructed_observations.jsonl').read_text().splitlines()
            self.assertEqual(len(obs),2)
            self.assertIn('rash',obs[1])
            self.assertEqual(json.loads(obs[0])['top_concepts'][0]['name'],'cough')

    def test_no_label_leakage(self):
        obj={'observation_id':'Medical_COVID::c1','disease_label':0,'source':'Journal_COVID',
             'top_concepts':[{'name':'fever'},{'name':'cough'}],'country_name':'France',
             'relation_dates':['2020-01']}
        self.assertEqual(embed.observation_text(obj,'concepts'),'Biomedical concepts: fever; cough.')
        self.assertNotIn('Journal_COVID',embed.observation_text(obj,'concepts_country_time'))
        self.assertIn('France',embed.observation_text(obj,'concepts_country_time'))

    def test_tag_and_embedding_payload(self):
        self.assertEqual(embed.model_slug('qwen3-embedding:0.6b'),'qwen3-embedding_0.6b')
        with patch.object(embed,'post_json',return_value={'embeddings':[[0.1,0.2],[0.2,0.3]]}) as mocked:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                objs=[{'observation_id':f'x{i}','top_concepts':[{'name':'fever'}],
                       'source':'Journal_COVID','disease_label':0} for i in range(2)]
                path=root/'observations.jsonl'
                path.write_text(''.join(json.dumps(x)+'\n' for x in objs))
                dest=embed.embed_one(objs,path,'test-embed','concepts',root/'out','http://localhost:11434',2,False,5)
                import numpy as np
                x=np.load(dest/'embeddings.npy')
                self.assertEqual(x.shape,(2,2))
                self.assertEqual(mocked.call_args[0][1]['input'],['Biomedical concepts: fever.']*2)
                self.assertEqual(mocked.call_args[0][1]['truncate'],False)

    def test_matrix_alignment_guard(self):
        self.assertTrue(callable(benchmark.read_matrix))
        self.assertTrue(callable(benchmark.metrics_for))

    def test_multi_seed_clustering(self):
        import numpy as np
        X=np.asarray([[1,0],[1,0.2],[0,1],[0.1,1]],dtype='float32')
        result=benchmark.metrics_for(X,np.asarray([0,0,1,1]),[9,42])
        self.assertEqual(len(result['runs']),2)
        self.assertAlmostEqual(result['aggregate']['ARI']['mean'],1.0,places=4)

    def test_masked_empty_concepts_remain_an_observation(self):
        obj={'observation_id':'e','top_concepts':[{'name':'Mpox'}]}
        self.assertIn('[none]',embed.observation_text(obj,'concepts',True))

    def test_mismatched_model_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/'emb'
            folder.mkdir()
            (folder/'manifest.json').write_text(json.dumps({'source_sha256':'wrong','sample':False}))
            f=root/'rows.jsonl';f.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'hash'):
                benchmark.read_matrix(folder,[{'observation_id':'x'}],f)


if __name__=='__main__':unittest.main()
