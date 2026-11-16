"""Regression tests: masked observations may legitimately have zero graph neighbors."""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE=Path(__file__).resolve().parents[2]/'scripts'/'realdata'/'structural_observation_embeddings.py'
spec=importlib.util.spec_from_file_location('structural_fix',MODULE)
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def rows():
    def item(i,name):
        return {
            'observation_id':f'obs{i}',
            'country_id':f'country{i%2}',
            'source':'Journal_COVID',
            'disease_label':int(i%2==1),
            'top_concepts':[{'id':name,'name':name}],
            'provenance':{'sampled':False},
        }
    return [item(0,'COVID-19'),item(1,'Monkeypox'),
            item(2,'Generic asthma'),item(3,'Generic fever'),
            item(4,'Generic cough'),item(5,'Generic fatigue')]


class IsolatedStructuralTests(unittest.TestCase):
    def test_fastrp_isolated_are_neutral_and_deterministic(self):
        r=rows()
        _,A,stats=s.build_graph(r,masked=True,with_country=False)
        degree=np.asarray(A.sum(axis=1)).ravel()
        self.assertEqual(np.flatnonzero(degree[:len(r)]==0).tolist(),[0,1])
        a=s.fastrp_style_embed(A,len(r),8,42)
        b=s.fastrp_style_embed(A,len(r),8,42)
        self.assertEqual(a.shape,(6,8))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a[:2],np.zeros((2,8),dtype='float32'))
        self.assertTrue(np.any(a[2:]!=0))
        np.testing.assert_array_equal(a,b)

    def test_country_edges_can_rescue_isolated_observation(self):
        r=rows()
        _,A,_=s.build_graph(r,masked=True,with_country=True)
        degree=np.asarray(A.sum(axis=1)).ravel()
        self.assertTrue((degree[:len(r)]>0).all())
        a=s.fastrp_style_embed(A,len(r),8,42)
        self.assertTrue(np.isfinite(a).all())

    def test_node2vec_zeroes_isolated_random_initialization(self):
        r=rows()
        _,A,_=s.build_graph(r,masked=True,with_country=False)
        a=s.node2vec_embed(A,len(r),8,42,walks=2,walk_length=6,epochs=1)
        np.testing.assert_array_equal(a[:2],np.zeros((2,8),dtype='float32'))
        self.assertTrue(np.isfinite(a).all())

    def test_graphsage_zeroes_isolated_self_features(self):
        r=rows()
        _,A,_=s.build_graph(r,masked=True,with_country=False)
        a=s.graphsage_embed(A,len(r),8,42,epochs=2)
        np.testing.assert_array_equal(a[:2],np.zeros((2,8),dtype='float32'))
        self.assertTrue(np.isfinite(a).all())

    def test_country_svd_uses_country_features_and_archives_legacy(self):
        import hashlib
        r=rows()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            inp=root/'source.jsonl'
            inp.write_text(''.join(json.dumps(x)+'\n' for x in r),encoding='utf8')
            B,A,stats=s.build_graph(r,masked=True,with_country=True)
            out=root/'svd_country';out.mkdir()
            # Simulate exactly the old version-1 cached SVD output.
            old_cfg={'input_sha256':s.sha256(inp),'algorithm':'svd','seed':42,
                     'dimension_requested':8,'mask_disease_terms':True,
                     'with_country_nodes':True,'graph_stats':stats,'version':1}
            fp=hashlib.sha256(json.dumps(old_cfg,sort_keys=True).encode()).hexdigest()
            old_matrix=s.svd_embed(B,8,42)
            np.save(out/'embeddings.npy',old_matrix)
            (out/'observations.csv').write_text('row,observation_id\n',encoding='utf8')
            (out/'manifest.json').write_text(json.dumps({**old_cfg,'fingerprint':fp,
                                                        'source_sha256':s.sha256(inp)}),encoding='utf8')
            s.write_result(out,'svd',True,True,8,42,r,B,A,stats,inp)
            archive=out/'archive_svd_country_ignored_v1'
            self.assertTrue((archive/'original_manifest.json.legacy').exists())
            self.assertTrue((archive/'original_embeddings.npy.legacy').exists())
            new=json.loads((out/'manifest.json').read_text())
            self.assertEqual(new['country_feature_policy'],'append_binary_country_nodes_v2')
            self.assertNotEqual(fp,new['fingerprint'])
            new_matrix=np.load(out/'embeddings.npy')
            self.assertEqual(new_matrix.shape[0],len(r))
            # Including country adds information: masked observations 0 and 1
            # have no concepts but a nonzero country-only representation.
            self.assertTrue(np.any(np.abs(new_matrix[:2])>0))
            s.write_result(out,'svd',True,True,8,42,r,B,A,stats,inp)

    def test_persistence_isolates_and_resume_skip_svd(self):
        r=rows()
        with tempfile.TemporaryDirectory() as d:
            t=Path(d)
            src=t/'observations.jsonl'
            src.write_text(''.join(json.dumps(x)+'\n' for x in r),encoding='utf8')
            B,A,stats=s.build_graph(r,masked=True,with_country=False)
            svd=t/'svd'; fastrp=t/'fastrp'
            s.write_result(svd,'svd',True,False,8,42,r,B,A,stats,src)
            before=json.loads((svd/'manifest.json').read_text())['fingerprint']
            s.write_result(svd,'svd',True,False,8,42,r,B,A,stats,src)
            after=json.loads((svd/'manifest.json').read_text())['fingerprint']
            self.assertEqual(before,after)
            s.write_result(fastrp,'fastrp_style',True,False,8,42,r,B,A,stats,src)
            manifest=json.loads((fastrp/'manifest.json').read_text())
            self.assertEqual(manifest['n_isolated_observations'],2)
            self.assertEqual(manifest['isolated_observation_ids'],['obs0','obs1'])
            self.assertEqual(manifest['isolated_vertex_policy'],'zero_embedding_v1')
            self.assertEqual(np.load(fastrp/'embeddings.npy').shape,(6,8))


if __name__=='__main__':unittest.main()
