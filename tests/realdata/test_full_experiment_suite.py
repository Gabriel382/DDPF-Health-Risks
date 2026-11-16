"""Unit tests for the all-experiments overlay (no server/network required)."""
from __future__ import annotations
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SRC=Path(__file__).resolve().parents[2]/'scripts'/'realdata'

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,SRC/file)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class FullSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.struct=module('struc_suite','structural_observation_embeddings.py')
        cls.bench=module('bench_suite','extended_embedding_benchmark.py')
        cls.runner=module('run_suite','run_all_predivid_experiments.py')
        cls.rows=[]
        for i in range(12):
            disease=0 if i<6 else 1
            concept='COVID-19' if disease==0 else 'Mpox'
            country=f'country-{i%6}'
            concepts=[{'name':concept,'id':concept,'intensity':10},
                      {'name':f'Common concept {i%3}','id':f'C{i%3}','intensity':7},
                      {'name':f'Extra {i%2}','id':f'E{i%2}','intensity':6}]
            cls.rows.append({'observation_id':f'Journal_{"COVID" if disease==0 else "Monkeypox"}::{country}',
                             'country_id':country,'country_name':country,
                             'source':f'Journal_{"COVID" if disease==0 else "Monkeypox"}',
                             'disease_label':disease,'top_concepts':concepts,
                             'provenance':{'sampled':False}})

    def test_graph_mask(self):
        B,A,meta=self.struct.build_graph(self.rows,True,True)
        self.assertEqual(B.shape[0],12)
        self.assertEqual(meta['country_vertices'],6)
        self.assertEqual((A!=A.T).nnz,0)
        self.assertTrue((B.sum(axis=1)>0).all())

    def test_svd_and_fastrp(self):
        import numpy as np
        B,A,meta=self.struct.build_graph(self.rows,False,False)
        svd=self.struct.svd_embed(B,8,42)
        fast=self.struct.fastrp_style_embed(A,12,8,42)
        self.assertEqual(svd.shape[0],12)
        self.assertEqual(fast.shape,(12,8))
        self.assertTrue(np.isfinite(fast).all())
        self.assertTrue(np.allclose(self.struct.fastrp_style_embed(A,12,8,42),fast))

    def test_structural_persistence_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            inp=Path(td)/'rows.jsonl'
            inp.write_text('\n'.join(json.dumps(r) for r in self.rows)+'\n',encoding='utf8')
            B,A,stats=self.struct.build_graph(self.rows,False,False)
            out=Path(td)/'embedding'
            self.struct.write_result(out,'fastrp_style',False,False,8,42,self.rows,B,A,stats,inp)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['source_sha256'],self.struct.sha256(inp))
            self.struct.write_result(out,'fastrp_style',False,False,8,42,self.rows,B,A,stats,inp)
            with self.assertRaises(ValueError):
                self.struct.write_result(out,'fastrp_style',False,False,8,13,self.rows,B,A,stats,inp)

    def test_diagnostics(self):
        d=self.bench.input_diagnostics(self.rows)
        self.assertEqual(d['n_countries'],6)
        self.assertEqual(d['countries_with_both_disease_labels'],6)
        self.assertEqual(d['explicit_disease_terms_by_label']['0']['affected'],6)

    def test_uncertainty(self):
        import numpy as np
        y=np.array([r['disease_label'] for r in self.rows])
        pred=y.copy()
        r=self.bench.uncertainty(y,pred,self.rows,20,20,42)
        self.assertEqual(r['bootstrap_valid_replicates'],20)
        self.assertAlmostEqual(r['country_block_bootstrap_CI_ARI_95'][0],1.0)
        self.assertTrue(0<r['label_permutation_p_within_source_modality']<=1)

    def test_source_alignment_refuses_mismatch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            inp=Path(td)/'rows.jsonl'; inp.write_text('\n'.join(json.dumps(r) for r in self.rows)+'\n')
            root=Path(td)/'structure';folder=root/'abc';folder.mkdir(parents=True)
            B,A,stats=self.struct.build_graph(self.rows,False,False)
            self.struct.write_result(folder,'fastrp_style',False,False,8,42,self.rows,B,A,stats,inp)
            arr=list(self.bench.source_matrices(self.rows,inp,Path(td)/'nothing',root))
            self.assertTrue(any(meta['family']=='projected_structural' for label,_,meta in arr))
            # Changing a source line invalidates its hash.
            inp.write_text(inp.read_text()+'\n')
            with self.assertRaises(ValueError):list(self.bench.source_matrices(self.rows,inp,Path(td)/'nothing',root))

    def test_storyforest_provenance_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'no_docs.jsonl'
            rows=[{'event_id':'e0','description':'a','occurred_at':'2020-01-01','published_at':'2020-01-01','evidence':[]},
                  {'event_id':'e1','description':'b','occurred_at':'2020-01-02','published_at':'2020-01-02','evidence':[]}]
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            with self.assertRaises(ValueError):self.runner.preflight_events(path)


if __name__=='__main__':unittest.main()
