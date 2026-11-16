"""Offline tests for the reproduction utilities. No Neo4j or download required."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / 'scripts' / 'realdata' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RealDataToolsTests(unittest.TestCase):
    def test_md5_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'test'
            p.write_bytes(b'abc')
            self.assertEqual(load('zenodo_fetch').md5_file(p), '900150983cd24fb0d6963f7d28e17f72')

    def test_path_checks_robust(self):
        mod=load('notebook_preflight')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'data'/'neo4j').mkdir(parents=True)
            (root/'data'/'neo4j'/'covid_observations_journal.csv').write_text('x')
            nb=root/'Experiments'/'Prediction'/'example.ipynb'
            nb.parent.mkdir(parents=True)
            match=mod.check_path(root,nb,'../data/neo4j/covid_observations_journal.csv')
            self.assertTrue(match['found'])

    def test_document_to_events_requires_real_text_and_dates(self):
        mod=load('documents_to_events')
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            inp=td/'docs.csv';out=td/'events.jsonl'
            inp.write_text('id,source,text,occurred,published,place,cuis\nA,WHO,Actual evidence,2022-05-01,2022-05-02,France,"[\'C123\']"\n',encoding='utf-8')
            n=mod.convert(inp,out,id_col='id',text_col='text',source_col='source',
                          occurred_col='occurred',published_col='published',location_col='place',concept_col='cuis')
            self.assertEqual(n,1)
            obj=json.loads(out.read_text())
            self.assertEqual(obj['evidence'][0]['text'],'Actual evidence')
            self.assertEqual(obj['concept_ids'],['C123'])
            self.assertEqual(obj['location'],'France')
            with self.assertRaises(FileExistsError):
                mod.convert(inp,out,id_col='id',text_col='text',source_col='source',
                            occurred_col='occurred',published_col='published')

    def test_bow_matches_six_csv_schema(self):
        mod=load('replay_bow')
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);src=td/'tables';src.mkdir()
            for fn,prefix,truth in mod.SOURCES:
                with (src/fn).open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=['System_Name','c.name','intensity','System_ID'])
                    writer.writeheader()
                    for i in range(3):
                        writer.writerow({'System_Name':f'SYS{i}','c.name':str(['corona','cough'] if truth==0 else ['rash','pox']),
                                         'intensity':str([3,5]),'System_ID':str(i)})
            result=mod.run(src,td/'result')
            self.assertEqual(result['n_observations'],18)
            self.assertGreaterEqual(result['ari'],0)
            self.assertTrue((td/'result'/'cluster_assignments.csv').exists())

    def test_duplicate_observation_id_fails(self):
        mod=load('replay_bow')
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            for fname,prefix,disease in mod.SOURCES:
                with (td/fname).open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=['System_Name','c.name','intensity']);w.writeheader()
                    w.writerow({'System_Name':'X','c.name':"['CUI']",'intensity':'[1]'})
                    if fname==mod.SOURCES[0][0]:
                        w.writerow({'System_Name':'X','c.name':"['CUI']",'intensity':'[1]'})
            with self.assertRaisesRegex(ValueError,'repeated'):
                mod.load_rows(td,25)

if __name__ == '__main__': unittest.main()
