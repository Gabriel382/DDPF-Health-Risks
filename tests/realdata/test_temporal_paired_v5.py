import sys
import tempfile
import unittest
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'realdata'))
from audit_temporal_paired_v5 import country_id, read_predictions, make_pairs, score, paired_bootstrap, run


class PairedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv = Path(self.temp.name)/'predictions.csv'
        origins = [
            ('KGWEEK:C0276180:wkg:1:2024-01-01','Mpox', 'flat', '1'),
            ('KGWEEK:C0276180:wkg:1:2024-01-08','Mpox', 'increase', '1'),
            ('KGWEEK:C5203670:wkg:2:2024-01-01','COVID-19', 'flat', '0'),
            ('KGWEEK:C5203670:wkg:3:2024-01-01','COVID-19', 'increase', '1'),
        ]
        predictions = {
            'M2_temporal_knn_noKG':['flat','flat','flat','flat'],
            'M2_temporal_staticKG_knn':['flat','increase','flat','increase'],
            'M3_llm_only':['flat','flat',None,'increase'],
            'M4_llm_kg_guard':['flat','increase',None,'flat'],
        }
        rows = []
        for method, preds in predictions.items():
            for i, pred in enumerate(preds):
                if pred is None:
                    continue
                eid, disease, truth, haskg = origins[i]
                rows.append({'method':method,
                             'split':'holdout_sample' if method.startswith(('M3','M4')) else 'holdout_all',
                             'origin_id':eid,'target_label':truth,'prediction':pred,
                             'disease':disease,'has_static_kg_context':haskg})
        with self.csv.open('w',newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_country_id_parsing(self):
        self.assertEqual(country_id('KGWEEK:C0276180:wkg:424314830:2024-01-22'),'wkg:424314830')
        with self.assertRaises(ValueError):
            country_id('bad')

    def test_pairs_and_kg_filter(self):
        r=read_predictions(self.csv)
        a=make_pairs(r,'M2_temporal_knn_noKG','M2_temporal_staticKG_knn')
        self.assertEqual(len(a),4)
        self.assertEqual(len(make_pairs(r,'M2_temporal_knn_noKG','M2_temporal_staticKG_knn',kg_only=True)),3)
        self.assertEqual(len(make_pairs(r,'M3_llm_only','M4_llm_kg_guard')),3)
        self.assertEqual(len({p['country_id'] for p in a}),3)

    def test_scoring(self):
        pairs=make_pairs(read_predictions(self.csv),'M2_temporal_knn_noKG','M2_temporal_staticKG_knn')
        self.assertEqual(score(pairs,'a')['accuracy'], .5)
        self.assertEqual(score(pairs,'b')['accuracy'], 1.)
        self.assertEqual(score(pairs,'b')['macro_f1'], 1.)

    def test_bootstrap_repeatability(self):
        pairs=make_pairs(read_predictions(self.csv),'M2_temporal_knn_noKG','M2_temporal_staticKG_knn')
        a=paired_bootstrap(pairs, 100, 42)
        b=paired_bootstrap(pairs, 100, 42)
        self.assertEqual(a,b)
        self.assertEqual(a['n_country_blocks'],3)

    def test_end_to_end(self):
        output=Path(self.temp.name)/'audit.json'
        result=run(self.csv,output,100,7)
        self.assertTrue(output.exists())
        self.assertEqual(result['comparisons'][0]['n'], 4)
        self.assertEqual(result['comparisons'][0]['net_correct_gain_b'], 2)
        self.assertEqual(result['comparisons'][3]['n'], 3)

    def test_revision_prediction_scored_wrong(self):
        rows=[{'true':'increase','a':'revision', 'b':'increase'},
              {'true':'flat','a':'flat','b':'flat'}]
        self.assertEqual(score(rows,'a')['accuracy'],0.5)
        self.assertEqual(score(rows,'b')['accuracy'],1.0)


if __name__=='__main__':
    unittest.main()
