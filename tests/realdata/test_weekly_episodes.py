"""Tests for temporal normalization and non-overclaiming guarantees."""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/realdata/build_weekly_episodes.py'
spec=importlib.util.spec_from_file_location('weekly_builder',SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def write_csv(path,rows):
    columns=['disease_id','disease','country_id','country_name','reported_date','cases_numeric','relation_internal_id']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def r(date,value,id='1',disease='COVID-19',country='wkg:1'):
    return {'disease_id':'C5203670' if disease=='COVID-19' else 'C0276180',
            'disease':disease,'country_id':country,'country_name':'Testland',
            'reported_date':date,'cases_numeric':value,'relation_internal_id':id}

class WeeklyTests(unittest.TestCase):
    def perform(self,rows):
        with TemporaryDirectory() as td:
            p=Path(td)/'sample.csv'
            write_csv(p,rows)
            return mod.build(p)

    def test_last_snapshot_not_sum(self):
        snaps,_,_,qa=self.perform([r('2024-03-04','10','1'),r('2024-03-06','11','2'),r('2024-03-10','12','3')])
        self.assertEqual(len(snaps),1)
        self.assertEqual(snaps[0]['reported_value'],'12')
        self.assertEqual(snaps[0]['number_of_reports_in_week'],3)
        self.assertEqual(snaps[0]['observation_date'],'2024-03-10')

    def test_one_week_change_and_adjacent_pair(self):
        snaps,events,links,qa=self.perform([r('2024-03-03','10','1'),r('2024-03-10','15','2')])
        self.assertEqual(len(links),1)
        self.assertEqual(snaps[-1]['difference_from_previous_snapshot'],'5')
        self.assertEqual(links[0]['relation'],'consecutive_snapshots_same_country_disease')
        self.assertIn('not_a_document_level_event',events[-1]['limitations'])

    def test_gap_does_not_impute(self):
        snaps,_,links,qa=self.perform([r('2024-03-03','10','1'),r('2024-03-17','15','2')])
        self.assertEqual(len(links),0)
        self.assertEqual(snaps[-1]['difference_from_previous_snapshot'],'')
        self.assertEqual(snaps[-1]['weeks_since_previous_snapshot'],'2')
        self.assertEqual(qa['counts']['gaps'],1)

    def test_negative_revision_preserved(self):
        snaps,_,_,qa=self.perform([r('2024-03-03','20','1'),r('2024-03-10','19','2')])
        self.assertEqual(snaps[-1]['difference_from_previous_snapshot'],'-1')
        self.assertEqual(snaps[-1]['quality_flags'],'negative_revision')
        self.assertEqual(qa['counts']['negative_weekly_differences'],1)

    def test_iso_year_week_crossing(self):
        snaps,_,links,_=self.perform([r('2023-12-31','10','1'),r('2024-01-01','11','2')])
        self.assertEqual([s['week_start'] for s in snaps],['2023-12-25','2024-01-01'])
        self.assertEqual(len(links),1)

    def test_invalid_records_excluded(self):
        snaps,_,_,qa=self.perform([r('bad-date','11','1'),r('2024-03-10','5','2')])
        self.assertEqual(len(snaps),1)
        self.assertEqual(qa['counts']['invalid_date_or_value'],1)

    def test_duplicate_country_date_fails(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.perform([r('2024-03-03','10','1'),r('2024-03-03','10','2')])

    def test_diseases_do_not_mix(self):
        snaps,_,links,_=self.perform([r('2024-03-03','10','1'),r('2024-03-10','12','2'),
                                       r('2024-03-03','10','3','Mpox'),r('2024-03-10','11','4','Mpox')])
        self.assertEqual(len(snaps),4)
        self.assertEqual(len(links),2)

if __name__=='__main__':unittest.main()
