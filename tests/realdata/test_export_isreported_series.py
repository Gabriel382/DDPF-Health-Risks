import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "realdata"))
from export_isreported_series import normalize_record, report_from_rows


class ExportTests(unittest.TestCase):
    def sample(self, disease_id, date, cases, relation_id):
        return {
            "disease_id": disease_id,
            "disease_node_name": None,
            "disease_node_internal_id": 1,
            "country_id": "Q1",
            "country_raw_name": '"Ukraine"',
            "country_node_internal_id": 42,
            "reported_date": date,
            "cases_raw": cases,
            "relation_internal_id": relation_id,
        }

    def test_property_mapping_and_decimal(self):
        covid = normalize_record(self.sample("C5203670", "2020-03-01", "7", 8))
        mpox = normalize_record(self.sample("C0276180", "2023-01-04", "1.0", 9))
        self.assertEqual(covid["original_date_property"], "Date_reported")
        self.assertEqual(mpox["original_date_property"], "date")
        self.assertEqual(mpox["cases_numeric"], "1.0")
        self.assertEqual(mpox["country_name"], "Ukraine")

    def test_missing_and_conflicting_duplicates(self):
        rows = [
            normalize_record(self.sample("C0276180", "2023-01-04", "1.0", 8)),
            normalize_record(self.sample("C0276180", "2023-01-04", "2.0", 9)),
            normalize_record(self.sample("C5203670", None, None, 10)),
        ]
        qa = report_from_rows(rows)
        self.assertEqual(qa["duplicate_disease_country_date_keys"], 1)
        self.assertEqual(qa["keys_with_conflicting_case_values"], 1)
        self.assertEqual(qa["results_by_disease"]["COVID-19"]["missing_or_invalid_dates"], 1)


if __name__ == "__main__":
    unittest.main()
