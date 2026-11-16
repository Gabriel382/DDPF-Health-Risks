#!/usr/bin/env python3
"""Build country-level weekly snapshots from exported Neo4j :isReported edges.

A snapshot is the LAST observed value within a Monday-Sunday ISO week.
Nothing is summed, imputed, forward-filled, or inferred as document-level evidence.
Differences between consecutive snapshots are descriptive, not validated incidence.

Input: outputs/realdata/who_series/isreported_country_raw.csv
Output: outputs/realdata/who_series/weekly/{snapshots.csv,episodes.jsonl,
        adjacency_baseline.jsonl,quality.json}

Python 3.9+ stdlib only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from datetime import date, timedelta, datetime, timezone
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Optional

REQUIRED = {
    'disease_id','disease','country_id','country_name','reported_date',
    'cases_numeric','relation_internal_id'
}

CSV_FIELDS = [
    'episode_id','disease','disease_id','country_id','country_name','week_start',
    'week_end','observation_date','reported_value','number_of_reports_in_week',
    'source_relation_id','previous_episode_id','previous_observation_date',
    'weeks_since_previous_snapshot','difference_from_previous_snapshot',
    'difference_kind','quality_flags'
]


def parse_rows(path: Path):
    groups = defaultdict(list)
    seen_keys = set()
    quality = Counter()
    first_invalid = []
    sha256 = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha256.update(chunk)
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f'Missing CSV columns: {sorted(REQUIRED - set(reader.fieldnames or []))}')
        for row in reader:
            quality['raw_rows'] += 1
            key = (row['disease_id'],row['country_id'],row['reported_date'])
            if key in seen_keys:
                raise ValueError(f'Duplicate disease/country/date key: {key}; audit conflicts first')
            seen_keys.add(key)
            if not row['country_id'] or not row['disease_id']:
                quality['invalid_identifier'] += 1
                continue
            try:
                d = date.fromisoformat(str(row['reported_date']).strip())
                val = Decimal(str(row['cases_numeric']).strip())
                if not val.is_finite() or val < 0:
                    raise ValueError('nonfinite or negative')
            except (ValueError, InvalidOperation, TypeError):
                quality['invalid_date_or_value'] += 1
                if len(first_invalid) < 5:
                    first_invalid.append({k: row.get(k) for k in ('disease','country_name','reported_date','cases_numeric')})
                continue
            start = d - timedelta(days=d.weekday())
            grp_key = (row['disease_id'], row['country_id'], start)
            groups[grp_key].append((d, val, row))
            quality['valid_rows'] += 1
    return groups, quality, first_invalid, sha256.hexdigest()


def serialize_decimal(v: Decimal) -> str:
    return format(v, 'f')


def build(path: Path):
    groups, quality, first_invalid, input_sha = parse_rows(path)
    by_series = defaultdict(list)
    all_snapshots = []
    all_episodes = []
    adjacency = []
    for (disease_id,country_id,week_start), events in groups.items():
        events.sort(key=lambda r:(r[0],str(r[2]['relation_internal_id'])))
        day, value, row = events[-1]
        week_end = week_start + timedelta(days=6)
        episode_id = f'KGWEEK:{disease_id}:{country_id}:{week_start.isoformat()}'
        snap = {
            'episode_id': episode_id,
            'disease': row['disease'],
            'disease_id': disease_id,
            'country_id': country_id,
            'country_name': row['country_name'],
            'week_start': week_start.isoformat(),
            'week_end': week_end.isoformat(),
            'observation_date': day.isoformat(),
            'reported_value': serialize_decimal(value),
            'number_of_reports_in_week': len(events),
            'source_relation_id': str(row['relation_internal_id']),
            'previous_episode_id': '',
            'previous_observation_date': '',
            'weeks_since_previous_snapshot': '',
            'difference_from_previous_snapshot': '',
            'difference_kind': 'no_previous_snapshot',
            'quality_flags': '',
        }
        by_series[(disease_id,country_id)].append((week_start,value,snap))

    gaps = Counter()
    per_disease = defaultdict(Counter)
    for key, series in sorted(by_series.items()):
        series.sort(key=lambda s:s[0])
        prev = None
        for week, value, snap in series:
            if prev is not None:
                prev_week, prev_value, prev_snap = prev
                weeks = (week - prev_week).days // 7
                snap['previous_episode_id'] = prev_snap['episode_id']
                snap['previous_observation_date'] = prev_snap['observation_date']
                snap['weeks_since_previous_snapshot'] = str(weeks)
                if weeks == 1:
                    difference = value - prev_value
                    snap['difference_from_previous_snapshot'] = serialize_decimal(difference)
                    snap['difference_kind'] = 'consecutive_week_snapshot_difference_NOT_validated_incidence'
                    if difference < 0:
                        snap['quality_flags'] = 'negative_revision'
                        quality['negative_weekly_differences'] += 1
                    elif difference == 0:
                        quality['zero_weekly_differences'] += 1
                    else:
                        quality['positive_weekly_differences'] += 1
                    adjacency.append({
                        'from_episode_id':prev_snap['episode_id'],
                        'to_episode_id':snap['episode_id'],
                        'relation':'consecutive_snapshots_same_country_disease',
                        'status':'deterministic_temporal_adjacency_NOT_epidemiological_link',
                        'from_relation_id':prev_snap['source_relation_id'],
                        'to_relation_id':snap['source_relation_id'],
                        'note':'This edge is only chronological adjacency of aggregate report snapshots; it is not a documented event continuation.'
                    })
                else:
                    snap['difference_kind'] = 'gap_no_weekly_difference_computed'
                    snap['quality_flags'] = 'missing_intermediate_weeks'
                    gaps[weeks] += 1
                    quality['gaps'] += 1
            all_snapshots.append(snap)
            per_disease[snap['disease']]['weekly_snapshots'] += 1
            episode = {
                'episode_id': snap['episode_id'],
                'entity_type':'aggregate_country_week_snapshot',
                'disease_id':snap['disease_id'],
                'disease':snap['disease'],
                'country_id':snap['country_id'],
                'country_name':snap['country_name'],
                'week_start':snap['week_start'],
                'week_end':snap['week_end'],
                'observed_at':snap['observation_date'],
                'reported_value':snap['reported_value'],
                'snapshot_difference':snap['difference_from_previous_snapshot'] or None,
                'difference_kind':snap['difference_kind'],
                'quality_flags':snap['quality_flags'].split(';') if snap['quality_flags'] else [],
                'provenance':{
                    'source':'BioPropaPhenKG :isReported relationship',
                    'neo4j_relationship_internal_id':snap['source_relation_id'],
                    'source_csv_sha256':input_sha,
                    'original_property_names':{
                        'COVID-19':['Date_reported','Cumulative_cases'],
                        'Mpox':['date','total_cases'],
                    }.get(snap['disease'],[]),
                },
                'limitations':[
                    'not_a_document_level_event',
                    'case_count_semantics_not_independently_verified',
                    'no_claim_of_transmission_or_causality'
                ]
            }
            all_episodes.append(episode)
            prev = (week,value,snap)
    all_snapshots.sort(key=lambda s:(s['disease_id'],s['country_id'],s['week_start']))
    all_episodes.sort(key=lambda s:(s['disease_id'],s['country_id'],s['week_start']))
    adjacency.sort(key=lambda s:(s['from_episode_id'],s['to_episode_id']))
    quality['weekly_snapshots'] = len(all_snapshots)
    quality['country_disease_series'] = len(by_series)
    quality['adjacent_week_pairs'] = len(adjacency)
    quality['weeks_collapsed'] = quality['valid_rows'] - len(all_snapshots)
    return all_snapshots, all_episodes, adjacency, {
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'input_csv':str(path),
        'input_csv_sha256':input_sha,
        'method':'latest_observed_report_per_ISO_week_Monday_to_Sunday',
        'no_aggregation_of_case_counts':True,
        'no_imputation':True,
        'weekly_difference_requires_consecutive_week_snapshots':True,
        'weekly_difference_is_incidence':False,
        'document_level_event_gold_available':False,
        'counts':dict(quality),
        'by_disease':{k:dict(v) for k,v in sorted(per_disease.items())},
        'gap_lengths_weeks':{str(k):v for k,v in sorted(gaps.items())},
        'invalid_row_examples':first_invalid,
        'warnings':[
            'Cumulative_cases and total_cases semantics have not been independently verified.',
            'Do not treat snapshot differences as validated incident cases.',
            'Do not use these aggregate snapshots as gold labels for Story Forest documented_continuation.',
            'Neo4j internal relationship IDs are valid only within the restored database instance.',
            'The hasPresence source-specific timestamps are fixed collection periods; do not copy their concepts into arbitrary weeks as time-specific evidence.'
        ]
    }


def write_jsonl(path: Path, rows):
    with path.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='outputs/realdata/who_series/isreported_country_raw.csv')
    ap.add_argument('--output-dir',default='outputs/realdata/who_series/weekly')
    args=ap.parse_args(argv)
    inp=Path(args.input)
    if not inp.exists():
        ap.error(f'Missing input file: {inp}')
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    snapshots,episodes,links,quality=build(inp)
    with (out/'snapshots.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(snapshots)
    write_jsonl(out/'episodes.jsonl',episodes)
    write_jsonl(out/'adjacency_baseline.jsonl',links)
    (out/'quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'counts':quality['counts'],'by_disease':quality['by_disease']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
