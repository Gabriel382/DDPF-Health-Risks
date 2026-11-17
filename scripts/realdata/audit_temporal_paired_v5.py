#!/usr/bin/env python3
"""Audit paired temporal forecasts from PreDiViD V5 without rerunning models.

Country-block bootstrap CIs are conditional on fixed predictions and do not
measure independent training-run uncertainty or documentary Story Forest links.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

PRIMARY = ('increase', 'flat')
M1 = 'M1_persistence'
M2N = 'M2_temporal_knn_noKG'
M2K = 'M2_temporal_staticKG_knn'
M3 = 'M3_llm_only'
M4 = 'M4_llm_kg_guard'


def country_id(origin_id: str) -> str:
    """Parse KGWEEK:<disease>:<country-id with colons>:<week> format."""
    parts = origin_id.split(':', 2)
    if len(parts) < 3 or parts[0] != 'KGWEEK':
        raise ValueError(f'Unexpected origin ID format: {origin_id!r}')
    rest = parts[2]
    if ':' not in rest:
        raise ValueError(f'Origin ID missing week: {origin_id!r}')
    return rest.rsplit(':', 1)[0]


def read_predictions(path: Path) -> dict[tuple[str, str], dict]:
    records = {}
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'method', 'split', 'origin_id', 'target_label', 'prediction', 'disease', 'has_static_kg_context'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing prediction columns: {sorted(missing)}')
        for row in reader:
            key = (row['origin_id'], row['method'])
            if key in records:
                raise ValueError(f'Duplicate method prediction for {key}')
            if row['method'] in (M1, M2N, M2K) and row['split'] != 'holdout_all':
                raise ValueError(f'Unexpected baseline split: {key}')
            if row['method'] in (M3, M4) and row['split'] != 'holdout_sample':
                raise ValueError(f'Unexpected LLM split: {key}')
            if row['target_label'] not in (*PRIMARY, 'revision'):
                raise ValueError(f'Unexpected true label at {key}')
            records[key] = row
    return records


def make_pairs(records: dict, method_a: str, method_b: str, *, kg_only: bool | None = None,
               disease: str | None = None) -> list[dict]:
    pairs = []
    origins = sorted({origin for origin, method in records if method == method_a})
    for origin in origins:
        a = records.get((origin, method_a))
        b = records.get((origin, method_b))
        if not a or not b:
            continue
        for field in ('target_label', 'disease', 'has_static_kg_context'):
            if a[field] != b[field]:
                raise ValueError(f'Inconsistent {field} for {origin}')
        if kg_only is not None and (a['has_static_kg_context'] == '1') != kg_only:
            continue
        if disease is not None and a['disease'] != disease:
            continue
        pairs.append({
            'origin': origin,
            'country_id': country_id(origin),
            'disease': a['disease'],
            'true': a['target_label'],
            'a': a['prediction'],
            'b': b['prediction'],
        })
    return pairs


def score(rows: list[dict], pred_key: str) -> dict:
    # Matches the V5 two-class scoring: revision truth is excluded from primary;
    # unknown/revision predictions against primary labels count as errors.
    rows = [r for r in rows if r['true'] in PRIMARY]
    if not rows:
        return {'accuracy': None, 'macro_f1': None, 'n': 0}
    accuracy = sum(r[pred_key] == r['true'] for r in rows) / len(rows)
    f1s = []
    for label in PRIMARY:
        tp = sum(r['true'] == label and r[pred_key] == label for r in rows)
        fn = sum(r['true'] == label and r[pred_key] != label for r in rows)
        fp = sum(r['true'] != label and r[pred_key] == label for r in rows)
        den = 2 * tp + fn + fp
        f1s.append(2 * tp / den if den else 0.0)
    return {'accuracy': round(accuracy, 6), 'macro_f1': round(sum(f1s) / 2, 6), 'n': len(rows)}


def percentile_sorted(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError('Empty percentile input')
    pos = q * (len(xs) - 1)
    low, high = math.floor(pos), math.ceil(pos)
    return xs[low] * (high - pos) + xs[high] * (pos - low) if low != high else xs[low]


def paired_bootstrap(rows: list[dict], n: int = 2000, seed: int = 20261117) -> dict:
    """Resample countries, keeping all their weeks and both diseases together."""
    by_country = defaultdict(list)
    for row in rows:
        by_country[row['country_id']].append(row)
    countries = sorted(by_country)
    if not countries or n <= 0:
        return {}
    rng = random.Random(seed)
    diffs = {'accuracy': [], 'macro_f1': []}
    for _ in range(n):
        sample = []
        for _ in range(len(countries)):
            sample.extend(by_country[countries[rng.randrange(len(countries))]])
        sa, sb = score(sample, 'a'), score(sample, 'b')
        for metric in diffs:
            diffs[metric].append(sb[metric] - sa[metric])
    result = {'block_unit': 'country_id', 'n_country_blocks': len(countries),
              'resamples': n, 'seed': seed}
    for metric, vals in diffs.items():
        vals.sort()
        result[f'delta_{metric}_95pct_CI'] = [
            round(percentile_sorted(vals, 0.025), 6),
            round(percentile_sorted(vals, 0.975), 6),
        ]
    return result


def summarize(pairs: list[dict], a_name: str, b_name: str,
              cohort: str, n_boot: int, seed: int) -> dict:
    if not pairs:
        return {'comparison': f'{b_name} minus {a_name}', 'cohort': cohort, 'n': 0}
    sa, sb = score(pairs, 'a'), score(pairs, 'b')
    discordant_a_only = sum(r['a'] == r['true'] and r['b'] != r['true'] for r in pairs)
    discordant_b_only = sum(r['a'] != r['true'] and r['b'] == r['true'] for r in pairs)
    out = {
        'comparison': f'{b_name} minus {a_name}', 'cohort': cohort,
        'n': len(pairs), 'countries': len({r['country_id'] for r in pairs}),
        'diseases': dict(Counter(r['disease'] for r in pairs)),
        'true_labels': dict(Counter(r['true'] for r in pairs)),
        'predictions_a': dict(Counter(r['a'] for r in pairs)),
        'predictions_b': dict(Counter(r['b'] for r in pairs)),
        'method_a': sa, 'method_b': sb,
        'delta_accuracy': round(sb['accuracy'] - sa['accuracy'], 6),
        'delta_macro_f1': round(sb['macro_f1'] - sa['macro_f1'], 6),
        'a_only_correct': discordant_a_only,
        'b_only_correct': discordant_b_only,
        'net_correct_gain_b': discordant_b_only - discordant_a_only,
        'bootstrap_conditional_fixed_predictions': paired_bootstrap(pairs, n_boot, seed),
    }
    return out


def run(path: Path, output: Path, bootstraps: int, seed: int) -> dict:
    records = read_predictions(path)
    comparisons = []
    configs = [
        (M2N, M2K, 'all_holdout', None, None),
        (M2N, M2K, 'kg_eligible', True, None),
        (M2N, M2K, 'kg_ineligible', False, None),
        (M3, M4, 'llm_common_sample', True, None),
        (M2K, M3, 'llm_common_sample', True, None),
        (M2K, M4, 'llm_common_sample', True, None),
    ]
    for disease in ('COVID-19', 'Mpox'):
        configs.extend([
            (M2N, M2K, f'kg_eligible_{disease}', True, disease),
            (M3, M4, f'llm_common_sample_{disease}', True, disease),
        ])
    for i, (a, b, cohort, kg, disease) in enumerate(configs):
        pairs = make_pairs(records, a, b, kg_only=kg, disease=disease)
        comparisons.append(summarize(pairs, a, b, cohort, bootstraps, seed+i))
    out = {
        'input': str(path),
        'scope': 'direction of the next weekly aggregate reported value',
        'limitations': [
            'No documentary event-link gold; not a Story Forest link evaluation.',
            'COVID and Mpox count field equivalence not independently verified.',
            'Paired country-block bootstrap is conditional on predictions already fitted; not a repeated-training CI.',
            'Weekly observations within a country are not independent.',
            'LLM comparisons are from a deterministic KG-covered sample, not random population sampling.',
            'M4 KG guard is not an OWL reasoner.',
            'No revisions occurred in the evaluated holdout: three-class performance is not estimable.',
        ],
        'comparisons': comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for item in comparisons:
        if not item['n']:
            continue
        print(f"{item['cohort']}: {item['comparison']} | n={item['n']} "
              f"countries={item['countries']} | "
              f"ΔF1={item['delta_macro_f1']:+.4f} "
              f"CI={item['bootstrap_conditional_fixed_predictions'].get('delta_macro_f1_95pct_CI')} | "
              f"ΔAcc={item['delta_accuracy']:+.4f} "
              f"CI={item['bootstrap_conditional_fixed_predictions'].get('delta_accuracy_95pct_CI')} | "
              f"a-only={item['a_only_correct']} b-only={item['b_only_correct']}")
    print('Saved:', output)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path, default=Path('outputs/realdata/temporal_trajectory_study_v5/predictions.csv'))
    ap.add_argument('--output', type=Path, default=Path('outputs/realdata/temporal_trajectory_study_v5/paired_stat_audit.json'))
    ap.add_argument('--bootstraps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261117)
    args = ap.parse_args()
    if args.bootstraps < 100:
        ap.error('--bootstraps must be >= 100')
    run(args.predictions, args.output, args.bootstraps, args.seed)


if __name__ == '__main__':
    main()
