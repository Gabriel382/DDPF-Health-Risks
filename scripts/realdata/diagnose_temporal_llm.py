#!/usr/bin/env python3
"""Read-only summary of existing M3/M4 cache and KG profile coverage."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def diagnose(output: Path) -> dict:
    cache = output / 'llm_predictions.jsonl'
    manifest = output / 'manifest.json'
    counts = defaultdict(Counter)
    reasons = defaultdict(Counter)
    sample_reasons = defaultdict(list)
    if cache.is_file():
        for line in cache.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get('method', '?'))
            counts[method][str(row.get('prediction', '?'))] += 1
            reason = str(row.get('reason', ''))
            reasons[method][reason] += 1
            if reason and len(sample_reasons[method]) < 3:
                sample_reasons[method].append(reason[:200])
    results = {'output': str(output), 'cached_predictions': {k: dict(v) for k, v in counts.items()},
               'reasons': {k: dict(v) for k, v in reasons.items()},
               'example_reasons': dict(sample_reasons)}
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding='utf-8'))
        results['manifest_context'] = {k: m.get(k) for k in (
            'n_train_examples', 'n_holdout_examples',
            'n_train_static_profiles', 'n_holdout_static_profiles',
            'n_kg_eligible_holdout', 'n_kg_eligible_by_disease',
            'n_llm_selected_by_disease')}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/realdata/temporal_trajectory_study')
    args = p.parse_args()
    print(json.dumps(diagnose(Path(args.output)), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
