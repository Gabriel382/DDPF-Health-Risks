#!/usr/bin/env python3
"""PreDiViD V6: one-shot, resumable Qwen3 parameter-scale study on V5's fixed 40 episodes.

Reuses the EXACT V5 KG-covered 20-per-disease sample, M3/M4 prompts and constrained
Ollama JSON class. Compares model sizes, not new datasets or Story Forest links.
All results are observational aggregate next-week reported-value classifications.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temporal_trajectory_study_v5 as v5

base = v5.base
METHODS = ('M3_llm_only', 'M4_llm_kg_guard')
CLASSES = ('increase', 'flat', 'revision')
DEFAULT_MODELS = ('qwen3:0.6b', 'qwen3:1.7b', 'qwen3:4b', 'qwen3:8b', 'qwen3:14b', 'qwen3:32b')
BASELINES = ('M1_persistence', 'M2_temporal_knn_noKG', 'M2_temporal_staticKG_knn')


def sha_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tag_id(tag: str) -> str:
    """File-safe name of the exact model tag, no collisions for ordinary tags."""
    return ''.join(c if c.isalnum() else '_' for c in tag)


def model_billions(model: str) -> float | None:
    try:
        s = model.split(':', 1)[1].lower()
        return float(s[:-1]) if s.endswith('b') else None
    except (IndexError, ValueError):
        return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as f:
        for n, ln in enumerate(f, 1):
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except ValueError as e:
                    raise ValueError(f'Corrupted JSONL {path}:{n}: {e}') from e
    return rows


def append_jsonl(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def export_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(rows: list[dict]) -> dict:
    """Counts + macro-F1 on observed classes; do not count failed requests as 'flat'."""
    labels = sorted({r['target_label'] for r in rows} | {r['prediction'] for r in rows})
    actual = Counter(r['target_label'] for r in rows)
    predicted = Counter(r['prediction'] for r in rows)
    if not rows:
        return {'n': 0, 'accuracy': None, 'macro_f1': None,
                'balanced_accuracy': None, 'recall_increase': None, 'actual': {},
                'predicted': {}, 'n_correct': 0}
    f1s = []
    recalls = []
    for label in labels:
        tp = sum(r['target_label'] == label and r['prediction'] == label for r in rows)
        fp = predicted[label] - tp
        fn = actual[label] - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0)
        if actual[label]:
            recalls.append(tp / actual[label])
    n_correct = sum(r['prediction'] == r['target_label'] for r in rows)
    return {
        'n': len(rows), 'n_correct': n_correct,
        'accuracy': round(n_correct / len(rows), 6),
        'macro_f1': round(sum(f1s) / len(f1s), 6),
        'balanced_accuracy': round(sum(recalls) / len(recalls), 6),
        'recall_increase': round(
            sum(r['prediction'] == 'increase' and r['target_label'] == 'increase' for r in rows)
            / actual['increase'], 6) if actual['increase'] else None,
        'actual': dict(actual), 'predicted': dict(predicted),
    }


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.] * len(values)
    for _, group in itertools.groupby(order, key=lambda i: values[i]):
        g = list(group)
        avg = sum(range(1 + sum(1 for x in values if x < values[g[0]]),
                        1 + sum(1 for x in values if x <= values[g[0]]))) / len(g)
        for i in g:
            result[i] = avg
    return result


def pearson(x, y):
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - xm for v in x], [v - ym for v in y]
    den = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return sum(a*b for a, b in zip(dx, dy)) / den if den else None


def spearman_exact(xs: list[float], ys: list[float]):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    rho = pearson(rx, ry)
    if rho is None:
        return {'rho': None, 'p_two_sided': None, 'models': len(xs)}
    rng = random.Random(20261117)
    if len(xs) <= 7:
        perms = itertools.permutations(ry)
        count, tail = 0, 0
        for p in perms:
            val = pearson(rx, p)
            count += 1
            tail += val is not None and abs(val) >= abs(rho) - 1e-12
        p_value = tail / count
        kind = 'exact_permutation'
    else:
        count, tail = 10000, 0
        for _ in range(count):
            p = rng.sample(ry, len(ry))
            val = pearson(rx, p)
            tail += val is not None and abs(val) >= abs(rho) - 1e-12
        p_value = (tail + 1) / (count + 1)
        kind = 'monte_carlo_permutation'
    return {'rho': round(rho, 6), 'p_two_sided': round(p_value, 6),
            'models': len(xs), 'permutation_method': kind,
            'exploratory': True}


def country_bootstrap_paired(a: list[dict], b: list[dict], resamples=2000, seed=20261117):
    amap = {r['episode_id']: r for r in a}
    bmap = {r['episode_id']: r for r in b}
    ids = sorted(set(amap) & set(bmap))
    if not ids:
        return {'n': 0}
    groups = defaultdict(list)
    for ep in ids:
        assert amap[ep]['target_label'] == bmap[ep]['target_label']
        country = amap[ep]['country_id']
        groups[country].append(ep)
    countries = sorted(groups)
    rng = random.Random(seed)
    full_a, full_b = [amap[i] for i in ids], [bmap[i] for i in ids]
    delta_acc = metric_rows(full_b)['accuracy'] - metric_rows(full_a)['accuracy']
    delta_f1 = metric_rows(full_b)['macro_f1'] - metric_rows(full_a)['macro_f1']
    ba, bf = [], []
    for _ in range(resamples):
        chosen = [ep for c in rng.choices(countries, k=len(countries)) for ep in groups[c]]
        if not chosen:
            continue
        aa = metric_rows([amap[ep] for ep in chosen])
        bb = metric_rows([bmap[ep] for ep in chosen])
        ba.append(bb['accuracy'] - aa['accuracy'])
        bf.append(bb['macro_f1'] - aa['macro_f1'])
    def ci(x):
        if not x:
            return None
        x.sort()
        return [round(x[int(.025 * (len(x)-1))], 6), round(x[int(.975 * (len(x)-1))], 6)]
    block_sum = [sum((int(bmap[i]['prediction'] == bmap[i]['target_label'])
                      - int(amap[i]['prediction'] == amap[i]['target_label'])) for i in groups[c])
                 for c in countries]
    observed = abs(sum(block_sum))
    flip_rng = random.Random(seed + 101)
    nperm = 5000
    tail = 0
    for _ in range(nperm):
        stat = abs(sum(sign * val for sign, val in zip(
            (flip_rng.choice((-1, 1)) for __ in countries), block_sum)))
        tail += stat >= observed
    return {'n': len(ids), 'countries': len(countries),
            'delta_accuracy': round(delta_acc, 6), 'delta_macro_f1': round(delta_f1, 6),
            'accuracy_95pct_CI': ci(ba), 'macro_f1_95pct_CI': ci(bf),
            'country_block_signflip_p_accuracy': round((tail+1)/(nperm+1), 6),
            'n_bootstrap': resamples, 'n_signflip': nperm,
            'conditional_on_fixed_predictions': True}


def validate_v5_and_select(v5_output: Path, max_per_disease: int):
    manifest_path = v5_output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('llm_study_complete'):
        raise RuntimeError('V5 sample is not complete; refusing mixed/incomplete baseline.')
    if manifest.get('sample_all'):
        raise RuntimeError('V5 used --sample-all: not the expected KG-covered cohort.')
    if manifest.get('cutoff_date') != '2024-01-01':
        raise RuntimeError('V5 holdout cutoff differs; need identical sample definition.')
    snap = Path(manifest['input'])
    profiles = Path(manifest['profiles'])
    if not snap.exists() or not profiles.exists():
        raise FileNotFoundError(f'V5 dataset paths do not exist: {snap}, {profiles}')
    if base.hash_file(snap) != manifest['sha256_snapshots']:
        raise RuntimeError('Snapshots SHA256 differs from V5; would change evaluation.')
    if base.hash_file(profiles) != manifest['sha256_profiles']:
        raise RuntimeError('Profiles SHA256 differs from V5; would change evaluation.')
    series = base.load_snapshots(snap)
    prof = base.load_static_profiles(profiles)
    examples = base.make_examples(series, prof)
    pool = [ex for ex in examples if ex.target.week >= date.fromisoformat(manifest['cutoff_date'])
            and bool(ex.semantic_keys)]
    selected = base.stable_sample(pool, max_per_disease)
    if len(selected) != 2 * max_per_disease:
        raise RuntimeError(f'Expected {2*max_per_disease} samples, found {len(selected)}')
    if max_per_disease == 20:
        if len(selected) != manifest['n_llm_selected']:
            raise RuntimeError('V5 sample count differs from 40-episode original')
        cache = base.load_cache(v5_output / 'llm_predictions.jsonl')
        for ex in selected:
            for method in METHODS:
                if (method, ex.origin.episode_id) not in cache:
                    raise RuntimeError('Original V5 cache missing one of the 40 episodes: '
                                       + ex.origin.episode_id)
    return manifest, selected


def original_v5_signature(manifest: dict, temperature: float, max_tokens: int,
                          max_output_tokens: int, num_ctx: int):
    cfg = {
        'study_version': 'v5_schema_enum_isolated_failures',
        'backend': 'ollama_native', 'model': 'ollama/qwen3:4b',
        'provider': 'ollama', 'temperature': temperature,
        'think': False, 'output_schema': v5.SCHEMA,
        'prompt_sha256': hashlib.sha256(base.PROMPT_SYSTEM.encode()).hexdigest(),
        'num_ctx': num_ctx, 'max_tokens': max_tokens,
        'max_output_tokens': max_output_tokens,
        'input_sha256': manifest['sha256_snapshots'],
        'profile_sha256': manifest['sha256_profiles'],
        'holdout_start': manifest['cutoff_date'], 'sample_all': False,
    }
    return sha_obj(cfg)


def import_4b_cache(v5_output, model_cache_path, selected, signature,
                    baseline_signature, config: dict):
    """Only reuse V5 baseline if its full prompt/settings/data signature matches."""
    original = base.load_cache(v5_output / 'llm_predictions.jsonl')
    entries = []
    for ex in selected:
        for method in METHODS:
            r = original.get((method, ex.origin.episode_id))
            if not r or r.get('signature') != baseline_signature \
                    or r.get('model') != 'ollama/qwen3:4b' \
                    or r.get('target_episode_id') != ex.target.episode_id \
                    or r.get('prediction') not in CLASSES:
                return False
            entries.append({
                'signature': signature, 'model': 'qwen3:4b',
                'method': method, 'episode_id': ex.origin.episode_id,
                'target_episode_id': ex.target.episode_id,
                'prediction': r['prediction'], 'elapsed_seconds': r.get('elapsed_seconds'),
                'ollama_stats': r.get('ollama_stats') or {}, 'reused_from_v5': True,
                'config_sha256': sha_obj(config),
            })
    if not model_cache_path.exists():
        for item in entries:
            append_jsonl(model_cache_path, item)
    return True


def model_metadata(server: str, model: str) -> dict:
    try:
        with urlopen(server.rstrip('/') + '/api/tags', timeout=15) as resp:
            data = json.loads(resp.read())
        for item in data.get('models', []):
            if item.get('model', item.get('name')) == model or item.get('name') == model:
                d = item.get('details', {})
                return {'digest': item.get('digest'), 'bytes': item.get('size'),
                        'parameter_size': d.get('parameter_size'),
                        'quantization_level': d.get('quantization_level'),
                        'family': d.get('family')}
    except Exception as e:
        return {'metadata_error': f'{type(e).__name__}: {str(e)[:140]}'}
    return {'not_present_in_ollama_tags': True}


def cli_pull(model: str, skip_pull: bool):
    if skip_pull:
        return True, None
    if not shutil.which('ollama'):
        return False, 'ollama executable not found; use --skip-pull for already installed models'
    print(f'\n=== DOWNLOADING/VERIFYING {model} ===', flush=True)
    code = subprocess.run(['ollama', 'pull', model], check=False).returncode
    return (True, None) if code == 0 else (False, f'ollama pull exited with status {code}')


def free_previous_model(model: str):
    if shutil.which('ollama'):
        subprocess.run(['ollama', 'stop', model], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)


def load_model_cache(path: Path, signature: str, selected: list):
    items = load_jsonl(path)
    if any(row.get('signature') != signature for row in items):
        raise RuntimeError(f'Stale/incompatible cache: {path}. Choose a new --output.')
    valid_ids = {e.origin.episode_id: e.target.episode_id for e in selected}
    out = {}
    for row in items:
        key = (row.get('method'), row.get('episode_id'))
        if row.get('prediction') not in CLASSES or key[0] not in METHODS:
            raise RuntimeError(f'Invalid cached prediction in {path}')
        if key[1] not in valid_ids or valid_ids[key[1]] != row.get('target_episode_id'):
            raise RuntimeError(f'Cached example not in current sample: {key}')
        if key in out and out[key]['prediction'] != row['prediction']:
            raise RuntimeError(f'Conflicting cached predictions for {key}')
        out[key] = row
    return out


def assemble_data(models, selected, model_cache, v5_output):
    examples_by_id = {x.origin.episode_id: x for x in selected}
    complete = defaultdict(set)
    predictions = []
    for model in models:
        for (method, ep), r in model_cache.get(model, {}).items():
            x = examples_by_id[ep]
            predictions.append({
                'model': model, 'parameters_b': model_billions(model), 'method': method,
                'episode_id': ep, 'country_id': x.origin.country_id,
                'country': x.origin.country, 'disease': x.origin.disease,
                'target_label': x.target_label, 'prediction': r['prediction'],
                'elapsed_seconds': r.get('elapsed_seconds'),
                'reused_from_v5': r.get('reused_from_v5', False),
            })
            complete[(model, method)].add(ep)
    per_method_common = {
        method: sorted(set.intersection(*(complete[(m, method)] for m in models))) if models else []
        for method in METHODS
    }
    common_both = sorted(set.intersection(*(complete[(m, method)] for m in models
                                            for method in METHODS))) if models else []
    baselines = {}
    pred_file = v5_output / 'predictions.csv'
    if pred_file.exists():
        with pred_file.open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                if r['method'] in BASELINES and r['origin_id'] in examples_by_id:
                    baselines[(r['method'], r['origin_id'])] = r['prediction']
    summary = []
    for model in models:
        for method in METHODS:
            rs = [r for r in predictions if r['model'] == model and r['method'] == method]
            for disease in ('ALL', 'COVID-19', 'Mpox'):
                group = [r for r in rs if disease == 'ALL' or r['disease'] == disease]
                total = sum(disease == 'ALL' or x.origin.disease == disease for x in selected)
                for cohort, included in [('completed', {x['episode_id'] for x in group}),
                                         ('common_all_models', set(common_both))]:
                    sub = [r for r in group if r['episode_id'] in included]
                    stats = metric_rows(sub)
                    summary.append({'model': model, 'parameters_b': model_billions(model),
                                    'method': method, 'disease': disease, 'cohort': cohort,
                                    'n_expected': total, 'coverage': round(len(group)/total, 6)
                                    if total else None, **{k:v for k,v in stats.items()
                                                        if k not in ('actual', 'predicted')},
                                    'actual': json.dumps(stats['actual'], sort_keys=True),
                                    'predicted': json.dumps(stats['predicted'], sort_keys=True)})
    baseline_rows = []
    for method in BASELINES:
        rs = []
        for ep in common_both:
            x = examples_by_id[ep]
            p = baselines.get((method, ep))
            if p is None:
                raise RuntimeError(f'Original V5 baseline prediction missing for {method}:{ep}')
            rs.append({'target_label': x.target_label, 'prediction': p})
        baseline_rows.append({'model': method, 'cohort':'common_all_models',
                              **metric_rows(rs)})
    return predictions, summary, common_both, per_method_common, baseline_rows


def comparison_report(models, predictions, common_both, selected, params):
    by = {(r['model'], r['method'], r['episode_id']): r for r in predictions}
    summaries = {}
    for method in METHODS:
        points = []
        for model in models:
            rows = [by[(model, method, ep)] for ep in common_both if (model, method, ep) in by]
            metric = metric_rows(rows)
            if params[model] is not None and metric['macro_f1'] is not None:
                points.append((params[model], metric['macro_f1'], model))
        if len(points) >= 3:
            xs, ys, _ = zip(*points)
            summaries[method] = {'trend_size_vs_macro_f1': spearman_exact(list(xs), list(ys))}
        else:
            summaries[method] = {'trend_size_vs_macro_f1': None}
    candidates = sorted((params[m], m) for m in models if params[m] is not None)
    if len(candidates) >= 2 and common_both:
        small, large = candidates[0][1], candidates[-1][1]
        for method in METHODS:
            aa = [by[(small, method, ep)] for ep in common_both]
            bb = [by[(large, method, ep)] for ep in common_both]
            summaries[method]['largest_minus_smallest'] = {
                'smallest': small, 'largest': large,
                **country_bootstrap_paired(aa, bb)
            }
        # Holm for precisely two predeclared method-specific accuracy tests.
        p_sorted = sorted([(summaries[m]['largest_minus_smallest']['country_block_signflip_p_accuracy'], m)
                           for m in METHODS])
        for i, (p, method) in enumerate(p_sorted):
            summaries[method]['largest_minus_smallest']['holm_p_accuracy_2_contrasts'] = \
                min(1.0, max(p_sorted[j][0] * (len(p_sorted)-j) for j in range(i+1)))
    return summaries


def report_markdown(report):
    rows = report['metrics']
    common_n = report['n_common_complete_all_models_both_methods']
    lines = [
        '# PreDiViD V6: Qwen3 LLM parameter-scale study', '',
        f"Status: **{report['status']}**. Selected KG-eligible episodes: **{report['n_selected']}** "
        f"(20 per disease). Episodes completed for **all models and both methods**: **{common_n}**.",
        '', '## Fixed-sample comparative results', '',
        '| Model | Parameters | Method | Completed / selected | Accuracy (common) | Macro-F1 (common) | Increase recall (common) |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    def fmt(v):
        return 'NA' if v is None else f'{v:.4f}'
    for m in report['models']:
        for method in METHODS:
            cm = next(r for r in rows if r['model'] == m and r['method'] == method
                      and r['cohort'] == 'common_all_models' and r['disease'] == 'ALL')
            done = next(r for r in rows if r['model'] == m and r['method'] == method
                        and r['cohort'] == 'completed' and r['disease'] == 'ALL')
            lines.append(f"| {m} | {report['model_parameters_b'].get(m)}B | {method} | "
                         f"{done['n']}/{report['n_selected']} | {fmt(cm['accuracy'])} | "
                         f"{fmt(cm['macro_f1'])} | {fmt(cm['recall_increase'])} |")
    lines += ['', '## Baselines on the exact same common episodes', '',
              '| Method | N | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for r in report['baseline_on_common']:
        lines.append(f"| {r['model']} | {r['n']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} |")
    lines += ['', '## Size trend and paired smallest vs largest', '',
              'Exploratory exact permutation correlation between model size and macro-F1; '
              'the two smallest-vs-largest country-block accuracy tests use Holm correction.']
    for method in METHODS:
        comp = report['comparisons'].get(method, {})
        lines.append(f'\n### {method}')
        lines.append('```json\n' + json.dumps(comp, indent=2, ensure_ascii=False) + '\n```')
    lines += ['', '## Per-disease and timing artifacts', '',
              '- `metrics_by_size.csv`: includes each disease, success-only coverage, common-cohort metrics.',
              '- `predictions_by_size.csv`: paired predictions and labels, no future value in model inputs.',
              '- `model_runtime.csv`: elapsed seconds and Ollama token statistics per model.',
              '- `study_manifest.json`: models, signatures, hashes, quantization and failures.', '',
              '## Interpretation and limitations', '',
              '- This is an exploratory sample of 40 deterministic KG-covered episodes; not random, not population inference.',
              '- Any missing model output remains missing; NEVER reinterpret a failure as `flat` or `unknown`.',
              '- Compare full-common cohort and report coverage; never compare metrics on mismatched episodes.',
              '- Qwen3 models have size-dependent pretraining, quantization and behavior; correlation is not a causal size effect.',
              '- The 4B V5 predictions are reused only if all input/prompt/settings signatures match.',
              '- Inference speed depends on GPU offload, CPU and quantization, not simply parameter count.',
              '- No documentary Story Forest link gold, no epidemiological incidence validation, and no OWL reasoning.', '',
    ]
    return '\n'.join(lines) + '\n'


def summarize(models, selected, model_cache, v5_output, out, metadata, failures, config):
    predictions, metrics, common, by_method, baselines = assemble_data(
        models, selected, model_cache, v5_output)
    params = {m:model_billions(m) for m in models}
    comparisons = comparison_report(models, predictions, common, selected, params)
    runtime = []
    for model in models:
        ms = [r for r in model_cache.get(model, {}).values()]
        durations = [float(x['elapsed_seconds']) for x in ms if x.get('elapsed_seconds') is not None]
        stats = [x.get('ollama_stats') or {} for x in ms]
        eval_tokens = sum(s.get('eval_count') or 0 for s in stats)
        eval_ns = sum(s.get('eval_duration') or 0 for s in stats)
        runtime.append({'model': model, 'n_calls': len(ms),
                        'total_elapsed_seconds': round(sum(durations), 2),
                        'avg_seconds_per_call': round(sum(durations)/len(durations), 3)
                        if durations else None,
                        'total_output_tokens': eval_tokens,
                        'estimated_output_tokens_per_second': round(eval_tokens*1e9/eval_ns, 3)
                        if eval_ns else None, 'model_bytes': metadata.get(model, {}).get('bytes'),
                        'quantization_level': metadata.get(model, {}).get('quantization_level'),
                        'parameter_size_tag': metadata.get(model, {}).get('parameter_size'),
                        'reused_calls_from_v5': sum(bool(x.get('reused_from_v5')) for x in ms)})
    status = 'COMPLETE' if len(common) == len(selected) and all(
        len(model_cache.get(m,{})) == len(selected) * len(METHODS) for m in models) else 'INCOMPLETE'
    manifest = {
        'study':'v6_qwen3_parameter_scaling_fixed_v5_sample', 'status':status,
        'models':models, 'model_parameters_b':params,
        'model_metadata':metadata, 'n_selected':len(selected),
        'n_common_complete_all_models_both_methods':len(common),
        'n_common_completed_by_method':{k:len(v) for k,v in by_method.items()},
        'baseline_on_common':baselines, 'comparisons':comparisons,
        'metrics':metrics, 'runtime':runtime, 'model_failures':failures,
        'config_sha256':sha_obj(config), 'config':config,
        'scientific_scope': 'Next-week aggregate reported-value direction; not documentary Story Forest',
        'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
    }
    export_csv(out/'predictions_by_size.csv', predictions)
    export_csv(out/'metrics_by_size.csv', metrics)
    export_csv(out/'model_runtime.csv', runtime)
    (out/'study_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'size_comparison_report.md').write_text(report_markdown(manifest),encoding='utf-8')
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS), help='Comma-separated exact Ollama model tags')
    ap.add_argument('--v5-output', default='outputs/realdata/temporal_trajectory_study_v5')
    ap.add_argument('--output', default='outputs/realdata/llm_size_study_v6')
    ap.add_argument('--per-disease', type=int, default=20,
                    help='Fixed V5 sample uses 20 episodes per disease')
    ap.add_argument('--server', default=os.environ.get('OLLAMA_BASE_URL','http://localhost:11434'))
    ap.add_argument('--temperature',type=float,default=0.)
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--initial-tokens',type=int,default=256)
    ap.add_argument('--max-output-tokens',type=int,default=1024)
    ap.add_argument('--num-ctx',type=int,default=4096)
    ap.add_argument('--retries',type=int,default=1)
    ap.add_argument('--skip-pull',action='store_true',help='Use only already installed models')
    ap.add_argument('--no-reuse-4b',action='store_true',help='Regenerate Qwen3 4B instead of importing matching V5 cache')
    args = ap.parse_args(argv)
    models = tuple(dict.fromkeys(v.strip() for v in args.models.split(',') if v.strip()))
    if not models or any(not m.startswith('qwen3:') for m in models):
        ap.error('Specify one or more exact qwen3:SIZE model tags')
    if args.per_disease != 20:
        ap.error('Only the original V5 20-per-disease sample is supported for strict comparability')
    if args.initial_tokens < 64 or args.max_output_tokens < args.initial_tokens:
        ap.error('Invalid output token budgets')
    if args.timeout < 10 or args.retries < 0:
        ap.error('Invalid timeout/retry limits')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    v5_output = Path(args.v5_output)
    v5manifest, selected = validate_v5_and_select(v5_output,args.per_disease)
    sample_ids = [x.origin.episode_id for x in selected]
    cfg = {
        'dataset_snapshots_sha256':v5manifest['sha256_snapshots'],
        'dataset_profiles_sha256':v5manifest['sha256_profiles'],
        'v5_cutoff_date':v5manifest['cutoff_date'], 'model_tags':list(models),
        'selected_episode_ids':sample_ids,
        'prompt_sha256':hashlib.sha256(base.PROMPT_SYSTEM.encode()).hexdigest(),
        'method_payload_version':'v5_prompt_payload_exact', 'schema':v5.SCHEMA,
        'temperature':args.temperature,'think':False,
        'num_ctx':args.num_ctx,'num_predict_initial':args.initial_tokens,
        'num_predict_max':args.max_output_tokens,
        'timeout_s':args.timeout,'retries':args.retries,
    }
    # A saved V6 run can be resumed even after a crash or restart.
    config_file = out/'run_config.json'
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding='utf-8'))
        if previous != cfg:
            raise RuntimeError('V6 config changed; use a NEW --output directory to avoid corrupt comparisons')
    else:
        config_file.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'V6 Qwen3 SCALE: {len(selected)} fixed episodes, models={models}', flush=True)
    print('No new KNN, KG, embeddings, weekly episode construction or V5 predictions.',flush=True)
    try:
        free_gb=shutil.disk_usage(os.environ.get('OLLAMA_MODELS',str(Path.home()))).free/1e9
        print(f'Free disk GB (approximate model storage filesystem): {free_gb:.1f}',flush=True)
    except (OSError,FileNotFoundError):
        pass
    model_cache = {}
    metadata = {}
    failures = {}
    last_model = None
    for model in models:
        print(f'\n==== MODEL {model} ==== ',flush=True)
        if last_model and last_model != model:
            free_previous_model(last_model)
        last_model = model
        cache_path = out / f'predictions_{tag_id(model)}.jsonl'
        error_path = out / f'failures_{tag_id(model)}.jsonl'
        signature = sha_obj({**cfg, 'model':model})
        cache = load_model_cache(cache_path,signature,selected)
        if model == 'qwen3:4b' and not args.no_reuse_4b and not cache:
            v5_sig=original_v5_signature(v5manifest, args.temperature,args.initial_tokens,
                                          args.max_output_tokens,args.num_ctx)
            reused = import_4b_cache(v5_output,cache_path,selected,signature,v5_sig,cfg)
            if reused:
                cache=load_model_cache(cache_path,signature,selected)
                print(f'Reused {len(cache)} EXACT matching V5 4B calls; no inference.',flush=True)
            else:
                print('Original V5 settings/signature mismatch: will evaluate 4B anew.',flush=True)
        pending = [(method, x) for x in selected for method in METHODS
                   if (method,x.origin.episode_id) not in cache]
        if not pending:
            model_cache[model]=cache
            print(f'Already complete: {len(cache)} cached. Skipping download/inference.',flush=True)
            metadata[model]=model_metadata(args.server,model)
            failures[model]=0
            summarize(models,selected,model_cache,v5_output,out,metadata,failures,cfg)
            continue
        ok,pull_error = cli_pull(model,args.skip_pull)
        metadata[model]=model_metadata(args.server,model)
        if not ok:
            print(f'FAILED to prepare {model}: {pull_error}',flush=True)
            append_jsonl(error_path,{'model':model,'error_type':'model_pull_failed','message':pull_error})
            model_cache[model]=cache;failures[model]=len(pending)
            summarize(models,selected,model_cache,v5_output,out,metadata,failures,cfg)
            continue
        if metadata[model].get('not_present_in_ollama_tags'):
            print(f'WARNING: {model} is not in /api/tags; API might fail.',flush=True)
        settings=SimpleNamespace(provider='ollama', model='ollama/'+model,
                                 ollama_base_url=args.server,temperature=args.temperature)
        llm=v5.StrictOllamaClient(settings,timeout_seconds=args.timeout,
                                  max_tokens=args.initial_tokens,num_ctx=args.num_ctx,
                                  max_output_tokens=args.max_output_tokens)
        failure_count=0
        for method, ex in pending:
            t0=time.monotonic()
            answer=None
            error=None
            for attempt in range(args.retries+1):
                try:
                    answer=base.llm_predict(ex,method,llm)
                    if answer['prediction'] not in CLASSES:
                        raise RuntimeError('Unexpected non-class label, not caching')
                    break
                except Exception as e:
                    error=e
                    print(f'ATTEMPT {attempt+1}/{args.retries+1} {model} {method} '
                          f'{ex.origin.episode_id} -> {type(e).__name__}: {str(e)[:150]}',flush=True)
            elapsed=round(time.monotonic()-t0,2)
            if answer is None:
                failure_count+=1
                append_jsonl(error_path,{
                    'model':model,'method':method,'episode_id':ex.origin.episode_id,
                    'error_type':type(error).__name__, 'message':str(error)[:300],
                    'diagnostics':getattr(error,'diagnostics',None),
                    'elapsed_seconds':elapsed,'attempts':args.retries+1,
                    'not_a_prediction':True})
                continue
            record={
                'signature':signature,'model':model,'method':method,
                'episode_id':ex.origin.episode_id,
                'target_episode_id':ex.target.episode_id,
                'prediction':answer['prediction'],'elapsed_seconds':elapsed,
                'ollama_stats':llm.last_stats, 'reused_from_v5':False,
            }
            append_jsonl(cache_path,record)
            cache[(method,ex.origin.episode_id)] = record
            print(f'OK {model} {method}: {len(cache)}/{len(selected)*2} '
                  f'label={answer["prediction"]} {elapsed:.2f}s',flush=True)
        model_cache[model]=cache
        failures[model]=failure_count
        print(f'FINISHED {model}: {len(cache)}/{len(selected)*2} valid, failures={failure_count}',flush=True)
        summarize(models,selected,model_cache,v5_output,out,metadata,failures,cfg)
        free_previous_model(model)
        last_model = None
    result = summarize(models,selected,model_cache,v5_output,out,metadata,failures,cfg)
    print('\n=== FINAL V6 ===',flush=True)
    print(json.dumps({'status':result['status'],
                      'models':list(models),
                      'common_complete_episodes':result['n_common_complete_all_models_both_methods'],
                      'report':str(out/'size_comparison_report.md'),
                      'metrics_csv':str(out/'metrics_by_size.csv'),
                      'manifest_json':str(out/'study_manifest.json')},indent=2),flush=True)
    if result['status']!='COMPLETE':
        print('Some models incomplete; same command resumes cached calls. '
              'See failures_*.jsonl and compare only common episodes.',flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
