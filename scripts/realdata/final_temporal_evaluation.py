#!/usr/bin/env python3
"""One-shot, read-only final evaluation of PreDiViD V5 aggregate temporal study.

Checks all existing experiments, without refitting or calling Ollama. Contains a
separate static-KG country-profile permutation negative control, reusing V4's
numeric-neighbor candidate algorithm (not stochastic refitting).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from audit_temporal_paired_v5 import (
    M1, M2N, M2K, M3, M4, PRIMARY, country_id, read_predictions,
    make_pairs, score, summarize
)

METHODS = (M1, M2N, M2K, M3, M4)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def two_class_metrics(rows: list[dict], pred_field='prediction') -> dict:
    usable = [r for r in rows if r['target_label'] in PRIMARY]
    if not usable:
        return {'n': 0, 'accuracy': None, 'macro_f1': None}
    shaped = [{'true': r['target_label'], 'p': r[pred_field]} for r in usable]
    out = score(shaped, 'p')
    confusion = {t: dict(Counter(r[pred_field] for r in usable if r['target_label'] == t)) for t in PRIMARY}
    recalls, precision = {}, {}
    for cls in PRIMARY:
        tp = sum(r['target_label'] == cls and r[pred_field] == cls for r in usable)
        true_total = sum(r['target_label'] == cls for r in usable)
        pred_total = sum(r[pred_field] == cls for r in usable)
        recalls[cls] = round(tp / true_total, 6) if true_total else None
        precision[cls] = round(tp / pred_total, 6) if pred_total else None
    available = [v for v in recalls.values() if v is not None]
    out.update({'balanced_accuracy_observed_classes': round(sum(available)/len(available),6) if available else None,
                'recall_by_class':recalls, 'precision_by_class':precision, 'confusion':confusion})
    return out


def validate_dataset(study: Path, snapshots_path: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((study / 'manifest.json').read_text(encoding='utf-8'))
    rows = read_csv(study / 'predictions.csv')
    snaps = read_csv(snapshots_path)
    if not rows or not snaps:
        raise ValueError('Missing snapshots or predictions data')
    if manifest.get('sha256_snapshots') != file_sha256(snapshots_path):
        raise ValueError('Snapshot SHA256 differs from the V5 manifest; refusing stale data')
    cutoff = date.fromisoformat(manifest['cutoff_date'])
    sidx = {}
    by_series = defaultdict(list)
    for r in snaps:
        eid = r['episode_id']
        if eid in sidx:
            raise ValueError('Duplicate weekly episode ID: '+eid)
        if not r['source_relation_id']:
            raise ValueError('Missing source relation provenance for '+eid)
        if date.fromisoformat(r['week_start']).weekday() != 0:
            raise ValueError('Weekly snapshot not on Monday: '+eid)
        sidx[eid] = r
        by_series[(r['disease_id'], r['country_id'])].append(r)
    for key, series in by_series.items():
        series.sort(key=lambda r: r['week_start'])
        for i in range(1, len(series)):
            prev, curr = series[i-1], series[i]
            if (date.fromisoformat(curr['week_start']) - date.fromisoformat(prev['week_start'])).days != 7:
                raise ValueError('Gap found in weekly series '+str(key))
            if curr['previous_episode_id'] != prev['episode_id']:
                raise ValueError('Broken previous_episode_id in '+str(key))
    prediction_count = Counter()
    seen = set()
    origin_label = {}
    llm_rows = []
    for r in rows:
        method = r['method']
        if method not in METHODS:
            raise ValueError('Unknown method in predictions: '+method)
        unique = (r['origin_id'], method)
        if unique in seen:
            raise ValueError('Duplicate prediction '+str(unique))
        seen.add(unique)
        a = sidx.get(r['origin_id']); b = sidx.get(r['target_id'])
        if not a or not b:
            raise ValueError('Prediction references unknown snapshots: '+str(unique))
        if a['country_id'] != b['country_id'] or a['disease_id'] != b['disease_id']:
            raise ValueError('Origin/target countries or diseases do not match: '+str(unique))
        at, bt = date.fromisoformat(r['origin_week']), date.fromisoformat(r['target_week'])
        if at != date.fromisoformat(a['week_start']) or bt != date.fromisoformat(b['week_start']):
            raise ValueError('Episode dates disagree with snapshot file: '+str(unique))
        if (bt-at).days != 7 or bt < cutoff:
            raise ValueError('Not a holdout consecutive-week episode: '+str(unique))
        if b['previous_episode_id'] != a['episode_id']:
            raise ValueError('Episode does not follow origin: '+str(unique))
        if r['source_relation_id_origin'] != a['source_relation_id'] or r['source_relation_id_target'] != b['source_relation_id']:
            raise ValueError('Provenance mismatch in prediction: '+str(unique))
        delta = float(b['reported_value']) - float(a['reported_value'])
        expected = 'increase' if delta > 1e-9 else ('revision' if delta < -1e-9 else 'flat')
        if r['target_label'] != expected:
            raise ValueError('Target label disagrees with actual future snapshot: '+str(unique))
        if r['prediction'] not in ('increase', 'flat', 'revision', 'unknown'):
            raise ValueError('Unknown predicted label: '+str(unique))
        orig = r['origin_id']
        if orig in origin_label and origin_label[orig] != expected:
            raise ValueError('Inconsistent target across methods: '+orig)
        origin_label[orig] = expected
        if r['disease'] != a['disease']:
            raise ValueError('Disease name inconsistent: '+str(unique))
        if method in (M3, M4):
            if r['split'] != 'holdout_sample':
                raise ValueError('LLM row incorrectly labeled as all holdout')
            llm_rows.append(r)
        elif r['split'] != 'holdout_all':
            raise ValueError('Baseline prediction not on full holdout')
        prediction_count[method] += 1
    n_holdout = int(manifest['n_holdout_examples'])
    if any(prediction_count[m] != n_holdout for m in (M1, M2N, M2K)):
        raise ValueError('Missing baseline holdout predictions')
    if len(origin_label) != n_holdout:
        raise ValueError('Number of unique holdout episodes inconsistent with manifest')
    expected = int(manifest.get('expected_llm_calls', 0))
    completed = int(manifest.get('completed_llm_calls', 0))
    if completed != len(llm_rows) or expected != completed:
        raise ValueError('Incomplete LLM calls, or manifest and predictions disagree')
    paired = {r['origin_id'] for r in llm_rows if r['method'] == M3} & {r['origin_id'] for r in llm_rows if r['method'] == M4}
    if len(paired) * 2 != completed or len(paired) != int(manifest.get('paired_complete_examples', -1)):
        raise ValueError('LLM paired comparisons do not match manifest')
    # Cross-check actual output cache, not only row-level predictions.
    cache_path = study / 'llm_predictions.jsonl'
    cache = {}
    with cache_path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            cache[(c['episode_id'], c['method'])] = c
    for r in llm_rows:
        c = cache.get((r['origin_id'], r['method']))
        if not c or c['prediction'] != r['prediction']:
            raise ValueError('LLM cache and prediction CSV disagree')
    # The true label must be computed from target only, never used as input;
    # we check split and source consistency, not claim formal feature-leak proof.
    count_class = Counter(origin_label.values())
    return ({
        'status': 'PASS',
        'v5_manifest_complete': True,
        'snapshots': len(snaps),
        'country_disease_series': len(by_series),
        'holdout': n_holdout,
        'prediction_rows': len(rows),
        'pairwise_llm_episodes': len(paired),
        'llm_cache_entries': len(cache),
        'cutoff_date': cutoff.isoformat(),
        'last_train_target': manifest.get('last_train_target'),
        'first_holdout_target': manifest.get('first_holdout_target'),
        'train_target_before_holdout': date.fromisoformat(manifest['last_train_target']) < cutoff <= date.fromisoformat(manifest['first_holdout_target']),
        'valid_consecutive_origin_target_links': n_holdout,
        'all_snapshot_series_without_gaps': True,
        'all_target_labels_recomputed_from_snapshots': True,
        'provenance_ids_verified': True,
        'class_counts_holdout': dict(count_class),
        'KG_covered_holdout': sum(r['has_static_kg_context'] == '1' for r in rows if r['method'] == M1),
        'LLM_by_disease': dict(Counter(r['disease'] for r in llm_rows if r['method'] == M3)),
    }, rows, snaps)


def cohort_table(rows: list[dict]) -> list[dict]:
    groups = {}
    for m in METHODS:
        batch = [r for r in rows if r['method'] == m]
        if m in (M3, M4):
            cohorts = {'llm_sample': batch}
        else:
            cohorts = {'full_holdout': batch, 'kg_eligible': [r for r in batch if r['has_static_kg_context'] == '1'],
                       'kg_ineligible': [r for r in batch if r['has_static_kg_context'] != '1']}
        for c, part in cohorts.items():
            for d in ('ALL', 'COVID-19', 'Mpox'):
                sub = [r for r in part if d == 'ALL' or r['disease'] == d]
                if sub:
                    result = {'method': m, 'cohort': c, 'disease': d,
                              'classes': dict(Counter(r['target_label'] for r in sub)),
                              'predicted': dict(Counter(r['prediction'] for r in sub)),
                              **two_class_metrics(sub)}
                    groups[m,c,d] = result
    # Always-flat & always-increase controls for appropriate primary cohorts.
    for cohort in ('full_holdout', 'kg_eligible', 'kg_ineligible', 'llm_sample'):
        reference_method = M3 if cohort == 'llm_sample' else M1
        ref = [r for r in rows if r['method'] == reference_method and
               (cohort == 'llm_sample' or (cohort == 'full_holdout' or
                (r['has_static_kg_context'] == '1') == (cohort == 'kg_eligible')))]
        if not ref:
            continue
        for d in ('ALL', 'COVID-19', 'Mpox'):
            base = [r for r in ref if d == 'ALL' or r['disease'] == d]
            if not base:
                continue
            for default in ('flat','increase'):
                fake = [{**r, 'prediction': default} for r in base]
                groups['always_'+default,cohort,d] = {
                    'method': 'always_'+default, 'cohort': cohort, 'disease': d,
                    'classes': dict(Counter(r['target_label'] for r in fake)),
                    'predicted': {default: len(fake)}, **two_class_metrics(fake),
                }
    return [groups[k] for k in sorted(groups)]


def paired_comparisons(rows: list[dict], bootstraps: int, signflips: int, seed: int) -> list[dict]:
    records = read_predictions_from_rows(rows)
    conf = [
        (M2N,M2K,'full_holdout',None,None),
        (M2N,M2K,'kg_eligible',True,None),
        (M2N,M2K,'kg_ineligible',False,None),
        (M3,M4,'llm_paired',True,None),
        (M2K,M3,'llm_paired',True,None),
        (M2K,M4,'llm_paired',True,None),
    ]
    for disease in ('COVID-19','Mpox'):
        conf.extend([
            (M2N,M2K,'kg_eligible_'+disease,True,disease),
            (M3,M4,'llm_paired_'+disease,True,disease),
            (M2K,M3,'llm_paired_'+disease,True,disease),
            (M2K,M4,'llm_paired_'+disease,True,disease),
        ])
    results = []
    for j,(a,b,cohort,kg,disease) in enumerate(conf):
        pairs = make_pairs(records,a,b,kg_only=kg,disease=disease)
        result = summarize(pairs,a,b,cohort,bootstraps,seed+j)
        if pairs:
            result['cluster_signflip_accuracy'] = cluster_signflip_accuracy(pairs,signflips,seed+j)
        results.append(result)
    # Holm family-wise correction across the pre-specified report contrasts.
    valid = sorted([(i, r['cluster_signflip_accuracy']['two_sided_p'])
                    for i,r in enumerate(results) if r.get('n',0)], key=lambda x:x[1])
    maximum = 0.
    for rank,(idx,pval) in enumerate(valid):
        maximum = max(maximum, min(1.,pval*(len(valid)-rank)))
        results[idx]['cluster_signflip_accuracy']['holm_adjusted_p_across_report_contrasts'] = round(maximum,6)
    return results


def read_predictions_from_rows(rows: list[dict]) -> dict:
    return {(r['origin_id'],r['method']): r for r in rows}


def cluster_signflip_accuracy(pairs: list[dict], permutations: int, seed: int) -> dict:
    """Cluster paired-label exchange test, two-sided, with within-country dependence.

    Tests equality of paired correctness *conditionally on these country clusters*.
    Not a test of external generalization or retraining stochasticity.
    """
    c = defaultdict(int)
    for p in pairs:
        c[p['country_id']] += int(p['b'] == p['true']) - int(p['a'] == p['true'])
    vals = list(c.values())
    obs = abs(sum(vals))
    if not obs:
        return {'two_sided_p': 1., 'n_country_blocks': len(vals), 'permutations': permutations}
    rnd = random.Random(seed)
    exceed = 0
    for _ in range(permutations):
        stat = abs(sum(v if rnd.random() < .5 else -v for v in vals))
        exceed += stat >= obs
    return {'two_sided_p': round((exceed+1)/(permutations+1),6),
            'n_country_blocks': len(vals), 'permutations': permutations,
            'assumption': 'exchangeable paired algorithm labels at country-block level'}


def kg_negative_control(study: Path, rows: list[dict], permutations: int, seed: int) -> dict:
    """Permute static KG profiles across countries within each disease, preserving
    KG coverage, fixed numeric features, and numeric nearest-neighbor pool.
    No model training, LLM or Neo4j calls; uses real V4 KNN mechanics.
    """
    import final_knn_reference_v4 as base
    manifest = json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    series = base.load_snapshots(Path(manifest['input']))
    profiles = base.load_static_profiles(Path(manifest['profiles']))
    if manifest.get('sha256_profiles') and base.hash_file(Path(manifest['profiles'])) != manifest['sha256_profiles']:
        raise ValueError('Static KG source hash differs from V5 manifest')
    examples = base.make_examples(series,profiles)
    cutoff = date.fromisoformat(manifest['cutoff_date'])
    train = [e for e in examples if e.target.week < cutoff]
    by_id = {e.origin.episode_id: e for e in examples if e.target.week >= cutoff and e.semantic_keys}
    kg_rows = [r for r in rows if r['method'] == M2K and r['has_static_kg_context'] == '1']
    test = [by_id[r['origin_id']] for r in kg_rows]
    if not train or not test:
        raise ValueError('Cannot build KG placebo: no training/test examples')
    if 'n_train_examples' in manifest and len(train) != int(manifest['n_train_examples']):
        raise ValueError('Chronological training count differs from V5 manifest')
    if 'n_kg_eligible_holdout' in manifest and len(test) != int(manifest['n_kg_eligible_holdout']):
        raise ValueError('KG-covered holdout count differs from V5 manifest')
    K = int(manifest.get('knn_k', 9))
    X = np.asarray([e.features for e in train],dtype=float)
    Y = np.asarray([e.features for e in test],dtype=float)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd<1.e-8]=1.
    X, Y = (X-mean)/sd, (Y-mean)/sd
    by_disease = defaultdict(list)
    for i,e in enumerate(train):
        by_disease[e.origin.disease_id].append(i)
    candidate_lists=[]
    for i,ex in enumerate(test):
        indexes = np.asarray(by_disease[ex.origin.disease_id],dtype=int)
        dist = np.linalg.norm(X[indexes]-Y[i],axis=1) / math.sqrt(X.shape[1])
        order = np.argsort(dist,kind='stable')[:min(max(K*15,80),len(dist))]
        candidate_lists.append([(float(dist[o]),int(indexes[o])) for o in order])
    def rerank(mapping=None):
        out=[]
        for ex,candidates in zip(test,candidate_lists):
            a = ex.semantic_keys if mapping is None else mapping.get((ex.origin.disease_id,ex.origin.country_id),ex.semantic_keys)
            scored=[]
            for rank,j in candidates:
                tr=train[j]
                b=tr.semantic_keys if mapping is None else (mapping.get((tr.origin.disease_id,tr.origin.country_id),tr.semantic_keys) if tr.semantic_keys else set())
                if a and b:
                    rank+=0.3*(1-len(a&b)/max(len(a|b),1))
                scored.append((rank,j))
            choices=[train[j].target_label for _,j in sorted(scored)[:K]]
            votes=Counter(choices)
            out.append(sorted(votes,key=lambda label:(-votes[label],('flat','increase','revision').index(label)))[0])
        return out
    original = rerank()
    expected = [r['prediction'] for r in kg_rows]
    if original != expected:
        mismatch = [(r['origin_id'],o,e) for r,o,e in zip(kg_rows,original,expected) if o!=e]
        raise ValueError(f'Placebo KNN reconstruction does not match original M2 KG (first mismatch: {mismatch[:2]})')
    baseline = two_class_metrics(kg_rows)
    original_f1 = baseline['macro_f1']
    original_accuracy = baseline['accuracy']
    # Stable KG sets are assigned at country level, never at individual week.
    country_features=defaultdict(dict)
    for ex in train+test:
        if ex.semantic_keys:
            country_features[ex.origin.disease_id][ex.origin.country_id] = ex.semantic_keys
    rnd = random.Random(seed)
    perm_f1=[]
    perm_acc=[]
    for _ in range(permutations):
        mapping={}
        for disease, items in sorted(country_features.items()):
            country_ids=sorted(items)
            values=[items[x] for x in country_ids]
            rnd.shuffle(values)
            mapping.update({(disease,country):value for country,value in zip(country_ids,values)})
        predictions=rerank(mapping)
        shuffled=[{**r,'prediction':p} for r,p in zip(kg_rows,predictions)]
        m=two_class_metrics(shuffled)
        perm_f1.append(m['macro_f1']);perm_acc.append(m['accuracy'])
    return {
        'status':'PASS', 'n_train_examples':len(train), 'n_kg_eligible_episodes':len(test),
        'n_country_profiles_per_disease':{d:len(c) for d,c in country_features.items()},
        'verified_exact_reproduction_of_original_M2KG_predictions':True,
        'original_macro_f1':original_f1, 'original_accuracy':original_accuracy,
        'placebo_permutations':permutations,
        'placebo_f1_mean':round(float(np.mean(perm_f1)),6),
        'placebo_f1_std':round(float(np.std(perm_f1)),6),
        'placebo_f1_min':min(perm_f1), 'placebo_f1_max':max(perm_f1),
        'placebo_accuracy_mean':round(float(np.mean(perm_acc)),6),
        'empirical_upper_tail_p_f1':round((1+sum(x>=original_f1 for x in perm_f1))/(permutations+1),6),
        'note':'Exploratory profile-shuffling negative control; keeps country-level profile identity but permutes biomedical content within each disease. Not a randomized train/test re-fit.',
    }


def save_summary_md(report: dict, destination: Path) -> None:
    lines=[
        '# PreDiViD V5: final one-shot evaluation',
        '',
        '## Integrity',
        f"- Status: **{report['integrity']['status']}**",
        f"- Weekly snapshots: {report['integrity']['snapshots']}; full holdout: {report['integrity']['holdout']}",
        f"- Paired LLM examples: {report['integrity']['pairwise_llm_episodes']}; cached outputs: {report['integrity']['llm_cache_entries']}",
        f"- Holdout true labels: {report['integrity']['class_counts_holdout']}",
        '', '## Baselines and ablations',
        '| Method | Cohort | Disease | n | Macro-F1 | Accuracy |',
        '|---|---|---|---:|---:|---:|',
    ]
    for row in report['performance']:
        if row['disease']=='ALL' and row['cohort'] in ('full_holdout','kg_eligible','llm_sample'):
            lines.append(f"| {row['method']} | {row['cohort']} | {row['disease']} | {row['n']} | {row['macro_f1']} | {row['accuracy']} |")
    lines+=['', '## Paired statistical comparisons',
            '| Comparison (B minus A) | Cohort | n | Δmacro-F1 | 95% country-block CI | Δaccuracy | Cluster sign-flip p |',
            '|---|---|---:|---:|---|---:|---:|']
    for p in report['paired_comparisons']:
        if p.get('n',0):
            lines.append(f"| {p['comparison']} | {p['cohort']} | {p['n']} | {p['delta_macro_f1']:+.4f} | {p['bootstrap_conditional_fixed_predictions']['delta_macro_f1_95pct_CI']} | {p['delta_accuracy']:+.4f} | {p['cluster_signflip_accuracy']['two_sided_p']} |")
    pl=report['KG_placebo']
    lines += ['', '## KG negative control',
              f"- Original M2KG F1: {pl['original_macro_f1']}; shuffled KG mean F1: {pl['placebo_f1_mean']}",
              f"- Shuffled profile runs: {pl['placebo_permutations']}; empirical F1 upper-tail p: {pl['empirical_upper_tail_p_f1']}",
              '', '## Interpretation boundaries',
              '- This is a benchmark of **direction of next aggregate reported count**; it is not a validation of documentary Story Forest links.',
              '- M2 is supervised on historical next-week labels; M3/M4 use prompt-only inference. Do not imply a controlled apples-to-apples architecture comparison.',
              '- KG source profiles are static historically collected concepts, not evidence specific to the target week.',
              '- Holdout has no downward revisions; results assess the binary increase/flat subtask only.',
              '- Country-block bootstrap conditions on predictions and selected countries; LLM selection is deterministic and not population-random.',
              '- No causal inference, transmission claims, verified incidence, OWL reasoner, or document-linked event gold.',
              '- Multiple planned contrasts are adjusted using Holm correction in the JSON; raw p-values remain in the table.',
              '']
    destination.write_text('\n'.join(lines),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, default=Path('outputs/realdata/temporal_trajectory_study_v5'))
    ap.add_argument('--snapshots', type=Path, default=Path('outputs/realdata/who_series/weekly/snapshots.csv'))
    ap.add_argument('--bootstraps', type=int, default=2000)
    ap.add_argument('--country-signflips', type=int, default=5000)
    ap.add_argument('--kg-placebos', type=int, default=99)
    ap.add_argument('--seed', type=int, default=20261117)
    args=ap.parse_args()
    if min(args.bootstraps,args.country_signflips,args.kg_placebos) < 1:
        ap.error('Counts for resampling must all be positive')
    print('1/4 Integrity, chronology, provenance, V5 cache...',flush=True)
    integrity,rows,_=validate_dataset(args.study,args.snapshots)
    print('   PASS: '+str(integrity['holdout'])+' holdout, '+str(integrity['pairwise_llm_episodes'])+' LLM episodes',flush=True)
    print('2/4 Baselines, two diseases, covered/uncovered KG...',flush=True)
    perf=cohort_table(rows)
    print('3/4 Paired country-block bootstrap and cluster sign flips...',flush=True)
    comparisons=paired_comparisons(rows,args.bootstraps,args.country_signflips,args.seed)
    print('4/4 KG-shuffling negative control, exact M2 KNN reconstruction...',flush=True)
    placebo=kg_negative_control(args.study,rows,args.kg_placebos,args.seed)
    report={
        'scope':'Aggregate reported-value next-week direction benchmark, NOT documentary Story Forest',
        'integrity':integrity,'performance':perf,'paired_comparisons':comparisons,
        'KG_placebo':placebo,
        'uncertainty_methods': {
            'bootstrap':'country-block resampling conditional on already fitted predictions',
            'country_signflip':'country-level paired-method label-exchange permutation for accuracy',
            'KG_placebo':'shuffled static country KG profile identities within each disease, fixed numerical neighbor candidates',
            'caveat':'exploratory multiple comparisons; p-values unadjusted; not repeated trainings'
        },
        'scientific_boundaries':[
            'No source-backed documentary events, no gold link labels, no Story Forest event-link evaluation',
            'Semantic profiles from source-specific collection dates are static, not evidence specific to weekly snapshots',
            'M4 type and provenance guards do not implement complete OWL reasoning',
            'Labels are higher/equal/lower observed aggregate values, not validated incidence',
            'Only increase versus flat labels in holdout; revision-class evaluation not possible',
            'Deterministic disease-balanced LLM sample of KG-covered examples; not random representative sampling',
            'M2 has supervised historical next-week labels, unlike prompting M3/M4',
            'Source count semantics COVID versus Mpox not independently verified',
        ]
    }
    dest=args.study/'final_one_shot_audit.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=args.study/'final_one_shot_report.md'
    save_summary_md(report,md)
    perfpath=args.study/'final_one_shot_performance.csv'
    with perfpath.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','cohort','disease','n','macro_f1','accuracy','balanced_accuracy_observed_classes','recall_by_class','precision_by_class','confusion','classes','predicted'])
        writer.writeheader()
        for p in perf:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in p.items()})
    print('\n=== FINAL SUMMARY ===')
    for x in perf:
        if x['disease']=='ALL' and x['cohort']=='full_holdout':
            print(f"{x['method']}: n={x['n']} F1={x['macro_f1']} Acc={x['accuracy']}")
    for x in comparisons:
        if x.get('n') and x['cohort'] in ('kg_eligible','llm_paired'):
            print(f"{x['cohort']}: {x['comparison']} n={x['n']} ΔF1={x['delta_macro_f1']:+.4f} "
                  f"CI={x['bootstrap_conditional_fixed_predictions']['delta_macro_f1_95pct_CI']} "
                  f"p_country={x['cluster_signflip_accuracy']['two_sided_p']}")
    print(f"KG placebo: F1 real={placebo['original_macro_f1']}, shuffled mean={placebo['placebo_f1_mean']}, "
          f"upper-tail p={placebo['empirical_upper_tail_p_f1']}")
    print('\nSAVED:',dest,md,perfpath,sep='\n')


if __name__=='__main__':
    main()
