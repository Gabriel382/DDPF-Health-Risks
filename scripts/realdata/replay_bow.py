#!/usr/bin/env python3
"""Reproduce the BoW -> K-Means branch in the original COVID/Mpox observation notebook.

Faithfully follows notebook cells 14-20 and 22-39 with explicit handling of
sparse features, stable concept ordering, cluster-label permutation and a
reproducibility manifest. Requires six observation CSVs from Detection.
This is disease clustering, NOT Story Forest event-link evaluation.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

SOURCES = (
    ('covid_observations_journal.csv', 'Journal_COVID', 0),
    ('covid_observations_medical.csv', 'Medical_COVID', 0),
    ('covid_observations_social.csv', 'Social_COVID', 0),
    ('monkeypox_observations_journal.csv', 'Journal_Monkeypox', 1),
    ('monkeypox_observations_medical.csv', 'Medical_Monkeypox', 1),
    ('monkeypox_observations_social.csv', 'Social_Monkeypox', 1),
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fin:
        while chunk := fin.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def parse_list(text: str) -> list[str]:
    if not text or text.strip().lower() in {'nan', 'none', 'null'}:
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple, set)): return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, str): return [obj.strip()] if obj.strip() else []
    except (ValueError, SyntaxError):
        pass
    # Compatible with the notebook's fallback string splitting.
    return [v.strip().strip('"\'') for v in text.strip('[]').split(',') if v.strip().strip('"\'')]


def select_top_25(row: dict, top: int = 25) -> list[str]:
    """Original notebook cell 17: remove duplicate names, score by intensity, top 25.

    The raw order and exact ties can be dataset-specific. Use deterministic
    sorting by numeric intensity and concept name to keep auditability.
    """
    names = parse_list(row.get('c.name', ''))
    intensity = parse_list(row.get('intensity', ''))
    if not names:
        return []
    if not intensity:
        raise ValueError('Observation has concepts but no intensity column; cannot replay cell 17')
    # zip mirrors the historical code when lengths do not match.
    ranked: dict[str, tuple[float, int]] = {}
    for idx, (name, score_raw) in enumerate(zip(names, intensity)):
        if name not in ranked:
            try: rank = float(score_raw)
            except ValueError: raise ValueError(f'Invalid intensity {score_raw!r}')
            ranked[name] = (rank, idx)
    return [name for name, _ in sorted(ranked.items(), key=lambda kv: (kv[1][0], kv[0]))[-top:]]


def load_rows(directory: Path, top: int):
    rows = []
    checksums = {}
    for fname, prefix, disease in SOURCES:
        filepath = directory / fname
        if not filepath.is_file():
            raise FileNotFoundError(f'Missing Detection observation table: {filepath}')
        checksums[fname] = file_hash(filepath)
        with filepath.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'System_Name', 'c.name', 'intensity'} <= set(reader.fieldnames or []):
                raise ValueError(f'{fname} lacks System_Name, c.name, or intensity; got {reader.fieldnames}')
            for line_idx, row in enumerate(reader, start=2):
                concepts = select_top_25(row, top)
                obs_id = prefix + '_' + row['System_Name'].replace('"', '')
                rows.append({'observation_id': obs_id, 'concepts': concepts,
                             'source_file': fname, 'source_line': line_idx, 'truth': disease,
                             'System_ID': row.get('System_ID', '')})
    if not rows: raise ValueError('All six source CSVs are empty')
    duplicate_ids = len(rows) - len({r['observation_id'] for r in rows})
    if duplicate_ids:
        raise ValueError(f'{duplicate_ids} repeated System_Name identifiers. The original notebook dictionary would silently overwrite them; fix input provenance before comparing.')
    return rows, checksums


def run(directory: Path, outdir: Path, top: int = 25, seed: int = 9) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, classification_report, f1_score
    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    import sklearn

    rows, hashes = load_rows(directory, top)
    if len(rows) < 2: raise ValueError('Need at least two observations to form two clusters')
    mlb = MultiLabelBinarizer(sparse_output=True)
    x = mlb.fit_transform([r['concepts'] for r in rows]).astype('float32')
    if x.shape[1] == 0: raise ValueError('No extracted concept names in six source files')
    y = np.array([r['truth'] for r in rows], dtype=int)
    if len(set(y.tolist())) < 2: raise ValueError('Need both diseases for two-class benchmark')
    pred = KMeans(n_clusters=2, random_state=seed, n_init='auto').fit_predict(x)
    # Clusters are unlabeled: choose the better mapping as the original comparative work does.
    f1_normal = f1_score(y, pred, average='weighted', zero_division=0)
    f1_swapped = f1_score(y, 1 - pred, average='weighted', zero_division=0)
    flipped = f1_swapped > f1_normal
    aligned = 1 - pred if flipped else pred
    results = {
        'method': 'historical_observation_bow_kmeans',
        'source_notebook': 'Experiments/Prediction/Prediction Module - COVID vs Monkeypox Observation.ipynb',
        'source_cells': [14,15,17,18,19,20,22,23,24,26,27,30,35,36,38,39],
        'n_observations': len(rows), 'n_concepts': int(x.shape[1]),
        'truth_support': {'COVID': int((y == 0).sum()), 'Monkeypox': int((y == 1).sum())},
        'k': 2, 'random_state': seed, 'n_init': 'auto', 'top_concepts': top,
        'best_cluster_mapping_flipped': bool(flipped),
        'weighted_f1': float(f1_score(y, aligned, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y, aligned)),
        'ari': float(adjusted_rand_score(y, pred)),
        'ami': float(adjusted_mutual_info_score(y, pred)),
        'classification_report': classification_report(y, aligned, labels=[0,1], target_names=['COVID','Monkeypox'], output_dict=True, zero_division=0),
        'input_sha256': hashes,
        'environment': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__},
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'limitations': ['Disease labels correlate with corpus collection periods (COVID 2019/2020 vs Monkeypox 2022).',
                        'No evidence that an observation cluster implies transmission or event continuity.',
                        'Not the Full Benchmark: this is only the historical BoW/KMeans branch.']
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (outdir / 'cluster_assignments.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['observation_id','source_file','source_line','System_ID','truth','cluster_raw','prediction_aligned'])
        for i,row in enumerate(rows):
            writer.writerow([row['observation_id'],row['source_file'],row['source_line'],row['System_ID'],row['truth'],int(pred[i]),int(aligned[i])])
    print(json.dumps({k:results[k] for k in ('n_observations','n_concepts','truth_support','weighted_f1','accuracy','ari','ami')}, indent=2))
    print('Outputs:', outdir)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/neo4j', help='Six source-specific observation CSVs')
    p.add_argument('--outdir', default='outputs/realdata/bow_kmeans')
    p.add_argument('--top', type=int, default=25)
    p.add_argument('--seed', type=int, default=9)
    p.add_argument('--check', action='store_true', help='Only verify six input files')
    args = p.parse_args()
    missing = [str(Path(args.data_dir) / fname) for fname,_,_ in SOURCES if not (Path(args.data_dir) / fname).is_file()]
    if missing:
        for name in missing: print('MISSING',name)
        raise SystemExit(2)
    if args.check:
        print('All six source-specific observation CSVs are available.')
        return
    run(Path(args.data_dir), Path(args.outdir), top=args.top, seed=args.seed)

if __name__ == '__main__':
    main()
