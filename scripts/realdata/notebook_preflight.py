#!/usr/bin/env python3
"""Static preflight of original Prediction notebooks; never executes notebook cells."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

NOTEBOOKS = {
    'observation': 'Experiments/Prediction/Prediction Module - COVID vs Monkeypox Observation.ipynb',
    'benchmark': 'Experiments/Prediction/Prediction Module - COVID vs Monkeypox Benchmark.ipynb',
    'full': 'Experiments/Prediction/Prediction Module - Full Benchmark.ipynb',
    'llm': 'Experiments/Prediction/LLM Based Prediction Module.ipynb',
}


def notebook_assignments(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding='utf-8'))
    found: list[dict] = []
    for cell_number, cell in enumerate(doc['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        try:
            tree = ast.parse('\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('%')))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (target.id.startswith('path_to_') or target.id.endswith('_path')):
                        val = node.value.value
                        if '/' in val or '\\' in val:
                            found.append({'cell': cell_number, 'variable': target.id, 'raw_path': val})
    return found


def check_path(root: Path, notebook_path: Path, raw_path: str) -> dict:
    path = Path(raw_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([notebook_path.parent / path, root / 'Experiments' / path, root / path])
        # Historic "../data" paths assume a working dir different from the notebook dir.
        trimmed = raw_path
        while trimmed.startswith('../') or trimmed.startswith('..\\'):
            trimmed = trimmed[3:]
        candidates.append(root / trimmed)
    unique = list(dict.fromkeys(str(p.resolve()) for p in candidates))
    existing = [p for p in unique if Path(p).exists()]
    return {'raw_path': raw_path, 'existing_paths': existing, 'found': bool(existing), 'candidate_paths': unique}


def inspect(root: Path, mode: str) -> dict:
    report = {'root': str(root), 'notebooks': {}}
    keys = NOTEBOOKS if mode == 'all' else {mode: NOTEBOOKS[mode]}
    for name, rel in keys.items():
        nb = root / rel
        if not nb.exists():
            report['notebooks'][name] = {'notebook': rel, 'error': 'file_missing'}
            continue
        assignments = notebook_assignments(nb)
        checks = [dict(cell=x['cell'], variable=x['variable'], **check_path(root, nb, x['raw_path'])) for x in assignments]
        report['notebooks'][name] = {'notebook': rel, 'path_checks': checks,
                                     'present': sum(c['found'] for c in checks),
                                     'missing': sum(not c['found'] for c in checks)}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--mode', choices=['all', *NOTEBOOKS], default='all')
    ap.add_argument('--output', default='outputs/realdata/notebook_preflight.json')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    report = inspect(root, args.mode)
    dest = root / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, item in report['notebooks'].items():
        print(f'{name:12s} available={item.get("present",0):2d} missing={item.get("missing",0):2d}  {item.get("notebook")}')
        for missing in item.get('path_checks', []):
            if not missing['found']:
                print(f'  cell {missing["cell"]:2d} {missing["variable"]}: {missing["raw_path"]} [MISSING]')
        if item.get('error'): print(' ', item['error'])
    print('Report:', dest)

if __name__ == '__main__':
    main()
