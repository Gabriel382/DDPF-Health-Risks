"""Evaluate only independently verifiable pair labels; UNKNOWN excluded.

Synthetic negatives and ontology-constraint tests must be reported separately.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from .models import Link


def read_jsonl(path:str):
    with Path(path).open(encoding='utf-8') as fh:
        for line_no,line in enumerate(fh,1):
            if not line.strip():continue
            try: yield json.loads(line)
            except ValueError as e:raise ValueError(f'{path}:{line_no}: {e}') from e

def evaluate_pairs(predicted:list[Link],gold_path:str,method:str|None=None)->dict:
    predicted_map={(p.parent_id,p.child_id):p for p in predicted if method is None or p.method==method}
    # Labeling semantics are exact; related_only is not the same as documented_continuation.
    confusion=Counter();excluded=0;compared=0
    verified_positive={'same_event','documented_continuation','related_only'}
    for row in read_jsonl(gold_path):
        if row.get('label')=='unknown': excluded+=1; continue
        if row.get('basis') not in ('independent_documented','manual_gold'):
            excluded+=1; continue
        pair=(row['parent_id'],row['child_id'])
        got=predicted_map.get(pair)
        target=row['label']
        pred=got.relation if got else 'unknown'
        compared+=1
        if target in verified_positive:
            if pred==target: confusion['tp']+=1
            else:
                confusion['fn']+=1
                if pred in verified_positive: confusion['fp']+=1
        elif pred in verified_positive:confusion['fp']+=1
        confusion['abstained']+=int(pred=='unknown')
        confusion['correct_exact']+=int(pred==target)
    tp=confusion['tp'];fp=confusion['fp'];fn=confusion['fn']
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=(2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None)
    return {'method':method,'verified_pairs':compared,'excluded_unknown_or_unverified':excluded,
            'precision':precision,'recall':recall,'f1':f1,
            'exact_accuracy':confusion['correct_exact']/compared if compared else None,
            'abstention_rate':confusion['abstained']/compared if compared else None,
            'counts':dict(confusion),
            'warning':'Partial-label metric: non-annotated pairs are NOT assumed negative. F1 only over verified pair annotations.'}
