"""Opt-in exact vector reuse with stable observation IDs; no inferred matrix row order."""
from .core import *
from .data import read_jsonl

def discover_vectors(root,level):
    preferred=[root/'data/reviewer_v8_ddpf'/f'{level}_embeddinggemma_country_time_vectors.jsonl',root/'outputs/realdata'/f'{level}_embeddinggemma_country_time_vectors.jsonl']
    candidates=[p for p in preferred if p.is_file()]
    if len(candidates)>1 and len({sha(p) for p in candidates})>1:raise ValueError('Multiple historical vector exports; specify --embedding-jsonl explicitly: '+str(candidates))
    if candidates:
        print('Reusing ID-keyed DDPF vector export:',candidates[0],flush=True);return str(candidates[0])
    return None
