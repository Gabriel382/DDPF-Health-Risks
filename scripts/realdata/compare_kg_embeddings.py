#!/usr/bin/env python3
"""Compare recovered KG observations using BoW and one or more Ollama matrices.

Clustering is fitted on the *full embedding*, never on a 2D t-SNE visualization.
The COVID-vs-Mpox labels are imperfect retrospective corpus labels, NOT event-link ground truth.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def rows_from_jsonl(path):
    objs=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip():objs.append(json.loads(line))
    return objs


def metrics_for(X,y,seeds):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score,normalized_mutual_info_score,
                                 f1_score,balanced_accuracy_score,accuracy_score)
    import numpy as np
    if X.shape[0]!=len(y):raise ValueError('Shape mismatch')
    if X.shape[0]<3 or len(set(y))!=2:raise ValueError('Need >=3 observations from both COVID and Mpox')
    if len(np.unique(X,axis=0))<2:raise ValueError('Embeddings contain <2 distinct vectors')
    runs=[]
    for seed in seeds:
        predictions=KMeans(n_clusters=2,random_state=seed,n_init=10).fit_predict(X)
        a=f1_score(y,predictions,average='macro',zero_division=0)
        b=f1_score(y,1-predictions,average='macro',zero_division=0)
        aligned=1-predictions if b>a else predictions
        runs.append({'seed':seed,'ARI':float(adjusted_rand_score(y,predictions)),
                     'NMI':float(normalized_mutual_info_score(y,predictions)),
                     'macro_f1_aligned':float(f1_score(y,aligned,average='macro',zero_division=0)),
                     'accuracy_aligned':float(accuracy_score(y,aligned)),
                     'balanced_accuracy_aligned':float(balanced_accuracy_score(y,aligned)),
                     'mapping_flipped':bool(b>a),'clusters':predictions.tolist()})
    aggregate={}
    for k in ('ARI','NMI','macro_f1_aligned','accuracy_aligned','balanced_accuracy_aligned'):
        vals=[r[k] for r in runs]
        aggregate[k]={'mean':sum(vals)/len(vals),'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
    return {'runs':runs,'aggregate':aggregate}


def read_matrix(folder,source_rows,source_path,allow_sample=False):
    import numpy as np
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('source_sha256')!=hash_file(source_path):
        raise ValueError(f'Input content hash does not match the model matrix {folder}')
    if meta.get('sample') and not allow_sample:
        raise ValueError(f'{folder} is from a sample run, refuse official benchmark. Pass --allow-sample for smoke testing')
    X=np.load(folder/'embeddings.npy',mmap_mode='r')
    with (folder/'observations.csv').open('r',encoding='utf-8',newline='') as f:
        recs=list(csv.DictReader(f))
    expected=[r['observation_id'] for r in source_rows]
    actual=[r['observation_id'] for r in recs]
    if expected!=actual:
        raise ValueError(f'Order/identity mismatch for {folder}. Refusing silent row reordering.')
    if X.shape[0]!=len(recs):raise ValueError(f'X rows mismatch for {folder}')
    return X,meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('outputs/realdata/reconstructed_observations.jsonl'))
    ap.add_argument('--embeddings-root',type=Path,default=Path('outputs/realdata/embeddings'))
    ap.add_argument('--out',type=Path,default=Path('outputs/realdata/embedding_benchmark'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[9,15,42,99,123])
    ap.add_argument('--allow-sample',action='store_true')
    a=ap.parse_args()
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer
    rows=rows_from_jsonl(a.input)
    if not rows:ap.error('No rows in observation JSONL')
    if not a.allow_sample:
        if any(row.get('provenance',{}).get('sampled') for row in rows):
            ap.error('Input is a partial graph sample. Pass --allow-sample for smoke tests.')
    y=np.asarray([r['disease_label'] for r in rows],dtype=int)
    concepts=[[str(c.get('name') or c.get('id')) for c in row.get('top_concepts',[])] for row in rows]
    bow=MultiLabelBinarizer(sparse_output=False).fit_transform(concepts).astype('float32')
    benchmark={'created_utc':datetime.now(timezone.utc).isoformat(),
               'n_observations':len(rows),'truth_support':{'COVID':int((y==0).sum()),'Mpox':int((y==1).sum())},
               'seeds':a.seeds,'input':str(a.input),'methods':{},
               'limitations':['Disease labels are correlated with collection period.',
                              'This is retrospective cluster separation; not causal prediction or event relation validity.',
                              'Explicit disease names may naturally occur among observed KG concepts.',
                              'Historical notebook clustered TSNE-reduced observations in some cells; this comparison clusters full vectors.',
                              'Source and truth label are not embedded as direct text inputs.']}
    benchmark['methods']['binary_BoW_concepts']={'dimension':int(bow.shape[1]),
        'metrics':metrics_for(bow,y,a.seeds)}
    # Leakage ablation: exclude concept names that directly identify the disease.
    from importlib.util import spec_from_file_location, module_from_spec
    model_file=Path(__file__).with_name('embed_kg_observations.py')
    spec=spec_from_file_location('local_ollama_embedding_text',model_file)
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    masked=[[name for name in cs if not mod.DISEASE_TERMS.search(name)] for cs in concepts]
    if any(masked) and any(concepts!=masked for concepts,masked in zip(concepts,masked)):
        masked_bow=MultiLabelBinarizer(sparse_output=False).fit_transform(masked).astype('float32')
        if masked_bow.shape[1]>=2:
            benchmark['methods']['binary_BoW_mask_disease_terms']={'dimension':int(masked_bow.shape[1]),
                'metrics':metrics_for(masked_bow,y,a.seeds)}
    folders=sorted(a.embeddings_root.glob('*/**/manifest.json'))
    for file in folders:
        folder=file.parent
        X,meta=read_matrix(folder,rows,a.input,a.allow_sample)
        label=f"ollama_{meta['model']}_{meta['features']}"
        benchmark['methods'][label]={'dimension':int(X.shape[1]),'manifest':str(file),
                                     'metrics':metrics_for(X,y,a.seeds)}
    if len(benchmark['methods'])==1:
        print('WARNING: No completed Ollama matrices found. Only BoW evaluated.')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'results.json').write_text(json.dumps(benchmark,indent=2,ensure_ascii=False),encoding='utf-8')
    with (a.out/'results.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['method','dimension','n_observations','ARI_mean','ARI_sd','NMI_mean','macro_F1_mean','macro_F1_sd','balanced_accuracy_mean'])
        for method,vals in benchmark['methods'].items():
            agg=vals['metrics']['aggregate']
            writer.writerow([method,vals['dimension'],len(rows),
                agg['ARI']['mean'],agg['ARI']['sd'],agg['NMI']['mean'],
                agg['macro_f1_aligned']['mean'],agg['macro_f1_aligned']['sd'],
                agg['balanced_accuracy_aligned']['mean']])
    print(json.dumps({'methods':list(benchmark['methods']), 'output':str(a.out/'results.csv')},indent=2))

if __name__=='__main__':main()
