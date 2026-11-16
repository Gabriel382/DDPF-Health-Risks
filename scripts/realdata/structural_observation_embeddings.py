#!/usr/bin/env python3
"""Unsupervised graph embeddings on the *projected observation-concept bipartite graph*.

NOT full BioPropaPhenKG embeddings. No disease/source labels are included in the graph.
Supports SVD, FastRP-inspired random-projection propagation, optional biased random
walk skip-gram (node2vec-like), and optional GraphSAGE link reconstruction.
No mutation or connection to Neo4j is necessary.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()


def load_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        rows=[json.loads(line) for line in f if line.strip()]
    ids=[x['observation_id'] for x in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate observation_id')
    if any(r.get('provenance',{}).get('sampled') for r in rows):
        raise ValueError('Sampled input is not allowed in a full experimental run')
    return rows


def build_graph(rows,masked=False,with_country=False):
    import numpy as np
    from scipy.sparse import coo_matrix
    # import local module from the previous drop-in, without importing the __main__ block
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        f=Path(__file__).with_name('embed_kg_observations.py')
        spec=spec_from_file_location('source_masking_rule',f)
        mod=module_from_spec(spec);spec.loader.exec_module(mod)
        mask_rule=mod.DISEASE_TERMS
    except FileNotFoundError:
        raise RuntimeError('Missing scripts/realdata/embed_kg_observations.py from previous package')
    concepts={}
    edges={}
    for i,obs in enumerate(rows):
        for c in obs.get('top_concepts',[]):
            name=str(c.get('name') or '').strip()
            if masked and mask_rule.search(name):continue
            key=str(c.get('id') or name).strip()
            if not key:continue
            if key not in concepts:concepts[key]=len(concepts)
            j=concepts[key]
            # Numeric intensity is not a disease label, and here is not used to make
            # primary structural baselines comparable to binary BoW.
            edges[(i,j)]=1.0
    n=len(rows)
    m=len(concepts)
    if m<2:raise ValueError('Fewer than two concepts survived the mask')
    I=[i for i,j in edges]
    J=[j for i,j in edges]
    V=[1.0]*len(edges)
    B=coo_matrix((V,(I,J)),shape=(n,m),dtype=np.float32).tocsr()
    graph_keys=['obs:'+r['observation_id'] for r in rows]+['concept:'+key for key in concepts]
    extra=[]
    if with_country:
        countries={}
        for obs in rows:
            c=str(obs.get('country_id') or obs.get('country_name') or '').strip()
            if c and c not in countries:countries[c]=len(countries)
        for i,obs in enumerate(rows):
            c=str(obs.get('country_id') or obs.get('country_name') or '').strip()
            if c:extra.append((i,n+m+countries[c]))
        graph_keys.extend('country:'+c for c in countries)
    row=np.asarray([i for i,j in edges]+[n+j for i,j in edges]+[a for a,b in extra]+[b for a,b in extra],dtype=int)
    col=np.asarray([n+j for i,j in edges]+[i for i,j in edges]+[b for a,b in extra]+[a for a,b in extra],dtype=int)
    vals=np.ones(len(row),dtype=np.float32)
    N=len(graph_keys)
    adjacency=coo_matrix((vals,(row,col)),shape=(N,N),dtype=np.float32).tocsr()
    if not (adjacency!=adjacency.T).nnz==0:raise ValueError('Adjacency is not symmetric')
    return B,adjacency,{'vertices':N,'observation_vertices':n,'concept_vertices':m,
                         'country_vertices':N-n-m,'undirected_edges':adjacency.nnz//2,
                         'disease_names_masked':masked,'with_country_nodes':with_country}


def svd_embed(B,dim,seed):
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    X=TfidfTransformer().fit_transform(B)
    components=min(dim,max(1,min(X.shape)-1))
    if components<2:raise ValueError('Insufficient rank for SVD')
    Z=TruncatedSVD(n_components=components,random_state=seed).fit_transform(X)
    return Z.astype('float32')


def fastrp_style_embed(A,n,dim,seed):
    """Random node initialization + row-stochastic multi-hop propagation.
    FastRP-inspired research baseline; NOT the Neo4j GDS FastRP algorithm.
    """
    import numpy as np
    from scipy.sparse import diags
    rng=np.random.default_rng(seed)
    N=A.shape[0]
    degree=np.asarray(A.sum(axis=1)).reshape(-1)
    if (degree==0).any():raise ValueError('Graph contains isolated nodes')
    P=diags(1.0/degree)@A
    H=rng.normal(0,1/math.sqrt(dim),size=(N,dim)).astype('float32')
    snapshots=[]
    for _ in range(4):
        H=(P@H).astype('float32')
        snapshots.append(H)
    Z=.15*snapshots[0]+.35*snapshots[1]+.35*snapshots[2]+.15*snapshots[3]
    return Z[:n].astype('float32')


def node2vec_embed(A,n,dim,seed,walks=8,walk_length=24,epochs=5):
    """Biased random walks + skip-gram. Requires gensim (optional dependency)."""
    import numpy as np
    try:
        from gensim.models import Word2Vec
    except ImportError as ex:
        raise RuntimeError('node2vec requires gensim; install with python -m pip install gensim') from ex
    rng=np.random.default_rng(seed)
    neighbors=[A.indices[A.indptr[i]:A.indptr[i+1]].tolist() for i in range(A.shape[0])]
    sets=[set(v) for v in neighbors]
    # Node2Vec transition preference: return 1/p, neighbor-of-prev 1, outward 1/q.
    p,q=1.0,2.0
    walks_out=[]
    order=list(range(A.shape[0]))
    for _ in range(walks):
        rng.shuffle(order)
        for start in order:
            trace=[start]
            while len(trace)<walk_length:
                current=trace[-1]; options=neighbors[current]
                if not options:break
                if len(trace)==1:choice=int(rng.choice(options))
                else:
                    prev=trace[-2]
                    weights=[(1/p if v==prev else 1.0 if v in sets[prev] else 1/q) for v in options]
                    w=np.asarray(weights,dtype=float);w/=w.sum()
                    choice=int(rng.choice(options,p=w))
                trace.append(choice)
            walks_out.append([str(z) for z in trace])
    model=Word2Vec(sentences=walks_out,vector_size=dim,window=5,sg=1,
                   min_count=0,workers=1,seed=seed,epochs=epochs,negative=5)
    return np.asarray([model.wv[str(i)] for i in range(n)],dtype=np.float32)


def graphsage_embed(A,n,dim,seed,epochs=65):
    """Projected-graph self-supervised GraphSAGE-style link reconstruction.
    Not an inductive out-of-graph evaluation, and NOT the original notebook model.
    Requires torch (optional dependency)."""
    import numpy as np
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as ex:
        raise RuntimeError('graphsage requires torch; install PyTorch matching the machine and CUDA') from ex
    import scipy.sparse as sp
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    degree=np.asarray(A.sum(axis=1)).reshape(-1)
    norm=sp.diags(1.0/np.maximum(degree,1))@A
    adj=torch.from_numpy(norm.toarray().astype('float32'))
    N=A.shape[0]
    features=rng.normal(0,.4,size=(N,64)).astype('float32')
    features[:,0]=np.log1p(degree)
    features[:n,1]=1.0  # node category, not disease label
    features[n:,2]=1.0
    x=torch.from_numpy(features)
    positives=np.vstack(A[:n].nonzero()).T.astype('int64')
    if len(positives)==0:raise ValueError('No positive observation graph edges')
    positives[:,1]+=0   # global column indices already in A
    actual=set(map(tuple,positives.tolist()))
    # Deterministic negative pairs restricted to the graph's non-observation vertices.
    neg=[];target=len(positives)
    maxtries=max(2000,target*50)
    for _ in range(maxtries):
        i=int(rng.integers(0,n));j=int(rng.integers(n,N))
        if (i,j) not in actual:neg.append((i,j))
        if len(neg)>=target:break
    if len(neg)<target:raise ValueError('Insufficient negative non-edges')
    pos=torch.tensor(positives,dtype=torch.long)
    negatives=torch.tensor(neg,dtype=torch.long)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.self1=nn.Linear(64,dim)
            self.neigh1=nn.Linear(64,dim,bias=False)
            self.self2=nn.Linear(dim,dim)
            self.neigh2=nn.Linear(dim,dim,bias=False)
        def forward(self,x):
            h=F.relu(self.self1(x)+self.neigh1(adj@x))
            return F.normalize(self.self2(h)+self.neigh2(adj@h),dim=-1)
    net=Model()
    optimizer=torch.optim.Adam(net.parameters(),lr=.008,weight_decay=1e-4)
    for ep in range(epochs):
        optimizer.zero_grad()
        z=net(x)
        a=(z[pos[:,0]]*z[pos[:,1]]).sum(dim=1)*5
        b=(z[negatives[:,0]]*z[negatives[:,1]]).sum(dim=1)*5
        loss=F.binary_cross_entropy_with_logits(a,torch.ones_like(a))+F.binary_cross_entropy_with_logits(b,torch.zeros_like(b))
        loss.backward();optimizer.step()
    with torch.no_grad():return net(x)[:n].detach().cpu().numpy().astype('float32')


def write_result(path,method,masked,with_country,dim,seed,rows,B,A,graph_stats,input_path):
    import numpy as np
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    meta=path/'manifest.json'
    h=sha256(input_path)
    cfg={'input_sha256':h,'algorithm':method,'seed':seed,'dimension_requested':dim,
         'mask_disease_terms':masked,'with_country_nodes':with_country,'graph_stats':graph_stats,
         'version':1}
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if meta.exists():
        old=json.loads(meta.read_text(encoding='utf-8'))
        if old['fingerprint']!=fingerprint:raise ValueError(f'Existing output differs: {path}')
        if not (path/'embeddings.npy').exists():raise FileNotFoundError('Missing existing matrix')
        print('SKIP existing',method,masked,with_country,flush=True)
        return
    if (path/'embeddings.npy').exists():raise FileExistsError(f'Orphan matrix at {path}')
    n=len(rows)
    if method=='svd':Z=svd_embed(B,dim,seed)
    elif method=='fastrp_style':Z=fastrp_style_embed(A,n,dim,seed)
    elif method=='node2vec':Z=node2vec_embed(A,n,dim,seed)
    elif method=='graphsage':Z=graphsage_embed(A,n,dim,seed)
    else:raise ValueError(method)
    if Z.shape[0]!=n or not np.isfinite(Z).all():raise ValueError('Invalid matrix')
    np.save(path/'embeddings.npy',Z)
    with (path/'observations.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['row','observation_id','country_id','source','disease_label'])
        for i,r in enumerate(rows):writer.writerow([i,r['observation_id'],r.get('country_id',''),r.get('source',''),r.get('disease_label','')])
    cfg.update({'fingerprint':fingerprint,'created_utc':datetime.now(timezone.utc).isoformat(),
                'method':'structural_projected_bipartite','model':method,
                'features':('obs_concept_country' if with_country else 'obs_concept')+('_masked' if masked else ''),
                'source_sha256':h,'dimension':int(Z.shape[1]),'sample':False,
                'warning':'Projected country-source observation-concept graph, not full BioPropaPhenKG. GraphSAGE/fastrp_style are local research baselines, not Neo4j GDS outputs.'})
    meta.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    print('DONE',method,'mask',masked,'country',with_country, Z.shape,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/realdata/reconstructed_observations.jsonl'))
    p.add_argument('--out',type=Path,default=Path('outputs/realdata/structural_embeddings'))
    p.add_argument('--methods',nargs='+',choices=['svd','fastrp_style','node2vec','graphsage'],default=['svd','fastrp_style','node2vec','graphsage'])
    p.add_argument('--dimension',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--strict',action='store_true',help='Fail if gensim or torch unavailable')
    args=p.parse_args()
    if args.dimension<4:p.error('--dimension must be >=4')
    rows=load_rows(args.input)
    summary={}
    failures=[]
    for masked in (False,True):
      for with_country in (False,True):
        B,A,stats=build_graph(rows,masked,with_country)
        for method in args.methods:
            label=method+('_masked' if masked else '')+('_country' if with_country else '')
            directory=args.out/label
            try:
                write_result(directory,method,masked,with_country,args.dimension,args.seed,rows,B,A,stats,args.input)
                summary[label]='complete'
            except (ImportError,RuntimeError) as exc:
                # Never suppress errors other than explicitly missing optional dependencies.
                msg=str(exc)
                if ('requires gensim' in msg or 'requires torch' in msg) and not args.strict:
                    print('SKIP',label,msg,flush=True)
                    summary[label]='skipped_missing_optional_dependency: '+msg
                else:raise
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'run_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Structure status:',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
