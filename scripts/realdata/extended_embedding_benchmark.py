#!/usr/bin/env python3
"""Evaluate text + projected-graph embeddings on identical observation IDs.

Includes country-block bootstrap CI (sampling uncertainty), label-permutation
null within source modality, KMeans seeds (optimization variability), and
optional cosine-agglomerative clustering. These are corpus labels, not event gold.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()


def read_rows(path):
    with path.open(encoding='utf8') as f:rows=[json.loads(x) for x in f if x.strip()]
    if not rows:raise ValueError('Empty input')
    if any(r.get('provenance',{}).get('sampled') for r in rows):
        raise ValueError('Sampled source observations are not a scientific benchmark')
    if len(set(r['observation_id'] for r in rows))!=len(rows):raise ValueError('Duplicate observation IDs')
    return rows


def input_diagnostics(rows):
    from importlib.util import spec_from_file_location,module_from_spec
    f=Path(__file__).with_name('embed_kg_observations.py')
    spec=spec_from_file_location('masking_diagnostics',f)
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    groups={}
    for r in rows:groups.setdefault(str(r['country_id']),set()).add(r['disease_label'])
    a=collections.Counter(r['source'].split('_')[0] for r in rows)
    explicit=collections.defaultdict(lambda:dict(affected=0,total=0,concepts_explicit=0,concepts_total=0))
    for r in rows:
        label=str(r['disease_label']); rec=explicit[label]
        rec['total']+=1
        names=[str(c.get('name','')) for c in r.get('top_concepts',[])]
        exp=sum(bool(mod.DISEASE_TERMS.search(name)) for name in names)
        rec['concepts_explicit']+=exp;rec['concepts_total']+=len(names)
        rec['affected']+=int(exp>0)
    return {'n_observations':len(rows),'n_countries':len(groups),
            'countries_with_both_disease_labels':sum(len(v)>1 for v in groups.values()),
            'source_modality_counts':dict(a), 'explicit_disease_terms_by_label':dict(explicit),
            'note':'Country and disease/corpus labels may be confounded; no independently verified event-level gold.'}


def source_matrices(rows,input_path,text_root,struct_root):
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer,normalize
    from sklearn.feature_extraction.text import TfidfTransformer
    from importlib.util import spec_from_file_location,module_from_spec
    f=Path(__file__).with_name('embed_kg_observations.py')
    spec=spec_from_file_location('masking_rules_2',f)
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    concepts=[[str(c.get('name') or c.get('id') or '') for c in row['top_concepts']] for row in rows]
    ids=[r['observation_id'] for r in rows]
    for mask in (False,True):
        lists=[[name for name in sub if name and (not mask or not mod.DISEASE_TERMS.search(name))] for sub in concepts]
        B=MultiLabelBinarizer().fit_transform(lists).astype('float32')
        label='masked' if mask else 'all'
        if B.shape[1]<2:continue
        yield 'bow_binary_'+label,B,{'family':'bag_of_words','mask':mask}
        yield 'bow_tfidf_'+label,TfidfTransformer().fit_transform(B).toarray().astype('float32'),{'family':'tfidf','mask':mask}
    h=sha(input_path)
    for root,family in [(text_root,'ollama_text'),(struct_root,'projected_structural')]:
        if not root.is_dir():continue
        for manifest in sorted(root.glob('**/manifest.json')):
            folder=manifest.parent
            meta=json.loads(manifest.read_text(encoding='utf8'))
            if meta.get('source_sha256')!=h:
                raise ValueError(f'Dataset hash mismatch: {manifest}')
            if meta.get('sample'):
                raise ValueError(f'Refusing sampled matrix: {manifest}')
            fp=folder/'observations.csv'
            if not fp.is_file() or not (folder/'embeddings.npy').is_file():
                raise FileNotFoundError(f'Incomplete matrix: {folder}')
            with fp.open(newline='',encoding='utf8') as stream:
                actual=[r['observation_id'] for r in csv.DictReader(stream)]
            if ids!=actual:
                raise ValueError(f'Observation order mismatch: {folder}')
            X=np.load(folder/'embeddings.npy',mmap_mode='r')
            if X.ndim!=2 or X.shape[0]!=len(ids) or not np.isfinite(X).all():
                raise ValueError(f'Invalid matrix: {folder}')
            label=family+'__'+folder.relative_to(root).as_posix().replace('/','__')
            yield label,X,{'family':family,'manifest':str(manifest),'mask':bool(meta.get('mask_disease_terms',False))}


def align_predictions(y,pred):
    from sklearn.metrics import f1_score
    s1=f1_score(y,pred,average='macro',zero_division=0)
    s2=f1_score(y,1-pred,average='macro',zero_division=0)
    return 1-pred if s2>s1 else pred


def evaluate_single(y,pred):
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score,balanced_accuracy_score
    p=align_predictions(y,pred)
    return {'ARI':float(adjusted_rand_score(y,pred)),
            'NMI':float(normalized_mutual_info_score(y,pred)),
            'macro_F1_oracle_aligned':float(f1_score(y,p,average='macro',zero_division=0)),
            'balanced_accuracy_oracle_aligned':float(balanced_accuracy_score(y,p))}


def uncertainty(y,pred,rows,bootstrap,permutations,seed):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, f1_score
    rng=np.random.default_rng(seed)
    country_ids=[str(r.get('country_id') or r['observation_id']) for r in rows]
    by_country=collections.defaultdict(list)
    for i,c in enumerate(country_ids):by_country[c].append(i)
    names=list(by_country)
    b_ari=[];b_f1=[]
    for _ in range(bootstrap):
        chosen=rng.choice(names,size=len(names),replace=True)
        idx=np.asarray([idx for c in chosen for idx in by_country[c]],dtype=int)
        if len(set(y[idx]))<2:continue
        yr=y[idx];pr=pred[idx]
        b_ari.append(float(adjusted_rand_score(yr,pr)))
        b_f1.append(float(f1_score(yr,align_predictions(yr,pr),average='macro',zero_division=0)))
    source_groups=collections.defaultdict(list)
    for i,r in enumerate(rows):source_groups[r['source'].split('_')[0]].append(i)
    ref=float(adjusted_rand_score(y,pred))
    larger=0
    for _ in range(permutations):
        perm=np.array(y,copy=True)
        for ids in source_groups.values():perm[ids]=rng.permutation(perm[ids])
        larger+=float(adjusted_rand_score(perm,pred))>=ref-1e-12
    return {'country_block_bootstrap_CI_ARI_95':
            [float(np.quantile(b_ari,.025)),float(np.quantile(b_ari,.975))] if b_ari else None,
            'country_block_bootstrap_CI_macro_F1_95':
            [float(np.quantile(b_f1,.025)),float(np.quantile(b_f1,.975))] if b_f1 else None,
            'label_permutation_p_within_source_modality':(larger+1)/(permutations+1) if permutations else None,
            'bootstrap_valid_replicates':len(b_ari),'permutations':permutations,
            'warning':'Bootstrap resamples countries with fixed cluster assignments; it is not a retrained-clustering CI. '
                      'Permutation null shuffles retrospective disease/corpus labels within source modality.'}


def run(args):
    import numpy as np
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.preprocessing import normalize
    rows=read_rows(args.input)
    y=np.asarray([r['disease_label'] for r in rows],dtype=int)
    if set(y)!={0,1}:raise ValueError('Expected labels 0 and 1')
    diagnostics=input_diagnostics(rows)
    results=[];details={}
    for label,X,meta in source_matrices(rows,args.input,args.text_root,args.structural_root):
        if args.max_matrices and len(details)>=args.max_matrices:break
        print('BENCHMARK',label, X.shape, flush=True)
        X=np.asarray(X,dtype='float32')
        for normalized in [False,True]:
            V=normalize(X) if normalized else X
            if len(np.unique(V,axis=0))<2:
                print('SKIP degenerate',label,normalized,flush=True);continue
            run_name=label+('__unitnorm' if normalized else '__raw')
            seed_results=[]
            for seed in args.seeds:
                pred=KMeans(n_clusters=2,n_init=10,random_state=seed).fit_predict(V)
                seed_results.append({'seed':seed,**evaluate_single(y,pred)})
            first_pred=KMeans(n_clusters=2,n_init=10,random_state=args.seeds[0]).fit_predict(V)
            ci=uncertainty(y,first_pred,rows,args.bootstrap,args.permutations,args.seeds[0])
            row={'method':run_name,'family':meta['family'],'n':len(y),'dimension':X.shape[1],
                 'n_seeds':len(args.seeds),'ARI_mean':float(np.mean([z['ARI'] for z in seed_results])),
                 'ARI_sd':float(np.std([z['ARI'] for z in seed_results],ddof=1)) if len(seed_results)>1 else 0.0,
                 'NMI_mean':float(np.mean([z['NMI'] for z in seed_results])),
                 'macro_F1_mean':float(np.mean([z['macro_F1_oracle_aligned'] for z in seed_results])),
                 'macro_F1_sd':float(np.std([z['macro_F1_oracle_aligned'] for z in seed_results],ddof=1)) if len(seed_results)>1 else 0.0,
                 'balanced_accuracy_mean':float(np.mean([z['balanced_accuracy_oracle_aligned'] for z in seed_results])),
                 'ARI_CI_95_low':ci['country_block_bootstrap_CI_ARI_95'][0] if ci['country_block_bootstrap_CI_ARI_95'] else '',
                 'ARI_CI_95_high':ci['country_block_bootstrap_CI_ARI_95'][1] if ci['country_block_bootstrap_CI_ARI_95'] else '',
                 'F1_CI_95_low':ci['country_block_bootstrap_CI_macro_F1_95'][0] if ci['country_block_bootstrap_CI_macro_F1_95'] else '',
                 'F1_CI_95_high':ci['country_block_bootstrap_CI_macro_F1_95'][1] if ci['country_block_bootstrap_CI_macro_F1_95'] else '',
                 'permutation_p':ci['label_permutation_p_within_source_modality']}
            results.append(row)
            details[run_name]={'meta':meta,'seeds':seed_results,'uncertainty':ci}
            # Cosine hierarchical is deterministic; only on L2-normalized representation.
            if normalized:
                try:
                    if (np.linalg.norm(V,axis=1)==0).any():raise ValueError('Zero vectors')
                    pred=AgglomerativeClustering(n_clusters=2,linkage='average',metric='cosine').fit_predict(V)
                    metrics=evaluate_single(y,pred)
                    agg={**row,'method':run_name+'__agglomerative_cosine','n_seeds':1,'ARI_mean':metrics['ARI'],
                         'ARI_sd':0.0,'NMI_mean':metrics['NMI'],'macro_F1_mean':metrics['macro_F1_oracle_aligned'],
                         'macro_F1_sd':0.0,'balanced_accuracy_mean':metrics['balanced_accuracy_oracle_aligned']}
                    for col in ('ARI_CI_95_low','ARI_CI_95_high','F1_CI_95_low','F1_CI_95_high','permutation_p'):agg[col]=''
                    results.append(agg)
                    details[agg['method']]={'meta':meta,'metrics':metrics}
                except ValueError as ex:
                    details[run_name]['agglomerative_skip']=str(ex)
    args.out.mkdir(parents=True,exist_ok=True)
    if not results:raise RuntimeError('No evaluable matrices')
    with (args.out/'results.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    data={'created_utc':datetime.now(timezone.utc).isoformat(),
          'source_sha256':sha(args.input),'diagnostics':diagnostics,
          'seeds':args.seeds,'bootstrap_replicates':args.bootstrap,'label_permutations':args.permutations,
          'notes':['Cluster labels are aligned to known disease labels ONLY for retrospective evaluation.',
                   'These labels are derived from the two sampled corpora and are NOT independently adjudicated clinical outcomes.',
                   'Optimization-seed SD is not dataset uncertainty. Country-block CI and modality-stratified label permutation reported separately.',
                   'Temporal and spatial fields can encode corpus provenance rather than epidemiological signal.',
                   'This benchmark does not validate event linking, causation or story correctness.'],
          'results':details}
    (args.out/'results.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    (args.out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')
    print('DONE',len(results),'benchmark rows',args.out/'results.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/realdata/reconstructed_observations.jsonl'))
    p.add_argument('--text-root',type=Path,default=Path('outputs/realdata/embeddings'))
    p.add_argument('--structural-root',type=Path,default=Path('outputs/realdata/structural_embeddings'))
    p.add_argument('--out',type=Path,default=Path('outputs/realdata/full_benchmark'))
    p.add_argument('--seeds',type=int,nargs='+',default=[9,15,42,99,123])
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--permutations',type=int,default=200)
    p.add_argument('--max-matrices',type=int,default=0)
    a=p.parse_args()
    if a.bootstrap<0 or a.permutations<0:p.error('Invalid bootstrap or permutations')
    run(a)

if __name__=='__main__':main()
