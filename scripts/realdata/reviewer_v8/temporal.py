"""Fixed original cohorts, new cross-family prompts and numerical baselines.
No target information enters a prediction payload. All legacy metrics are kept separately.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import date,timedelta
import json,math,random
import numpy as np
from .core import *

SNAP='outputs/realdata/who_series/weekly/snapshots.csv'
LEGACY='data/reviewer_v8/llm_v6_summary.json'

def direction(d):return 'increase' if d>0 else 'revision' if d<0 else 'flat'

def build_examples(rows):
    g=defaultdict(list)
    for r in rows:g[r['disease_id'],r['country_id']].append(r)
    ex=[];excluded=Counter();eligible_pairs=0
    for (d,c),rs in g.items():
        rs=sorted(rs,key=lambda r:r['week_start']);seen=set()
        for r in rs:
            if r['week_start'] in seen:raise ValueError('duplicate weekly snapshot '+str((d,c,r['week_start'])))
            seen.add(r['week_start']);date.fromisoformat(r['week_start']);float(r['reported_value'])
        for i in range(len(rs)-1):
            now,next_=rs[i],rs[i+1];t=date.fromisoformat(now['week_start']);t1=date.fromisoformat(next_['week_start'])
            if (t1-t).days!=7:excluded['nonadjacent_target']+=1;continue
            eligible_pairs+=1
            if i<3:excluded['insufficient_four_snapshot_history']+=1;continue
            window=rs[i-3:i+1];ds=[date.fromisoformat(x['week_start']) for x in window]
            if any((b-a).days!=7 for a,b in zip(ds,ds[1:])):excluded['history_gap']+=1;continue
            vals=[float(x['reported_value']) for x in window];future=float(next_['reported_value']);diff=[b-a for a,b in zip(vals,vals[1:])]
            ex.append({'origin_id':now['episode_id'],'disease_id':d,'disease':now['disease'],'country_id':c,'country':now['country_name'],'origin_week':str(t),'target_week':str(t1),'history':vals,'historical_differences':diff,'label':direction(future-vals[-1]),'previous_direction':direction(diff[-1])})
    return ex,{'snapshots':len(rows),'series':len(g),'adjacent_target_pairs':eligible_pairs,'included':len(ex),'excluded':dict(excluded)}

def profile_candidates(root,expected):
    paths=[]
    for base in [root/'data',root/'outputs'/'realdata']:
        if base.exists():paths.extend(p for p in base.rglob('*') if p.is_file() and 'profil' in p.name.lower() and p.suffix.lower() in ('.json','.jsonl','.csv','.tsv') and 'reviewer_v8' not in p.parts)
    for p in paths:
        if sha(p)==expected:return p
    # Fallback to content hashes, even if legacy naming does not mention profiles.
    for base in [root/'data',root/'outputs'/'realdata']:
        if base.exists():
            for p in base.rglob('*'):
                if p.is_file() and p not in paths and p.suffix.lower() in ('.json','.jsonl','.csv','.tsv') and 'reviewer_v8' not in p.parts and p.stat().st_size<64*1024*1024 and sha(p)==expected:return p
    return None

def profile_records(obj):
    """Index only records with explicit disease + country IDs (or compound keys).
    Never send arbitrary metadata, case counts, labels, or target columns to the model.
    """
    found={};allowed={'concepts','concept_ids','concept_names','concept_labels','concept_counts','concept_weights','concept_profile','profile','terms','cuis','features','top_concepts','biomedical_concepts','concept_intensities','concept_freq','concept_frequencies','vector','concept_vector','source_counts'}
    def walk(x,hints=()):
        if isinstance(x,dict):
            d=first(x,'disease_id','disease_cui','disease','phenomenon_id',default='');c=first(x,'country_id','location_id','country',default='');alltext=' '.join([str(d),str(c),*map(str,hints)])
            ds=re.findall(r'C\d{7}',alltext);cs=re.findall(r'wkg:\d+',alltext)
            vals={k:v for k,v in x.items() if k in allowed and v not in (None,'',[],{})}
            for k,v in list(vals.items()):
                if isinstance(v,str) and v[:1] in ('[','{'):
                    try:vals[k]=json.loads(v)
                    except ValueError:pass
            # Accept a pure CUI->weight map, with parent keys carrying the identities.
            if not vals and x and all(re.fullmatch(r'C\d{7}',str(k)) and isinstance(v,(int,float,str)) for k,v in x.items()):vals={'concept_weights':x}
            if ds and cs and vals:
                found[(ds[0],cs[0])]=vals;return
            for k,v in x.items():
                if isinstance(v,(list,dict)):walk(v,hints+(str(k),))
        elif isinstance(x,list):
            for v in x:walk(v,hints)
    walk(obj);return found

def load_profiles(root,expected):
    path=profile_candidates(root,expected)
    if not path:return {},{'status':'exact_legacy_profile_file_not_found','expected_sha256':expected}
    if path.suffix=='.csv':obj=csvread(path)
    elif path.suffix=='.jsonl':obj=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    elif path.suffix=='.tsv':
        import csv
        with path.open() as f:obj=list(csv.DictReader(f,delimiter='\t'))
    else:obj=load(path)
    profiles=profile_records(obj)
    # Bound supplied features, keep raw data hash and do not claim same old prompt.
    def safe(v,depth=0):
        if depth>5:return None
        if isinstance(v,dict):return {str(k):safe(x,depth+1) for k,x in sorted(v.items(),key=lambda t:str(t[0]))[:80] if not re.search(r'(target|label|future|report|cases|date|prediction|outcome)',str(k),re.I) or k=='concept_labels'}
        if isinstance(v,list):return [safe(x,depth+1) for x in v[:80]]
        return v if isinstance(v,(int,float,bool)) else str(v)[:300]
    profiles={k:safe(v) for k,v in profiles.items()}
    return profiles,{'status':'parsed' if profiles else 'exact_file_found_schema_not_recognized','file':str(path.relative_to(root)),'sha256':sha(path),'n_profiles':len(profiles)}

def harmonize_legacy(summary):
    """Exact binary macro F1 from archived support, prediction marginals, accuracy and recall.
    The 8B revision prediction counts as an error, never silently recoded as flat.
    """
    rows=[]
    for r in summary['metrics']:
        if r['cohort']!='common_all_models':continue
        a=r['actual'];p=r['predicted'];a=json.loads(a) if isinstance(a,str) else a;p=json.loads(p) if isinstance(p,str) else p
        n=r['n'];ni=a.get('increase',0);nf=a.get('flat',0)
        if ni+nf!=n or sum(p.values())!=n or not ni or not nf or r['recall_increase'] is None:continue
        tpi=round(ni*r['recall_increase']);tpf=r['n_correct']-tpi
        if not (0<=tpi<=min(ni,p.get('increase',0)) and 0<=tpf<=min(nf,p.get('flat',0))):raise ValueError('Inconsistent aggregate confusion counts')
        fi=2*tpi/(ni+p.get('increase',0));ff=2*tpf/(nf+p.get('flat',0));rows.append({'model':r['model'],'method':r['method'],'disease':r['disease'],'n':n,'legacy_macro_f1':r['macro_f1'],'fixed_two_class_macro_f1':(fi+ff)/2,'accuracy':r['accuracy'],'predicted_revision':p.get('revision',0),'flat_tp':tpf,'increase_tp':tpi,'source':'exact reconstruction from support, prediction counts, correct counts and rounded class recall'})
    return rows

def vector(x):
    def logsign(n):return math.copysign(math.log1p(abs(n)),n)
    return [float(x['disease_id']=='C5203670')]+[logsign(v) for v in x['history']]+[logsign(v) for v in x['historical_differences']]
def score_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        for cohort in ['ALL',r['disease']]:groups[r['model'],r['condition'],r['cohort'],cohort].append(r)
    res=[]
    for (m,c,s,d),rr in groups.items():
        labels=['flat','increase'];q=multiclass([r['label'] for r in rr],[r['prediction'] for r in rr],labels);res.append({'model':m,'condition':c,'cohort':s,'disease':d,**q})
    return res

def block_compare(rows,a,b,seed=20261124,nboot=1000,nperm=2000):
    """Conditional paired country-block intervals and sign-flip accuracy tests."""
    A={r['origin_id']:r for r in rows if (r['model'],r['condition'])==a};B={r['origin_id']:r for r in rows if (r['model'],r['condition'])==b};ids=sorted(set(A)&set(B));ids=[k for k in ids if A[k]['prediction'] is not None and B[k]['prediction'] is not None]
    if not ids:return None
    y=np.array([A[k]['label'] for k in ids]);pa=np.array([A[k]['prediction'] for k in ids]);pb=np.array([B[k]['prediction'] for k in ids]);groups=defaultdict(list)
    for i,k in enumerate(ids):groups[A[k]['country_id']].append(i)
    blocks=[np.array(v) for v in groups.values()];rng=np.random.default_rng(seed)
    def f1(y,p):
        return np.mean([2*np.sum((y==l)&(p==l))/max(1,np.sum(y==l)+np.sum(p==l)) for l in ('flat','increase')])
    observed=float(f1(y,pb)-f1(y,pa));diff=(pb==y).astype(int)-(pa==y).astype(int);dacc=float(diff.mean());boots=[];accs=[]
    for _ in range(nboot):
        ii=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]);boots.append(float(f1(y[ii],pb[ii])-f1(y[ii],pa[ii])));accs.append(float(diff[ii].mean()))
    blockdiff=np.array([diff[v].sum() for v in blocks]);null=(rng.choice([-1,1],size=(nperm,len(blocks)))*blockdiff).sum(axis=1)/len(ids);p=(1+np.sum(np.abs(null)>=abs(dacc)-1e-12))/(nperm+1)
    return {'a':list(a),'b':list(b),'n_paired_complete':len(ids),'countries':len(blocks),'delta_macro_f1':observed,'delta_accuracy':dacc,'macro_f1_95pct_CI':np.quantile(boots,[.025,.975]).tolist(),'accuracy_95pct_CI':np.quantile(accs,[.025,.975]).tolist(),'p_accuracy':float(p),'scope':'conditional on fixed predictions, country-block resampling'}

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_accuracy']);prev=0
    for j,i in enumerate(order):prev=max(prev,min(1,(len(rows)-j)*rows[i]['p_accuracy']));rows[i]['holm_p_accuracy']=prev
    return rows

def refit_profile_control(train,test,profiles,out,nperm=19):
    """Refit supervised classifiers under fixed within-disease profile permutations.
    Control country identity separately; use a matched KG-eligible cohort only.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack,csr_matrix
    key=lambda x:(x['disease_id'],x['country_id'])
    tr=[x for x in train if key(x) in profiles];te=[x for x in test if key(x) in profiles]
    if not tr or not te:return {'status':'UNAVAILABLE','reason':'no matched profile cohort'}
    def concept_counts(obj):
        # Explicit CUI content only, not arbitrary profile record metadata.
        found=Counter(re.findall(r'C\d{7}',json.dumps(obj)))
        return {k:float(v) for k,v in found.items()}
    bag={k:concept_counts(v) for k,v in profiles.items()}
    if not any(bag.values()):return {'status':'UNAVAILABLE','reason':'no identifiable CUI content in parsed profile records'}
    scaler=StandardScaler();A=csr_matrix(scaler.fit_transform([vector(x) for x in tr]));B=csr_matrix(scaler.transform([vector(x) for x in te]));y=[x['label'] for x in tr];true=[x['label'] for x in te]
    dv=DictVectorizer();D=dv.fit_transform([{'country:'+x['country_id']:1.0} for x in tr]);E=dv.transform([{'country:'+x['country_id']:1.0} for x in te]);cv=DictVectorizer();cv.fit(list(bag.values()))
    def fit(a,b):
        m=LogisticRegression(max_iter=500,random_state=42);m.fit(a,y);return multiclass(true,m.predict(b).tolist(),['flat','increase'])
    def fit_profile(mapping):
        return fit(hstack([A,cv.transform([bag[mapping[key(x)]] for x in tr])]).tocsr(),hstack([B,cv.transform([bag[mapping[key(x)]] for x in te])]).tocsr())
    observed=fit_profile({k:k for k in profiles});base=fit(A,B);country=fit(hstack([A,D]).tocsr(),hstack([B,E]).tocsr());groups=defaultdict(list)
    for k in profiles:groups[k[0]].append(k)
    rng=random.Random(20261124);perm=[]
    for i in range(nperm):
        mapping={}
        for v in groups.values():
            z=list(v);rng.shuffle(z);mapping.update(dict(zip(v,z)))
        r=fit_profile(mapping);perm.append({'permutation':i+1,'macro_f1':r['macro_f1'],'accuracy':r['accuracy'],'changed_profiles':sum(k!=v for k,v in mapping.items())})
    csvwrite(out/'refitted_profile_permutations.csv',perm)
    result={'status':'COMPLETE','n_train':len(tr),'n_test':len(te),'n_permutations':nperm,'numerical_only':base,'numerical_plus_country_identity':country,'numerical_plus_actual_CUI_profile':observed,'permuted_macro_f1_mean':float(np.mean([r['macro_f1'] for r in perm])),'empirical_upper_tail_p':(1+sum(r['macro_f1']>=observed['macro_f1'] for r in perm))/(1+nperm),'scope':'independent refits of logistic regression, fixed split; static retrospective profiles, not dated evidence'}
    dump(out/'refitted_profile_control.json',result);return result

def run(root,out,client,models,preflight=False):
    out.mkdir(parents=True,exist_ok=True);legacy=load(root/LEGACY);config=legacy['config'];p=locate(root,'snapshots.csv',[SNAP]);expected=config['dataset_snapshots_sha256']
    if sha(p)!=expected:raise ValueError('Snapshot hash differs from V6: '+sha(p)+' expected '+expected+'; do not compare unmatched cohorts')
    ex,quality=build_examples(csvread(p));train=[x for x in ex if x['target_week']<config['v5_cutoff_date']];test=[x for x in ex if x['target_week']>=config['v5_cutoff_date']];index={x['origin_id']:x for x in test};selection=config['selected_episode_ids'];selected=[index[x] for x in selection];assert len(set(selection))==40
    if len(train)!=24640 or len(test)!=1565 or Counter(x['label'] for x in test)!=Counter({'flat':1155,'increase':410}):raise ValueError('Reconstructed cohort differs from V5: '+str((len(train),len(test),Counter(x['label'] for x in test))))
    # Verify exact holdout origin IDs if the old row file exists.
    oldpred=root/'outputs/realdata/temporal_trajectory_study_v5/predictions.csv';old_ids_verified=False
    if oldpred.exists():
        old=csvread(oldpred);ids={first(r,'origin_episode_id','origin_id','episode_id','origin') for r in old if r.get('split')=='holdout_all'}
        if None not in ids and ids:
            if ids!=set(index):raise ValueError('Holdout ID mismatch vs V5')
            old_ids_verified=True
    profiles,paudit=load_profiles(root,config['dataset_profiles_sha256']);covered=[x for x in test if (x['disease_id'],x['country_id']) in profiles];matched_profile_cohort=len(covered)==578 and all((x['disease_id'],x['country_id']) in profiles for x in selected)
    quality.update(n_train=len(train),n_holdout=len(test),n_selected=len(selected),holdout_labels=dict(Counter(x['label'] for x in test)),sample_labels=dict(Counter(x['label'] for x in selected)),old_holdout_ids_verified=old_ids_verified,snapshot_sha256=sha(p),profile_audit=paudit,parsed_kg_eligible=len(covered),original_profile_cohort_matched=matched_profile_cohort)
    dump(out/'cohort_audit.json',quality);dump(out/'selected_episodes.json',selected);csvwrite(out/'legacy_v6_harmonized.csv',harmonize_legacy(legacy));print('Temporal cohort:',quality,flush=True)
    if preflight:return {'status':'PREFLIGHT_OK','profile_cohort_matched':matched_profile_cohort,'quality':quality}
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    X=np.array([vector(x) for x in train]);Y=[x['label'] for x in train];Xtest=np.array([vector(x) for x in test]);rows=[];sel=set(selection)
    def add(m,c,x,p):
        for s in ['all_holdout']+(['fixed_40'] if x['origin_id'] in sel else [])+(['kg_eligible'] if matched_profile_cohort and (x['disease_id'],x['country_id']) in profiles else []):rows.append({k:x[k] for k in ('origin_id','disease','country_id','target_week','label')}|{'model':m,'condition':c,'cohort':s,'prediction':p})
    for x in test:add('persistence','no_context',x,x['previous_direction']);add('always_flat','no_context',x,'flat')
    for name,model in [('logistic_regression',make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))),('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.08,random_state=42))]:
        model.fit(X,Y)
        for x,y in zip(test,model.predict(Xtest)):add(name,'no_context',x,str(y))
    csvwrite(out/'predictions.csv',rows);dump(out/'metrics.json',score_rows(rows))
    try:control=refit_profile_control(train,test,profiles,out) if matched_profile_cohort else {'status':'UNAVAILABLE','reason':'exact KG profile cohort not reconstructed'}
    except Exception as e:control={'status':'FAILED','error':str(e)[:500]}
    dump(out/'refitted_profile_control.json',control)
    def val(x):
        if set(x)!= {'prediction'} or x['prediction'] not in DIR:raise ValueError('invalid direction schema')
    # Same new prompt for both families, same sample; never combine with old prompt results.
    permuted={};groups=defaultdict(list);rng=random.Random(20261124)
    for k in sorted(profiles):groups[k[0]].append(k)
    for keys in groups.values():
        v=list(keys);rng.shuffle(v);permuted.update(dict(zip(keys,v)))
    dump(out/'llm_profile_permutation.json',{'mapping':[{'disease':k[0],'country':k[1],'assigned_country':v[1]} for k,v in permuted.items()], 'changed_profiles':sum(stable(profiles[k])!=stable(profiles[v]) for k,v in permuted.items())})
    for m in models:
        for x in selected:
            base={'disease':x['disease'],'country':x['country'],'weekly_reported_values_oldest_to_latest':x['history'],'historical_differences':x['historical_differences'],'origin_week':x['origin_week'],'target_definition':'next reported value compared with latest observed value; increase if larger, flat if equal, revision if smaller. Use supplied historical values only; no retrospective factual recall.'}
            conditions={'temporal_only':base}
            if matched_profile_cohort:
                k=(x['disease_id'],x['country_id']);conditions['static_kg_context']=dict(base,static_profile=profiles[k]);conditions['shuffled_static_kg_context']=dict(base,static_profile=profiles[permuted[k]])
            for c,payload in conditions.items():
                ans=client.call(m,'Predict next-week reported-value direction',payload,schema_enum('prediction',DIR),val,budget=128);rows.append({k:x[k] for k in ('origin_id','disease','country_id','target_week','label')}|{'model':m,'condition':c,'cohort':'fixed_40','prediction':ans['prediction'] if ans else None})
        csvwrite(out/'predictions.csv',rows);dump(out/'metrics.json',score_rows(rows))
    sample=[r for r in rows if r['cohort']=='fixed_40'];comparisons=[]
    for a,b in [('qwen3:4b','gemma3:4b'),('qwen3:32b','gemma3:27b')]:
        if a in models and b in models:
            for c in ['temporal_only']+(['static_kg_context'] if matched_profile_cohort else []):
                v=block_compare(sample,(a,c),(b,c))
                if v:comparisons.append(v)
    if matched_profile_cohort:
        for m in models:
            v=block_compare(sample,(m,'temporal_only'),(m,'static_kg_context'))
            if v:comparisons.append(v)
            v=block_compare(sample,(m,'shuffled_static_kg_context'),(m,'static_kg_context'))
            if v:comparisons.append(v)
    dump(out/'paired_comparisons.json',holm(comparisons));return {'status':'COMPLETE' if matched_profile_cohort and control['status']=='COMPLETE' and all(r['prediction'] is not None for r in rows) else 'PARTIAL_CONTEXT_OR_TECHNICAL','quality':quality,'new_baselines':['persistence','always_flat','logistic_regression','hist_gradient_boosting'],'models':models,'refitted_profile_control':control,'llm_scope':'identical new prompt on exact original fixed 40; previous V6 metrics separate'}
