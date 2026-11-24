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

def disease_id(value):
    """Map explicit disease metadata, never infer disease from observation concepts."""
    text=str(value or '').casefold()
    found=set(re.findall(r'C\d{7}',str(value or '')))&{'C5203670','C0276180'}
    if re.search(r'covid|2019.novel.coronavirus|sars.cov.2',text):found.add('C5203670')
    if re.search(r'monkeypox|mpox',text):found.add('C0276180')
    return next(iter(found)) if len(found)==1 else None

def country_id(value):
    ids=set(re.findall(r'wkg:\d+',str(value or ''),re.I))
    return next(iter(ids)).lower() if len(ids)==1 else None

# These are explicit semantic-content fields, not arbitrary record metadata.
PROFILE_FIELDS={'concepts','concept_ids','concept_names','concept_labels','concept_counts','concept_weights','concept_profile','profile','terms','cuis','features','top_concepts','biomedical_concepts','concept_intensities','concept_freq','concept_frequencies','vector','concept_vector','source_counts','concept_set','concept_list','concepts_by_source','concepts_by_id','concepts_by_cui','selected_concepts','top_terms','tokens','cui_counts','cui_set','cui_list','entities','entity_ids','entity_concepts','observed_concepts','observations','concept_records','concept_ids_masked','concept_names_masked','concept_tokens','concepts_with_intensity','concepts_with_weights','concepts_with_names','concepts_with_intensities','concepts_and_intensities','concept_data','semantic_concepts','semantic_profile','profile_concepts','concept_features','cui_features','cui_names','cui_weights','cui_terms','observed_entities','concept_entries','concept_items','concept_values','concept_ids_and_names','concept_ids_and_intensities'}

def profile_records(obj):
    """Read flat/nested observation records or legacy disease-country maps.

    Repeated observations are accumulated rather than overwritten. Identity is
    established only by explicit metadata, not by scanning biomedical concepts.
    """
    found=defaultdict(list)
    def walk(x,d=None,c=None):
        if isinstance(x,dict):
            dv=[x[k] for k in ('disease_id','disease_cui','disease','disease_name','phenomenon_id','phenomenon','source','source_id','source_name','source_type','source_label','source_key','source_category','source_dataset','dataset','dataset_name','label','observation_id','episode_id','_key') if k in x]
            cv=[x[k] for k in ('country_id','country','location_id','location','country_node_id','country_wkg_id','country_wkg','country_uri','location_uri','geographic_id','geo_id','system_id','system','System_ID','System_Id','System_Name','location_cui','observation_id','episode_id','_key','id') if k in x]
            ds={z for v in dv if (z:=disease_id(v))};cs={z for v in cv if (z:=country_id(v))}
            if len(ds)>1 or len(cs)>1:return
            d=next(iter(ds),d);c=next(iter(cs),c)
            vals={k:v for k,v in x.items() if str(k).casefold() in PROFILE_FIELDS and v not in (None,'',[],{})}
            # Legacy profiles may be direct CUI -> numeric weight maps.
            direct={str(k):v for k,v in x.items() if re.fullmatch(r'C\d{7}',str(k)) and isinstance(v,(int,float,str))}
            if direct:vals['concept_weights']=direct
            if d and c and vals:
                found[d,c].append(vals);return
            for k,v in x.items():
                if isinstance(v,(list,dict,set,tuple)):
                    walk(v,disease_id(k) or d,country_id(k) or c)
        elif isinstance(x,(list,tuple,set)):
            # A profile loader may return bare concept lists/sets for tuple keys.
            if d and c and x and all(isinstance(v,str) for v in x):
                found[d,c].append({'concepts':sorted(x) if isinstance(x,set) else list(x)});return
            for v in x:walk(v,d,c)
    walk(obj)
    return {k: (v[0] if len(v)==1 else {'observation_profiles':v}) for k,v in found.items()}

def legacy_profile_adapter(root,path):
    """Prefer the actual V5 aggregation when the original function is available.
    Discover only profile-loading/building functions accepting a single path/root.
    Import the user's existing module without executing its CLI main.
    """
    import ast,importlib.util,inspect,sys
    attempts=[];results=[]
    for filename in ('temporal_trajectory_study_v5.py','llm_size_study_v6.py'):
        script=root/'scripts'/'realdata'/filename
        if not script.is_file():continue
        try:
            tree=ast.parse(script.read_text(encoding='utf-8-sig'))
            names=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and ('profil' in n.name.lower() or ('observation' in n.name.lower() and 'profil' in (ast.get_source_segment(script.read_text(encoding='utf-8-sig'),n) or '').lower())) and any(w in n.name.lower() for w in ('load','build','read','collect','recover','make','aggregate'))]
            if not names:continue
            name='_reviewer_v8_legacy_'+script.stem
            spec=importlib.util.spec_from_file_location(name,script);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
            for name in names:
                f=getattr(mod,name);sig=inspect.signature(f);params=list(sig.parameters.values())
                required=[p for p in params if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]
                if len(required)!=1:continue
                param=required[0].name.lower()
                if not any(w in param for w in ('root','path','file','profile','observation','input','rows','records','data')):continue
                if 'root' in param:arg=root
                elif param in ('rows','records','observations','data'):arg=[json.loads(l) for l in path.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
                else:arg=path
                try:
                    result=f(arg);objects=[result]+list(result) if isinstance(result,tuple) else [result]
                    for obj in objects:
                        parsed=profile_records(obj)
                        if parsed:results.append((parsed,{'adapter':'legacy_function','function':filename+':'+name,'script_sha256':sha(script)}))
                    attempts.append({'function':filename+':'+name,'status':'called','return_type':type(result).__name__})
                except Exception as e:attempts.append({'function':filename+':'+name,'status':'failed','error':str(e)[:200]})
        except Exception as e:attempts.append({'file':filename,'status':'import_failed','error':str(e)[:200]})
    return results,attempts

def load_profiles(root,expected,expected_keys=None,holdout_keys=None):
    path=profile_candidates(root,expected)
    if not path:return {},{'status':'exact_legacy_profile_file_not_found','expected_sha256':expected}
    if path.suffix=='.csv':obj=csvread(path)
    elif path.suffix=='.jsonl':obj=[json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    elif path.suffix=='.tsv':
        import csv
        with path.open() as f:obj=list(csv.DictReader(f,delimiter='\t'))
    else:obj=load(path)
    options,attempts=legacy_profile_adapter(root,path)
    options.append((profile_records(obj),{'adapter':'explicit_observation_schema'}))
    # Do not select an adapter merely because it returns some data. Compare to
    # the original eligible disease-country keys when the old predictions exist.
    def rank(item):
        p,info=item
        relevant=set(p)&holdout_keys if holdout_keys is not None else set(p)
        return (-len(relevant^expected_keys) if expected_keys is not None else len(relevant), bool(any(re.findall(r'C\d{7}',json.dumps(v,default=str)) for v in p.values())),info['adapter']=='legacy_function')
    profiles,adapter=max(options,key=rank)
    def safe(v,depth=0):
        if depth>8:return None
        if isinstance(v,dict):return {str(k):safe(x,depth+1) for k,x in sorted(v.items(),key=lambda t:str(t[0])) if not re.search(r'(target|label|future|report|cases|date|prediction|outcome)',str(k),re.I) or k=='concept_labels'}
        if isinstance(v,(list,tuple,set)):return [safe(x,depth+1) for x in (sorted(v,key=str) if isinstance(v,set) else v)]
        return v if isinstance(v,(int,float,bool)) else str(v)[:500]
    profiles={k:safe(v) for k,v in profiles.items()}
    # Expose actual record structure for diagnostics without printing full data.
    def shape(x,depth=0):
        if depth>2:return type(x).__name__
        if isinstance(x,dict):return {str(k):shape(v,depth+1) for k,v in list(x.items())[:30]}
        if isinstance(x,(list,tuple)):return {'type':'list','length':len(x),'first':shape(x[0],depth+1) if x else None}
        return str(x)[:120] if isinstance(x,(str,int,float)) else type(x).__name__
    return profiles,{'status':'parsed' if profiles else 'exact_file_found_schema_not_recognized','file':str(path.relative_to(root)),'sha256':sha(path),'n_profiles':len(profiles),**adapter,'legacy_adapter_attempts':attempts,'candidate_adapters':[dict(info,n_profiles=len(p)) for p,info in options],'first_record_schema':shape(obj[0] if isinstance(obj,list) and obj else obj)}

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
    old_eligible_ids=None
    if oldpred.exists():
        eligible={first(r,'origin_episode_id','origin_id','episode_id','origin') for r in old if r.get('split')=='holdout_all' and str(r.get('has_static_kg_context','')).lower() in ('1','1.0','true')}
        if len(eligible)==578 and eligible<=set(index):old_eligible_ids=eligible
    expected_keys={(index[i]['disease_id'],index[i]['country_id']) for i in old_eligible_ids} if old_eligible_ids is not None else None
    profiles,paudit=load_profiles(root,config['dataset_profiles_sha256'],expected_keys,{(x['disease_id'],x['country_id']) for x in test});covered=[x for x in test if (x['disease_id'],x['country_id']) in profiles];actual_ids={x['origin_id'] for x in covered}
    matched_profile_cohort=len(covered)==578 and all((x['disease_id'],x['country_id']) in profiles for x in selected) and (old_eligible_ids is None or actual_ids==old_eligible_ids)
    paudit['legacy_eligibility_ids_verified']=old_eligible_ids is not None and actual_ids==old_eligible_ids
    if old_eligible_ids is not None:paudit.update(missing_eligible_ids=sorted(old_eligible_ids-actual_ids)[:10],extra_eligible_ids=sorted(actual_ids-old_eligible_ids)[:10])
    quality.update(n_train=len(train),n_holdout=len(test),n_selected=len(selected),holdout_labels=dict(Counter(x['label'] for x in test)),sample_labels=dict(Counter(x['label'] for x in selected)),old_holdout_ids_verified=old_ids_verified,snapshot_sha256=sha(p),profile_audit=paudit,parsed_kg_eligible=len(covered),original_profile_cohort_matched=matched_profile_cohort)
    dump(out/'cohort_audit.json',quality);dump(out/'selected_episodes.json',selected);csvwrite(out/'legacy_v6_harmonized.csv',harmonize_legacy(legacy));print('Temporal cohort:',quality,flush=True)
    if preflight:return {'status':'PREFLIGHT_OK' if matched_profile_cohort else 'PROFILE_COHORT_BLOCKED','profile_cohort_matched':matched_profile_cohort,'quality':quality}
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
    def compact(v,depth=0):
        if depth>8:return None
        if isinstance(v,dict):return {k:compact(x,depth+1) for k,x in list(v.items())[:80]}
        if isinstance(v,list):return [compact(x,depth+1) for x in v[:80]]
        return v
    for m in models:
        for x in selected:
            base={'disease':x['disease'],'country':x['country'],'weekly_reported_values_oldest_to_latest':x['history'],'historical_differences':x['historical_differences'],'origin_week':x['origin_week'],'target_definition':'next reported value compared with latest observed value; increase if larger, flat if equal, revision if smaller. Use supplied historical values only; no retrospective factual recall.'}
            conditions={'temporal_only':base}
            if matched_profile_cohort:
                k=(x['disease_id'],x['country_id']);conditions['static_kg_context']=dict(base,static_profile=compact(profiles[k]));conditions['shuffled_static_kg_context']=dict(base,static_profile=compact(profiles[permuted[k]]))
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
