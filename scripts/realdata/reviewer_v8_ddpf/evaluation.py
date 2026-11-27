"""Independent reference handling. Never imported by the inference worker."""
from .data import *
import random,itertools,math
import numpy as np
import networkx as nx
from sklearn.metrics import cohen_kappa_score

JUDGED=['same_event','continuation','related_only','unrelated']
ALL=JUDGED+['insufficient_evidence']

def sample_review(out,gold,n,seed):
    events=read_jsonl(out/'events.jsonl');ei={e['observation_id']:e for e in events};ids=sorted(ei,key=lambda i:(ei[i]['available_at'],i));r=random.Random(seed)
    # Draw independently of retrieval scores. Include non-retrieved and outside-window pairs.
    strata=defaultdict(list)
    for a,b in itertools.combinations(ids,2):
        x,y=ei[a],ei[b];
        if chronology_order(y,x):a,b=b,a;x,y=y,x
        stratum='|'.join([str(x['country_id']==y['country_id']),str(x['collection_start'][:7]==y['collection_start'][:7]),str(x['source_corpus']==y['source_corpus']),':'.join(sorted([x['modality'],y['modality']]))]);strata[stratum].append((a,b))
    picked=[];pools={k:r.sample(v,len(v)) for k,v in sorted(strata.items())}
    while len(picked)<n and any(pools.values()):
        for k,v in pools.items():
            if v and len(picked)<n:picked.append((k,*v.pop()))
    counts=Counter(k for k,a,b in picked);rows=[{'a':a,'b':b,'stratum':k,'stratum_population':len(strata[k]),'stratum_sample':counts[k],'label':'','evidence_ids':'','justification':'','certainty':'','block_id':''} for k,a,b in picked]
    for name in ('annotator_A.csv','annotator_B.csv','adjudication.csv'):
        target=gold/name
        if target.exists():raise FileExistsError(f'{target} already exists; choose separate gold folder')
    for name in ('annotator_A.csv','annotator_B.csv','adjudication.csv'):csvwrite(gold/name,rows)
    csvwrite(gold/'mapping_template.csv',[{'observation_id':e['observation_id'],'reference_event_id':'','match_kind':'','evidence_ids':'','justification':''} for e in events]);csvwrite(gold/'reference_events_template.csv',[],['reference_event_id','disease','country_id','occurrence_start','occurrence_end','event_type','description','source_id','evidence_anchor','block_id']);csvwrite(gold/'reference_edges_template.csv',[],['source_event','target_event','evidence_ids','block_id']);dump(gold/'sampling_manifest.json',{'seed':seed,'n':len(rows),'universe_size':sum(map(len,strata.values())),'stratum_populations':{k:len(v) for k,v in strata.items()},'gold_independent_of_predictions':True,'note':'Add all independently identified positives in adjudication, including not sampled. This sheet is a probability sample within strata, not an exhaustive edge gold.'})
    if (out/'edges.jsonl').is_file() and (out/'nodes.jsonl').is_file():
        fs={f['fact_id']:f for f in read_jsonl(out/'edges.jsonl')};ns={n['node_id']:n for n in read_jsonl(out/'nodes.jsonl')}
        jsonl(gold/'observation_cards.jsonl',[{'observation_id':e['observation_id'],'country':e['country_label'],'collection_start':e['collection_start'],'collection_end':e['collection_end'],'source_corpus':e['source_corpus'],'facts':[dict(fs[i],subject_label=ns[fs[i]['subject']]['label'],object_label=ns[fs[i]['object']]['label']) for i in e['fact_ids']]} for e in events])
    return {'pairs':len(rows),'events_to_map':len(events)}

def read_labels(p):
    rows=csvread(p);d={}
    for r in rows:
        k=(r['a'],r['b']);l=r.get('label','').strip()
        if not l:continue
        if l not in ALL:raise ValueError('invalid label '+l)
        if k in d:raise ValueError('duplicate label '+str(k))
        if not r.get('evidence_ids') or not r.get('justification'):raise ValueError('reviewed label missing evidence/justification '+str(k))
        d[k]=r
    return d

def eval_pairs(ref,rows):
    pred={(r['a'],r['b']):r.get('validated') for r in rows};keys=[k for k,r in ref.items() if r['label'] in JUDGED];y=[ref[k]['label'] for k in keys];p=[(pred[k]['relation'] if pred.get(k) else None) for k in keys]
    z=multiclass(y,p,JUDGED);z.update(judged=len(keys),gold_insufficient=sum(r['label']=='insufficient_evidence' for r in ref.values()),missing=sum(x is None for x in p),abstentions=sum(x=='insufficient_evidence' for x in p),decision_coverage=sum(x in JUDGED for x in p)/len(y) if y else None,continuation=binary([int(x=='continuation') for x in y],[int(x=='continuation') if x in JUDGED else None for x in p]))
    for l in JUDGED:
        tp=sum(a==l and b==l for a,b in zip(y,p));den=sum(b==l for b in p);z['per_class'][l]['precision']=tp/den if den else None
    z['confusion']={l:dict(Counter(str(b) for a,b in zip(y,p) if a==l)) for l in JUDGED};return z

def edge_metrics(ref,edges):
    judged={k for k,r in ref.items() if r['label'] in JUDGED};pos={k for k,r in ref.items() if r['label']=='continuation'};pred={(e['a'],e['b']) for e in edges};tp=len(pos&pred);fp=len((judged-pos)&pred);fn=len(pos-pred);unknown=len(pred-judged);den=2*tp+fp+fn
    return {'judged_tp':tp,'judged_fp':fp,'judged_fn':fn,'predicted_unjudged':unknown,'judged_edge_f1':2*tp/den if den else None,'judged_precision':tp/(tp+fp) if tp+fp else None,'judged_recall':tp/(tp+fn) if tp+fn else None,'prediction_judgment_coverage':(tp+fp)/len(pred) if pred else None,'precision_lower_if_unknown_false':tp/len(pred) if pred else None,'precision_upper_if_unknown_true':(tp+unknown)/len(pred) if pred else None,'scope':'judged pairs, not complete forest recall'}

def graph_gold(gold,graphs,ei):
    # Optional exhaustive reference graph: exact adjudicated mapping only. Partial/many-to-many stays explicit.
    files=['reference_events.csv','reference_edges.csv','mapping.csv','complete_universe.json']
    if not all((gold/f).is_file() for f in files):return {'status':'not_available'}
    u=load(gold/'complete_universe.json')
    if not u.get('exhaustive_adjudicated') or not u.get('universe_id') or not u.get('observation_ids'):return {'status':'not_exhaustive'}
    obs=set(u['observation_ids']);refs={r['reference_event_id'] for r in csvread(gold/'reference_events.csv')};mapping=defaultdict(set)
    if not obs<=ei.keys():raise ValueError('unknown universe observations')
    for r in csvread(gold/'mapping.csv'):
        if r['match_kind']=='exact' and r['observation_id'] in obs:
            if r['reference_event_id'] not in refs:raise ValueError('unknown gold event')
            mapping[r['observation_id']].add(r['reference_event_id'])
    one={i:next(iter(v)) for i,v in mapping.items() if len(v)==1};matched=set(one.values());event_tp=len(matched);unmapped=len(obs)-len(one)
    gold_e={(r['source_event'],r['target_event']) for r in csvread(gold/'reference_edges.csv')};
    if any(a not in refs or b not in refs for a,b in gold_e):raise ValueError('unknown reference edge endpoint')
    results={}
    for kind in ('forest','dag'):
        pred=set();unresolved=0
        for r in graphs[kind]:
            a,b=r['a'],r['b']
            if a not in obs or b not in obs:continue
            if a not in one or b not in one:unresolved+=1;continue
            if one[a]!=one[b]:pred.add((one[a],one[b]))
        tp=len(pred&gold_e);fp=len(pred-gold_e)+unresolved;fn=len(gold_e-pred);den=2*tp+fp+fn
        results[kind]={'edge_f1':2*tp/den if den else None,'tp':tp,'fp':fp,'fn':fn,'unmapped_prediction_edges_penalized':unresolved}
        gg=nx.DiGraph();gg.add_nodes_from(refs);gg.add_edges_from(gold_e);pg=nx.DiGraph();pg.add_nodes_from(refs);pg.add_edges_from(pred)
        def anc(g):return {(a,b) for a in g for b in nx.descendants(g,a) if a!=b}
        def costory(g):return {tuple(sorted((a,b))) for comp in nx.weakly_connected_components(g) for a,b in itertools.combinations(comp,2)}
        def f1(a,b):return 2*len(a&b)/(len(a)+len(b)) if a or b else None
        results[kind].update(ancestor_f1=f1(anc(gg),anc(pg)),co_story_f1=f1(costory(gg),costory(pg)),components=nx.number_weakly_connected_components(pg),gold_components=nx.number_weakly_connected_components(gg))
    return {'status':'exhaustive_declared','universe_id':u['universe_id'],'mapping_scope':'exact unambiguous matches; partial mappings remain unresolved','reference_events':len(refs),'observations':len(obs),'mapped_reference_events':event_tp,'unmapped_observations':unmapped,'event_f1':2*event_tp/(len(refs)+len(obs)) if refs or obs else None,'structures':results}

def evaluate(out,gold):
    if not (out/'run_manifest.json').exists():raise ValueError('freeze predictions first')
    manifest=load(out/'run_manifest.json');
    if manifest.get('status')=='RUNNING':raise ValueError('Inference not complete; freeze completed outputs first')
    variants=manifest['variants'];ref=read_labels(gold/'adjudication.csv');a=read_labels(gold/'annotator_A.csv');b=read_labels(gold/'annotator_B.csv');common=sorted(set(a)&set(b));ei={e['observation_id']:e for e in read_jsonl(out/'events.jsonl')}
    if not ref:raise ValueError('No adjudicated labels; do not fabricate gold')
    if any(x not in ei or y not in ei or x==y for x,y in ref):raise ValueError('gold contains unknown IDs')
    agreement={'n':len(common),'raw_agreement':sum(a[k]['label']==b[k]['label'] for k in common)/len(common) if common else None,'kappa':float(cohen_kappa_score([a[k]['label'] for k in common],[b[k]['label'] for k in common],labels=ALL)) if common and len({a[k]['label'] for k in common}|{b[k]['label'] for k in common})>1 else None,'coverage_of_adjudication':len(set(ref)&set(common))/len(ref)}
    allpairs=read_jsonl(out/'eligible_pairs.jsonl');rank={(p['a'],p['b']):p['rank'] for p in allpairs};positive={k for k,r in ref.items() if r['label']=='continuation'};eligible=positive&rank.keys()
    retrieval={'positive_reference_pairs':len(positive),'eligible_positive_pairs':len(eligible),'outside_candidate_universe':len(positive-eligible),'recall_at_k':{str(k):sum(rank[x]<=k for x in eligible)/len(eligible) if eligible else None for k in (5,10,20,30)},'mrr_over_eligible_positive_pairs':sum(1/rank[x] for x in eligible)/len(eligible) if eligible else None,'scope':'judged reference positives; ranks within ordered vs overlapping retrieval strata; no unjudged negatives assumed'}
    for ch in ('cosine','jaccard','geo','time','cluster','tfidf','hybrid','no_embedding','no_cluster','no_time','no_geography'):
        cr={(p['a'],p['b']):p['rank_'+ch] for p in allpairs}
        retrieval.setdefault('channel_recall_at_k',{})[ch]={str(k):sum(cr[x]<=k for x in eligible)/len(eligible) if eligible else None for k in (5,10,20,30)}
    table=[];result={};preds={}
    for name,v in variants.items():
        path=out/'runs'/name/'decisions.jsonl'
        if sha(path)!=v['decisions_sha256']:raise ValueError('prediction hash mismatch '+name)
        rows=read_jsonl(path);g=load(path.parent/'graphs.json');m=eval_pairs(ref,rows);row_keys={(r['a'],r['b']) for r in rows};controlled=eval_pairs({k:r for k,r in ref.items() if k in row_keys},rows);result[name]={'pair':m,'controlled_retrieved_pairs':controlled,'forest':edge_metrics(ref,g['forest']),'dag':edge_metrics(ref,g['dag']),'exhaustive':graph_gold(gold,g,ei)};table.append({'variant':name,'macro_f1':m['macro_f1'],'decision_coverage':m['decision_coverage'],'controlled_macro_f1':controlled['macro_f1'],'controlled_n':controlled['n'],'judged_forest_f1':result[name]['forest']['judged_edge_f1'],'judged_dag_f1':result[name]['dag']['judged_edge_f1'],'missing':m['missing'],'abstentions':m['abstentions']});preds[name]={(r['a'],r['b']):r.get('validated') for r in rows}
    # Conditional paired block bootstrap for C3 vs C2, only if independently supplied block IDs.
    groups=defaultdict(list)
    for k,r in ref.items():
        if r['label'] in JUDGED:groups[r.get('block_id','')].append(k)
    contrasts=[];rng=np.random.default_rng(42)
    if '' not in groups and len(groups)>=2:
        keys=list(groups);draws=rng.integers(0,len(keys),(int(manifest['config']['bootstrap']),len(keys)))
        def f(name,kk):
            pp=preds[name];return multiclass([ref[k]['label'] for k in kk],[pp[k]['relation'] if pp.get(k) else None for k in kk],JUDGED)['macro_f1']
        for name in preds:
            if not name.endswith('__C3'):continue
            base=name[:-2]+'C2'
            if base not in preds:continue
            obs=[k for v in groups.values() for k in v];delta=f(name,obs)-f(base,obs);values=[]
            for draw in draws:
                kk=[k for i in draw for k in groups[keys[i]]];values.append(f(name,kk)-f(base,kk))
            diffs=[]
            for k in keys:
                diffs.append(sum(int((preds[name].get(i) or {}).get('relation')==ref[i]['label'])-int((preds[base].get(i) or {}).get('relation')==ref[i]['label']) for i in groups[k]))
            diffs=np.array(diffs);obs_acc=abs(diffs.sum());perms=rng.choice([-1,1],(max(2000,len(draws)),len(keys)))@diffs;pval=float((1+np.sum(np.abs(perms)>=obs_acc))/(len(perms)+1))
            contrasts.append({'contrast':name+' minus '+base,'delta_macro_f1':delta,'ci95':np.quantile(values,[.025,.975]).tolist(),'blocks':len(keys),'accuracy_block_signflip_p':pval,'note':'conditional on fixed predictions; F1 bootstrap interval and accuracy sign-flip p-value measure distinct quantities'})
    ordered=sorted(range(len(contrasts)),key=lambda i:contrasts[i]['accuracy_block_signflip_p']);prev=0.
    for j,i in enumerate(ordered):
        prev=max(prev,min(1.,(len(ordered)-j)*contrasts[i]['accuracy_block_signflip_p']));contrasts[i]['accuracy_holm_p']=prev
    dest=out/'evaluation';dest.mkdir(exist_ok=True);dump(dest/'results.json',{'agreement':agreement,'retrieval':retrieval,'variants':result,'block_bootstrap':contrasts,'status':'ADJUDICATED_PAIR_SCORES','note':'No global edge recall without exhaustive reference universe.'});csvwrite(dest/'metrics.csv',table);return {'agreement':agreement,'retrieval':retrieval,'variants':len(table),'output':str(dest)}

def audit_independence(out,reference_manifest,gold):
    """Offline exact/near-duplicate check over available source text, separate from inference.
    Manifest CSV requires source_id, URL, publication_date, text_path, sha256.
    Missing raw texts cannot be certified independent by a marker scan.
    """
    refs=[];base=reference_manifest.parent
    def shingles(t):
        w=re.findall(r'\w+',norm(t));return {tuple(w[i:i+5]) for i in range(max(0,len(w)-4))}
    for r in csvread(reference_manifest):
        p=base/r['text_path'];h=sha(p)
        if h!=r['sha256']:raise ValueError('Reference hash mismatch '+r['source_id'])
        txt=p.read_text(encoding='utf8');refs.append((r,stable(norm(txt)),shingles(txt)))
    events=read_jsonl(out/'events.jsonl');hits=[];missing=[]
    for e in events:
        t=e.get('source_text')
        if not t or not e['source_ids']:missing.append(e['observation_id']);continue
        h=stable(norm(t));sh=shingles(t)
        for r,rh,rsh in refs:
            score=len(sh&rsh)/max(1,len(sh|rsh));contain=len(sh&rsh)/max(1,min(len(sh),len(rsh)))
            if h==rh or r['source_id'] in e['source_ids'] or r['url'] in e['source_ids'] or score>=.5 or (len(sh&rsh)>=10 and contain>=.8):hits.append({'observation_id':e['observation_id'],'reference_source':r['source_id'],'jaccard':score,'containment':contain,'exact':h==rh})
    x={'status':'REVIEW_REQUIRED' if missing or hits or not refs else 'NO_MATCHES_IN_AVAILABLE_TEXTS_NOT_CERTIFICATION','reference_manifest_sha256':sha(reference_manifest),'reference_documents':len(refs),'observations':len(events),'missing_source_text_or_ids':missing,'possible_overlap':hits,'note':'Audit cannot exclude copied facts/paraphrases or contamination in legacy aggregate KG. Rebuild affected representations; obtain human provenance review before confirmatory approval.'};dump(gold/'independence_audit.json',x);return x
