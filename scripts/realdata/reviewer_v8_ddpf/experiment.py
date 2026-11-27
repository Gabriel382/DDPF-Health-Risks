"""Frozen-prefix candidate retrieval, LLM ablations, and graph projections.
This module deliberately does not import the benchmark/evaluation module.
"""
from .data import *
import math,random,itertools,platform
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

LABELS=['same_event','continuation','related_only','unrelated','insufficient_evidence']
SCHEMA={'type':'object','properties':{'relation':{'type':'string','enum':LABELS},'subtype':{'type':'string','enum':['within_location','cross_location','other','none']},'score':{'type':'number','minimum':0,'maximum':1},'evidence_ids':{'type':'array','items':{'type':'string'}},'rationale':{'type':'string'},'abstain_reason':{'type':'string'}},'required':['relation','subtype','score','evidence_ids','rationale','abstain_reason'],'additionalProperties':False}
TASK=('Classify the relationship between the two DDPF observations. same_event means the same occurrence, not merely same country or disease. continuation means evidence supports evolution of a situation, not mere similarity or chronology. Collection dates are not occurrence dates. related_only means a supported association without continuity. unrelated requires support for distinct unrelated occurrences. insufficient_evidence is not a negative label. Cite supplied fact IDs from both observations for positive links; never use external knowledge or assume transmission. Scores are confidence rankings, not calibrated probabilities.')

def check_answer(a):
    if set(a)!=set(SCHEMA['required']) or a['relation'] not in LABELS or a['subtype'] not in SCHEMA['properties']['subtype']['enum']:raise ValueError('bad schema')
    if isinstance(a['score'],bool) or not isinstance(a['score'],(int,float)) or not math.isfinite(a['score']) or not 0<=a['score']<=1:raise ValueError('bad score')
    if not isinstance(a['evidence_ids'],list) or not all(isinstance(x,str) for x in a['evidence_ids']):raise ValueError('bad evidence IDs')
    if not all(isinstance(a[k],str) for k in ('rationale','abstain_reason')):raise ValueError('bad text')

def load_export(out):
    rows=[read_jsonl(out/(n+'.jsonl')) for n in ('events','edges','nodes')];return validate(*rows)

def fact_text(f,nodes):
    return ' '.join([nodes[f['subject']]['label'],f['predicate'],nodes[f['object']]['label']])

def concepts(e,fi):return {fi[x]['subject'] for x in e['fact_ids'] if fi[x]['kind'] in {'observed_aggregate','observed','extracted'}}

def representations(ei,fi,ni,cfg,client=None,external=None):
    ids=sorted(ei);texts=[' '.join(sorted({fact_text(fi[x],ni) for x in ei[i]['fact_ids']})) for i in ids]
    # No disease-label, corpus disease suffix or target is ever used as a feature.
    if external:
        records=read_jsonl(external);mp={r['observation_id']:r for r in records}
        if len(mp)!=len(records) or set(mp)!=set(ids):raise ValueError('embedding IDs must exactly match DDPF observations')
        mat=np.array([mp[i]['vector'] for i in ids],dtype=float);method='external_DDPF_embeddings';meta={'source_sha256':sha(external),'availability':'requires frozen encoder/provenance attestation for replay'}
    elif cfg['embedding']=='ollama':
        if client is None:raise ValueError('Ollama embeddings require client')
        mat=np.array(client.embed(cfg['embedding_model'],texts));method=cfg['embedding_model'];meta={'model':client.meta[cfg['embedding_model']]}
    else:
        mat=HashingVectorizer(n_features=cfg['embedding_dimensions'],alternate_sign=False,token_pattern=r'(?u)\b\w+\b').transform(texts).toarray();method='regenerated_stateless_hashing_bow';meta={'not_original_embedding':True}
    if mat.ndim!=2 or len(mat)!=len(ids) or not np.isfinite(mat).all():raise ValueError('invalid embedding matrix')
    mat=normalize(mat);return ids,mat,texts,dict(meta,method=method,dimension=int(mat.shape[1]))

def ordered(a,b):
    return bool(a.get('occurrence_end') and b.get('occurrence_start') and a['occurrence_end']<b['occurrence_start'])

def pair_id(a,b):return 'P'+stable([a,b])[:24]

def candidates(ei,fi,ids,mat,texts,cfg):
    # Prefix-only clustering; ties in availability use the same batch, not future data.
    pos={x:i for i,x in enumerate(ids)};ordered_ids=sorted(ids,key=lambda x:(ei[x]['available_at'],x));co={x:concepts(ei[x],fi) for x in ids};out=[];allpairs=[];cluster_rows=[]
    for t,batch in itertools.groupby(ordered_ids,key=lambda x:ei[x]['available_at']):
        batch=list(batch);prefix=[x for x in ordered_ids if ei[x]['available_at']<=t];v=mat[[pos[x] for x in prefix]];k=min(cfg['clusters'],len(prefix),len(np.unique(v,axis=0)))
        labels=KMeans(n_clusters=k,random_state=cfg['seed'],n_init=5).fit_predict(v) if k>1 else np.zeros(len(prefix),dtype=int);cl=dict(zip(prefix,map(int,labels)))
        for b in batch:
            cluster_rows.append({'observation_id':b,'as_of':t,'cluster_id':cl[b],'prefix_size':len(prefix)})
            # TF-IDF fit only on available observations.
            try:
                tf=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').fit_transform([texts[pos[x]] for x in prefix]);lex=dict(zip(prefix,(tf@tf[prefix.index(b)].T).toarray().ravel()))
            except ValueError:lex={x:0. for x in prefix}
            choices=[]
            for a in prefix:
                if (ei[a]['available_at'],a)>=(t,b):continue
                gap=(date.fromisoformat(t)-date.fromisoformat(ei[a]['available_at'])).days
                if gap>cfg['window_days']:continue
                j=len(co[a]&co[b])/max(1,len(co[a]|co[b]));cos=float(np.clip(mat[pos[a]]@mat[pos[b]],-1,1));geo=float(ei[a]['country_id']==ei[b]['country_id']);time_score=math.exp(-gap/180);cluster=float(cl[a]==cl[b]);lexical=float(lex[a]);hybrid=(max(0,cos)+j+geo+time_score+cluster)/5
                choices.append({'pair_id':pair_id(a,b),'a':a,'b':b,'arrival_id':b,'as_of':t,'gap_days':gap,'cosine':cos,'jaccard':j,'geo':geo,'time':time_score,'cluster':cluster,'tfidf':lexical,'hybrid':hybrid,'known_occurrence_order':ordered(ei[a],ei[b])})
                if ordered(ei[b],ei[a]):choices[-1].update(a=b,b=a,pair_id=pair_id(b,a),known_occurrence_order=True)
            for p in choices:
                for name,drop in [('no_embedding','cosine'),('no_cluster','cluster'),('no_time','time'),('no_geography','geo')]:
                    p[name]=sum(max(0,p[k]) for k in ('cosine','jaccard','geo','time','cluster') if k!=drop)/4
            channels=['cosine','jaccard','geo','time','cluster','tfidf','hybrid','no_embedding','no_cluster','no_time','no_geography'];ranked={ch:sorted(choices,key=lambda p:(-p[ch],p['pair_id'])) for ch in channels}
            for ch,rs in ranked.items():
                for n,p in enumerate(rs,1):p['rank_'+ch]=n
            union={p['pair_id']:p for ch,rs in ranked.items() if not ch.startswith('no_') for p in rs[:cfg['top_k']]}
            # Hybrid final ranking over union; retain per-channel ranks for retrieval evaluation.
            mode=cfg.get('retrieval_mode','hybrid')
            selected=(sorted(union.values(),key=lambda p:(-p['hybrid'],p['pair_id'])) if mode=='hybrid' else ranked[mode])[:cfg['top_k']]
            for n,p in enumerate(sorted(choices,key=lambda p:(-p['hybrid'],p['pair_id'])),1):p['rank']=n
            allpairs.extend(choices);out.extend(selected)
    return out,allpairs,cluster_rows

class StaticContext:
    """Explicit static allowlist, limited RDFS inference; no isReported, no WHO.
    Static snapshot independence/as-of must be established separately.
    """
    def __init__(self,root,cfg,neo=False,ontology=()):
        import rdflib
        from rdflib.namespace import RDFS,OWL,RDF
        self.rdf=rdflib;self.RDFS=RDFS;self.OWL=OWL;self.RDF=RDF;self.g=rdflib.Graph();self.memo={};self.driver=None;self.cfg=cfg;self.audit={'ontology':[],'errors':[],'neo4j':False}
        for p in ontology:
            p=Path(p);self.g.parse(p,format='xml');self.audit['ontology'].append({'file':str(p),'sha256':sha(p)})
        self.class_index=defaultdict(set);self.explicit_mapping={}
        mapping=root/'data/reviewer_v8_ddpf/ontology_node_map.json'
        if mapping.is_file():
            for id,uri in load(mapping).items():
                cls=rdflib.URIRef(uri)
                if (cls,RDF.type,OWL.Class) not in self.g and (cls,RDF.type,RDFS.Class) not in self.g:raise ValueError('unverified ontology mapping: '+id)
                self.explicit_mapping[id]=cls
            self.audit['mapping_sha256']=sha(mapping)
        for s in set(self.g.subjects(RDF.type,OWL.Class))|set(self.g.subjects(RDF.type,RDFS.Class)):
            if isinstance(s,rdflib.URIRef):self.class_index[str(s).rsplit('#',1)[-1].rsplit('/',1)[-1]].add(s);self.class_index[str(s)].add(s)
        if neo:
            from neo4j import GraphDatabase
            from dotenv import load_dotenv
            load_dotenv(root/'.env');self.driver=GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.getenv('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD']));self.driver.verify_connectivity();self.audit['neo4j']=True
    def get(self,cid):
        if cid in self.memo:return self.memo[cid]
        rows=[]
        if self.driver:
            q="MATCH (n {id:$id})-[r]-(m) WHERE type(r) IN $rels RETURN DISTINCT n.id AS node_id, type(r) AS predicate, CASE WHEN startNode(r)=n THEN 'out' ELSE 'in' END AS direction, m.id AS neighbor_id,m.name AS neighbor_label ORDER BY predicate,neighbor_id,neighbor_label LIMIT 12"
            with self.driver.session(database=os.getenv('NEO4J_DATABASE','neo4j'),default_access_mode='READ') as s:
                for r in s.run(q,id=cid,rels=self.cfg['static_relations']):
                    x=dict(r)
                    if x['neighbor_id'] and not REF.search(json.dumps(x)):rows.append({'kind':'static_kg','origin':'BioPropaPhenKG static allowlist',**x})
        classes={self.explicit_mapping[cid]} if cid in self.explicit_mapping else self.class_index.get(cid,set())
        if len(classes)==1:
            c=next(iter(classes));todo=[c];seen={c}
            while todo and len(seen)<100:
                child=todo.pop()
                for parent in self.g.objects(child,self.RDFS.subClassOf):
                    if not isinstance(parent,self.rdf.URIRef):continue
                    rows.append({'kind':'asserted_rdfs','subject':str(child),'predicate':'subClassOf','object':str(parent),'origin':'ontology'})
                    if parent not in seen:seen.add(parent);todo.append(parent)
            direct=set(self.g.objects(c,self.RDFS.subClassOf))
            for parent in sorted(seen-{c}-direct,key=str):rows.append({'kind':'inferred_rdfs','subject':str(c),'predicate':'subClassOf','object':str(parent),'rule':'transitive named-class subsumption','origin':'ontology'})
            for p in self.g.objects(c,self.OWL.equivalentClass):
                if isinstance(p,self.rdf.URIRef):rows.append({'kind':'asserted_rdfs','subject':str(c),'predicate':'equivalentClass','object':str(p),'origin':'ontology'})
        self.memo[cid]=[dict(x,fact_id='S'+stable(x)[:24]) for x in rows if not REF.search(json.dumps(x))];return self.memo[cid]
    def for_event(self,e,fi):
        rows=[]
        for c in sorted(concepts(e,fi)):rows.extend(self.get(c))
        uniq={x['fact_id']:x for x in rows};return sorted(uniq.values(),key=lambda x:(x['kind']!='inferred_rdfs',x['fact_id']))[:self.cfg['max_static_facts']]
    def check_fact(self,f,ni):
        # Only explicit URI mappings and explicit disjointness justify rejection.
        rd=self.rdf;predicate=rd.URIRef(f['predicate']) if '://' in f['predicate'] or f['predicate'].startswith('urn:') else None
        if predicate is None:return []
        reasons=[]
        def ancestors(c):
            seen={c};todo=[c]
            while todo:
                for parent in self.g.objects(todo.pop(),self.RDFS.subClassOf):
                    if isinstance(parent,rd.URIRef) and parent not in seen:seen.add(parent);todo.append(parent)
            return seen
        for endpoint,relation,code in [('subject',self.RDFS.domain,'domain_disjoint'),('object',self.RDFS.range,'range_disjoint')]:
            n=ni[f[endpoint]];c=rd.URIRef(n['class_uri']) if n.get('class_uri') else self.explicit_mapping.get(f[endpoint]);
            if c is None:continue
            for required in self.g.objects(predicate,relation):
                if not isinstance(required,rd.URIRef):continue
                if any((a,self.OWL.disjointWith,b) in self.g or (b,self.OWL.disjointWith,a) in self.g for a in ancestors(c) for b in ancestors(required)):reasons.append(code)
        return reasons
    def close(self):
        if self.driver:self.driver.close()

def selected_facts(e,fi,cfg):return [fi[x] for x in e['fact_ids'][:cfg['max_facts']]]
def block(e,fi,ni,cfg,condition,static,donor=None):
    e2=donor or e;fs=selected_facts(e2,fi,cfg)
    metadata={'country':e['country_label'],'collection_interval':[e['collection_start'],e['collection_end']],'occurrence_interval':[e.get('occurrence_start'),e.get('occurrence_end')],'modality':e['modality'],'granularity':e['RE_level'],'availability_basis':e['availability_basis']}
    d={'metadata':metadata}
    if condition=='C0':return d,[]
    if condition=='C1':d['facts']=[{'fact_id':f['fact_id'],'text':fact_text(f,ni)+' '+json.dumps({k:v for k,v in f['properties'].items() if k in ('intensity','frequency','degree','collection_periods','relation_type')},sort_keys=True,ensure_ascii=False)} for f in fs]
    else:
        d['facts']=[{'fact_id':f['fact_id'],'subject':f['subject'],'subject_label':ni[f['subject']]['label'],'subject_type':ni[f['subject']].get('type'),'predicate':f['predicate'],'object':f['object'],'object_label':ni[f['object']]['label'],'object_type':ni[f['object']].get('type'),'properties':{k:v for k,v in f['properties'].items() if k in ('intensity','frequency','degree','collection_periods','relation_type')},'kind':f['kind']} for f in fs]
    if condition in ('C3','C4','C5','C3_no_guard','C3_no_inference'):
        d['static_facts']=[x for x in static.get(e2['observation_id'],[]) if condition!='C3_no_inference' or x['kind']!='inferred_rdfs']
    return d,[f['fact_id'] for f in fs]

def anonymize(payload,seed):
    # Consistent entity aliases and date shift across both inputs; no original names/IDs.
    payload=json.loads(json.dumps(payload));shift=timedelta(days=365*17+seed);mapping={}
    def alias(s):
        if s not in mapping:mapping[s]='E'+stable([seed,s])[:12]
        return mapping[s]
    def rec(x,key=''):
        if isinstance(x,dict):return {k:rec(v,k) for k,v in x.items()}
        if isinstance(x,list):return [rec(v,key) for v in x]
        if isinstance(x,str):
            if key in {'country','subject','object','subject_label','object_label','node_id','neighbor_id','neighbor_label'}:return alias(x)
            if key in {'collection_interval','occurrence_interval','collection_periods'}:
                try:
                    if len(x)==7:return (date.fromisoformat(x+'-01')+shift).isoformat()[:7]
                    return (date.fromisoformat(x)+shift).isoformat()
                except ValueError:return alias(x)
        return x
    return rec(payload)

def donors(ei,seed):
    # Permute within corpus/modality/collection month/granularity, no gold labels.
    groups=defaultdict(list);out={};r=random.Random(seed)
    for i,e in sorted(ei.items()):groups[(e['source_corpus'],e['modality'],e['collection_start'][:7],e['RE_level'],e['available_at'])].append(i)
    for group in groups.values():
        if len(group)<2:continue
        r.shuffle(group);out.update(zip(group,group[1:]+group[:1]))
    return out

def guard(a,b,ans,aid,bid,static_ids=(),violations=None):
    if ans is None:return None,['technical_failure']
    result=dict(ans);why=[];ids=set(ans['evidence_ids']);allowed=set(aid)|set(bid)|set(static_ids)
    if not ids<=allowed:why.append('invalid_evidence_id')
    for id in sorted(ids & set(violations or {})):why.extend(violations[id])
    if ans['relation'] in ('same_event','continuation'):
        if not (ids&set(aid) and ids&set(bid)):why.append('no_evidence_from_both_observations')
        if ans['relation']=='continuation' and not ordered(a,b):why.append('occurrence_order_not_established')
        if ans['relation']=='same_event' and a.get('occurrence_end') and b.get('occurrence_start') and a['occurrence_end']<b['occurrence_start']:why.append('disjoint_occurrence_intervals')
        if ans['relation']=='same_event' and b.get('occurrence_end') and a.get('occurrence_start') and b['occurrence_end']<a['occurrence_start']:why.append('disjoint_occurrence_intervals')
    if why:result.update(relation='insufficient_evidence',abstain_reason=';'.join(why))
    # The guard validates admissibility, not factual entailment.
    return result,why

def graphs(ei,rows,threshold):
    nodes=sorted(ei);g=nx.DiGraph();g.add_nodes_from(nodes);raw=[];identity=[];related=[];rejected=[]
    for r in rows:
        ans=r.get('validated');a,b=r['a'],r['b']
        if not ans:continue
        rel=ans['relation'];score=ans['score'];edge={'a':a,'b':b,'pair_id':r['pair_id'],'score':score,'evidence_ids':ans['evidence_ids']}
        if rel=='same_event':identity.append(edge) # never automatically collapse ambiguous aggregate events
        elif rel=='related_only':related.append(edge)
        elif rel=='continuation':
            raw.append(edge)
            if score<threshold or not ordered(ei[a],ei[b]):rejected.append(dict(edge,reason='threshold_or_unknown_occurrence_order'));continue
            g.add_edge(a,b,weight=score-threshold,**edge)
    # A strict occurrence order guarantees acyclicity; maximum branching permits roots.
    if not nx.is_directed_acyclic_graph(g):raise ValueError('chronology violation')
    f=nx.maximum_branching(g,attr='weight',preserve_attrs=True);f.add_nodes_from(nodes)
    assert nx.is_directed_acyclic_graph(f) and all(d<=1 for _,d in f.in_degree())
    dag=[dict(v) for _,_,v in g.edges(data=True)];forest=[dict(v) for _,_,v in f.edges(data=True)]
    rejected.extend(dict(v,reason='forest_parent_projection') for a,b,v in g.edges(data=True) if not f.has_edge(a,b))
    return {'nodes':nodes,'raw_continuations':raw,'identity_proposals':identity,'related_only':related,'dag':dag,'forest':forest,'roots':[x for x,d in f.in_degree() if d==0],'rejected':rejected,'identity_policy':'singleton observations; same_event proposals not merged without independent authorized identity evidence'}

def baseline_rows(ei,pairs,cfg,fi=None):
    out={};th=cfg['threshold']
    for name in ('B0_no_links','B1_temporal_geo','B2_tfidf','B2_jaccard','B3_ddpf_cosine','B4_hybrid','B5_adapted_similarity_insertion','B6_conservative_rules','B_timeline'):
        rows=[]
        for p in pairs:
            score={'B0_no_links':0.,'B1_temporal_geo':(p['time']+p['geo'])/2,'B2_tfidf':p['tfidf'],'B2_jaccard':p['jaccard'],'B3_ddpf_cosine':max(0,p['cosine']),'B4_hybrid':p['hybrid'],'B5_adapted_similarity_insertion':(p['tfidf']+p['time'])/2,'B6_conservative_rules':0.,'B_timeline':1. if p['rank_time']==1 and p['known_occurrence_order'] else 0.}[name]
            rel='continuation' if score>=th else 'insufficient_evidence'
            if name=='B0_no_links':rel='insufficient_evidence'
            evidence=[]
            if name=='B6_conservative_rules' and fi:
                evidence=[id for id in set(ei[p['a']]['fact_ids'])|set(ei[p['b']]['fact_ids']) if fi[id]['predicate']=='documented_continuation' and fi[id]['subject']==p['a'] and fi[id]['object']==p['b'] and fi[id]['kind']=='observed' and fi[id]['provenance'].get('source_id')]
                if evidence and p['known_occurrence_order']:rel='continuation';score=1.
            a={'relation':rel,'score':float(score),'evidence_ids':evidence,'subtype':'none','rationale':'heuristic similarity proxy, not documentary evidence','abstain_reason':''};rows.append({**p,'raw':a,'validated':a})
        out[name]=rows
    return out

def write_variant(out,name,ei,rows,cfg):
    base=out/'runs'/name;base.mkdir(parents=True,exist_ok=True);jsonl(base/'decisions.jsonl',rows);jsonl(base/'decisions_raw.jsonl',[{**{k:r[k] for k in ('a','b','pair_id')},'decision':r['raw']} for r in rows]);jsonl(base/'decisions_validated.jsonl',[{**{k:r[k] for k in ('a','b','pair_id')},'decision':r['validated'],'guard_reasons':r.get('guard_reasons',[])} for r in rows]);g=graphs(ei,rows,cfg['threshold']);dump(base/'graphs.json',g);dump(base/'forest.json',{'nodes':g['nodes'],'edges':g['forest'],'roots':g['roots']})
    # Graph structure at every availability checkpoint, with static/frozen features only.
    rep=[];prev=set()
    for t in sorted({e['available_at'] for e in ei.values()}):
        sub={i:e for i,e in ei.items() if e['available_at']<=t};gg=graphs(sub,[r for r in rows if r['a'] in sub and r['b'] in sub],cfg['threshold']);cur={(r['a'],r['b']) for r in gg['forest']};rep.append({'as_of':t,'nodes':len(sub),'edges':len(cur),'added':len(cur-prev),'removed':len(prev-cur),'edge_pairs':sorted(cur)});prev=cur
    jsonl(base/'replay.jsonl',rep)
    csvwrite(base/'evidence_audit.csv',[{'pair_id':r['pair_id'],'a':r['a'],'b':r['b'],'raw_relation':(r.get('raw') or {}).get('relation'),'validated_relation':(r.get('validated') or {}).get('relation'),'cited_ids':(r.get('raw') or {}).get('evidence_ids',[]),'guard_reasons':r.get('guard_reasons',[]),'fact_ids_a':r.get('fact_ids_a',[]),'fact_ids_b':r.get('fact_ids_b',[]),'status':r.get('status','baseline')} for r in rows],['pair_id','a','b','raw_relation','validated_relation','cited_ids','guard_reasons','fact_ids_a','fact_ids_b','status'])
    usage=[r['inference'] for r in rows if r.get('inference')];times=[x['elapsed_seconds'] for x in usage if x.get('elapsed_seconds') is not None];return {'pairs':len(rows),'inference_records':len(usage),'mean_original_call_seconds':float(np.mean(times)) if times else None,'input_tokens':sum(x.get('usage',{}).get('prompt_eval_count') or 0 for x in usage),'output_tokens':sum(x.get('usage',{}).get('eval_count') or 0 for x in usage),'cache_hits':sum(x.get('cache_hit',False) for x in usage),'technical_missing':sum(r.get('raw') is None for r in rows),'forest_edges':len(g['forest']),'dag_edges':len(g['dag']),'roots':len(g['roots']),'guard_changes':sum(bool(r.get('guard_reasons')) for r in rows),'decisions_sha256':sha(base/'decisions.jsonl')}

def create_plan(out,ei,fi,ni,cfg,client=None,external=None):
    ids,mat,texts,meta=representations(ei,fi,ni,cfg,client,external)
    np.save(out/'embeddings.npy',mat);jsonl(out/'embedding_ids.jsonl',[{'observation_id':i,'row':j,'embedding_ref':f'embeddings.npy#{j}'} for j,i in enumerate(ids)])
    selected,universe,clusters=candidates(ei,fi,ids,mat,texts,cfg);jsonl(out/'candidates.jsonl',selected);jsonl(out/'eligible_pairs.jsonl',universe);jsonl(out/'clusters.jsonl',clusters)
    cp={r['observation_id']:r for r in clusters};jsonl(out/'events_with_prediction.jsonl',[dict(e,original_embedding_ref=e.get('embedding_ref'),original_cluster_id=e.get('cluster_id'),embedding_ref=f'embeddings.npy#{j}',cluster_id=cp[i]['cluster_id'],cluster_as_of=cp[i]['as_of']) for j,i in enumerate(ids) for e in [ei[i]]])
    result={'embedding':meta,'n_observations':len(ids),'n_eligible_pairs':len(universe),'n_selected_pairs':len(selected),'known_occurrence_order_pairs':sum(p['known_occurrence_order'] for p in selected),'max_main_calls':(len(universe) if cfg.get('all_pairs') else len(selected))*len(cfg['models'])*(len(cfg['conditions'])+(len(cfg['shuffle_seeds'])-1 if 'C5' in cfg['conditions'] else 0)),'extra_no_inference_calls_upper_bound':len(universe) if cfg.get('all_pairs') else len(selected),'clustering':'prefix KMeans on available observations; regenerated, not exact original clusters','pair_id_order':'known occurrence order when established, otherwise availability then observation_id; arrival_id identifies latest availability; equal availability is not causal order','hashes':{n:sha(out/n) for n in ('embeddings.npy','embedding_ids.jsonl','events_with_prediction.jsonl','candidates.jsonl','eligible_pairs.jsonl','clusters.jsonl')}}
    dump(out/'plan.json',result);return result

def run_models(root,out,ei,fi,ni,cfg,client,pairs,static,scope,violations=None):
    # Separate output for every frozen configuration and variant. Cache is shared within run.
    summary={};seed=cfg['seed'];shuffles={s:donors(ei,s) for s in cfg['shuffle_seeds']};variants=[]
    for c in cfg['conditions']:
        if c=='C5':variants.extend((f'C5_seed{s}',c,s) for s in cfg['shuffle_seeds'])
        else:variants.append((c,c,seed))
    # Guard-only and no-inference variants on the designated model; raw calls reused for guard-only.
    for model in cfg['models']:
        for variant,cond,s in variants+([('C3_no_inference','C3_no_inference',seed)] if model==cfg['models'][0] and scope!='pilot' else []):
            rows=[];skips=0
            for p in pairs:
                a,b=ei[p['a']],ei[p['b']];d1=d2=None
                if cond=='C5':
                    mp=shuffles[s]
                    if p['a'] not in mp or p['b'] not in mp:
                        rows.append(dict(p,raw=None,validated=None,guard_reasons=[],status='shuffle_not_eligible'));skips+=1;continue
                    d1,d2=ei[mp[p['a']]],ei[mp[p['b']]]
                x,aid=block(a,fi,ni,cfg,cond,static,d1);y,bid=block(b,fi,ni,cfg,cond,static,d2)
                payload={'observation_a':x,'observation_b':y}
                if cond=='C4':payload=anonymize(payload,s)
                if REF.search(json.dumps(payload,ensure_ascii=False)):raise ValueError('reserved reference marker in prompt payload')
                raw=client.call(model,TASK,payload,SCHEMA,check_answer,budget=512)
                static_ids=[f['fact_id'] for z in (x,y) for f in z.get('static_facts',[])]
                if cond in ('C3','C4','C5','C3_no_inference'):validated,reasons=guard(a,b,raw,aid,bid,static_ids,violations)
                else:validated,reasons=raw,[]
                rows.append(dict(p,raw=raw,validated=validated,guard_reasons=reasons,status='ok' if raw else 'technical_failure',inference=getattr(client,'last_usage',None),input_hash=stable(payload),fact_ids_a=aid,fact_ids_b=bid,static_fact_ids=static_ids,shuffle_donors=[d1['observation_id'],d2['observation_id']] if d1 else None))
            name=model.replace(':','_').replace('/','_')+'__'+variant;summary[name]=dict(write_variant(out,name,ei,rows,cfg),shuffle_unavailable=skips,scope=scope)
            if cond=='C2':
                guarded=[]
                for r in rows:
                    vv,why=guard(ei[r['a']],ei[r['b']],r['raw'],r['fact_ids_a'],r['fact_ids_b'])
                    guarded.append(dict(r,validated=vv,guard_reasons=why))
                name2=model.replace(':','_').replace('/','_')+'__C2_guard';summary[name2]=dict(write_variant(out,name2,ei,guarded,cfg),scope=scope)
            if cond=='C3':
                raw_rows=[dict(r,validated=r['raw'],guard_reasons=[]) for r in rows];name2=model.replace(':','_').replace('/','_')+'__C3_no_guard';summary[name2]=dict(write_variant(out,name2,ei,raw_rows,cfg),scope=scope)
            dump(out/'model_run_progress.json',summary)
            print('Finished',name,summary[name],flush=True)
    return summary


def natural_baselines(ei,universe,cfg,fi):
    """Use each method's own retrieval channel; comparison on same DDPF universe."""
    by=baseline_rows(ei,universe,cfg,fi);out={}
    channels={'B1_temporal_geo':lambda p:(p['time']+p['geo'])/2,'B2_tfidf':lambda p:p['tfidf'],'B2_jaccard':lambda p:p['jaccard'],'B3_ddpf_cosine':lambda p:p['cosine'],'B4_hybrid':lambda p:p['hybrid'],'B5_adapted_similarity_insertion':lambda p:(p['tfidf']+p['time'])/2,'B_timeline':lambda p:p['time']}
    for name,key in channels.items():
        groups=defaultdict(list)
        for p in by[name]:groups[p.get('arrival_id',p['b'])].append(p)
        out[name+'__natural_retrieval']=[p for g in groups.values() for p in sorted(g,key=lambda p:(-key(p),p['pair_id']))[:cfg['top_k']]]
    return out
