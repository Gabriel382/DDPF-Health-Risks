"""Documentary track: all candidate decisions, matched models, blind reference review,
source-anchored extraction, semantic retrieval and independently constrained forest insertion.
The new forest is a separately specified implementation, not the old SQLite replay.
"""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations
from datetime import date,datetime
import re,json,random,sqlite3,os
from bs4 import BeautifulSoup
from .core import *

BASE='data/realdata/who_storyforest_v7'
OUT='outputs/realdata/who_storyforest_final_v7'
def don(x):
    m=re.search(r'DON\s*(\d+)',str(x),re.I);return 'DON'+m.group(1) if m else None

def event(x):
    if not isinstance(x,dict):raise ValueError('Event must be an object')
    # Support legacy event cards, including nested evidence and typed metadata.
    meta=x.get('metadata') or x.get('attributes') or {}; meta=meta if isinstance(meta,dict) else {}
    d=dict(meta,**x);ev=first(d,'evidence','evidence_text','source_text','quote','evidence_span','supporting_text','text_span','passage','source_passage','evidence_quote','source_excerpt',default='' )
    evidence=ev if isinstance(ev,list) else [ev]
    quotes=[];ids=[];sources=[]
    for z in evidence:
        if isinstance(z,dict):
            quotes.append(str(first(z,'quote','text','excerpt','span','evidence_text','passage',default='')));ids.append(str(first(z,'evidence_id','id',default='')));sources.append(str(first(z,'document_id','doc_id','source_id','source','url',default='')))
        elif z:quotes.append(str(z))
    src=don(first(d,'document_id','doc_id','source_document','source_doc','source_doc_id','source_id','source','don_id','url',default='')) or don(sources) or don(d)
    id_=first(d,'event_id','id','uid','_key','mention_id'); day=iso(first(d,'date','event_date','occurrence_date','reported_date','time','event_time','timestamp','date_iso','event_date_iso','start_date'))
    typ=first(d,'event_type','type','kind','category','event_kind'); desc=first(d,'description','title','summary','event_description','text','label','name','event')
    concepts=first(d,'concept_ids','concepts','umls_cuis','disease_id','concept_id','cui',default=[])
    if isinstance(concepts,str):concepts=re.findall(r'C\d{7}',concepts)
    concepts=sorted(set(re.findall(r'C\d{7}',json.dumps(concepts))))
    disease=first(d,'disease','phenomenon','disease_name',default=''); loc=first(d,'location','country','country_name','place','locations',default='')
    if not id_ or not day or not typ or not desc or not src:raise ValueError('Missing required event fields: '+json.dumps(x,default=str)[:850])
    return {'id':str(id_),'date':day,'type':str(typ),'description':str(desc),'doc':src,'disease':str(disease),'location':str(loc),'concepts':concepts,'quotes':[q for q in quotes if q], 'evidence_ids':[i for i in ids if i]}

def pairids(x):
    def ident(v):return str(first(v,'event_id','id','uid')) if isinstance(v,dict) else str(v)
    a=first(x,'source_event_id','from_event_id','from_event','source_event','parent_id','event_a','event_a_id','a_id','source_id','from_id','source','from','a','left','parent','earlier_event_id','event1','event_id_a');b=first(x,'target_event_id','to_event_id','to_event','target_event','child_id','event_b','event_b_id','b_id','target_id','to_id','target','to','b','right','child','later_event_id','event2','event_id_b')
    if a is None or b is None:raise ValueError('Unrecognized pair: '+str(x)[:350])
    return ident(a),ident(b)
def ref_label(x):
    l=first(x,'gold_relation','reference_relation','expected_relation','relation_type','relation','label','gold_label','reference_label','expected','silver_label','target_label','is_positive','gold','positive')
    if isinstance(l,bool):return 'core_positive' if l else 'core_negative'
    l=norm(l).replace(' ','_').replace('-','_')
    aliases={'continuation':'documented_continuation','same':'same_event','coreference':'same_event','coreferent':'same_event','same_event_identity':'same_event','negative':'core_negative','not_related':'core_negative','not_continuation':'core_negative','unrelated':'core_negative','unsupported':'core_negative','no_link':'core_negative','not_linked':'core_negative','positive':'core_positive','1':'core_positive','0':'core_negative','false':'core_negative','true':'core_positive'}
    l=aliases.get(l,l)
    if l not in REL+['core_positive','core_negative']:raise ValueError('Unrecognized reference label: '+str(x)[:400])
    return l

def read_documents(root,expected):
    # Never download newer versions over the original archived input.
    srcroot=root/OUT
    candidates=list(srcroot.rglob('*.html'))+list(srcroot.rglob('*.htm'))+list(srcroot.rglob('*.txt'))
    candidates=[p for p in candidates if not any('final_v7_1'==x for x in p.parts)]
    docs={}
    for d in expected:
        hits=[p for p in candidates if d.lower() in p.name.lower()]
        if not hits:
            # Some archives use full WHO slugs rather than DON filenames.
            hits=[p for p in candidates if p.suffix.lower() in ('.html','.htm') and d.lower() in p.read_text(encoding='utf-8',errors='replace')[:40000].lower()]
        if not hits:raise FileNotFoundError('Archived source HTML/text missing for '+d+' in '+str(srcroot))
        html=sorted(hits,key=lambda p:(p.suffix.lower() not in ('.html','.htm'),str(p)))[0]
        if len({sha(p) for p in hits if p.suffix==html.suffix})>1:raise ValueError('Multiple different source versions for '+d+': '+str(hits))
        raw=html.read_text(encoding='utf-8',errors='replace')
        if html.suffix.lower() in ('.html','.htm'):
            soup=BeautifulSoup(raw,'html.parser')
            for z in soup(['script','style','nav','footer','header','noscript']):z.decompose()
            body=soup.select_one('.sf-detail-body-wrapper') or soup.select_one('article') or soup.select_one('main') or soup.body or soup
            raw=body.get_text(' ',strip=True)
        text=' '.join(raw.split())
        if len(text)<300:raise ValueError('Source text too short: '+d)
        docs[d]={'text':text,'file':str(html.relative_to(root)),'sha256':sha(html),'url':'https://www.who.int/emergencies/disease-outbreak-news/item/'+('2020-' if int(d[3:])<300 else '2022-')+d}
    return docs

def validate_sources(events,docs):
    for e in events:
        src=docs[e['doc']]['text'];e['verified_quotes']=[q for q in e['quotes'] if norm(q) in norm(src)]
        # A verified quote is preferable; otherwise supply the source itself without fabricating a passage.
        e['evidence']=e['verified_quotes'] or [src]
        e['source_hash']=docs[e['doc']]['sha256'];e['quote_verified']=bool(e['verified_quotes'])
    return events

def candidates(events,k=4):
    ordered=sorted(events,key=lambda e:(e['date'],e['id']));return [(a['id'],b['id']) for i,b in enumerate(ordered) for a in ordered[max(0,i-k):i]]

def recover_candidates(root,events):
    ids={e['id'] for e in events};trees=root/OUT/'final_v7_1'/'trees'
    def visit(x,out):
        if isinstance(x,dict):
            try:
                a,b=pairids(x)
                if a in ids and b in ids and a!=b:out.add((a,b))
            except ValueError:pass
            for v in x.values():visit(v,out)
        elif isinstance(x,list):
            for v in x:visit(v,out)
    for p in sorted(trees.glob('*.json')):
        found=set();visit(load(p),found)
        if len(found)==106:return sorted(found),{'kind':'recovered_original_106','source':str(p.relative_to(root)),'sha256':sha(p)}
    # Recover exact full candidate decisions from SQLite if JSON exports only retained edges.
    for p in sorted(trees.glob('*.sqlite3')):
        with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            for (tab,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                if not re.fullmatch(r'[A-Za-z0-9_]+',tab):continue
                found=set()
                for row in db.execute('SELECT * FROM "'+tab+'" LIMIT 2000'):
                    try:
                        a,b=pairids(dict(row))
                        if a in ids and b in ids and a!=b:found.add((a,b))
                    except ValueError:pass
                if len(found)==106:return sorted(found),{'kind':'recovered_original_106','source':str(p.relative_to(root))+':'+tab,'sha256':sha(p)}
    return candidates(events),{'kind':'new_top4_chronological_schedule','note':'Exact legacy candidate list not recovered. This is a new deterministic schedule, not an exact replay.'}

class Semantic:
    """Read-only static ontology and optional KG retrieval. Do not query temporal case/observation edges."""
    def __init__(self,root,out,use_neo4j):
        self.out=out;self.audit={'ontology_files':[],'neo4j_status':'not_requested','retrieved_concepts':{},'class_mappings':{}};self.graph=None;self.driver=None;self.memo={}
        import rdflib
        from rdflib.namespace import RDF,RDFS,OWL
        self.RDF=RDF;self.RDFS=RDFS;self.OWL=OWL;self.graph=rdflib.Graph()
        files=[]
        for name in ['PropaPhen.owl','BioPropaPhen.rdf']:
            try:p=locate(root,name,required=False)
            except ValueError as e:self.audit.setdefault('errors',[]).append(str(e));continue
            if p:
                try:self.graph.parse(p,format='xml');files.append(p);self.audit['ontology_files'].append({'path':str(p.relative_to(root)),'sha256':sha(p)})
                except Exception as e:self.audit.setdefault('errors',[]).append(str(e))
        self.labels=defaultdict(set)
        for c in set(self.graph.subjects(RDF.type,OWL.Class))|set(self.graph.subjects(RDF.type,RDFS.Class)):
            if not isinstance(c,rdflib.URIRef):continue
            names=[str(v) for v in self.graph.objects(c,RDFS.label)]+[str(c).rsplit('#',1)[-1].rsplit('/',1)[-1]]
            for s in names:self.labels[re.sub(r'[^a-z0-9]','',s.lower())].add(c)
        self.explicit_mapping={}
        mp=root/'data'/'reviewer_v8'/'ontology_event_class_map.json'
        if mp.exists():
            for typ,uri in load(mp).items():
                c=rdflib.URIRef(uri)
                if (c,RDF.type,OWL.Class) not in self.graph and (c,RDF.type,RDFS.Class) not in self.graph:raise ValueError('Invalid mapped ontology class '+str(uri))
                self.explicit_mapping[typ]=c
        self.audit['ontology_triples']=len(self.graph);self.audit['explicit_mapping_file']=str(mp.relative_to(root)) if mp.exists() else None
        if use_neo4j:
            try:
                from neo4j import GraphDatabase
                from dotenv import load_dotenv
                load_dotenv(root/'.env')
                self.driver=GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.getenv('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD']));self.driver.verify_connectivity();self.audit['neo4j_status']='connected_read_only'
            except Exception as e:self.audit['neo4j_status']='unavailable';self.audit.setdefault('errors',[]).append(type(e).__name__);self.driver=None
    def classes(self,e):
        if e['type'] in self.explicit_mapping:return [self.explicit_mapping[e['type']]]
        key=re.sub(r'[^a-z0-9]','',e['type'].lower());c=self.labels.get(key,set());return list(c) if len(c)==1 else []
    def ancestors(self,c):
        seen={c};todo=[c]
        while todo:
            for p in self.graph.objects(todo.pop(),self.RDFS.subClassOf):
                if p not in seen:seen.add(p);todo.append(p)
        return seen
    def context(self,e):
        rows=[]
        for c in self.classes(e):
            self.audit['class_mappings'][e['type']]=str(c)
            for a in sorted(self.ancestors(c),key=str)[:30]:rows.append({'source':'ontology','class':str(c),'subclass_of_or_self':str(a)})
            for a in self.graph.objects(c,self.OWL.disjointWith):rows.append({'source':'ontology','class':str(c),'disjoint_with':str(a)})
        for cui in e['concepts']:
            if cui not in self.memo:
                results=[]
                if self.driver:
                    try:
                        with self.driver.session(database=os.getenv('NEO4J_DATABASE','neo4j'),default_access_mode='READ') as s:
                            # Strict allowlist: no isReported/hasPresence or future observations.
                            q="MATCH (n {id:$id})-[r]-(m) WHERE type(r) IN $rels RETURN DISTINCT n.name AS concept,type(r) AS relation,CASE WHEN startNode(r)=n THEN 'out' ELSE 'in' END AS direction,m.id AS neighbor_id,m.name AS neighbor_name ORDER BY relation,neighbor_id,neighbor_name LIMIT 12"
                            for x in s.run(q,id=cui,rels=['SCO','ISA','IS_A','subClassOf','PAR','CHD','RB','RN','SY','RO']):results.append(dict(x))
                    except Exception as ex:self.audit.setdefault('retrieval_errors',[]).append({'cui':cui,'error':type(ex).__name__})
                self.memo[cui]=results;self.audit['retrieved_concepts'][cui]=len(results)
            rows.extend({'source':'BioPropaPhenKG','cui':cui,**r} for r in self.memo[cui])
        return rows
    def disjoint(self,a,b):
        ca=self.classes(a);cb=self.classes(b)
        if not ca or not cb:return False
        aa=self.ancestors(ca[0]);bb=self.ancestors(cb[0])
        return any((x,self.OWL.disjointWith,y) in self.graph or (y,self.OWL.disjointWith,x) in self.graph for x in aa for y in bb)
    def close(self):
        if self.driver:self.driver.close()
        dump(self.out/'semantic_audit.json',self.audit)

LINK_SCHEMA={'type':'object','properties':{'relation':{'type':'string','enum':REL},'reason':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string'}}},'required':['relation','reason','evidence_ids'],'additionalProperties':False}
def link_validate(x):
    if set(x)!=set(LINK_SCHEMA['required']) or x['relation'] not in REL or not isinstance(x['reason'],str) or not isinstance(x['evidence_ids'],list) or any(not isinstance(v,str) for v in x['evidence_ids']):raise ValueError('invalid link schema')
def link_payload(a,b,condition,semantic_context=None):
    def card(e):
        r={'id':e['id'],'description':e['description'],'evidence':e['evidence'],'document_id':e['doc']}
        if condition!='text_only':r.update({k:e[k] for k in ('date','type','disease','location','concepts')})
        if semantic_context is not None:r['retrieved_static_knowledge']=semantic_context[e['id']]
        return r
    return {'earlier_candidate':card(a),'later_candidate':card(b),'definition':'same_event = the same occurrence described twice; documented_continuation = distinct occurrences with explicitly supported development of one reported story; related_only = topical association without documented continuation; unknown = insufficient evidence. Return IDs of both cards when supporting a positive relation. A shared disease or temporal order alone is insufficient. Do not infer contagion.'}
def guard(a,b,ans,sem):
    if not ans:return None,'technical_failure'
    r=ans['relation'];why=[]
    if r in ('same_event','documented_continuation'):
        if not {a['id'],b['id']}<=set(ans['evidence_ids']):why.append('missing_card_evidence_ids')
        if not a['evidence'] or not b['evidence']:why.append('missing_source')
        if r=='documented_continuation' and a['date']>b['date']:why.append('reverse_chronology')
        if r=='same_event' and sem.disjoint(a,b):why.append('explicit_ontology_disjointness')
    return ('unknown' if why else r),';'.join(why)

def forest(events,preds,schedule):
    """Exactly one parent per node, forward total order; latest eligible parent tie-break."""
    e={x['id']:x for x in events};order={x['id']:i for i,x in enumerate(sorted(events,key=lambda x:(x['date'],x['id'])))}
    parents={}; rejected=Counter();edges=[];identity=[]
    for a,b in schedule:
        r=preds.get((a,b))
        if r=='same_event':identity.append([a,b]);continue
        if r!='documented_continuation':continue
        if order[a]>=order[b]:rejected['non_forward_order']+=1;continue
        if b not in parents or order[a]>order[parents[b]]:
            if b in parents:rejected['alternate_parent']+=1
            parents[b]=a
        else:rejected['alternate_parent']+=1
    edges=sorted((a,b) for b,a in parents.items());roots=[x for x in order if x not in parents];assert len(edges)==len(events)-len(roots)
    return {'nodes':events,'edges':[list(x) for x in edges],'roots':roots,'identity_relations':identity,'n_nodes':len(events),'n_edges':len(edges),'n_roots':len(roots),'acyclic':True,'max_indegree':int(bool(edges)),'rejected':dict(rejected),'rule':'latest eligible predecessor in total chronological/id order; identity retained separately'}

def forest_score(f,refs):
    pos={p for p,l in refs.items() if l=='documented_continuation'};neg={p for p,l in refs.items() if l in ('related_only','same_event','core_negative')};edges=set(map(tuple,f['edges']));tp=len(edges&pos);fp=len(edges&neg);unknown=len(edges-pos-neg)
    return {'reference_direct_edges':len(pos),'matched':tp,'known_negative_edges':fp,'unlabelled_edges':unknown,'direct_edge_recall':tp/len(pos) if pos else None,'precision_on_labelled_edges':tp/(tp+fp) if tp+fp else None,'edge_precision_lower_bound':tp/len(edges) if edges else None,'edge_precision_upper_bound':(tp+unknown)/len(edges) if edges else None}

def score_pairs(preds,refs):
    keys=[k for k,v in refs.items() if v!='unknown'];gold=[int(refs[k] in ('same_event','documented_continuation','core_positive')) for k in keys];pred=[None if preds.get(k) is None else int(preds[k] in ('same_event','documented_continuation')) for k in keys];d=binary(gold,pred);d['decision_coverage']=sum(preds.get(k) not in (None,'unknown') for k in keys)/len(keys) if keys else None;d['unknown_decisions']=sum(preds.get(k)=='unknown' for k in keys);d['technical_missing']=sum(preds.get(k) is None for k in keys);d['accuracy_all']=sum(a==b and preds.get(k) not in (None,'unknown') for k,a,b in zip(keys,gold,pred))/len(keys) if keys else None;multi=[k for k in keys if refs[k] in REL];d['four_class_metrics']=multiclass([refs[k] for k in multi],[preds.get(k) for k in multi],REL);return d

def baseline(events,pairs):
    # Fixed, predeclared threshold; not optimized on the 39 silver references.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    x=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit_transform([e['description'] for e in events]);idx={e['id']:i for i,e in enumerate(events)};sim=cosine_similarity(x);out={}
    for a,b in pairs:
        s=float(sim[idx[a],idx[b]]);out[a,b]={'similarity':s,'relation':'documented_continuation' if s>=.35 else 'related_only'}
    return out

def review_files(out,events,pairs,docs):
    ids={e['id']:e for e in events};rows=[]
    for a,b in pairs:
        A,B=ids[a],ids[b];rows.append({'a':a,'b':b,'date_a':A['date'],'date_b':B['date'],'description_a':A['description'],'description_b':B['description'],'document_a':A['doc'],'document_b':B['doc'],'source_a':docs[A['doc']]['url'],'source_b':docs[B['doc']]['url'],'evidence_a':' | '.join(A['evidence']),'evidence_b':' | '.join(B['evidence']),'label':'','supporting_passage':'','reviewer_notes':''})
    random.Random(20261124).shuffle(rows)
    for name in ('reviewer_A.csv','reviewer_B.csv','consensus.csv'):
        p=out/'human_review'/name
        if not p.exists():csvwrite(p,rows)
    erows=[{'event_id':e['id'],'document':e['doc'],'url':docs[e['doc']]['url'],'date':e['date'],'type':e['type'],'description':e['description'],'evidence':' | '.join(e['evidence']),'valid_event':'','reviewer_notes':''} for e in events]
    for name in ('event_reviewer_A.csv','event_reviewer_B.csv'):
        p=out/'human_review'/name
        if not p.exists():csvwrite(p,erows)

def human_refs(out,pairs):
    from sklearn.metrics import cohen_kappa_score
    d={};allowed=set(REL);expected=set(pairs)
    for name in ('reviewer_A','reviewer_B','consensus'):
        rows=csvread(out/'human_review'/(name+'.csv'));seen=set();ans={}
        for x in rows:
            key=(x['a'],x['b']);l=x['label'].strip()
            if key in seen or key not in expected:raise ValueError('Invalid review key '+str(key))
            seen.add(key)
            if l:
                if l not in allowed:raise ValueError('Invalid reviewer label '+l)
                if l in ('same_event','documented_continuation') and not x.get('supporting_passage','').strip():raise ValueError('Positive review needs source passage '+str(key))
                ans[key]=l
        d[name]=ans
    a,b=d['reviewer_A'],d['reviewer_B'];common=sorted(set(a)&set(b));dis=[k for k in common if a[k]!=b[k]];cons=d['consensus'];resolved={k:a[k] for k in common if a[k]==b[k]};resolved.update({k:cons[k] for k in dis if k in cons});
    kappa=float(cohen_kappa_score([a[k] for k in common],[b[k] for k in common],labels=REL)) if len(common)>1 and len(set(a[k] for k in common)|set(b[k] for k in common))>1 else None
    event_status={}
    for prefix in ('event','generated_event'):
        pp=[out/'human_review'/(prefix+'_reviewer_'+x+'.csv') for x in ('A','B')]
        if all(p.exists() for p in pp):
            keys_name='event_id' if prefix=='event' else 'review_id';rr=[csvread(p) for p in pp];maps=[]
            for rows in rr:
                m={}
                for x in rows:
                    label=x.get('valid_event','').strip().lower()
                    if label:
                        if label not in ('yes','no','uncertain'):raise ValueError('Event review valid_event must be yes/no/uncertain')
                        m[x[keys_name]]=label
                maps.append(m)
            aa,bb=maps;cc=sorted(set(aa)&set(bb));event_status[prefix]={'expected':len(rr[0]),'reviewer_A_completed':len(aa),'reviewer_B_completed':len(bb),'paired':len(cc),'agreement':sum(aa[k]==bb[k] for k in cc)/len(cc) if cc else None,'agreement_yes':sum(aa[k]==bb[k]=='yes' for k in cc),'disagreements':sum(aa[k]!=bb[k] for k in cc)}
    return resolved,{'event_review':event_status,'expected':len(pairs),'reviewer_A_completed':len(a),'reviewer_B_completed':len(b),'paired':len(common),'agreement':sum(a[k]==b[k] for k in common)/len(common) if common else None,'cohen_kappa':kappa,'disagreements':len(dis),'resolved':len(resolved),'complete':len(resolved)==len(pairs),'note':'Reviewer identity and independence require author confirmation; completion alone does not establish gold quality.'}

def extract(client,model,doc,source,types):
    # Chunked extraction prevents silent context truncation. Quotes must occur in the original source.
    schema={'type':'object','properties':{'events':{'type':'array','items':{'type':'object','properties':{'date':{'type':'string'},'type':{'type':'string','enum':types},'description':{'type':'string'},'quote':{'type':'string'},'disease':{'type':'string'},'location':{'type':'string'},'concept_ids':{'type':'array','items':{'type':'string'}}},'required':['date','type','description','quote','disease','location','concept_ids'],'additionalProperties':False}}},'required':['events'],'additionalProperties':False}
    def val(x):
        if set(x)!= {'events'} or not isinstance(x['events'],list):raise ValueError('bad extraction schema')
        for e in x['events']:
            if not isinstance(e,dict) or set(e)!=set(schema['properties']['events']['items']['required']) or e['type'] not in types or not all(isinstance(e[k],str) for k in ('date','description','quote','disease','location')) or not isinstance(e['concept_ids'],list):raise ValueError('bad event schema')
    text=source['text'];result=[];audit=[];seen=set();chunks=[text[i:i+8000] for i in range(0,len(text),7300)]
    for ci,chunk in enumerate(chunks):
        ans=client.call(model,'Extract explicitly dated occurrences only, with a verbatim supporting quote, ISO date, type from schema. Return no events if uncertain. Do not assign inferred dates or invented concepts.',{'document_id':doc,'text':chunk},schema,val,budget=1536)
        if ans is None:audit.append({'chunk':ci,'status':'technical_failure'});continue
        for e in ans['events']:
            day=iso(e['date']);quote=e['quote'];qnorm=norm(quote);reason=None
            if not day:reason='unresolved_date'
            elif len(qnorm)<15 or qnorm not in norm(text):reason='quote_not_verbatim'
            elif not e['description'].strip():reason='empty_description'
            # Date support is separately checked, not assumed from a matching event-type slot.
            if day:
                dt=date.fromisoformat(day);variants=[day,dt.strftime('%d %B %Y'),dt.strftime('%d %b %Y'),dt.strftime('%B %d, %Y'),f'{dt.day} {dt.strftime("%B")} {dt.year}',f'{dt.strftime("%B")} {dt.day}, {dt.year}'];date_supported=any(norm(v) in qnorm for v in variants)
            else:date_supported=False
            if reason:audit.append({'chunk':ci,'status':'rejected','reason':reason,'output':e});continue
            key=(day,e['type'],norm(e['description']))
            if key in seen:continue
            seen.add(key);item=event(dict(e,id=f'{doc}-G{len(result)+1:03d}',document_id=doc,evidence_text=quote));item.update(evidence=[quote],verified_quotes=[quote],quote_verified=True,source_hash=source['sha256'],date_explicit_in_quote=date_supported);item['model_proposed_concepts']=item['concepts'];disease=norm(item['disease']);item['concepts']=['C5203670'] if ('covid' in disease or 'coronavirus' in disease) and ('coronavirus' in norm(text) or 'covid' in norm(text)) else ['C0276180'] if ('mpox' in disease or 'monkeypox' in disease) and ('monkeypox' in norm(text) or 'mpox' in norm(text)) else [];result.append(item)
        audit.append({'chunk':ci,'status':'complete'})
    return result,audit

def align(generated,reference):
    # Exact date/type/document match; ambiguous slots are not treated as resolved identities.
    G=defaultdict(list);R=defaultdict(list)
    for e in generated:G[e['doc'],e['date'],e['type']].append(e['id'])
    for e in reference:R[e['doc'],e['date'],e['type']].append(e['id'])
    mapping={g[0]:R[k][0] for k,g in G.items() if len(g)==1 and len(R.get(k,[]))==1}
    return mapping,{'reference_slots':len(R),'matched_slots':len(set(G)&set(R)),'unique_event_alignment':len(mapping),'reference_mentions':len(reference),'generated_mentions':len(generated),'ambiguous_slots':sum(len(G[k])!=1 or len(R[k])!=1 for k in set(G)&set(R))}

def run(root,out,client,models,sem,review_only=False,preflight=False,embedding_model='embeddinggemma'):
    out.mkdir(parents=True,exist_ok=True)
    ep=locate(root,'silver_events.json',[BASE+'/silver_events.json']);pp=locate(root,'silver_pairs.json',[BASE+'/silver_pairs.json'])
    events=[event(x) for x in records(load(ep),['events','mentions','reference_events','silver_events','event_cards'])];ids={x['id']:x for x in events}
    if len(events)!=29 or len(ids)!=29:raise ValueError('Expected 29 unique V7 reference cards, found '+str(len(ids)))
    refs={pairids(x):ref_label(x) for x in records(load(pp),['pairs','links','references','silver_pairs','reference_pairs'])}
    if len(refs)!=39 or any(a not in ids or b not in ids for a,b in refs):raise ValueError('Expected 39 valid reference pairs')
    if sum(l in ('same_event','documented_continuation','core_positive') for l in refs.values())!=22:raise ValueError('Original V7 fixed-pair positive count differs from 22; inspect reference label mapping')
    docs=read_documents(root,{e['doc'] for e in events});
    if len(docs)!=8:raise ValueError('Expected 8 archived WHO docs')
    events=validate_sources(events,docs);schedule,origin=recover_candidates(root,events);pairs=sorted(set(schedule)|set(refs))
    inputs={'event_sha256':sha(ep),'pair_sha256':sha(pp),'documents':{k:{x:y for x,y in v.items() if x!='text'} for k,v in docs.items()},'candidate_origin':origin,'schedule_count':len(schedule),'fixed_pairs':len(refs),'union_pairs':len(pairs),'verified_card_quotes':sum(e['quote_verified'] for e in events),'schedule_sha256':stable(schedule)}
    ip=out/'input_manifest.json'
    if ip.exists() and stable(load(ip))!=stable(inputs):raise ValueError('Documentary inputs changed; select a new --output to avoid mixing reviewed references or predictions')
    dump(ip,inputs);review_files(out,events,pairs,docs)
    dump(out/'candidate_schedule.json',{'forest_candidates':schedule,'fixed_reference_pairs':list(refs),'all_pairs':pairs})
    resolved,review=human_refs(out,pairs);dump(out/'human_review_status.json',review)
    if preflight:return {'status':'PREFLIGHT_OK','schedule':len(schedule),'candidate_origin':origin,'review':review}
    if review_only:
        p=out/'pair_predictions.csv'
        if p.exists():
            groups=defaultdict(dict)
            for x in csvread(p):groups[x['model'],x['condition']][x['a'],x['b']]=x['prediction'] or None
            dump(out/'human_review_metrics.json',[{'model':m,'condition':c,**score_pairs(ps,resolved)} for (m,c),ps in groups.items()])
        forest_review=[]
        for f in sorted((out/'forests').glob('*.json')):
            forest_review.append({'file':f.name,**forest_score(load(f),resolved)})
        for f in sorted((out/'end_to_end').glob('*/forest.json')):
            m=load(f.parent/'alignment.json')['mapping'];z=load(f);mapped={'edges':[[m[a],m[b]] for a,b in z['edges'] if a in m and b in m]};forest_review.append({'file':str(f.relative_to(out)),**forest_score(mapped,resolved),'unmatched_generated_edges':len(z['edges'])-len(mapped['edges'])})
        dump(out/'human_review_forest_metrics.json',forest_review)
        return {'status':'HUMAN_REVIEW_COMPLETE' if review['complete'] else 'HUMAN_REVIEW_PENDING',**review}
    contexts={e['id']:sem.context(e) for e in events};has_kg=any(any(x.get('source')=='BioPropaPhenKG' for x in v) for v in contexts.values());has_sem=any(contexts.values());dump(out/'retrieved_context.json',contexts)
    # Permute actual context within disease; report whether the control is informative.
    shuffled=dict(contexts);g=defaultdict(list)
    for e in events:g[tuple(e['concepts']) or (e['disease'],)].append(e['id'])
    rng=random.Random(20261124)
    for members in g.values():
        v=[contexts[i] for i in members];rng.shuffle(v);shuffled.update(dict(zip(members,v)))
    n_changed=sum(stable(shuffled[k])!=stable(v) for k,v in contexts.items());dump(out/'semantic_control.json',{'actual_kg_retrieved':has_kg,'any_context':has_sem,'within_group_shuffled_changed_cards':n_changed,'shuffle_informative':n_changed>0})
    conditions=[('text_only',None),('typed',None)]
    if has_sem:conditions.append(('semantic',contexts))
    if n_changed:conditions.append(('semantic_shuffled',shuffled))
    predictions=[];metrics=[];fm=[];end_metrics=[];forests=out/'forests';forests.mkdir(exist_ok=True)
    def evaluate(m,c,ps):
        metrics.append({'model':m,'condition':c,'reference':'silver_fixed_pairs',**score_pairs(ps,refs)})
        if resolved:metrics.append({'model':m,'condition':c,'reference':'human_reviewed_subset',**score_pairs(ps,resolved)})
        f=forest(events,ps,schedule);s=forest_score(f,refs);f['metrics']=s;f['candidate_coverage']=sum(ps.get(k) is not None for k in schedule)/len(schedule);dump(forests/(m.replace(':','_').replace('/','_')+'__'+c+'.json'),f);fm.append({'model':m,'condition':c,'candidate_count':len(schedule),'candidate_coverage':f['candidate_coverage'],'n_edges':f['n_edges'],'n_roots':f['n_roots'],'max_indegree':f['max_indegree'],**s})
    base=baseline(events,pairs);evaluate('TFIDF_char_3_5','fixed_threshold_0.35',{k:v['relation'] for k,v in base.items()});predictions.extend({'model':'TFIDF_char_3_5','condition':'fixed_threshold_0.35','a':a,'b':b,'prediction':v['relation'],'reason':'similarity threshold','guard_reason':'','evidence_ids':[]} for (a,b),v in base.items());dump(out/'similarity_baseline.json',[{'a':a,'b':b,**v} for (a,b),v in base.items()])
    if embedding_model:
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.metrics import average_precision_score
            matrix=cosine_similarity(np.asarray(client.embed(embedding_model,[e['description'] for e in events])))
            idx={e['id']:i for i,e in enumerate(events)};scores={k:float(matrix[idx[k[0]],idx[k[1]]]) for k in pairs}
            ps={k:'documented_continuation' if v>=.65 else 'related_only' for k,v in scores.items()};evaluate(embedding_model,'cosine_fixed_threshold_0.65',ps);predictions.extend({'model':embedding_model,'condition':'cosine_fixed_threshold_0.65','a':a,'b':b,'prediction':v,'reason':'cosine threshold','guard_reason':'','evidence_ids':[]} for (a,b),v in ps.items())
            positives=[k for k,l in refs.items() if l!='unknown'];y=[int(refs[k] in ('same_event','documented_continuation','core_positive')) for k in positives]
            dump(out/'embedding_baseline.json',{'model':embedding_model,'threshold':.65,'average_precision_core_link':float(average_precision_score(y,[scores[k] for k in positives])),'scores':[{'a':a,'b':b,'similarity':s} for (a,b),s in scores.items()]})
        except Exception as e:dump(out/'embedding_baseline.json',{'status':'UNAVAILABLE','error':str(e)[:600]})
    csvwrite(out/'pair_predictions.csv',predictions);csvwrite(out/'pair_metrics.csv',metrics);csvwrite(out/'forest_metrics.csv',fm)
    for model in models:
        client.prepare(model)
        for condition,context in conditions:
            raw={};guarded={}
            for a,b in pairs:
                ans=client.call(model,'classify documentary relation',link_payload(ids[a],ids[b],condition,context),LINK_SCHEMA,link_validate,budget=384)
                pred=ans['relation'] if ans else None;gd,reason=guard(ids[a],ids[b],ans,sem);raw[a,b]=pred;guarded[a,b]=gd
                for c,v in ((condition,pred),(condition+'_guard',gd)):
                    predictions.append({'model':model,'condition':c,'a':a,'b':b,'prediction':v,'guard_reason':reason if c.endswith('_guard') else '', 'reason':ans['reason'] if ans else '', 'evidence_ids':ans['evidence_ids'] if ans else []})
            evaluate(model,condition,raw);evaluate(model,condition+'_guard',guarded);csvwrite(out/'pair_predictions.csv',predictions);csvwrite(out/'pair_metrics.csv',metrics);csvwrite(out/'forest_metrics.csv',fm)
        # end-to-end: all models use the same source documents, generated cards only, no reference labels in candidate decisions
        gen=[];aud=[];types=sorted({e['type'] for e in events})
        for doc,source in sorted(docs.items()):
            x,a=extract(client,model,doc,source,types);gen.extend(x);aud.extend({'document':doc,**v} for v in a)
        modelout=out/'end_to_end'/model.replace(':','_').replace('/','_');modelout.mkdir(parents=True,exist_ok=True);dump(modelout/'extracted_events.json',gen);dump(modelout/'extraction_audit.json',aud)
        raw_gen=gen;gen=[e for e in raw_gen if e['date_explicit_in_quote']];dump(modelout/'forest_eligible_events.json',gen)
        mapping,alignment=align(gen,events);alignment['all_quote_verified_extracted_mentions']=len(raw_gen);alignment['excluded_without_explicit_date_in_quote']=len(raw_gen)-len(gen);dump(modelout/'alignment.json',{'mapping':mapping,**alignment});gp=candidates(gen);gids={e['id']:e for e in gen};gcontext={e['id']:sem.context(e) for e in gen};preds={};grows=[]
        for a,b in gp:
            ans=client.call(model,'classify documentary relation',link_payload(gids[a],gids[b],'semantic' if has_sem else 'typed',gcontext if has_sem else None),LINK_SCHEMA,link_validate,budget=384);gd,why=guard(gids[a],gids[b],ans,sem);preds[a,b]=gd;grows.append({'a':a,'b':b,'prediction':gd,'reason':ans['reason'] if ans else '', 'guard_reason':why})
        f=forest(gen,preds,gp);mapped={'edges':[[mapping[a],mapping[b]] for a,b in f['edges'] if a in mapping and b in mapping]};fs=forest_score(mapped,refs);fs['unmatched_generated_edges']=len(f['edges'])-len(mapped['edges']);fs['mapped_edges']=len(mapped['edges']);fs['candidate_count']=len(gp);fs['candidate_coverage']=sum(v is not None for v in preds.values())/len(gp) if gp else None;fs['failed_extraction_chunks']=sum(x['status']=='technical_failure' for x in aud);fs['date_explicit_in_quote']=sum(e['date_explicit_in_quote'] for e in gen);f['evaluation']=fs;dump(modelout/'forest.json',f);csvwrite(modelout/'candidate_predictions.csv',grows,fields=['a','b','prediction','reason','guard_reason']);row={**alignment,**fs,'scope':'strict unique date/type/source alignment; unmatched events are unadjudicated, not false positives'};dump(modelout/'metrics.json',row);end_metrics.append(row)
    pool={}
    for f in sorted((out/'end_to_end').glob('*/extracted_events.json')):
        for e in load(f):
            k=stable([e['doc'],e['date'],e['type'],norm(e['description']),norm(' '.join(e['evidence']))])[:18]
            pool[k]={'review_id':k,'document':e['doc'],'url':docs[e['doc']]['url'],'date':e['date'],'type':e['type'],'description':e['description'],'quote':' | '.join(e['evidence']),'valid_event':'','reviewer_notes':''}
    if pool:
        rows=list(pool.values());random.Random(20261124).shuffle(rows)
        for name in ('generated_event_reviewer_A.csv','generated_event_reviewer_B.csv'):
            p=out/'human_review'/name
            if not p.exists():csvwrite(p,rows)
    return {'status':'COMPLETE' if all(x['candidate_coverage']==1 for x in fm) and all(x['failed_extraction_chunks']==0 and x['candidate_coverage'] in (1,None) for x in end_metrics) and has_kg and has_sem and n_changed else 'PARTIAL_CONTEXT_OR_TECHNICAL','fixed_pairs':39,'candidate_schedule':len(schedule),'all_pairs':len(pairs),'candidate_origin':origin,'semantic_context':has_sem,'actual_kg_retrieved':has_kg,'human_review':review,'models':models,'end_to_end_metrics':end_metrics,'semantic_shuffle_informative':bool(n_changed)}
