"""DDPF-only export, provenance gates, and read-only graph recovery."""
from .core import *
from datetime import timedelta
from calendar import monthrange
from collections import defaultdict
import subprocess

LEGACY='outputs/realdata/reconstructed_observations.jsonl'
LEGACY_SHA='738b3b304bd0cb4f2b1dd84c057fce8b02e645197d60d4932e2362aa01efa004'
LEVELS={'sentence','paragraph','document'}
REF=re.compile(r'(?i)(\bWHO\b|\bECDC\b|who\.int|world health organization|\bDON\d+\b|\bisReported\b)')

def jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,default=str)+'\n')
    t.replace(p)

def read_jsonl(p):
    with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(l) for l in f if l.strip()]

def interval(s):
    if re.fullmatch(r'\d{4}-\d{2}',str(s)):
        y,m=map(int,s.split('-'));return date(y,m,1).isoformat(),date(y,m,monthrange(y,m)[1]).isoformat()
    v=date.fromisoformat(s).isoformat();return v,v

def require_date(x):return date.fromisoformat(x).isoformat()
def fact(subject,predicate,obj,properties,provenance,kind,available_at):
    body=dict(subject=subject,predicate=predicate,object=obj,properties=properties,provenance=provenance,kind=kind,available_at=available_at)
    return {'fact_id':'F'+stable(body)[:24],**body}

def inventory(root,out,hash_large=False):
    result=[]
    for base in (root/'outputs'/'realdata',root/'Experiments',root/'data'):
        if not base.exists():continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or any('reviewer_v8_ddpf' in x or 'reviewer_revision_v8' in x or x=='gold' for x in p.parts):continue
            if not re.search(r'(?i)observation|embedding|cluster|matrix|prediction|detection|propaphen|manifest|\.ipynb$',p.name):continue
            if p.suffix.lower() not in {'.csv','.json','.jsonl','.ipynb','.npy','.npz','.owl','.rdf','.dump','.parquet','.pkl','.tsv'}:continue
            result.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p) if hash_large or p.stat().st_size<100_000_000 else None,'hash_deferred':not hash_large and p.stat().st_size>=100_000_000})
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    dirty=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,capture_output=True,text=True)
    dump(out/'inventory.json',{'git_commit':git.stdout.strip() or None,'tracked_dirty':bool(dirty.stdout.strip()),'files':result,'large_files':'hash on explicit selection; not verified by inventory'})
    return result

def legacy_adapter(rows,path,level):
    events=[];facts=[];nodes={};seen=set()
    for r in rows:
        oid=r['observation_id'];country=r['country_id'];source=r['source'];name=str(r.get('country_name',country)).strip('"')
        if oid in seen:raise ValueError('duplicate observation '+oid)
        seen.add(oid)
        if REF.search(source):raise ValueError('reserved source '+source)
        modality=source.split('_')[0].casefold()
        if modality not in {'journal','medical','social'}:raise ValueError('unknown source modality '+source)
        periods=r['relation_dates'];bounds=[interval(x) for x in periods];start=min(x[0] for x in bounds);end=max(x[1] for x in bounds)
        # End of collection is a declared proxy, not a document publication/availability date.
        avail=end;ids=[]
        nodes[country]={'node_id':country,'type':'Country','label':name,'origin':'DDPF export'}
        for c in r['top_concepts']:
            cid=c['id'];label=c['name'];nodes[cid]={'node_id':cid,'type':'concept','label':label,'origin':'DDPF export'}
            f=fact(cid,'hasPresence',country,{'intensity':c['intensity'],'collection_periods':periods}, {'observation_id':oid,'source':source,'file_sha256':path,'aggregation':'top_concepts summary; not original relationship ID'},'observed_aggregate',avail)
            ids.append(f['fact_id']);facts.append(f)
        events.append({'observation_id':oid,'canonical_event_id':oid,'identity_basis':'singleton; no identity adjudication','country_id':country,'country_label':name,'collection_start':start,'collection_end':end,'occurrence_start':None,'occurrence_end':None,'available_at':avail,'availability_basis':'collection_end_proxy','modality':modality,'RE_level':level,'granularity_basis':'legacy recovered file associated with sentence experiment; source units absent','cluster_id':None,'embedding_ref':None,'subgraph_ref':oid,'fact_ids':ids,'source_ids':[],'extraction_config_hash':stable(r.get('provenance',{})),'source_corpus':source,'source_text':None,'provenance_status':'aggregate_only_unverified_independence','origin':path})
    return events,facts,list(nodes.values())

def canonical_adapter(folder):
    folder=Path(folder);events=read_jsonl(folder/'events.jsonl');facts=read_jsonl(folder/'edges.jsonl');nodes=read_jsonl(folder/'nodes.jsonl')
    keys={'observation_id','country_id','collection_start','collection_end','available_at','modality','RE_level','fact_ids','source_ids','extraction_config_hash','subgraph_ref'}
    for e in events:
        if not keys<=e.keys():raise ValueError('missing event fields '+str(keys-e.keys()))
        e.setdefault('canonical_event_id',e['observation_id']);e.setdefault('identity_basis','singleton');e.setdefault('country_label',e['country_id']);e.setdefault('occurrence_start',None);e.setdefault('occurrence_end',None);e.setdefault('availability_basis','unspecified');e.setdefault('source_text',None);e.setdefault('source_corpus',e['modality']);e.setdefault('provenance_status','unverified');e.setdefault('cluster_id',None);e.setdefault('embedding_ref',None)
        # Gold labels and arbitrary exporter fields never enter model payloads.
    return events,facts,nodes

def validate(events,facts,nodes):
    ni={n['node_id']:n for n in nodes};fi={f['fact_id']:f for f in facts};ei={e['observation_id']:e for e in events}
    if len(ni)!=len(nodes) or len(fi)!=len(facts) or len(ei)!=len(events):raise ValueError('duplicate node/fact/observation IDs')
    if not events:raise ValueError('no DDPF observations')
    for e in events:
        if e['RE_level'] not in LEVELS:raise ValueError('invalid RE_level')
        for k in ('collection_start','collection_end','available_at'):require_date(e[k])
        if not e['collection_start']<=e['collection_end']<=e['available_at']:raise ValueError('invalid collection availability '+e['observation_id'])
        a,b=e.get('occurrence_start'),e.get('occurrence_end')
        if (a is None)!=(b is None) or a and (require_date(a)>require_date(b) or b>e['available_at']):raise ValueError('invalid occurrence interval')
        if not e['fact_ids'] or not set(e['fact_ids'])<=fi.keys():raise ValueError('missing facts '+e['observation_id'])
        if REF.search(json.dumps({k:e.get(k) for k in ('observation_id','source_ids','source_corpus','source_text')},ensure_ascii=False)):raise ValueError('reserved reference marker in DDPF source; segregate before running')
        for id in e['fact_ids']:
            f=fi[id]
            if f['subject'] not in ni or f['object'] not in ni:raise ValueError('orphan graph fact '+id)
            if not f.get('available_at') or require_date(f['available_at'])>e['available_at']:raise ValueError('future fact '+id)
            if REF.search(json.dumps(f,ensure_ascii=False)):raise ValueError('reserved reference marker in fact '+id)
    return ei,fi,ni

def verify_neo4j(root,events,facts):
    """Recover original hasPresence relationship identities, never case/report edges.
    The summary facts are retained unchanged; exact matches are separately audited.
    """
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv(root/'.env');driver=GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.getenv('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD']))
    audit=[];fm={f['fact_id']:f for f in facts}
    try:
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j'),default_access_mode='READ') as s:
            for e in events:
                source=e['source_corpus'];want=[fm[x] for x in e['fact_ids'] if fm[x]['predicate']=='hasPresence'];ids=list({f['subject'] for f in want})
                if not ids:continue
                q="MATCH (c)-[r:hasPresence]->(p:Country {id:$country}) WHERE c.id IN $ids AND r.source=$source RETURN c.id AS concept, id(r) AS relationship_id, r.date AS period, r.intensity AS intensity"
                rows=[dict(x) for x in s.run(q,country=e['country_id'],ids=ids,source=source)]
                for f in want:
                    ps=f['properties'].get('collection_periods',[]);matches=[r for r in rows if r['concept']==f['subject'] and str(r['period']) in [str(x) for x in ps]]
                    audit.append({'observation_id':e['observation_id'],'fact_id':f['fact_id'],'matches':matches,'status':'matching_relationships_found' if matches else 'missing','note':'internal IDs valid only for this restored DB; summary intensity may aggregate multiple relations'})
    finally:driver.close()
    return audit

# Exact notebook thresholds recovered from reconstruction_manifest.json (16 Nov 2026).
THRESHOLDS={'Journal_COVID':5,'Medical_COVID':8,'Social_COVID':10,'Journal_Monkeypox':5,'Medical_Monkeypox':3,'Social_Monkeypox':10}

def chronology_order(a,b):
    """Strictly ordered observation windows; occurrence is a distinct optional task."""
    if a.get('chronology_basis')=='collection' and b.get('chronology_basis')=='collection':
        return bool(a.get('collection_end') and b.get('collection_start') and a['collection_end']<b['collection_start'])
    return bool(a.get('occurrence_end') and b.get('occurrence_start') and a['occurrence_end']<b['occurrence_start'])

def graph_inventory(root,out):
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv(root/'.env');driver=GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.getenv('NEO4J_USERNAME','neo4j'),os.environ['NEO4J_PASSWORD']))
    # Known six source groups only; no isReported, WHO or reference data.
    q="""MATCH (c)-[r:hasPresence]->(n:Country)
    WHERE r.source IN $sources AND toInteger(r.intensity) >= $thresholds[r.source]
    RETURN r.source AS source, n.id AS country_id, n.wkgs_nameEn AS country_name,
           c.id AS concept_id, c.name AS concept_name, r.date AS relation_date,
           r.intensity AS intensity, id(r) AS relationship_id, properties(r) AS properties"""
    try:
        with driver.session(database=os.getenv('NEO4J_DATABASE','neo4j'),default_access_mode='READ') as session:
            rows=[dict(x) for x in session.run(q,sources=list(THRESHOLDS),thresholds=THRESHOLDS)]
            schema=[dict(x) for x in session.run("MATCH ()-[r:hasPresence]->(:Country) RETURN r.source AS source, keys(r) AS property_keys, count(*) AS n ORDER BY n DESC")]
    finally:driver.close()
    rows.sort(key=lambda x:(str(x['source']),str(x['country_id']),str(x['concept_id']),str(x['relation_date']),str(x['relationship_id'])))
    jsonl(out/'neo4j_notebook_threshold_rows.jsonl',rows)
    groups={(x['source'],x['country_id']) for x in rows if x['country_id']}
    audit={'query':q,'thresholds':THRESHOLDS,'rows':len(rows),'source_country_groups':len(groups),'by_source':dict(Counter(a for a,b in groups)),'relationship_schema':schema,'sha256':sha(out/'neo4j_notebook_threshold_rows.jsonl'),'note':'Notebook threshold cohort; source-country groups are aggregate observations, not proof of document/paragraph granularity. Other levels require independently identifiable extraction exports or notebook settings.'}
    dump(out/'neo4j_recovery_inventory.json',audit);return audit

def rehydrate(events,facts,audit):
    """Keep the original DDPF selected concepts; attach exact graph properties and periods.
    Do not substitute all threshold-matching concepts for the original top-x selection."""
    fm={f['fact_id']:f for f in facts};am={(r['observation_id'],r['fact_id']):r for r in audit};new=[]
    for e in events:
        ids=[];periods=[]
        for id in e['fact_ids']:
            old=fm[id];r=am.get((e['observation_id'],id));matches=(r or {}).get('matches',[])
            if old['predicate']!='hasPresence':new.append(old);ids.append(id);continue
            if not matches:raise ValueError('Missing original graph relationship for '+e['observation_id']+' '+id)
            # Preserve top-concept aggregate intensity and actual individual relationship values separately.
            dates=sorted({str(m['period']) for m in matches});periods.extend(dates)
            props=dict(old['properties'],collection_periods=dates,graph_relationships=sorted([{'relationship_id':str(m['relationship_id']),'date':str(m['period']),'intensity':m['intensity']} for m in matches],key=lambda m:m['relationship_id']))
            by_period=defaultdict(list)
            for m in matches:by_period[str(m['period'])].append(m['intensity'])
            props['period_intensities']={k:sorted(v,key=str) for k,v in sorted(by_period.items())}
            pr=dict(old['provenance'],graph_verified=True,aggregation='DDPF selected concept; exact matching hasPresence relationships retained')
            f=fact(old['subject'],old['predicate'],old['object'],props,pr,'observed_aggregate',max(interval(x)[1] for x in dates));new.append(f);ids.append(f['fact_id'])
        if periods:
            bounds=[interval(x) for x in periods];e['collection_start']=min(x[0] for x in bounds);e['collection_end']=max(x[1] for x in bounds)
            if e['availability_basis']=='collection_end_proxy':e['available_at']=e['collection_end']
            e['chronology_provenance']='Neo4j hasPresence.date on the original selected DDPF concepts'
        e['fact_ids']=ids
    return new


def prepare(root,out,args,cfg):
    out.mkdir(parents=True,exist_ok=True);inventory(root,out,args.hash_large)
    p=Path(args.observations) if args.observations else root/LEGACY
    if args.export_dir:events,facts,nodes=canonical_adapter(args.export_dir);origin={'export_dir':str(args.export_dir),'hashes':{n:sha(Path(args.export_dir)/n) for n in ('events.jsonl','edges.jsonl','nodes.jsonl')}}
    else:
        if args.level!='sentence' and not args.observations:raise ValueError('Other granularities require --observations or --export-dir with independently verified DDPF extraction level; see inventory.json and neo4j_recovery_inventory.json.')
        if not p.is_file():raise FileNotFoundError(f'{p}; provide --observations or --export-dir. See inventory.json.')
        h=sha(p);events,facts,nodes=legacy_adapter(read_jsonl(p),h,args.level);origin={'file':str(p),'sha256':h,'exact_legacy_file':h==LEGACY_SHA}
    validate(events,facts,nodes)
    for e in events:
        e['chronology_basis']=cfg.get('chronology_basis','collection')
        e.setdefault('chronology_provenance','original DDPF relation dates or canonical collection interval')
    graph_audit=None;recovery=None
    if args.neo4j and not args.skip_neo4j_audit:
        if not args.export_dir:
            recovery=graph_inventory(root,out)
        graph_audit=verify_neo4j(root,events,facts)
        jsonl(out/'neo4j_relationship_audit.jsonl',graph_audit)
        if not args.export_dir:
            facts=rehydrate(events,facts,graph_audit)
    validate(events,facts,nodes)
    levels=Counter(e['RE_level'] for e in events)
    if len(levels)!=1:raise ValueError('run each RE granularity separately')
    coverage=Counter((e['source_corpus'],e['collection_start'][:7],e['modality'],e['country_id']) for e in events)
    csvwrite(out/'coverage.csv',[{'source_corpus':a,'month':b,'modality':c,'country_id':d,'observations':n} for (a,b,c,d),n in sorted(coverage.items())])
    csvwrite(out/'dataset_manifest.csv',[{k:e.get(k) for k in ('observation_id','source_corpus','source_ids','collection_start','collection_end','available_at','availability_basis','RE_level','country_id','extraction_config_hash','provenance_status')} for e in events])
    for name,rows in [('events',events),('edges',facts),('nodes',nodes)]:jsonl(out/(name+'.jsonl'),rows)
    status={'status':'PREPARED','origin':origin,'observations':len(events),'facts':len(facts),'nodes':len(nodes),'levels':dict(levels),'historical_expected':cfg['history_expected'],'unrecovered_levels':{k:v for k,v in cfg['history_expected'].items() if levels.get(k,0)!=v},'availability_proxy_count':sum(e['availability_basis']!='verified_publication' for e in events),'missing_source_ids':sum(not e['source_ids'] for e in events),'missing_source_text':sum(not e.get('source_text') for e in events),'known_occurrence_intervals':sum(bool(e.get('occurrence_start')) for e in events),'note':'WHO/ECDC marker scan is not a full provenance or near-duplicate audit. No benchmark or official events read.'}
    if graph_audit is not None:status['neo4j']={'checked':len(graph_audit),'matched':sum(x['status']=='matching_relationships_found' for x in graph_audit),'rehydrated':not bool(args.export_dir)}
    if recovery is not None:status['graph_recovery']=recovery
    status['chronology_basis']=cfg.get('chronology_basis','collection')
    status['source_text_required_for_observation_analysis']=False
    status['hashes']={n:sha(out/n) for n in ('events.jsonl','edges.jsonl','nodes.jsonl')};dump(out/'export_audit.json',status);return status
