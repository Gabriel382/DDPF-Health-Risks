import unittest,tempfile,json,os
from pathlib import Path
from scripts.realdata.reviewer_v8_ddpf.data import *
from scripts.realdata.reviewer_v8_ddpf.experiment import *
from scripts.realdata.reviewer_v8_ddpf.evaluation import *

CFG=load(Path(__file__).resolve().parents[2]/'configs/reviewer_v8_ddpf/default.json')
def make(n=4,known=False):
    rows=[{'observation_id':f'Journal_COVID::wkg:{i}','country_name':f'"Country {i}"','country_id':f'wkg:{i}','source':'Journal_COVID','disease_label':'0','top_concepts':[{'id':'C0000001','name':'Fever','intensity':3},{'id':f'C{i+100:07}','name':f'Symptom {i}','intensity':2}], 'relation_dates':['2019-11' if i<2 else '2019-12'],'provenance':{'entity_granularity':'country_by_source','source':'BioPropaPhenKG-COVIDvsMPOX.dump','selection':'hasPresence with notebook original source-specific thresholds','aggregation':'by country and source; NOT by document'}} for i in range(n)]
    es,fs,ns=legacy_adapter(rows,'abc','sentence')
    if known:
        for i,e in enumerate(es):e['occurrence_start']=e['occurrence_end']=f'2019-11-{i+1:02}'
    return es,fs,ns

def ans(rel='continuation',ids=()):return {'relation':rel,'subtype':'other','score':.9,'evidence_ids':list(ids),'rationale':'test','abstain_reason':''}
class TestV8DDPF(unittest.TestCase):
    def test_exact_legacy_shape(self):
        e,f,n=make();self.assertEqual(len(e),4);self.assertEqual(len(f),8);self.assertIsNone(e[0]['occurrence_start']);self.assertEqual(e[0]['available_at'],'2019-11-30');self.assertNotIn('disease_label',e[0]);validate(e,f,n)
    def test_month(self):self.assertEqual(interval('2020-02'),('2020-02-01','2020-02-29'))
    def test_reference_rejected(self):
        e,f,n=make();e[0]['source_ids']=['DON233'];self.assertRaises(ValueError,validate,e,f,n)
    def test_future_rejected(self):
        e,f,n=make();f[0]['available_at']='2026-01-01';self.assertRaises(ValueError,validate,e,f,n)
    def test_orphan(self):
        e,f,n=make();f[0]['subject']='missing';self.assertRaises(ValueError,validate,e,f,n)
    def test_guard_chronology(self):
        e,f,n=make();a,b=e[:2];v,r=guard(a,b,ans(ids=[a['fact_ids'][0],b['fact_ids'][0]]),a['fact_ids'],b['fact_ids']);self.assertEqual(v['relation'],'insufficient_evidence');self.assertIn('occurrence_order_not_established',r)
    def test_guard_ids(self):
        e,f,n=make(known=True);a,b=e[:2];v,r=guard(a,b,ans(ids=['fake']),a['fact_ids'],b['fact_ids']);self.assertEqual(v['relation'],'insufficient_evidence')
    def test_guard_accepts_admissible(self):
        e,f,n=make(known=True);a,b=e[:2];v,r=guard(a,b,ans(ids=[a['fact_ids'][0],b['fact_ids'][0]]),a['fact_ids'],b['fact_ids']);self.assertEqual(v['relation'],'continuation');self.assertFalse(r)
    def test_prefix_retrieval(self):
        ei,fi,ni=validate(*make());ids,mat,txt,meta=representations(ei,fi,ni,CFG);sel,allp,c=candidates(ei,fi,ids,mat,txt,CFG);self.assertEqual(len(allp),6);self.assertEqual(len(sel),6);self.assertTrue(all(ei[p['a']]['available_at']<=p['as_of']==ei[p['b']]['available_at'] for p in sel));self.assertTrue(all(not p['known_occurrence_order'] for p in sel))
        early={i:e for i,e in ei.items() if e['available_at']=='2019-11-30'};ii,mm,tt,me=representations(early,fi,ni,CFG);_,ap,cl=candidates(early,fi,ii,mm,tt,CFG);self.assertEqual([x for x in allp if x['as_of']=='2019-11-30'],ap)
    def test_shuffle_no_future(self):
        ei,fi,ni=validate(*make());m=donors(ei,42);self.assertTrue(all(i!=j and ei[i]['available_at']==ei[j]['available_at'] for i,j in m.items()))
    def test_alias(self):
        p={'metadata':{'country':'China','collection_interval':['2020-01-01','2020-02-01']},'facts':[{'subject':'C5203670','subject_label':'COVID-19','fact_id':'F1'}]};z=anonymize(p,42);self.assertNotIn('China',json.dumps(z));self.assertNotIn('COVID-19',json.dumps(z));self.assertEqual(z['facts'][0]['fact_id'],'F1')
    def test_payload_no_gold(self):
        ei,fi,ni=validate(*make());e=next(iter(ei.values()));e['disease_label']='secret_label';x,ids=block(e,fi,ni,CFG,'C3',{});self.assertNotIn('secret_label',json.dumps(x));self.assertNotIn('Journal_COVID',json.dumps(x))
    def test_forest_dag(self):
        ei,fi,ni=validate(*make(known=True));a,b,c,d=sorted(ei);rs=[{'a':a,'b':c,'pair_id':'p1','validated':ans()},{'a':b,'b':c,'pair_id':'p2','validated':dict(ans(),score=.8)}];g=graphs(ei,rs,.65);self.assertEqual(len(g['dag']),2);self.assertEqual(len(g['forest']),1);self.assertEqual(g['forest'][0]['a'],a);self.assertEqual(len(g['roots']),3)
    def test_unknown_not_negative(self):
        ref={('a','b'):{'label':'continuation'},('a','c'):{'label':'insufficient_evidence'}};x=edge_metrics(ref,[{'a':'a','b':'c'}]);self.assertEqual(x['judged_fp'],0);self.assertEqual(x['predicted_unjudged'],1);self.assertEqual(x['judged_fn'],1)
    def test_gold_sample_includes_unretrieved(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';gold=Path(d)/'gold';e,f,n=make();jsonl(out/'events.jsonl',e);z=sample_review(out,gold,6,42);self.assertEqual(z['pairs'],6);self.assertEqual(len(csvread(gold/'annotator_A.csv')),6);self.assertRaises(FileExistsError,sample_review,out,gold,6,42)
    def test_mock_six_condition_run(self):
        class Mock:
            def call(self,model,task,payload,schema,validate,budget):
                x=ans('related_only');validate(x);return x
        ei,fi,ni=validate(*make(known=True));cfg=dict(CFG,models=['mock'],shuffle_seeds=[42],bootstrap=10)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);plan=create_plan(out,ei,fi,ni,cfg);pairs=read_jsonl(out/'candidates.jsonl');self.assertEqual(plan['n_selected_pairs'],6);z=run_models(out,out,ei,fi,ni,cfg,Mock(),pairs,{},'exploratory');self.assertIn('mock__C3',z);self.assertIn('mock__C5_seed42',z);self.assertIn('mock__C2_guard',z);self.assertEqual(z['mock__C3']['technical_missing'],0)
    def test_ontology_inference(self):
        with tempfile.TemporaryDirectory() as d:
            from rdflib import Graph,URIRef
            from rdflib.namespace import RDF,RDFS,OWL
            p=Path(d)/'ontology.owl';g=Graph();a,b,c=[URIRef('urn:test#'+x) for x in ('C0000001','B','C')];g.add((a,RDF.type,OWL.Class));g.add((a,RDFS.subClassOf,b));g.add((b,RDFS.subClassOf,c));g.serialize(p,format='xml');s=StaticContext(Path(d),CFG,False,[str(p)]);r=s.get('C0000001');self.assertTrue(any(x['kind']=='inferred_rdfs' and x['object']==str(c) for x in r));s.close()
    def test_evaluate_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';gold=Path(d)/'gold';out.mkdir();ei,fi,ni=validate(*make(known=True));jsonl(out/'events.jsonl',list(ei.values()));create_plan(out,ei,fi,ni,CFG);pairs=read_jsonl(out/'candidates.jsonl');name='B0';vs=write_variant(out,name,ei,baseline_rows(ei,pairs,CFG)["B0_no_links"],CFG);dump(out/'run_manifest.json',{'variants':{name:vs},'config':CFG});sample_review(out,gold,6,42)
            for f in ('annotator_A.csv','annotator_B.csv','adjudication.csv'):
                rows=csvread(gold/f)
                for i,r in enumerate(rows):r.update(label='continuation' if i%2 else 'unrelated',evidence_ids='test',justification='fixture',block_id=str(i%2))
                csvwrite(gold/f,rows)
            z=evaluate(out,gold);self.assertEqual(z['variants'],1);self.assertEqual(z['agreement']['raw_agreement'],1.)
if __name__=='__main__':unittest.main()
