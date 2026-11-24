import unittest,tempfile,json,sys,os
from pathlib import Path
from datetime import date,timedelta
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'/'realdata'))
from reviewer_v8 import core,temporal,documentary as d

class FakeClient:
    def prepare(self,m):return {'name':m,'digest':'fake'}
    def call(self,model,task,payload,schema,validate,budget=512):
        if 'Extract' in task:
            a={'events':[]}
        else:
            a={'relation':'documented_continuation','reason':'mock only','evidence_ids':[payload['earlier_candidate']['id'],payload['later_candidate']['id']]}
        validate(a);return a
class FakeSemantic:
    def context(self,e):return [{'source':'BioPropaPhenKG','cui':'C5203670','test_value':e['id']}]
    def disjoint(self,a,b):return False

class TestV8(unittest.TestCase):
    def test_fixed_macro_f1(self):
        x=core.multiclass(['flat','increase','increase'],['flat','revision',None],['flat','increase'])
        self.assertEqual(x['macro_f1'],.5);self.assertAlmostEqual(x['accuracy'],1/3)
    def test_zero_positive_f1(self):
        x=core.binary([1,1,0],[0,0,0]);self.assertEqual(x['f1'],0);self.assertIsNone(x['precision'])
    def test_date_normalization(self):
        self.assertEqual(core.iso('21 May 2022'),'2022-05-21');self.assertIsNone(core.iso('May 2022'))
    def test_history_and_no_leakage(self):
        rows=[]
        for i,v in enumerate([1,2,2,3,3,2]):
            day=str(date(2023,12,4)+timedelta(days=i*7));rows.append({'disease_id':'C5203670','country_id':'wkg:1','disease':'COVID-19','country_name':'X','week_start':day,'episode_id':str(i),'reported_value':str(v)})
        ex,a=temporal.build_examples(rows);self.assertEqual(len(ex),2);self.assertEqual(ex[0]['history'],[1,2,2,3]);self.assertEqual(ex[0]['label'],'flat');self.assertEqual(ex[1]['label'],'revision');self.assertEqual(a['excluded']['insufficient_four_snapshot_history'],3)
    def test_legacy_8b(self):
        j=core.load(ROOT/'data/reviewer_v8/llm_v6_summary.json');rows=temporal.harmonize_legacy(j);x=next(r for r in rows if r['model']=='qwen3:8b' and r['method']=='M3_llm_only' and r['disease']=='ALL')
        self.assertAlmostEqual(x['fixed_two_class_macro_f1'],(2*26/61+2*4/18)/2);self.assertEqual(x['predicted_revision'],1)
    def test_profile_index(self):
        p=temporal.profile_records({'C5203670':{'wkg:123':{'concept_counts':{'C0123456':2}}}});self.assertIn(('C5203670','wkg:123'),p)
    def test_forest_structure(self):
        e=[{'id':str(i),'date':str(date(2020,1,1)+timedelta(days=i))} for i in range(29)];pairs=d.candidates(e);self.assertEqual(len(pairs),106);preds={k:'documented_continuation' for k in pairs};f=d.forest(e,preds,pairs);self.assertEqual(f['n_edges'],28);self.assertEqual(f['n_roots'],1);self.assertEqual(f['max_indegree'],1)
    def test_identity_separate(self):
        e=[{'id':'a','date':'2020-01-01'},{'id':'b','date':'2020-01-02'}];f=d.forest(e,{('a','b'):'same_event'},[('a','b')]);self.assertEqual(f['n_edges'],0);self.assertEqual(len(f['identity_relations']),1)
    def test_guard_rejects_evidence(self):
        a={'id':'a','date':'2020-01-01','evidence':['x']};b={'id':'b','date':'2020-01-02','evidence':['y']};g,why=d.guard(a,b,{'relation':'documented_continuation','evidence_ids':['a']},FakeSemantic());self.assertEqual(g,'unknown');self.assertIn('missing_card_evidence',why)
    def test_unlabelled_not_false(self):
        f={'edges':[['a','b'],['b','c']]};x=d.forest_score(f,{('a','b'):'documented_continuation'});self.assertEqual(x['matched'],1);self.assertEqual(x['unlabelled_edges'],1);self.assertEqual(x['known_negative_edges'],0)
    def test_ambiguous_alignment(self):
        a={'id':'a','date':'2020-01-01','doc':'DON233','type':'notification'};g=[dict(a,id='x'),dict(a,id='y')];m,r=d.align(g,[a]);self.assertEqual(m,{});self.assertEqual(r['matched_slots'],1)
    def test_label_parser(self):
        self.assertEqual(d.ref_label({'gold_relation':'documented_continuation'}),'documented_continuation');self.assertEqual(d.ref_label({'is_positive':False}),'core_negative')
    def test_blind_full_mock_run(self):
        with tempfile.TemporaryDirectory() as z:
            root=Path(z);base=root/d.BASE;base.mkdir(parents=True);src=root/d.OUT;src.mkdir(parents=True);out=root/'review';cards=[]
            for i in range(29):
                doc='DON'+str(233+i%8);dt=str(date(2020,1,1)+timedelta(days=i));cards.append({'id':f'W{i:02d}','date':dt,'type':'notification','title':'Event '+str(i),'source_id':doc,'evidence_text':'A reported event was described.' ,'concept_ids':['C5203670']});(src/(doc+'.html')).write_text('<html><article>'+('A reported event was described. '*30)+'</article></html>')
            core.dump(base/'silver_events.json',cards);pairs=d.candidates([d.event(x) for x in cards])[:39];core.dump(base/'silver_pairs.json',[{'a':a,'b':b,'label':'documented_continuation' if i<21 else 'same_event' if i==21 else 'related_only'} for i,(a,b) in enumerate(pairs)])
            p=d.run(root,out,None,['mock'],None,preflight=True);self.assertEqual(p['schedule'],106)
            rows=core.csvread(out/'human_review/reviewer_A.csv');self.assertTrue(all(not r['label'] for r in rows));self.assertEqual(len(rows),106)
            r=d.run(root,out,FakeClient(),['mock'],FakeSemantic(),embedding_model='');self.assertEqual(r['status'],'COMPLETE');f=core.load(out/'forests/mock__semantic_guard.json');self.assertEqual(f['n_nodes'],29);self.assertEqual(f['candidate_coverage'],1);self.assertTrue((out/'end_to_end/mock/metrics.json').exists())
            self.assertEqual(d.run(root,out,None,[],None,review_only=True)['status'],'HUMAN_REVIEW_PENDING')
    def test_mock_ollama_cache(self):
        class R:
            def __init__(self,v):self.v=v
            def raise_for_status(self):pass
            def json(self):return self.v
        with tempfile.TemporaryDirectory() as z:
            c=core.Client(Path(z));schema=core.schema_enum('prediction',core.DIR)
            with patch('reviewer_v8.core.requests.get',return_value=R({'models':[{'name':'gemma3:4b','digest':'x'}]})),patch('reviewer_v8.core.requests.post',return_value=R({'message':{'content':'{"prediction":"flat"}'},'done_reason':'stop'})) as post:
                self.assertEqual(c.call('gemma3:4b','task',{},schema,lambda v:None),{'prediction':'flat'});c.call('gemma3:4b','task',{},schema,lambda v:None);self.assertEqual(post.call_count,1);self.assertNotIn('think',post.call_args.kwargs['json'])
            c.close()
if __name__=='__main__':unittest.main()
