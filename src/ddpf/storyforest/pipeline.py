"""Bounded incremental processing and conservative story linking."""
from __future__ import annotations
from dataclasses import dataclass
from .models import Event,Link
from .candidates import retrieve
from .classifier import classify
from .ontology import OntologyGuard
from .storage import Store

METHODS={'temporal','kg_rules','llm_only','llm_kg','full'}
@dataclass
class ProcessResult:
    event_id:str
    created:bool
    decision:Link|None
    candidates:int

def process_event(store:Store,event:Event,method:str='full',llm=None,
                  guard:OntologyGuard|None=None,top_k:int=8,
                  history_limit:int=2000)->ProcessResult:
    if method not in METHODS:raise ValueError('Invalid method '+method)
    if store.has_event(event.event_id):
        return ProcessResult(event.event_id,False,None,0)
    existing=store.recent_events(event.published_at,limit=history_limit)
    candidates=retrieve(event,existing,k=top_k)
    validator=guard or OntologyGuard()
    decisions=[]
    for prior,_score in candidates:
        decision=classify(prior,event,method,llm,validator)
        decisions.append(decision)
    # Store decisions, including rejected/unknown, for audit. Linking in the
    # actual forest will only use documented_continuation.
    inserted=store.insert_event(event)
    for decision in decisions:store.add_link(decision)
    strong=[x for x in decisions if x.relation in {'same_event','documented_continuation'} and x.status=='accepted']
    top=max(strong,key=lambda x:x.score,default=None)
    return ProcessResult(event.event_id,inserted,top,len(candidates))
