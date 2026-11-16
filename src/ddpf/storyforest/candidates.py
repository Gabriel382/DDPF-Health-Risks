"""Small-scale bounded candidate retrieval; replace with indexed KG retrieval for scale."""
from __future__ import annotations
from datetime import date
from difflib import SequenceMatcher
from .models import Event

def overlap(a: list[str], b: list[str]) -> float:
    aa=set(a);bb=set(b)
    return len(aa&bb)/len(aa|bb) if aa and bb else 0.0

def similarity(a: Event,b:Event) -> float:
    semantic=overlap(a.concept_ids,b.concept_ids)
    textual=SequenceMatcher(None,a.description.lower(),b.description.lower()).ratio()
    spatial=1.0 if a.location and a.location.casefold()==b.location.casefold() else 0.0
    days=abs((date.fromisoformat(a.occurred_at)-date.fromisoformat(b.occurred_at)).days)
    temporal=1/(1+days/14)
    return round(.45*semantic+.25*textual+.15*spatial+.15*temporal,6)

def retrieve(event: Event, existing: list[Event], k:int=8) -> list[tuple[Event,float]]:
    candidates=[(x,similarity(x,event)) for x in existing
                if x.event_id!=event.event_id and x.published_at<=event.published_at]
    return sorted(candidates,key=lambda q:(-q[1],q[0].event_id))[:k]
